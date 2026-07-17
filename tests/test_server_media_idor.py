"""Issue #69: /v1/media/{generate,jobs} IDOR — body.session_id bypasses _check_owner.

Both media routes compute ``session_id = body.session_id or header_sid or pool.new_session_id()``
and then gate the ownership check on ``if header_sid is not None:`` ONLY. A caller who
supplies another tenant's ``session_id`` in the JSON body (the real
``MediaGenerateRequest.session_id`` field) WITHOUT the ``X-Session-Id`` header skips
``_check_owner`` entirely and runs media generation in the victim's pooled agent
(acquires their session lock, mutates their state, billed generation in their context).

These tests pin the body-or-header ownership gate: under auth, a cross-tenant
``body.session_id`` is denied (403 ``forbidden``) on both the sync and async media
routes. Same class of IDOR as closed #52; the reference correct pattern lives in
``submit_job`` (``app.py`` ~1599-1604, gates on ``body.session_id``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.server import create_app  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402


def _config(storage_dir: str) -> Config:
    """Auth-on server config with mock media (image) enabled."""
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test",
                "base_url": "http://localhost:8080/v1",
            },
            "memory": {"backend": "in_memory"},
            "sandbox": {"backend": "passthrough"},
            "server": {"auth_required": True},
            "media": {"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": storage_dir}},
        },
        validate=True,
    )


def _app(cfg: Config):
    return create_app(
        cfg,
        client_factory=lambda: MockClient([make_mock_response(content="ok")]),
        enable_cors=False,
        api_keys=["key-alice", "key-bob"],
    )


class TestMediaRouteIDOR:
    """Issue #69: _check_owner must fire on body.session_id, not only the header."""

    async def test_body_session_id_bypasses_check_owner_sync(self, tmp_path):
        # Alice owns a session; Bob sends her session_id in the BODY (no header).
        app = _app(_config(str(tmp_path / "art")))
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            alice = await c.post("/v1/sessions", headers={"Authorization": "Bearer key-alice"})
            assert alice.status_code == 201, alice.text
            alice_sid = alice.json()["session_id"]

            r = await c.post(
                "/v1/media/generate",
                json={"modality": "image", "prompt": "x", "session_id": alice_sid},
                headers={"Authorization": "Bearer key-bob"},  # NO X-Session-Id header
            )
        # Today: 200 (IDOR — generation runs in Alice's agent). Fix: 403 forbidden.
        assert r.status_code == 403, f"expected 403 for cross-tenant body.session_id, got {r.status_code}"
        assert r.json()["error"]["code"] == "forbidden"

    async def test_body_session_id_bypasses_check_owner_async(self, tmp_path):
        # Same IDOR via the async media-jobs route.
        app = _app(_config(str(tmp_path / "art")))
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            alice = await c.post("/v1/sessions", headers={"Authorization": "Bearer key-alice"})
            assert alice.status_code == 201, alice.text
            alice_sid = alice.json()["session_id"]

            r = await c.post(
                "/v1/media/jobs",
                json={"modality": "image", "prompt": "x", "session_id": alice_sid},
                headers={"Authorization": "Bearer key-bob"},  # NO X-Session-Id header
            )
        # Today: 202 (IDOR — job accepted against Alice's session). Fix: 403 forbidden.
        assert r.status_code == 403, f"expected 403 for cross-tenant body.session_id, got {r.status_code}"
        assert r.json()["error"]["code"] == "forbidden"

    async def test_own_body_session_id_allowed_sync(self, tmp_path):
        # Regression guard: a caller who owns the session may pass it in the BODY.
        app = _app(_config(str(tmp_path / "art")))
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            alice = await c.post("/v1/sessions", headers={"Authorization": "Bearer key-alice"})
            assert alice.status_code == 201, alice.text
            alice_sid = alice.json()["session_id"]

            r = await c.post(
                "/v1/media/generate",
                json={"modality": "image", "prompt": "x", "session_id": alice_sid},
                headers={"Authorization": "Bearer key-alice"},  # owner, no header
            )
        assert r.status_code == 200, f"owner body.session_id should pass, got {r.status_code}"

    async def test_malformed_body_session_id_rejected(self, tmp_path):
        # Covers the new body-path is_safe_session_id guard (defense-in-depth on a
        # previously-unvalidated body field). "invalid@id" fails the segment regex.
        app = _app(_config(str(tmp_path / "art")))
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/v1/media/generate",
                json={"modality": "image", "prompt": "x", "session_id": "invalid@id"},
                headers={"Authorization": "Bearer key-alice"},  # NO X-Session-Id header
            )
        assert r.status_code == 400, f"expected 400 for malformed body.session_id, got {r.status_code}"
        assert r.json()["error"]["code"] == "bad_request"
