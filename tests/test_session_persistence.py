"""Regression: a persisted session must be retrievable after a serve restart.

``get_session`` used to check only the in-memory agent pool, so a session that had
persisted to SQLite 404'd ("session not found") after the server restarted with a
fresh pool on the same DB. That breaks koboi-range suspend/resume -- the Outrider
``dismount``->``remount`` cycle restores ``koboi_memory.db`` and restarts
``koboi serve``, but the resumed session was unreachable via GET /v1/sessions/{id}.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
import httpx
from httpx import ASGITransport

from koboi.config import Config
from koboi.server import create_app
from tests.conftest import MockClient, make_mock_response


def _cfg(db_path: str) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test",
                "base_url": "http://localhost:8080/v1",
            },
            "memory": {"backend": "sqlite", "db_path": db_path},
            "sandbox": {"backend": "restricted"},
            "server": {"auth_required": False},
        },
        validate=True,
    )


def _app(cfg: Config):
    return create_app(
        cfg,
        client_factory=lambda: MockClient([make_mock_response(content="hello")]),
        enable_cors=False,
    )


@pytest.mark.asyncio
async def test_get_session_survives_serve_restart(tmp_path):
    db = str(tmp_path / "mem.db")
    cfg = _cfg(db)

    # Boot #1: create a session and confirm it resolves while the pool is warm.
    app1 = _app(cfg)
    async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app1)) as c1:
        sid = (await c1.post("/v1/sessions")).json()["session_id"]
        assert (await c1.get(f"/v1/sessions/{sid}")).status_code == 200

    # Boot #2: a fresh server process (new in-memory pool) on the SAME SQLite DB.
    # The persisted session must still resolve -- previously 404 ("session not found").
    app2 = _app(cfg)
    async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app2)) as c2:
        r = await c2.get(f"/v1/sessions/{sid}")
        assert r.status_code == 200, r.text
        assert r.json()["session_id"] == sid

    # A truly-unknown session must still 404 (the fix must not over-reach).
    app3 = _app(cfg)
    async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app3)) as c3:
        unknown = "ffffffffffffffffffffffffffffffff"
        assert (await c3.get(f"/v1/sessions/{unknown}")).status_code == 404
