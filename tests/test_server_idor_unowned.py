"""Issue #52: unowned-session fail-open → cross-tenant IDOR.

A session that has persisted conversation history but NO owner row (e.g. created
via the CLI, or pre-existing before auth was enabled, or orphaned by a partial
eviction) is reachable by ANY authenticated caller via ``X-Session-Id``. Today
``_check_owner`` only blocks a session whose owner is set AND differs from the
caller; an unowned session is allowed to everyone, then auto-claimed by whoever
touches it first — so an attacker both reads the victim's history and steals
ownership.

These tests pin the history-aware fail-closed gate: under auth, an unowned
session WITH history is denied (403, no ownership transfer); dev mode and
genuinely-new (no-history) sessions are unaffected.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.memory_sqlite import SQLiteMemory  # noqa: E402
from koboi.server import create_app  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402


def _config(**overrides) -> Config:
    """Server-test config mirroring tests/test_server_app.py but sqlite-backed."""
    cfg = {
        "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "base_url": "http://localhost:8080/v1",
        },
        "memory": {"backend": "in_memory"},
        "sandbox": {"backend": "restricted"},
        "server": {"auth_required": False},
    }
    cfg.update(overrides)
    return Config.from_dict(cfg, validate=True)


def _seed_victim(db_path: str, session_id: str) -> None:
    """Write persisted messages for ``session_id`` with NO owner row.

    SQLiteMemory writes/commits into the ``messages`` + ``sessions`` tables but
    never touches the ``session_owners`` sidecar, so the victim has full history
    yet is unowned — exactly the IDOR precondition.
    """
    mem = SQLiteMemory(db_path=db_path, session_id=session_id)
    mem.add_user_message("secret corporate data")
    mem.close()


async def _chat_status(client: httpx.AsyncClient, sid: str, token: str | None) -> int:
    headers = {"X-Session-Id": sid}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    async with client.stream("POST", "/v1/chat/stream", json={"message": "hi"}, headers=headers) as r:
        await r.aread()
        return r.status_code


class TestIDORUnownedSession:
    """Issue #52: _check_owner must fail closed for unowned-with-history."""

    async def test_unowned_session_with_history_denied_under_auth(self, tmp_path):
        db = tmp_path / "mem.db"
        victim_sid = "victim-session-1"
        _seed_victim(str(db), victim_sid)
        app = create_app(
            _config(memory={"backend": "sqlite", "db_path": str(db)}),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
            api_keys=["key-alice", "key-bob"],
        )
        async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
            status = await _chat_status(c, victim_sid, "key-bob")
        # Today: 200 + ownership stolen by the attacker (key-bob). Fix: 403, no claim.
        assert status == 403, f"expected 403 for unowned-with-history, got {status}"
        assert app.state.ownership.get_owner(victim_sid) is None, "attacker stole the unowned session"

    async def test_dev_mode_unowned_session_allowed(self, tmp_path):
        """Regression guard: dev mode (auth off) must keep current behavior."""
        db = tmp_path / "mem.db"
        victim_sid = "victim-session-2"
        _seed_victim(str(db), victim_sid)
        app = create_app(
            _config(memory={"backend": "sqlite", "db_path": str(db)}),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
            status = await _chat_status(c, victim_sid, None)
        assert status == 200, f"dev mode should allow unowned session, got {status}"

    async def test_new_session_under_auth_is_claimed(self, tmp_path):
        """Regression guard: a genuinely-new (no-history) session is still claimable."""
        db = tmp_path / "mem.db"
        fresh_sid = "fresh-session-3"
        app = create_app(
            _config(memory={"backend": "sqlite", "db_path": str(db)}),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
            api_keys=["key-alice"],
        )
        async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
            status = await _chat_status(c, fresh_sid, "key-alice")
        assert status == 200, f"new session should be claimed, got {status}"
        assert app.state.ownership.get_owner(fresh_sid) is not None, "new session was not claimed"
