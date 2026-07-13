"""Issue #10a: REST session surface -- list, fork, delete clears DB rows."""

from __future__ import annotations

import httpx
from httpx import ASGITransport

from koboi.config import Config
from koboi.memory_sqlite import SQLiteMemory
from koboi.server.app import create_app
from tests.conftest import MockClient, make_mock_response


def _cfg(db_path: str, backend: str = "sqlite") -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
            "memory": {"backend": backend, "db_path": db_path},
            "sandbox": {"backend": "passthrough"},
            "server": {"auth_required": False},
        },
        validate=True,
    )


def _app(cfg: Config):
    factory = lambda: MockClient([make_mock_response(content="ok")])  # noqa: E731
    return create_app(cfg, client_factory=factory, enable_cors=False)


class TestSessionSurface:
    async def test_list_fork_delete(self, tmp_path):
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(_cfg(db))), base_url="http://t") as c:
            r = await c.post("/v1/sessions")
            assert r.status_code == 201
            sid = r.json()["session_id"]

            # seed a message directly in the DB for that session
            mem = SQLiteMemory(db_path=db, session_id=sid)
            mem.add_user_message("hello fork me")
            mem.close()

            # LIST includes it
            r = await c.get("/v1/sessions")
            assert r.status_code == 200
            sids = [s["session_id"] for s in r.json()["sessions"]]
            assert sid in sids

            # FORK copies messages to a new session
            r = await c.post(f"/v1/sessions/{sid}/fork")
            assert r.status_code == 201, r.text
            new_sid = r.json()["session_id"]
            assert new_sid != sid
            forked = SQLiteMemory.get_session_messages(db, new_sid)
            assert any("hello fork me" in str(m.get("content", "")) for m in forked)

            # DELETE clears persisted DB rows
            r = await c.delete(f"/v1/sessions/{sid}")
            assert r.status_code == 200
            assert SQLiteMemory.get_session_messages(db, sid) == []

    async def test_list_empty_for_non_sqlite(self, tmp_path):
        cfg = _cfg(str(tmp_path / "d.db"), backend="in_memory")
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(cfg)), base_url="http://t") as c:
            r = await c.get("/v1/sessions")
            assert r.status_code == 200
            assert r.json()["sessions"] == []

    async def test_fork_404_for_unknown(self, tmp_path):
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(_cfg(db))), base_url="http://t") as c:
            r = await c.post("/v1/sessions/nonexistent/fork")
            assert r.status_code == 404

    async def test_fork_rolls_back_on_pool_failure(self, tmp_path):
        # C1: if pool.get_or_create fails (any exception) AFTER fork_session
        # committed, the forked DB/owner rows must be rolled back (no ghost).
        db = str(tmp_path / "d.db")
        factory = lambda: MockClient([make_mock_response(content="ok")])  # noqa: E731
        app = create_app(_cfg(db), client_factory=factory, enable_cors=False)
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/v1/sessions")  # original session
            sid = r.json()["session_id"]
            mem = SQLiteMemory(db_path=db, session_id=sid)
            mem.add_user_message("seed")
            mem.close()

            # force get_or_create to fail for the fork's new session
            async def _fail(_session_id):
                raise RuntimeError("forced failure")

            app.state.pool.get_or_create = _fail
            r = await c.post(f"/v1/sessions/{sid}/fork")
            assert r.status_code == 500  # fork_failed, rolled back
            sids = [s["session_id"] for s in SQLiteMemory.list_sessions(db)]
            assert sids == [sid]  # only the original; no ghost fork
