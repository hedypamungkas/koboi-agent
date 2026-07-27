"""Regression: a persisted session must be retrievable after a serve restart.

``get_session`` used to check only the in-memory agent pool, so a session that had
persisted to SQLite 404'd ("session not found") after the server restarted with a
fresh pool on the same DB. That breaks koboi-range suspend/resume -- the Outrider
``dismount``->``remount`` cycle restores ``koboi_memory.db`` and restarts
``koboi serve``, but the resumed session was unreachable via GET /v1/sessions/{id}.

These tests also pin the security-relevant ordering of the fix: ``_check_owner`` runs
*before* the on-demand rehydrate, so a non-owner is 403'd before an agent is built
(otherwise a denied GET on a persisted session would materialize an agent -- and at
pool cap evict a legitimate tenant's warm agent).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.memory_sqlite import SQLiteMemory  # noqa: E402
from koboi.server import create_app  # noqa: E402
from koboi.server.pool import PoolFull  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402


def _cfg(db_path: str, *, auth: bool = False) -> Config:
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
            "server": {"auth_required": auth},
        },
        validate=True,
    )


def _app(cfg: Config, *, api_keys: list[str] | None = None):
    return create_app(
        cfg,
        client_factory=lambda: MockClient([make_mock_response(content="hello")]),
        enable_cors=False,
        api_keys=api_keys,
    )


async def test_get_session_survives_serve_restart(tmp_path):
    db = str(tmp_path / "mem.db")
    cfg = _cfg(db)

    # Boot #1: create a session, seed a persisted message, confirm it resolves warm.
    app1 = _app(cfg)
    async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app1)) as c1:
        sid = (await c1.post("/v1/sessions")).json()["session_id"]
        mem = SQLiteMemory(db_path=db, session_id=sid)
        mem.add_user_message("secret-marker-7")
        mem.close()
        assert (await c1.get(f"/v1/sessions/{sid}")).status_code == 200
    await app1.state.pool.close_all()

    # Boot #2: a fresh app instance (new in-memory pool) on the SAME SQLite DB --
    # simulates a serve restart. The persisted session must still resolve, AND the
    # persisted conversation must be read back (proving get_or_create points the agent
    # at the shared DB rather than returning []). Previously 404 ("session not found").
    app2 = _app(cfg)
    async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app2)) as c2:
        r = await c2.get(f"/v1/sessions/{sid}")
        assert r.status_code == 200, r.text
        assert r.json()["session_id"] == sid
        body = " ".join(m.get("content", "") for m in r.json()["messages"])
        assert "secret-marker-7" in body, "rehydrate did not read the persisted conversation"
    await app2.state.pool.close_all()

    # A truly-unknown session must still 404 (the fix must not over-reach).
    app3 = _app(cfg)
    async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app3)) as c3:
        unknown = "ffffffffffffffffffffffffffffffff"
        assert (await c3.get(f"/v1/sessions/{unknown}")).status_code == 404
    await app3.state.pool.close_all()


async def test_get_session_non_owner_no_build_after_restart(tmp_path):
    """A non-owner GET on a persisted session must 403 BEFORE an agent is built.

    Pins the owner-check-first ordering: a denied caller must never reach
    ``pool.get_or_create`` (which would materialize an agent and, at pool cap, evict a
    legitimate tenant's warm agent). ``KeyStore`` derives a deterministic key-id from
    the token, so ownership written by key-alice in boot #1 is still alice's in boot #2.
    """
    db = str(tmp_path / "mem.db")
    cfg = _cfg(db, auth=True)

    app1 = _app(cfg, api_keys=["key-alice", "key-bob"])
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app1)) as c1:
        sid = (await c1.post("/v1/sessions", headers={"Authorization": "Bearer key-alice"})).json()["session_id"]
    await app1.state.pool.close_all()

    # Fresh pool on the same DB: the session is persisted + owned by alice, not pooled.
    app2 = _app(cfg, api_keys=["key-alice", "key-bob"])
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app2)) as c2:
        assert app2.state.pool.get(sid) is None  # precondition: nothing materialized yet
        r = await c2.get(f"/v1/sessions/{sid}", headers={"Authorization": "Bearer key-bob"})
        assert r.status_code == 403, r.text
        # No agent built for a denied GET -- the rehydrate branch was never reached.
        assert app2.state.pool.get(sid) is None, "non-owner GET materialized an agent"
    await app2.state.pool.close_all()


async def test_get_session_429_when_pool_full_on_rehydrate(tmp_path):
    """The rehydrate branch returns a 429 pool_full envelope when the pool is full."""
    cfg = _cfg(str(tmp_path / "mem.db"))
    app = _app(cfg)
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
        sid = app.state.pool.new_session_id()
        # Persisted existence without an in-pool agent -> forces the rehydrate branch.
        app.state.ownership.set_owner(sid, "dev")

        async def _poolfull(_session_id):
            raise PoolFull("no room")

        app.state.pool.get_or_create = _poolfull
        r = await c.get(f"/v1/sessions/{sid}")
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "pool_full"
    await app.state.pool.close_all()


async def test_get_session_500_envelope_when_rehydrate_fails(tmp_path):
    """A non-PoolFull rehydrate failure returns a 500 error envelope, not an opaque 500."""
    cfg = _cfg(str(tmp_path / "mem.db"))
    app = _app(cfg)
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
        sid = app.state.pool.new_session_id()
        app.state.ownership.set_owner(sid, "dev")

        async def _boom(_session_id):
            raise RuntimeError("disk full")

        app.state.pool.get_or_create = _boom
        r = await c.get(f"/v1/sessions/{sid}")
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "rehydrate_failed"
    await app.state.pool.close_all()
