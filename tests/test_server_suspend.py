"""Opt-in POST /v1/sessions/{id}/suspend -- consistent DB snapshot for an external
suspend/snapshot step."""

from __future__ import annotations

import asyncio
import os
import sqlite3

import httpx
from httpx import ASGITransport

from koboi.config import Config
from koboi.server.app import create_app
from tests.conftest import MockClient, make_mock_response


def _cfg(db_path: str, backend: str = "sqlite", suspend: bool = False, auth: bool = False) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
            "memory": {"backend": backend, "db_path": db_path},
            "sandbox": {"backend": "passthrough"},
            "server": {"auth_required": auth, "suspend_enabled": suspend},
        },
        validate=True,
    )


def _app(cfg: Config, api_keys: list[str] | None = None):
    factory = lambda: MockClient([make_mock_response(content="ok")])  # noqa: E731
    kwargs: dict = {"client_factory": factory, "enable_cors": False}
    if api_keys:
        kwargs["api_keys"] = api_keys  # only pass when auth tests need real keys
    return create_app(cfg, **kwargs)


class TestSuspendRoute:
    async def test_disabled_by_default_returns_404(self, tmp_path):
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(_cfg(db))), base_url="http://t") as c:
            r = await c.post("/v1/sessions/sess-1/suspend")
            assert r.status_code == 404  # opt-in: route self-disables when unset

    async def test_enabled_writes_consistent_snapshot(self, tmp_path):
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(
            transport=ASGITransport(app=_app(_cfg(db, suspend=True))), base_url="http://t"
        ) as c:
            r = await c.post("/v1/sessions")
            assert r.status_code == 201
            sid = r.json()["session_id"]

            r = await c.post(f"/v1/sessions/{sid}/suspend")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["snapshot_path"].endswith(".suspend.db")
            assert body["snapshot_bytes"] > 0
            # typed accessor (not a substring check): the checkpoint ran cleanly
            assert body["checkpoint"]["ok"] is True
            assert isinstance(body["checkpoint"]["busy"], int)
            assert os.path.exists(body["snapshot_path"])
            # the snapshot file is standalone + integrity-clean
            v = sqlite3.connect(body["snapshot_path"])
            assert v.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            v.close()

    async def test_requires_sqlite_backend(self, tmp_path):
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(
            transport=ASGITransport(app=_app(_cfg(db, backend="in_memory", suspend=True))),
            base_url="http://t",
        ) as c:
            r = await c.post("/v1/sessions/sess-1/suspend")
            assert r.status_code == 409
            assert "not_persisted" in r.text

    async def test_suspend_unknown_session_returns_404(self, tmp_path):
        # I-2: parity with DELETE/fork/resume -- a never-existed session 404s instead of
        # silently snapshotting the whole shared DB off an arbitrary id.
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(
            transport=ASGITransport(app=_app(_cfg(db, suspend=True))), base_url="http://t"
        ) as c:
            r = await c.post("/v1/sessions/nosuchsession123/suspend")
            assert r.status_code == 404

    async def test_suspend_403_when_not_owner(self, tmp_path):
        # Ownership gate (I-4): the owner of session A must not be able to suspend via
        # another tenant's session. Mirrors tests/test_server_app.py::test_403_*.
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(
            transport=ASGITransport(app=_app(_cfg(db, suspend=True, auth=True), api_keys=["key-alice", "key-bob"])),
            base_url="http://t",
        ) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer key-alice"})).json()["session_id"]
            r = await c.post(f"/v1/sessions/{sid}/suspend", headers={"Authorization": "Bearer key-bob"})
            assert r.status_code == 403

    async def test_suspend_drains_inflight_lock(self, tmp_path):
        # I-4: suspend waits for an in-flight run on the session (existing_session_lock)
        # and only resolves once the lock is released -- it must not return while work is
        # in progress, and it must complete cleanly afterwards.
        db = str(tmp_path / "d.db")
        app = _app(_cfg(db, suspend=True))
        pool = app.state.pool
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            # hold the session lock as an in-flight /chat/stream would
            async with pool.existing_session_lock(sid):
                suspend_task = asyncio.create_task(c.post(f"/v1/sessions/{sid}/suspend"))
                await asyncio.sleep(0.1)  # let the request reach the lock-acquire point
                assert not suspend_task.done(), "suspend returned while the session lock was held"
            # lock released -> suspend proceeds to a clean turn boundary
            r = await asyncio.wait_for(suspend_task, timeout=5)
            assert r.status_code == 200, r.text

    async def test_concurrent_suspend_different_sessions_no_collision(self, tmp_path):
        # I-1: two sessions sharing one DB must not collide on the snapshot path. Before
        # the fix both resolved to the same {shared_db}.suspend.db and clobbered each other.
        db = str(tmp_path / "d.db")
        app = _app(_cfg(db, suspend=True))
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            sid_a = (await c.post("/v1/sessions")).json()["session_id"]
            sid_b = (await c.post("/v1/sessions")).json()["session_id"]
            ra, rb = await asyncio.gather(
                c.post(f"/v1/sessions/{sid_a}/suspend"),
                c.post(f"/v1/sessions/{sid_b}/suspend"),
            )
            assert ra.status_code == 200 and rb.status_code == 200
            pa, pb = ra.json()["snapshot_path"], rb.json()["snapshot_path"]
            assert pa != pb, "session-scoped snapshot paths must differ"
            assert os.path.exists(pa) and os.path.exists(pb)
            for p in (pa, pb):
                v = sqlite3.connect(p)
                assert v.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                v.close()

    async def test_backup_still_runs_when_checkpoint_fails(self, tmp_path, monkeypatch):
        # C-2: a wal_checkpoint failure must NOT abort the (guaranteed) backup. The route
        # isolates the best-effort checkpoint and proceeds; the response surfaces the
        # checkpoint failure (ok=False, error) alongside a real snapshot.
        db = str(tmp_path / "d.db")
        app = _app(_cfg(db, suspend=True))
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            from koboi.memory_sqlite import SQLiteMemory

            def _boom(*a, **k):
                raise sqlite3.OperationalError("simulated checkpoint contention")

            monkeypatch.setattr(SQLiteMemory, "wal_checkpoint", staticmethod(_boom))
            r = await c.post(f"/v1/sessions/{sid}/suspend")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["snapshot_bytes"] > 0
            assert os.path.exists(body["snapshot_path"])
            assert body["checkpoint"]["ok"] is False
            assert "error" in body["checkpoint"]

    async def test_suspend_failure_does_not_leak_filesystem_path(self, tmp_path, monkeypatch):
        # S-6: a backup failure must return a generic message -- never the raw exception
        # (which carries the server filesystem path). Full detail is logged server-side.
        db = str(tmp_path / "d.db")
        app = _app(_cfg(db, suspend=True))
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            from koboi.memory_sqlite import SQLiteMemory

            def _boom(*a, **k):
                raise OSError(28, "No space left on device", f"{db}.{sid}.suspend.db")

            monkeypatch.setattr(SQLiteMemory, "consistent_backup", staticmethod(_boom))
            r = await c.post(f"/v1/sessions/{sid}/suspend")
            assert r.status_code == 500
            assert "suspend_failed" in r.text
            assert ".suspend.db" not in r.text  # no filesystem path leaked
            assert "No space left" not in r.text  # no raw exception detail leaked
