"""tests/test_server_cov.py -- branch coverage for koboi/server/app.py + jobs.py.

Targets the error/guard/branch paths the existing server tests don't reach:
session-surface owner checks, fork rollback paths, resume errors, chat-stream
branches, approve resolution, job submit/cancel/stream guards, the webhook
delivery helpers, run_job terminal branches, and the module-level app helpers.
HTTP routes are driven via httpx ASGITransport (mocked LLM); jobs helpers are
unit-tested directly.
"""

from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.server import create_app  # noqa: E402
from koboi.server.app import (  # noqa: E402
    _build_key_store,
    _cancel_tasks,
    _cleanup_workdirs,
    _enrich_trace,
    _resolve_allowed_modes,
    serve_app,
)
from koboi.server.jobs import (  # noqa: E402
    DuplicateIdempotencyKey,
    JobRegistry,
    JobStore,
    _deliver_webhooks,
    _emit_job_webhooks,
    _on_webhook_task_done,
    _post_webhook,
    _webhook_payload,
    drain_webhook_tasks,
    resume_on_startup,
    run_job,
)
from tests.conftest import MockClient, make_mock_response  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers (mirror tests/test_server_app.py wiring + auth fixture pattern)
# ---------------------------------------------------------------------------


def _config(**overrides) -> Config:
    cfg = {
        "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "base_url": "http://localhost:8080/v1",
        },
        "memory": {"backend": "in_memory"},
        "sandbox": {"backend": "restricted"},  # C3: jobs require containment
        "server": {"auth_required": False},  # dev-open by default
    }
    cfg.update(overrides)
    return Config.from_dict(cfg, validate=True)


def _factory(responses=None):
    return lambda: MockClient(responses or [make_mock_response(content="ok")])  # noqa: E731


def _app(responses=None, **kw):
    return create_app(_config(), client_factory=_factory(responses), enable_cors=False, **kw)


def _client(app):
    return httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app))


def _parse_sse(text: str) -> list:
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            payload = line[6:]
            out.append("[DONE]" if payload == "[DONE]" else __import__("json").loads(payload))
    return out


# ---------------------------------------------------------------------------
# app.py: session-surface route guards + fork rollback
# ---------------------------------------------------------------------------


class TestSessionRouteGuards:
    async def test_delete_owner_mismatch_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer keyA"})).json()["session_id"]
            r = await c.delete(f"/v1/sessions/{sid}", headers={"Authorization": "Bearer keyB"})
            assert r.status_code == 403

    async def test_resume_owner_mismatch_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer keyA"})).json()["session_id"]
            r = await c.post(f"/v1/sessions/{sid}/resume", headers={"Authorization": "Bearer keyB"})
            assert r.status_code == 403

    async def test_resume_unsafe_session_400(self):
        async with _client(_app()) as c:
            assert (await c.post("/v1/sessions/bad.id/resume")).status_code == 400

    async def test_resume_exception_returns_500(self):
        app = _app()
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            agent = app.state.pool.get(sid)
            agent.resume = AsyncMock(side_effect=RuntimeError("boom"))
            r = await c.post(f"/v1/sessions/{sid}/resume")
            assert r.status_code == 500
            assert r.json()["error"]["code"] == "resume_failed"


class TestForkRoutes:
    async def test_fork_unsafe_session_400(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/sessions/a.b/fork")
            assert r.status_code == 400

    async def test_fork_owner_mismatch_403(self):
        # Owner check happens before the sqlite check, so in_memory backend is fine.
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer keyA"})).json()["session_id"]
            r = await c.post(f"/v1/sessions/{sid}/fork", headers={"Authorization": "Bearer keyB"})
            assert r.status_code == 403

    async def test_fork_non_sqlite_returns_409(self):
        async with _client(_app()) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            r = await c.post(f"/v1/sessions/{sid}/fork")
            assert r.status_code == 409
            assert r.json()["error"]["code"] == "not_persisted"

    async def test_fork_poolfull_rolls_back_429(self, tmp_path):
        from koboi.memory_sqlite import SQLiteMemory
        from koboi.server.pool import PoolFull

        db = str(tmp_path / "fork.db")
        cfg = _config(memory={"backend": "sqlite", "db_path": db})
        app = create_app(cfg, client_factory=_factory(), enable_cors=False)
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            mem = SQLiteMemory(db_path=db, session_id=sid)
            mem.add_user_message("seed")
            mem.close()

            async def _poolfull(_session_id):
                raise PoolFull("no room")

            app.state.pool.get_or_create = _poolfull
            r = await c.post(f"/v1/sessions/{sid}/fork")
            assert r.status_code == 429
            assert r.json()["error"]["code"] == "pool_full"
            # Ghost fork rolled back: only the original session remains.
            assert [s["session_id"] for s in SQLiteMemory.list_sessions(db)] == [sid]


# ---------------------------------------------------------------------------
# app.py: chat-stream branches
# ---------------------------------------------------------------------------


class TestChatStreamBranches:
    async def test_header_session_owner_mismatch_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer keyA"})).json()["session_id"]
            r = await c.post(
                "/v1/chat/stream",
                json={"message": "hi"},
                headers={"X-Session-Id": sid, "Authorization": "Bearer keyB"},
            )
            assert r.status_code == 403

    async def test_per_request_max_iterations_applied(self):
        app = _app([make_mock_response(content="ok")])
        async with _client(app) as c:
            async with c.stream("POST", "/v1/chat/stream", json={"message": "hi", "max_iterations": 5}) as r:
                await r.aread()
                assert r.status_code == 200

    async def test_per_request_mode_applied(self):
        # exercise the mode-switch branch (effective_mode is not None).
        app = _app([make_mock_response(content="ok")])
        async with _client(app) as c:
            async with c.stream("POST", "/v1/chat/stream", json={"message": "hi", "mode": "act"}) as r:
                await r.aread()
                assert r.status_code == 200

    async def test_run_stream_error_emits_error_event(self):
        app = _app([make_mock_response(content="ok")])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            agent = app.state.pool.get(sid)

            async def _boom(_message):
                raise RuntimeError("stream blew up")
                yield  # noqa: unreachable -- makes this an async generator

            agent.run_stream = _boom
            async with c.stream("POST", "/v1/chat/stream", json={"message": "hi"}, headers={"X-Session-Id": sid}) as r:
                text = (await r.aread()).decode()
            events = _parse_sse(text)
            types = [e.get("type") if isinstance(e, dict) else e for e in events]
            assert "error" in types
            assert types[-1] == "[DONE]"


# ---------------------------------------------------------------------------
# app.py: approve resolution paths (873-881)
# ---------------------------------------------------------------------------


class TestApproveRoutes:
    async def test_approve_unsafe_session_400(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/sessions/a.b/approve", json={"approval_id": "x"})
            assert r.status_code == 400

    async def test_approve_owner_mismatch_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer keyA"})).json()["session_id"]
            r = await c.post(
                f"/v1/sessions/{sid}/approve",
                json={"approval_id": "x"},
                headers={"Authorization": "Bearer keyB"},
            )
            assert r.status_code == 403

    async def test_approve_resolves_pending(self):
        from koboi.server.approvals import ApprovalCoordinator

        app = _app()
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            coord = ApprovalCoordinator(asyncio.Queue(), timeout=10)
            app.state.approvals.register(sid, coord)
            future = asyncio.get_event_loop().create_future()
            coord._futures["aid"] = future
            r = await c.post(
                f"/v1/sessions/{sid}/approve",
                json={"approval_id": "aid", "decision": "approve", "scope": "always"},
            )
            assert r.status_code == 200
            assert r.json()["resolved"] is True
            assert future.result().approved is True
            assert future.result().always_allow is True

    async def test_approve_unknown_approval_404(self):
        from koboi.server.approvals import ApprovalCoordinator

        app = _app()
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            coord = ApprovalCoordinator(asyncio.Queue(), timeout=10)
            app.state.approvals.register(sid, coord)
            r = await c.post(f"/v1/sessions/{sid}/approve", json={"approval_id": "nope", "decision": "approve"})
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# app.py: job submit / stream / cancel guards
# ---------------------------------------------------------------------------


class TestJobRouteGuards:
    async def test_submit_bad_session_id_400(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/jobs", json={"message": "x", "session_id": "bad.id"})
            assert r.status_code == 400

    async def test_submit_poolfull_on_dedicated_session_429(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, cap=0)
        async with _client(app) as c:
            r = await c.post("/v1/jobs", json={"message": "x"})
            assert r.status_code == 429
            assert r.json()["error"]["code"] == "pool_full"

    async def test_submit_duplicate_idem_returns_existing(self, monkeypatch):
        app = _app()
        app.state.job_store.insert("job_existing", "s1", "dev", "m")

        def _raise(*a, **k):
            raise DuplicateIdempotencyKey("job_existing")

        monkeypatch.setattr(app.state.job_store, "insert", _raise)
        async with _client(app) as c:
            r = await c.post("/v1/jobs", json={"message": "x"}, headers={"Idempotency-Key": "fresh-1"})
            assert r.status_code == 202
            assert r.json()["job_id"] == "job_existing"

    async def test_submit_duplicate_idem_no_existing_409(self, monkeypatch):
        app = _app()

        def _raise(*a, **k):
            raise DuplicateIdempotencyKey("job_missing")

        monkeypatch.setattr(app.state.job_store, "insert", _raise)
        async with _client(app) as c:
            r = await c.post("/v1/jobs", json={"message": "x"}, headers={"Idempotency-Key": "fresh-2"})
            assert r.status_code == 409
            assert r.json()["error"]["code"] == "duplicate_request"

    async def test_stream_other_owner_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            r = await c.post("/v1/jobs", json={"message": "x"}, headers={"Authorization": "Bearer keyA"})
            job_id = r.json()["job_id"]
            s = await c.get(f"/v1/jobs/{job_id}/stream", headers={"Authorization": "Bearer keyB"})
            assert s.status_code == 403

    async def test_cancel_other_owner_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            r = await c.post("/v1/jobs", json={"message": "x"}, headers={"Authorization": "Bearer keyA"})
            job_id = r.json()["job_id"]
            rc = await c.post(f"/v1/jobs/{job_id}/cancel", headers={"Authorization": "Bearer keyB"})
            assert rc.status_code == 403

    async def test_cancel_task_done_race_409(self):
        app = _app()
        async with _client(app) as c:
            job_id = "job_done_race"
            app.state.job_store.insert(job_id, "s1", "dev", "m")  # status pending (non-terminal)
            record = app.state.job_registry.register(job_id, "s1", "dev")

            async def _noop():
                return None

            record.task = asyncio.get_event_loop().create_task(_noop())
            await record.task  # mark done; store status still pending -> hits the 409 guard
            rc = await c.post(f"/v1/jobs/{job_id}/cancel")
            assert rc.status_code == 409

    async def test_stream_deadline_break(self):
        # A non-terminal job whose stream loop hits the monotonic deadline exits cleanly.
        # timeout_seconds must be > 0 (Pydantic); a tiny value forces a near-immediate deadline.
        cfg = _config(jobs={"timeout_seconds": 0.001})
        app = create_app(cfg, client_factory=_factory(), enable_cors=False)
        job_id = "job_deadline"
        app.state.job_store.insert(job_id, "s1", "dev", "m")
        app.state.job_store.update_status(job_id, "running")
        record = app.state.job_registry.register(job_id, "s1", "dev")
        record.status = "running"  # terminal event never set -> loop runs until deadline
        async with _client(app) as c:
            async with c.stream("GET", f"/v1/jobs/{job_id}/stream") as s:
                text = (await s.aread()).decode()
            assert s.status_code == 200
            assert _parse_sse(text)[-1] == "[DONE]"

    async def test_stream_job_not_in_registry_breaks(self):
        # Job exists in the store (passes _check_job_access) but has no in-memory
        # registry record -> the stream's else-branch breaks immediately.
        app = _app()
        job_id = "job_no_record"
        app.state.job_store.insert(job_id, "s1", "dev", "m")
        app.state.job_store.update_status(job_id, "completed")  # not registered in registry
        async with _client(app) as c:
            async with c.stream("GET", f"/v1/jobs/{job_id}/stream") as s:
                text = (await s.aread()).decode()
            assert s.status_code == 200
            assert _parse_sse(text)[-1] == "[DONE]"


# ---------------------------------------------------------------------------
# app.py: MCP route guards (unsafe-id 400, unknown-session 404, owner 403)
# ---------------------------------------------------------------------------


class _FakeMcpClient:
    """Minimal MCP client duck-type for SessionMcpRegistry.register/reconnect."""

    def __init__(self, name="fake", connected=True):
        self._name = name
        self._connected = connected

    @property
    def name(self):
        return self._name

    @property
    def endpoint(self):
        return "fake"

    @property
    def transport(self):
        return "stdio"

    @property
    def tool_names(self):
        return ["t1"]

    server_info = {}

    def is_connected(self):
        return self._connected

    def close(self):
        self._connected = False

    def connect(self):
        self._connected = True


class TestMcpRouteGuards:
    async def test_list_unsafe_session_400(self):
        async with _client(_app()) as c:
            r = await c.get("/v1/sessions/a.b/mcp/servers")
            assert r.status_code == 400

    async def test_list_owner_mismatch_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer keyA"})).json()["session_id"]
            r = await c.get(f"/v1/sessions/{sid}/mcp/servers", headers={"Authorization": "Bearer keyB"})
            assert r.status_code == 403

    async def test_add_unsafe_session_400(self):
        async with _client(_app()) as c:
            r = await c.post(
                "/v1/sessions/a.b/mcp/servers",
                json={"transport": "stdio", "command": "python3", "args": []},
            )
            assert r.status_code == 400

    async def test_add_unknown_session_404(self):
        async with _client(_app()) as c:
            r = await c.post(
                "/v1/sessions/nope/mcp/servers",
                json={"transport": "stdio", "command": "python3", "args": []},
            )
            assert r.status_code == 404

    async def test_add_owner_mismatch_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer keyA"})).json()["session_id"]
            r = await c.post(
                f"/v1/sessions/{sid}/mcp/servers",
                json={"transport": "stdio", "command": "python3", "args": []},
                headers={"Authorization": "Bearer keyB"},
            )
            assert r.status_code == 403

    async def test_add_register_close_failure_logs_502(self, monkeypatch):
        # 29-D variant: connect ok, register fails, and client.close() also fails -> warning + 502.
        app = _app()

        class _Client:
            name = "bad"
            endpoint = "bad"
            transport = "stdio"
            tool_names = []
            server_info = {}

            def is_connected(self):
                return True

            def connect(self):
                return {"serverInfo": {"name": "bad"}}

            def discover_tools(self):
                raise RuntimeError("bad tools/list")

            def close(self):
                raise RuntimeError("close failed")

        monkeypatch.setattr("koboi.facade._create_mcp_client", lambda *a, **k: _Client())
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            r = await c.post(
                f"/v1/sessions/{sid}/mcp/servers",
                json={"transport": "stdio", "command": "python3", "args": []},
            )
            assert r.status_code == 502
            assert r.json()["error"]["code"] == "mcp_register_failed"

    async def test_remove_unsafe_session_400(self):
        async with _client(_app()) as c:
            r = await c.delete("/v1/sessions/a.b/mcp/servers/whatever")
            assert r.status_code == 400

    async def test_remove_unknown_session_404(self):
        async with _client(_app()) as c:
            r = await c.delete("/v1/sessions/nope/mcp/servers/whatever")
            assert r.status_code == 404

    async def test_remove_owner_mismatch_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer keyA"})).json()["session_id"]
            r = await c.delete(f"/v1/sessions/{sid}/mcp/servers/whatever", headers={"Authorization": "Bearer keyB"})
            assert r.status_code == 403

    async def test_reconnect_unsafe_session_400(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/sessions/a.b/mcp/servers/whatever/reconnect")
            assert r.status_code == 400

    async def test_reconnect_unknown_session_404(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/sessions/nope/mcp/servers/whatever/reconnect")
            assert r.status_code == 404

    async def test_reconnect_owner_mismatch_403(self):
        app = create_app(_config(), client_factory=_factory(), enable_cors=False, api_keys=["keyA", "keyB"])
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer keyA"})).json()["session_id"]
            r = await c.post(
                f"/v1/sessions/{sid}/mcp/servers/whatever/reconnect",
                headers={"Authorization": "Bearer keyB"},
            )
            assert r.status_code == 403

    async def test_reconnect_failure_400(self, monkeypatch):
        app = _app()
        async with _client(app) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            await c.get(f"/v1/sessions/{sid}/mcp/servers")  # materialize the registry
            reg = app.state.mcp_registries[sid]
            server_id = reg.register(_FakeMcpClient("fake"))
            monkeypatch.setattr(reg, "reconnect", lambda *_a: (_ for _ in ()).throw(RuntimeError("boom")))
            r = await c.post(f"/v1/sessions/{sid}/mcp/servers/{server_id}/reconnect")
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "mcp_reconnect_failed"


# ---------------------------------------------------------------------------
# app.py: module-level helpers
# ---------------------------------------------------------------------------


class TestAppHelpers:
    async def test_extra_middleware_runs(self):
        async def mw(request, call_next):
            resp = await call_next(request)
            resp.headers["X-Test-MW"] = "yes"
            return resp

        app = create_app(_config(), client_factory=_factory(), enable_cors=False, extra_middleware=(mw,))
        async with _client(app) as c:
            r = await c.get("/healthz")
            assert r.headers.get("X-Test-MW") == "yes"

    def test_enrich_trace_sets_metadata(self):
        class LangfuseTracingHook:
            captured = None

            def set_serving_metadata(self, **kw):
                LangfuseTracingHook.captured = kw

        hook = LangfuseTracingHook()
        agent = MagicMock()
        agent._core.hooks.find_hook.return_value = hook
        _enrich_trace(agent, mode="interactive", owner="alice")
        assert LangfuseTracingHook.captured == {"mode": "interactive", "owner": "alice"}

    def test_enrich_trace_no_hook_noop(self):
        agent = MagicMock()
        agent._core.hooks.find_hook.return_value = None
        _enrich_trace(agent, mode="x")  # must not raise

    def test_enrich_trace_no_hooks_attr(self):
        agent = MagicMock()
        agent._core.hooks = None
        _enrich_trace(agent, mode="x")  # must not raise

    def test_build_key_store_from_env(self, monkeypatch):
        monkeypatch.setenv("KOBOI_API_KEYS", "envkey1,envkey2")
        monkeypatch.delenv("KOBOI_API_KEYS_FILE", raising=False)
        ks = _build_key_store(_config())
        assert ks.has_keys
        assert ks.validate("envkey1") is not None

    def test_build_key_store_from_config(self, monkeypatch):
        monkeypatch.delenv("KOBOI_API_KEYS", raising=False)
        monkeypatch.delenv("KOBOI_API_KEYS_FILE", raising=False)
        cfg = _config(server={"api_keys": ["cfgkey"]})
        ks = _build_key_store(cfg)
        assert ks.has_keys
        assert ks.validate("cfgkey") is not None

    def test_cleanup_workdirs_oserror_ignored(self, tmp_path, monkeypatch):
        from pathlib import Path

        ws = tmp_path / "workspace"
        ws.mkdir()
        bad = MagicMock()
        bad.is_dir.return_value = True
        bad.stat.side_effect = OSError("boom")
        orig_iterdir = Path.iterdir
        monkeypatch.setattr(Path, "iterdir", lambda self: [bad] if self == ws else orig_iterdir(self))
        assert _cleanup_workdirs(str(ws), ttl_seconds=60) == 0


class TestResolveAllowedModes:
    def test_default_when_none(self):
        assert _resolve_allowed_modes(None) == frozenset({"chat", "plan", "act", "auto"})

    def test_default_when_empty(self):
        assert _resolve_allowed_modes([]) == frozenset({"chat", "plan", "act", "auto"})

    def test_not_list_raises(self):
        with pytest.raises(ValueError):
            _resolve_allowed_modes("chat")

    def test_non_string_entry_raises(self):
        with pytest.raises(ValueError):
            _resolve_allowed_modes([123])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            _resolve_allowed_modes(["bogus"])

    def test_valid_list(self):
        assert _resolve_allowed_modes(["chat", "act"]) == frozenset({"chat", "act"})


class TestCancelTasksHelper:
    async def test_drains_and_clears_set(self):
        tasks: set[asyncio.Task] = set()

        async def _long():
            await asyncio.sleep(100)

        for _ in range(3):
            tasks.add(asyncio.create_task(_long()))
        await _cancel_tasks(tasks)
        assert tasks == set()

    async def test_empty_set_noop(self):
        await _cancel_tasks(set())


class TestServeAppWarning:
    def test_non_loopback_with_keys_warns_and_runs(self, tmp_path, monkeypatch):
        # Keys present -> serve_app logs the non-loopback warning (no SystemExit), then
        # calls uvicorn.run (mocked so it doesn't block).
        import uvicorn

        monkeypatch.setenv("KOBOI_API_KEYS", "realkey")
        monkeypatch.delenv("KOBOI_API_KEYS_FILE", raising=False)
        called = {}

        def _fake_run(*a, **k):
            called["ran"] = True

        monkeypatch.setattr(uvicorn, "run", _fake_run)
        cfg = tmp_path / "agent.yaml"
        cfg.write_text(
            "agent:\n  name: t\n  system_prompt: h\n  max_iterations: 3\n"
            "llm:\n  provider: openai\n  model: m\n  api_key: x\n  base_url: http://x\n"
            "memory:\n  backend: in_memory\n"
        )
        serve_app(str(cfg), host="0.0.0.0", port=8001)  # no SystemExit
        assert called.get("ran") is True


# ---------------------------------------------------------------------------
# jobs.py: JobStore + JobRegistry branch coverage
# ---------------------------------------------------------------------------


class TestJobStoreBranches:
    def test_duplicate_job_id_reraises_integrity_error(self, tmp_path):
        # Same primary key, no idempotency_key -> the non-duplicate IntegrityError re-raises.
        store = JobStore(str(tmp_path / "j.db"))
        store.insert("job_1", "s", "a", "m")
        with pytest.raises(sqlite3.IntegrityError):
            store.insert("job_1", "s", "a", "m")

    def test_list_by_owner_with_status_filter(self, tmp_path):
        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j1", "s", "a", "m")
        store.update_status("j1", "completed")
        store.insert("j2", "s", "a", "m")  # stays pending
        assert len(store.list_by_owner("a", status="completed")) == 1
        assert len(store.list_by_owner("a", status="pending")) == 1
        assert len(store.list_by_owner("a")) == 2


class TestJobRegistryExtra:
    async def test_list_by_owner_filters(self):
        reg = JobRegistry()
        reg.register("j1", "s", "alice")
        reg.register("j2", "s", "bob")
        assert [r.job_id for r in reg.list_by_owner("alice")] == ["j1"]

    async def test_cancel_returns_true_with_live_task(self):
        reg = JobRegistry()
        reg.register("j1", "s", "a")

        async def _long():
            await asyncio.sleep(100)

        task = asyncio.create_task(_long())
        reg.set_running("j1", task)
        try:
            assert await reg.cancel("j1") is True
        finally:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_cancel_returns_false_without_task(self):
        reg = JobRegistry()
        reg.register("j1", "s", "a")  # no task set
        assert await reg.cancel("j1") is False

    async def test_cancel_all_cancels_running(self):
        reg = JobRegistry()
        reg.register("j1", "s", "a")
        reg.register("j2", "s", "a")

        async def _long():
            await asyncio.sleep(100)

        t1 = asyncio.create_task(_long())
        t2 = asyncio.create_task(_long())
        reg.set_running("j1", t1)
        reg.set_running("j2", t2)
        assert reg.cancel_all() == 2
        for t in (t1, t2):
            try:
                await t
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# jobs.py: webhook delivery helpers
# ---------------------------------------------------------------------------


class TestWebhookPayload:
    def test_unknown_job_returns_none(self, tmp_path):
        store = JobStore(str(tmp_path / "j.db"))
        assert _webhook_payload(store, "nope", "completed") is None

    def test_valid_result_json_parsed(self, tmp_path):
        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j1", "s", "a", "m")
        store.update_status("j1", "completed", result_json='{"content":"done"}')
        payload = _webhook_payload(store, "j1", "completed")
        assert payload["job_id"] == "j1"
        assert payload["status"] == "completed"
        assert payload["event"] == "job.completed"
        assert payload["result"] == {"content": "done"}
        assert payload["retriable"] is False

    def test_invalid_result_json_kept_raw(self, tmp_path):
        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j1", "s", "a", "m")
        store.update_status("j1", "failed", result_json="not json{")
        payload = _webhook_payload(store, "j1", "failed")
        assert payload["result"] == "not json{"

    def test_none_result_json(self, tmp_path):
        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j1", "s", "a", "m")
        store.update_status("j1", "completed")
        payload = _webhook_payload(store, "j1", "completed")
        assert payload["result"] is None


class _FakeResp:
    def __init__(self, status):
        self.status_code = status


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient used by _post_webhook."""

    def __init__(self, *args, **kwargs):
        self._post_status = kwargs.pop("_post_status", 200)
        self._raise = kwargs.pop("_raise", None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        if self._raise is not None:
            raise self._raise
        return _FakeResp(self._post_status)


class _FakeAsyncClientBoom:
    """AsyncClient whose construction raises (exercises the outer except)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("client construction failed")


class TestPostWebhook:
    async def test_success_returns_immediately(self, monkeypatch):
        from koboi.server import jobs

        monkeypatch.setattr(jobs.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(_post_status=200))
        await _post_webhook("http://x", b"{}", {}, 10.0)

    async def test_5xx_retries_then_warns(self, monkeypatch):
        from koboi.server import jobs

        monkeypatch.setattr(jobs.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(_post_status=503))
        await _post_webhook("http://x", b"{}", {}, 10.0)  # 2 attempts -> warning

    async def test_connect_error_warns(self, monkeypatch):
        from koboi.server import jobs

        monkeypatch.setattr(
            jobs.httpx,
            "AsyncClient",
            lambda *a, **k: _FakeAsyncClient(_raise=jobs.httpx.ConnectError("nope")),
        )
        await _post_webhook("http://x", b"{}", {}, 10.0)

    async def test_timeout_error_warns(self, monkeypatch):
        from koboi.server import jobs

        monkeypatch.setattr(
            jobs.httpx,
            "AsyncClient",
            lambda *a, **k: _FakeAsyncClient(_raise=jobs.httpx.TimeoutException("slow")),
        )
        await _post_webhook("http://x", b"{}", {}, 10.0)

    async def test_unexpected_exception_caught(self, monkeypatch):
        from koboi.server import jobs

        monkeypatch.setattr(jobs.httpx, "AsyncClient", _FakeAsyncClientBoom)
        await _post_webhook("http://x", b"{}", {}, 10.0)  # outer except -> warning


class TestDeliverWebhooks:
    async def test_empty_webhooks_noop(self):
        await _deliver_webhooks([], None, "j1", "completed")

    async def test_unknown_job_noop(self, tmp_path):
        store = JobStore(str(tmp_path / "j.db"))
        await _deliver_webhooks([{"url": "http://x"}], store, "nope", "completed")

    async def test_secret_signs_and_posts(self, tmp_path, monkeypatch):
        import hashlib
        import hmac

        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j1", "s", "a", "m")
        store.update_status("j1", "completed")
        captured = []

        async def _fake_post(url, body, headers, timeout):
            captured.append((url, body, headers, timeout))

        monkeypatch.setattr("koboi.server.jobs._post_webhook", _fake_post)
        await _deliver_webhooks(
            [{"url": "http://hook", "events": ["completed"], "secret": "topsecret", "timeout": 5}],
            store,
            "j1",
            "completed",
        )
        assert len(captured) == 1
        url, body, headers, timeout = captured[0]
        assert url == "http://hook"
        assert timeout == 5.0
        assert headers["Content-Type"] == "application/json"
        expected = hmac.new(b"topsecret", body, hashlib.sha256).hexdigest()
        assert headers["X-Koboi-Signature"] == f"sha256={expected}"

    async def test_default_timeout_when_unset(self, tmp_path, monkeypatch):
        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j1", "s", "a", "m")
        store.update_status("j1", "completed")
        captured = []

        async def _fake_post(url, body, headers, timeout):
            captured.append(timeout)

        monkeypatch.setattr("koboi.server.jobs._post_webhook", _fake_post)
        # No "events" -> matches all statuses; no "timeout" -> default 10s.
        await _deliver_webhooks([{"url": "http://hook"}], store, "j1", "completed")
        assert captured == [10.0]

    async def test_event_filter_skips_non_matching(self, tmp_path, monkeypatch):
        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j1", "s", "a", "m")
        store.update_status("j1", "failed")
        posted = []

        async def _fake_post(*a):
            posted.append(a)

        monkeypatch.setattr("koboi.server.jobs._post_webhook", _fake_post)
        await _deliver_webhooks([{"url": "http://hook", "events": ["completed"]}], store, "j1", "failed")
        assert posted == []

    async def test_missing_url_skipped(self, tmp_path, monkeypatch):
        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j1", "s", "a", "m")
        store.update_status("j1", "completed")
        posted = []

        async def _fake_post(*a):
            posted.append(a)

        monkeypatch.setattr("koboi.server.jobs._post_webhook", _fake_post)
        await _deliver_webhooks([{"url": ""}], store, "j1", "completed")
        assert posted == []


class TestWebhookTaskCallbacks:
    async def test_done_clean_task_discards_ref(self):
        from koboi.server.jobs import _WEBHOOK_TASKS

        async def _ok():
            return None

        task = asyncio.create_task(_ok())
        _WEBHOOK_TASKS.add(task)
        await task
        _on_webhook_task_done(task)
        assert task not in _WEBHOOK_TASKS

    async def test_done_cancelled_task_returns(self):
        from koboi.server.jobs import _WEBHOOK_TASKS

        async def _long():
            await asyncio.sleep(100)

        task = asyncio.create_task(_long())
        _WEBHOOK_TASKS.add(task)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        _on_webhook_task_done(task)  # cancelled -> early return
        assert task not in _WEBHOOK_TASKS

    async def test_done_task_with_exception_logs(self):
        from koboi.server.jobs import _WEBHOOK_TASKS

        async def _bad():
            raise RuntimeError("deliver bug")

        task = asyncio.create_task(_bad())
        _WEBHOOK_TASKS.add(task)
        try:
            await task
        except RuntimeError:
            pass
        _on_webhook_task_done(task)  # exc -> error log
        assert task not in _WEBHOOK_TASKS


class TestEmitAndDrain:
    async def test_emit_no_webhooks_noop(self):
        _emit_job_webhooks(None, JobStore(":memory:"), "j1", "completed")

    async def test_emit_schedules_task_then_drains(self, tmp_path, monkeypatch):
        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j1", "s", "a", "m")
        store.update_status("j1", "completed")

        async def _fake_post(*a, **k):
            return None

        monkeypatch.setattr("koboi.server.jobs._post_webhook", _fake_post)
        _emit_job_webhooks([{"url": "http://h"}], store, "j1", "completed")
        await drain_webhook_tasks()

    async def test_drain_empty_noop(self):
        await drain_webhook_tasks()

    async def test_drain_timeout_abandons(self):
        from koboi.server.jobs import _WEBHOOK_TASKS

        async def _forever():
            await asyncio.sleep(100)

        task = asyncio.create_task(_forever())
        _WEBHOOK_TASKS.add(task)
        await drain_webhook_tasks(timeout=0.01)  # exceeds -> warning, task abandoned
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# jobs.py: run_job terminal branches + resume_on_startup pending loop
# ---------------------------------------------------------------------------


class TestRunJobBranches:
    async def test_record_none_returns_early(self):
        store = JobStore(":memory:")
        reg = JobRegistry()
        await run_job("nope", object(), reg, store, "m", timeout=5)  # no record -> return

    async def test_cancelled_branch_marks_cancelled(self, monkeypatch):
        from koboi.server import jobs

        store = JobStore(":memory:")
        reg = JobRegistry()
        reg.register("j1", "s", "a")
        store.insert("j1", "s", "a", "m")

        async def _cancel(*a, **k):
            raise asyncio.CancelledError()

        monkeypatch.setattr(jobs, "_execute_job", _cancel)
        with pytest.raises(asyncio.CancelledError):
            await run_job("j1", object(), reg, store, "m", timeout=5)
        assert store.get("j1")["status"] == "cancelled"
        assert reg.get("j1").status == "cancelled"

    async def test_timeout_branch_marks_timed_out(self, monkeypatch):
        from koboi.server import jobs

        store = JobStore(":memory:")
        reg = JobRegistry()
        reg.register("j1", "s", "a")
        store.insert("j1", "s", "a", "m")

        async def _to(*a, **k):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(jobs, "_execute_job", _to)
        await run_job("j1", object(), reg, store, "m", timeout=5)
        row = store.get("j1")
        assert row["status"] == "timed_out"
        assert row["error_class"] == "TimeoutError"
        assert row["retriable"] == 1

    async def test_generic_exception_marks_failed_and_redacts(self, monkeypatch):
        from koboi.server import jobs

        store = JobStore(":memory:")
        reg = JobRegistry()
        reg.register("j1", "s", "a")
        store.insert("j1", "s", "a", "m")

        async def _boom(*a, **k):
            raise RuntimeError("failed with sk-abcdefghijklmnopqrstuvwxyz leaked")

        monkeypatch.setattr(jobs, "_execute_job", _boom)
        await run_job("j1", object(), reg, store, "m", timeout=5)
        row = store.get("j1")
        assert row["status"] == "failed"
        assert row["error_class"] == "RuntimeError"
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in (row["error"] or "")
        assert reg.get("j1").status == "failed"


class TestResumeOnStartupPending:
    async def test_pending_job_requeued_fresh(self, tmp_path, monkeypatch):
        from koboi.server import jobs

        store = JobStore(str(tmp_path / "j.db"))
        store.insert("j_pending", "s1", "alice", "do")  # stays pending
        calls: list[tuple] = []

        async def fake_run_job(
            job_id,
            pool,
            reg,
            st,
            message,
            timeout=1800,
            mode=None,
            max_iterations=None,
            resume=False,
            webhooks=None,
            workflow_ref=None,
            workflow_store=None,
            replay_mode=None,
        ):
            calls.append((job_id, resume))

        monkeypatch.setattr(jobs, "run_job", fake_run_job)
        count = await resume_on_startup(store, object(), JobRegistry(), timeout=30)
        await asyncio.sleep(0.01)  # let the created task record its call
        assert count == 1
        assert ("j_pending", False) in calls  # pending -> requeued fresh (resume=False)
