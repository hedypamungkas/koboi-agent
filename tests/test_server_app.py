"""Integration tests for koboi/server (FastAPI app via httpx ASGI transport).

Gated on the ``api`` extra: ``pytest.importorskip("fastapi")`` skips the module
cleanly when fastapi isn't installed (CI without the extra).
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.server import create_app  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402


def _config(**overrides) -> Config:
    """Base server-test config: in-memory memory, restricted sandbox, dev-open
    auth (``server.auth_required: false``) so non-auth integration tests don't
    401. Pass top-level overrides, e.g. ``_config(sandbox={...})``."""
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
        "server": {"auth_required": False},  # C1: dev-open by default for non-auth tests
    }
    cfg.update(overrides)
    return Config.from_dict(cfg, validate=True)


def _app(responses=None, **kw):
    factory = lambda: MockClient(responses or [make_mock_response(content="hello")])  # noqa: E731
    return create_app(_config(), client_factory=factory, enable_cors=False, **kw)


def _parse_sse(text: str) -> list:
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            payload = line[6:]
            out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


async def _drain(client, sid, body):
    async with client.stream("POST", "/v1/chat/stream", json=body, headers={"X-Session-Id": sid}) as r:
        await r.aread()
        return r.status_code


class TestHealth:
    async def test_healthz(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            r = await c.get("/healthz")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

    async def test_readyz_reports_checks(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            r = await c.get("/readyz")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert any(ch["name"] == "pool" for ch in body["checks"])

    async def test_request_id_echoed_and_minted(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            r = await c.get("/healthz", headers={"X-Request-Id": "abc-123"})
            assert r.headers["X-Request-Id"] == "abc-123"
            r2 = await c.get("/healthz")
            assert r2.headers.get("X-Request-Id")  # minted when absent


class TestSessions:
    async def test_create_returns_id_in_header_and_body(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            r = await c.post("/v1/sessions")
            assert r.status_code == 201
            sid = r.json()["session_id"]
            assert r.headers["X-Session-Id"] == sid
            assert len(sid) == 32

    async def test_get_unknown_404(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            assert (await c.get("/v1/sessions/nope")).status_code == 404

    async def test_delete_evicts_then_404(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            r = await c.delete(f"/v1/sessions/{sid}")
            assert r.status_code == 200
            assert r.json()["evicted"] is True
            assert (await c.delete(f"/v1/sessions/{sid}")).status_code == 404

    async def test_create_session_429_when_pool_full(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="x")]),
            enable_cors=False,
            cap=0,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/sessions")
            assert r.status_code == 429
            assert r.json()["error"]["code"] == "pool_full"

    async def test_resume_session_returns_result(self):
        """POST /v1/sessions/:id/resume returns 200 with RunResult JSON."""
        responses = [make_mock_response(content="resumed answer")]
        app = create_app(
            _config(),
            client_factory=lambda: MockClient(responses),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            r = await c.post(f"/v1/sessions/{sid}/resume")
            assert r.status_code == 200
            body = r.json()
            assert body["session_id"] == sid
            assert body["success"] is True

    async def test_resume_unknown_session_404(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            assert (await c.post("/v1/sessions/nonexistent/resume")).status_code == 404

    async def test_resume_unsafe_session_id_400(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            r = await c.post("/v1/sessions/bad.id/resume")
            assert r.status_code == 400


class TestChatStream:
    async def test_sse_happy_path_message(self):
        async with httpx.AsyncClient(
            base_url="http://testserver", transport=ASGITransport(app=_app([make_mock_response(content="hello")]))
        ) as c:
            async with c.stream("POST", "/v1/chat/stream", json={"message": "hi"}) as r:
                assert r.status_code == 200
                assert "text/event-stream" in r.headers["content-type"]
                text = (await r.aread()).decode()
            events = _parse_sse(text)
            types = [e["type"] if isinstance(e, dict) else e for e in events]
            assert "text_delta" in types
            assert "complete" in types
            assert types[-1] == "[DONE]"
            complete = next(e for e in events if isinstance(e, dict) and e["type"] == "complete")
            assert complete["content"] == "hello"

    async def test_sse_messages_array_shape(self):
        async with httpx.AsyncClient(
            base_url="http://testserver", transport=ASGITransport(app=_app([make_mock_response(content="world")]))
        ) as c:
            async with c.stream(
                "POST", "/v1/chat/stream", json={"messages": [{"role": "user", "content": "hey"}]}
            ) as r:
                text = (await r.aread()).decode()
            events = _parse_sse(text)
            assert any(isinstance(e, dict) and e.get("type") == "complete" and e["content"] == "world" for e in events)

    async def test_empty_body_400(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            r = await c.post("/v1/chat/stream", json={"message": ""})
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "bad_request"

    async def test_auto_create_session_header(self):
        async with httpx.AsyncClient(
            base_url="http://testserver", transport=ASGITransport(app=_app([make_mock_response(content="ok")]))
        ) as c:
            async with c.stream("POST", "/v1/chat/stream", json={"message": "hi"}) as r:
                sid = r.headers.get("X-Session-Id")
                await r.aread()
            assert sid and len(sid) == 32

    async def test_multi_turn_reuse_persists_memory(self):
        responses = [make_mock_response(content="first"), make_mock_response(content="second")]
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app(responses))) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            async with c.stream("POST", "/v1/chat/stream", json={"message": "a"}, headers={"X-Session-Id": sid}) as r:
                await r.aread()
            msgs = (await c.get(f"/v1/sessions/{sid}")).json()["messages"]
            assert len(msgs) >= 2  # user + assistant from turn 1 persisted

    async def test_lock_serializes_concurrent_same_session(self):
        state = {"active": 0, "overlap": False}

        class _Detect(MockClient):
            async def complete_stream(self, messages, tools=None):
                state["active"] += 1
                if state["active"] > 1:
                    state["overlap"] = True
                try:
                    async for ev in super().complete_stream(messages, tools):
                        await asyncio.sleep(0)
                        yield ev
                finally:
                    state["active"] -= 1

        app = create_app(
            _config(), client_factory=lambda: _Detect([make_mock_response(content="x")]), enable_cors=False
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            await asyncio.gather(_drain(c, sid, {"message": "a"}), _drain(c, sid, {"message": "b"}))
            assert state["overlap"] is False

    async def test_429_when_pool_full_and_busy(self):
        # cap=1: materialize s1 and hold its lock directly, then POST s2 -> 429.
        # (Deterministic: avoids streaming-timing races; the PoolFull->429 mapping
        #  is the unit under test, the busy-lock state is set up out-of-band.)
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="x")]),
            enable_cors=False,
            cap=1,
        )
        pool = app.state.pool
        await pool.get_or_create("s1")
        await pool._locks["s1"].acquire()  # hold s1 busy
        try:
            async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
                r = await c.post("/v1/chat/stream", json={"message": "y"}, headers={"X-Session-Id": "s2"})
                assert r.status_code == 429
                assert r.json()["error"]["code"] == "pool_full"
        finally:
            pool._locks["s1"].release()
            await pool.close_all()


class TestExtras:
    async def test_extra_route_mounted(self):
        def ping(app, pool):
            @app.get("/ping")
            def _ping():
                return {"pong": True}

        async with httpx.AsyncClient(
            base_url="http://testserver", transport=ASGITransport(app=_app(extra_routes=(ping,)))
        ) as c:
            r = await c.get("/ping")
            assert r.status_code == 200
            assert r.json() == {"pong": True}

    async def test_extra_tools_registered_on_pooled_agent(self):
        def my_tool(x: str) -> str:
            return f"got {x}"

        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=False,
            extra_tools=(
                (
                    "my_tool",
                    my_tool,
                    "a tool",
                    {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                ),
            ),
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            async with c.stream("POST", "/v1/chat/stream", json={"message": "hi"}, headers={"X-Session-Id": sid}) as r:
                await r.aread()
            agent = app.state.pool.get(sid)
            assert "my_tool" in agent._core.tools


class TestSecurity:
    @pytest.mark.parametrize("bad", ["../etc", "/abs/path", "a/b", "a b", "a.b", ".."])
    async def test_unsafe_session_id_header_rejected(self, bad):
        # X-Session-Id with traversal/invalid chars must 400, never reach a path.
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            r = await c.post("/v1/chat/stream", json={"message": "hi"}, headers={"X-Session-Id": bad})
            assert r.status_code == 400, f"{bad!r} should be rejected"
            assert r.json()["error"]["code"] == "bad_request"

    async def test_unsafe_session_id_in_path_rejected(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            # "a.b" is a single path segment (routes match) but an invalid id (dot).
            assert (await c.get("/v1/sessions/a.b")).status_code == 400
            assert (await c.delete("/v1/sessions/a.b")).status_code == 400

    async def test_safe_session_id_accepted(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            r = await c.post("/v1/chat/stream", json={"message": "hi"}, headers={"X-Session-Id": "valid-id_123"})
            assert r.status_code == 200


class TestApprovals:
    """HTTP route tests for /approve.

    The full mid-stream flow (stream + concurrent POST /approve) can't be tested
    via httpx ASGITransport (asyncio.create_task inside anyio doesn't interleave).
    The queue-bridge + coordinator + handler integration is tested directly in
    ``test_server_approvals.py::TestQueueBridgeIntegration``. In production
    (uvicorn), concurrent connections work correctly.
    """

    async def test_approve_no_active_session_404(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            sid = (await c.post("/v1/sessions")).json()["session_id"]
            r = await c.post(f"/v1/sessions/{sid}/approve", json={"approval_id": "x"})
            assert r.status_code == 404

    async def test_approve_unknown_session_404(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            r = await c.post("/v1/sessions/nope/approve", json={"approval_id": "x"})
            assert r.status_code == 404


class TestAuth:
    """M3: API-key auth + session ownership integration tests."""

    async def test_dev_mode_no_auth_required(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            assert (await c.get("/healthz")).status_code == 200
            assert (await c.post("/v1/sessions")).status_code == 201

    async def test_fail_closed_when_auth_required_and_no_keys(self):
        # C1: default auth_required=true with no keys configured → 401 (was 201,
        # fully open). This is the core fail-closed fix. Health stays open.
        app = create_app(
            _config(server={"auth_required": True}),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            assert (await c.post("/v1/sessions")).status_code == 401
            assert (await c.get("/healthz")).status_code == 200

    async def test_keys_loaded_from_koboi_api_keys_file_env(self, tmp_path, monkeypatch):
        # M8: KOBOI_API_KEYS_FILE env (set by docker-compose) is honored even when
        # the YAML server.api_keys_file is absent.
        import hashlib
        import json as _json

        from koboi.server.app import _build_key_store

        token = "koboi_envfile_secret"
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(_json.dumps([{"id": "k1", "hash": hashlib.sha256(token.encode()).hexdigest()}]))
        monkeypatch.setenv("KOBOI_API_KEYS_FILE", str(keys_file))
        ks = _build_key_store(_config(server={"api_keys_file": None}))
        assert ks.has_keys
        assert ks.validate(token) == "k1"

    def test_serve_app_refuses_non_loopback_without_keys(self, tmp_path):
        # C1: serve_app refuses to start a non-loopback server that would fail
        # open (auth_required=true default + no keys). Sync -- raises before uvicorn.
        from koboi.server.app import serve_app

        cfg = tmp_path / "agent.yaml"
        cfg.write_text(
            "agent:\n  name: t\n  system_prompt: h\n  max_iterations: 3\n"
            "llm:\n  provider: openai\n  model: m\n  api_key: x\n  base_url: http://x\n"
            "memory:\n  backend: in_memory\n"
        )
        with pytest.raises(SystemExit):
            serve_app(str(cfg), host="0.0.0.0", port="0")

    async def test_401_without_bearer_when_keys_configured(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=False,
            api_keys=["secret-key"],
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            assert (await c.post("/v1/sessions")).status_code == 401

    async def test_200_with_valid_bearer(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=False,
            api_keys=["secret-key"],
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/sessions", headers={"Authorization": "Bearer secret-key"})
            assert r.status_code == 201

    async def test_401_with_invalid_bearer(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=False,
            api_keys=["secret-key"],
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/sessions", headers={"Authorization": "Bearer wrong"})
            assert r.status_code == 401

    async def test_healthz_open_even_with_auth(self):
        app = create_app(_config(), enable_cors=False, api_keys=["secret-key"])
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            assert (await c.get("/healthz")).status_code == 200

    async def test_403_accessing_other_owner_session(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=False,
            api_keys=["key-alice", "key-bob"],
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            sid = (await c.post("/v1/sessions", headers={"Authorization": "Bearer key-alice"})).json()["session_id"]
            r = await c.get(f"/v1/sessions/{sid}", headers={"Authorization": "Bearer key-bob"})
            assert r.status_code == 403
            r2 = await c.get(f"/v1/sessions/{sid}", headers={"Authorization": "Bearer key-alice"})
            assert r2.status_code == 200


async def _poll_job(client, job_id, timeout=10.0):
    """Poll GET /v1/jobs/:id until terminal. Returns the final JSON or None on timeout."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await client.get(f"/v1/jobs/{job_id}")
        body = r.json()
        if body["status"] in ("completed", "failed", "timed_out", "cancelled"):
            return body
        await asyncio.sleep(0.1)
    return None


class TestCORS:
    """C4: CORS is config-driven; default is no cross-origin reads (no wildcard)."""

    @staticmethod
    def _app_with(cors_cfg, enable_cors=True):
        return create_app(
            _config(server={"auth_required": False, "cors": cors_cfg}),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=enable_cors,
        )

    async def test_no_acao_when_cors_unconfigured(self):
        # No `server.cors` block (default) → no Access-Control-Allow-Origin header.
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=True,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.options(
                "/v1/sessions",
                headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
            )
            assert r.headers.get("access-control-allow-origin") is None

    async def test_configured_origin_reflected_on_preflight(self):
        app = self._app_with({"allow_origins": ["https://app.example"]})
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.options(
                "/v1/sessions",
                headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST"},
            )
            assert r.headers.get("access-control-allow-origin") == "https://app.example"

    async def test_disallowed_origin_not_reflected(self):
        app = self._app_with({"allow_origins": ["https://app.example"]})
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.options(
                "/v1/sessions",
                headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
            )
            # Disallowed origin is not echoed back (no wildcard reflection).
            assert r.headers.get("access-control-allow-origin") != "https://evil.example"


class TestDocs:
    """H7: interactive docs are off by default; opt-in via server.docs_enabled."""

    async def test_docs_disabled_by_default(self):
        app = create_app(_config(), enable_cors=False)
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            assert (await c.get("/docs")).status_code == 404
            assert (await c.get("/openapi.json")).status_code == 404

    async def test_docs_enabled_when_configured(self):
        app = create_app(_config(server={"auth_required": False, "docs_enabled": True}), enable_cors=False)
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            assert (await c.get("/docs")).status_code == 200
            assert (await c.get("/openapi.json")).status_code == 200


class TestSessionOwnershipH1:
    """H1: any newly-touched session acquires an owner (no unowned sessions)."""

    async def test_header_session_claims_owner(self):
        # A fresh X-Session-Id on /chat/stream now claims ownership (previously
        # left unowned → any caller could share it).
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=False,
            api_keys=["keyA"],
        )
        sid = "header-shared-sid"
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            async with c.stream(
                "POST",
                "/v1/chat/stream",
                json={"message": "hi"},
                headers={"X-Session-Id": sid, "Authorization": "Bearer keyA"},
            ) as r:
                await r.aread()
            assert app.state.ownership.get_owner(sid) is not None

    async def test_reused_job_session_blocks_other_owner(self):
        # A reused session_id on /v1/jobs is claimed by the first caller; a
        # different caller is rejected (403), not silently shared.
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
            api_keys=["keyA", "keyB"],
        )
        sid = "job-shared-sid"
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            rA = await c.post(
                "/v1/jobs",
                json={"message": "a", "session_id": sid},
                headers={"Authorization": "Bearer keyA"},
            )
            assert rA.status_code == 202
            rB = await c.post(
                "/v1/jobs",
                json={"message": "b", "session_id": sid},
                headers={"Authorization": "Bearer keyB"},
            )
            assert rB.status_code == 403


class TestResourceLimits:
    """H6: request-body caps reject oversized payloads (Pydantic 422)."""

    async def test_oversize_chat_message_rejected(self):
        app = create_app(_config(), enable_cors=False)
        big = "x" * 70000  # > 65536 cap
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/chat/stream", json={"message": big})
            assert r.status_code == 422

    async def test_oversize_messages_list_rejected(self):
        app = create_app(_config(), enable_cors=False)
        payload = {"messages": [{"role": "user", "content": "x"} for _ in range(60)]}  # > 50 cap
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/chat/stream", json=payload)
            assert r.status_code == 422

    async def test_oversize_job_message_rejected(self):
        app = create_app(_config(), enable_cors=False)
        big = "x" * 70000
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": big})
            assert r.status_code == 422


class TestJobStreamCap:
    """M3: per-owner concurrent job-stream cap (slowloris guard)."""

    @staticmethod
    def _owner(token: str) -> str:
        return "env:" + hashlib.sha256(token.encode()).hexdigest()[:12]

    async def test_stream_over_cap_returns_429(self):
        app = create_app(
            _config(server={"auth_required": True, "limits": {"job_streams_per_owner": 1}}),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
            api_keys=["keyA"],
        )
        owner = self._owner("keyA")
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "do"}, headers={"Authorization": "Bearer keyA"})
            job_id = r.json()["job_id"]
            app.state.job_streams[owner] = 1  # simulate one active stream (at cap)
            s = await c.get(f"/v1/jobs/{job_id}/stream", headers={"Authorization": "Bearer keyA"})
            assert s.status_code == 429

    async def test_stream_releases_slot_when_done(self):
        app = create_app(
            _config(server={"auth_required": True, "limits": {"job_streams_per_owner": 2}}),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
            api_keys=["keyA"],
        )
        owner = self._owner("keyA")
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "do"}, headers={"Authorization": "Bearer keyA"})
            job_id = r.json()["job_id"]
            app.state.job_registry.get(job_id).terminal.set()  # simulate terminal job
            async with c.stream("GET", f"/v1/jobs/{job_id}/stream", headers={"Authorization": "Bearer keyA"}) as s:
                await s.aread()
            assert app.state.job_streams.get(owner, 0) == 0


class TestJobs:
    """M4: autonomous background jobs integration tests."""

    async def test_submit_and_poll_until_completed(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="job done")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "do something"})
            assert r.status_code == 202
            job_id = r.json()["job_id"]
            assert r.json()["session_id"]  # dedicated session

            result = await _poll_job(c, job_id)
            assert result is not None
            assert result["status"] == "completed"

    async def test_list_jobs_by_owner(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            await c.post("/v1/jobs", json={"message": "job 1"})
            await c.post("/v1/jobs", json={"message": "job 2"})
            r = await c.get("/v1/jobs")
            assert r.status_code == 200
            assert len(r.json()) >= 2

    async def test_get_unknown_job_404(self):
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=_app())) as c:
            assert (await c.get("/v1/jobs/nonexistent")).status_code == 404

    async def test_cancel_job(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "long job"})
            job_id = r.json()["job_id"]
            r2 = await c.post(f"/v1/jobs/{job_id}/cancel")
            # The job may have completed before the cancel arrives (MockClient is instant).
            # Either outcome is valid: 200 (cancelled) or 409 (already terminal).
            assert r2.status_code in (200, 409)

    async def test_cancel_already_terminal_409(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="done")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "quick"})
            job_id = r.json()["job_id"]
            await _poll_job(c, job_id)  # wait for completion
            r2 = await c.post(f"/v1/jobs/{job_id}/cancel")
            assert r2.status_code == 409

    async def test_idempotency_key_returns_same_job(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r1 = await c.post("/v1/jobs", json={"message": "a"}, headers={"Idempotency-Key": "key-abc"})
            r2 = await c.post("/v1/jobs", json={"message": "a"}, headers={"Idempotency-Key": "key-abc"})
            assert r1.json()["job_id"] == r2.json()["job_id"]

    async def test_job_403_other_owner(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
            api_keys=["key-a", "key-b"],
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "secret"}, headers={"Authorization": "Bearer key-a"})
            job_id = r.json()["job_id"]
            r2 = await c.get(f"/v1/jobs/{job_id}", headers={"Authorization": "Bearer key-b"})
            assert r2.status_code == 403

    async def test_job_stream_replays_events(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="streamed result")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "go"})
            job_id = r.json()["job_id"]
            # Wait for completion so the stream replays the full buffer.
            await _poll_job(c, job_id)
            # Now stream → should replay buffered events + [DONE].
            async with c.stream("GET", f"/v1/jobs/{job_id}/stream") as resp:
                assert resp.status_code == 200
                text = (await resp.aread()).decode()
            events = _parse_sse(text)
            types = [e["type"] if isinstance(e, dict) else e for e in events]
            assert "complete" in types
            assert types[-1] == "[DONE]"


class TestWorkdirGC:
    """16.24: workdir TTL GC tests."""

    def test_cleanup_removes_old_dirs(self, tmp_path):
        import os
        import time

        from koboi.server.app import _cleanup_workdirs

        ws = tmp_path / "workspace"
        ws.mkdir()
        old_dir = ws / "old_session"
        old_dir.mkdir()
        new_dir = ws / "new_session"
        new_dir.mkdir()
        old_time = time.time() - 2 * 86400
        os.utime(old_dir, (old_time, old_time))
        removed = _cleanup_workdirs(str(ws), ttl_seconds=86400)
        assert removed == 1
        assert not old_dir.exists()
        assert new_dir.exists()

    def test_cleanup_no_root(self, tmp_path):
        from koboi.server.app import _cleanup_workdirs

        assert _cleanup_workdirs(str(tmp_path / "nonexistent"), ttl_seconds=60) == 0

    def test_cleanup_keeps_recent(self, tmp_path):
        from koboi.server.app import _cleanup_workdirs

        ws = tmp_path / "workspace"
        ws.mkdir()
        recent = ws / "recent"
        recent.mkdir()
        removed = _cleanup_workdirs(str(ws), ttl_seconds=86400)
        assert removed == 0
        assert recent.exists()


class TestChatIdempotency:
    """G6: /chat/stream rejects a duplicate Idempotency-Key with 409 (no replay)."""

    async def test_duplicate_key_returns_409(self):
        app = _app()
        headers = {"X-Session-Id": "sess-idem", "Idempotency-Key": "k1"}
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            async with c.stream("POST", "/v1/chat/stream", json={"message": "hi"}, headers=headers) as r1:
                await r1.aread()
                assert r1.status_code == 200
            r2 = await c.post("/v1/chat/stream", json={"message": "hi"}, headers=headers)
            assert r2.status_code == 409
            assert r2.json()["error"]["code"] == "duplicate_request"

    async def test_no_key_no_dedup(self):
        app = _app()
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            async with c.stream(
                "POST", "/v1/chat/stream", json={"message": "hi"}, headers={"X-Session-Id": "sess-a"}
            ) as r1:
                await r1.aread()
                assert r1.status_code == 200
            async with c.stream(
                "POST", "/v1/chat/stream", json={"message": "hi"}, headers={"X-Session-Id": "sess-a"}
            ) as r2:
                await r2.aread()
                assert r2.status_code == 200  # no key → no dedup


class TestJobSandboxGuard:
    """C3: autonomous jobs require a restricted sandbox; passthrough is refused."""

    async def test_job_refused_with_passthrough_sandbox(self):
        # Explicit passthrough → the job is refused at execute time (failed).
        app = create_app(
            _config(sandbox={"backend": "passthrough"}),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "do"})
            assert r.status_code == 202
            body = await _poll_job(c, r.json()["job_id"])
            assert body is not None and body["status"] == "failed"
            assert body.get("error_class") == "PermissionError" or "restricted" in (body.get("error") or "").lower()

    async def test_job_runs_with_restricted_sandbox(self):
        # Restricted (the _config default) → job completes normally.
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "do"})
            body = await _poll_job(c, r.json()["job_id"])
            assert body is not None and body["status"] == "completed"


class TestPerTenantLimit:
    """G5a: per_tenant_max enforced only with real auth; counts running jobs."""

    @staticmethod
    def _config(per_tenant: int = 1) -> Config:
        return Config.from_dict(
            {
                "agent": {"name": "t", "max_iterations": 1},
                "llm": {"provider": "openai", "model": "m", "api_key": "x", "base_url": "http://x"},
                "memory": {"backend": "in_memory"},
                "sandbox": {"backend": "restricted"},  # C3: jobs require containment
                "server": {"auth_required": False},  # C1: dev-open (no api_keys in these tests)
                "jobs": {"per_tenant_max": per_tenant},
            },
            validate=True,
        )

    @staticmethod
    def _mkapp(config: Config, *, api_keys=None) -> Config:
        return create_app(
            config,
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
            api_keys=api_keys,
        )

    async def test_skipped_in_dev_mode(self):
        # No auth → every owner is "dev". Pre-seed a running "dev" job so enforcement
        # WOULD yield 429 if applied; dev-skip means the submit still succeeds (202).
        app = self._mkapp(self._config())
        seed = app.state.job_registry.register("seed", "s", "dev")
        seed.status = "running"
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "do"})
            assert r.status_code == 202

    async def test_enforced_with_auth(self):
        owner = "env:" + hashlib.sha256(b"secret").hexdigest()[:12]
        app = self._mkapp(self._config(), api_keys=["secret"])
        seed = app.state.job_registry.register("seed", "s", owner)
        seed.status = "running"  # count=1 >= per_tenant_max(1) → next submit is 429
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/jobs", json={"message": "do"}, headers={"Authorization": "Bearer secret"})
            assert r.status_code == 429
            assert r.json()["error"]["code"] == "too_many_jobs_per_tenant"


class TestJobQueueBacklog:
    """G5c-b: overflow queues (up to queue_depth) then 429s; cancel + drain."""

    @staticmethod
    def _config(max_concurrent: int = 1, queue_depth: int = 2) -> Config:
        return Config.from_dict(
            {
                "agent": {"name": "t", "max_iterations": 1},
                "llm": {"provider": "openai", "model": "m", "api_key": "x", "base_url": "http://x"},
                "memory": {"backend": "in_memory"},
                "sandbox": {"backend": "restricted"},  # C3: jobs require containment
                "server": {"auth_required": False},  # C1: dev-open (no api_keys in these tests)
                "jobs": {"max_concurrent": max_concurrent, "queue_depth": queue_depth, "per_tenant_max": 64},
            },
            validate=True,
        )

    @staticmethod
    def _mkapp(config: Config):
        return create_app(
            config,
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
        )

    async def test_overflow_queues_then_rejects(self):
        app = self._mkapp(self._config())
        seed = app.state.job_registry.register("seed", "s", "dev")
        seed.status = "running"  # fill the single slot (no task → never completes/drains)
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            ra = await c.post("/v1/jobs", json={"message": "A"})
            rb = await c.post("/v1/jobs", json={"message": "B"})
            rc = await c.post("/v1/jobs", json={"message": "C"})
            assert ra.status_code == 202 and ra.json()["status"] == "pending"
            assert rb.status_code == 202 and rb.json()["status"] == "pending"
            assert rc.status_code == 429 and rc.json()["error"]["code"] == "queue_full"
            assert app.state.job_registry.pending_count == 2

    async def test_cancel_queued_job(self):
        app = self._mkapp(self._config())
        seed = app.state.job_registry.register("seed", "s", "dev")
        seed.status = "running"
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            ra = await c.post("/v1/jobs", json={"message": "A"})
            job_id = ra.json()["job_id"]
            assert app.state.job_registry.pending_count == 1
            rc = await c.post(f"/v1/jobs/{job_id}/cancel")
            assert rc.status_code == 200
            assert app.state.job_registry.pending_count == 0
            assert app.state.job_store.get(job_id)["status"] == "cancelled"

    async def test_drain_to_completion_under_burst(self):
        # 6 jobs, max_concurrent=2, queue_depth=10 → some queue, then all drain to completion.
        app = self._mkapp(self._config(max_concurrent=2, queue_depth=10))
        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
            responses = await asyncio.gather(*(c.post("/v1/jobs", json={"message": f"j{i}"}) for i in range(6)))
            ids = [r.json()["job_id"] for r in responses]
            for jid in ids:
                body = await _poll_job(c, jid)
                assert body is not None and body["status"] == "completed"
