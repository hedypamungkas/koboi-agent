"""Integration tests for koboi/server (FastAPI app via httpx ASGI transport).

Gated on the ``api`` extra: ``pytest.importorskip("fastapi")`` skips the module
cleanly when fastapi isn't installed (CI without the extra).
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.server import create_app  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402


def _config() -> Config:
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
        },
        validate=True,
    )


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
