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
