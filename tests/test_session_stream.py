"""Tests for B2 -- replayable session event stream (GET /v1/sessions/{id}/stream).

Closes the B1 post-handover blindness gap: a supervisor can replay a session's
buffered event history + live-tail. Uses a short ``session_stream_timeout`` so the
long-lived stream ends (``[DONE]``) for ``aread()``; the replay burst arrives in
the first poll, then the deadline closes the stream.
"""

from __future__ import annotations

import json

import pytest

from koboi.server.session_events import SessionEventRegistry


# ---------------------------------------------------------------------------
# Unit (no fastapi)
# ---------------------------------------------------------------------------


class TestSessionEventRegistry:
    def test_append_capped_keeps_newest(self):
        r = SessionEventRegistry(max_events=3)
        for e in ("a", "b", "c", "d", "e"):
            r.append_event("s1", e)
        assert r.get_events("s1") == ["c", "d", "e"]  # last 3 retained

    def test_get_events_returns_copy(self):
        r = SessionEventRegistry()
        r.append_event("s1", "x")
        got = r.get_events("s1")
        got.append("MUTATED")
        assert r.get_events("s1") == ["x"]  # internal list untouched

    def test_unseen_session_returns_empty(self):
        assert SessionEventRegistry().get_events("nope") == []

    def test_forget_drops_buffer(self):
        r = SessionEventRegistry()
        r.append_event("s1", "x")
        r.forget("s1")
        assert r.get_events("s1") == []
        r.forget("never")  # no-op, no error


# ---------------------------------------------------------------------------
# Server-level (fastapi) -- replay / retention / multi-turn / DELETE
# ---------------------------------------------------------------------------


def _need_fastapi():
    pytest.importorskip("fastapi")
    import httpx  # noqa: F401


def _config():
    from koboi.config import Config

    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "test"},
            "memory": {"backend": "in_memory"},
            "sandbox": {"backend": "restricted"},
            "server": {"auth_required": False, "limits": {"session_stream_timeout": 0.5}},
            "tools": {"builtin": ["transfer_to_human"]},
        },
        validate=True,
    )


def _parse_sse(text: str) -> list:
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            payload = line[6:]
            out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


class TestSessionStreamReplay:
    async def test_replays_completed_turn(self):
        _need_fastapi()
        import httpx
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response

        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hello there")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            # Turn 1: a normal completion. Drain the /chat/stream so it fully buffers.
            async with c.stream("POST", "/v1/chat/stream", json={"message": "hi"}, headers={"X-Session-Id": "s1"}) as r:
                await r.aread()
            # Replay: GET /sessions/s1/stream -> the turn's events replay, then [DONE] (deadline).
            async with c.stream("GET", "/v1/sessions/s1/stream", headers={"X-Session-Id": "s1"}) as r:
                body = (await r.aread()).decode()
        events = _parse_sse(body)
        types = [e.get("type") if isinstance(e, dict) else e for e in events]
        assert "text_delta" in types or "complete" in types, f"expected replay of turn events, got {types}"
        assert types[-1] == "[DONE]"

    async def test_retains_handover_event_after_stream_ends(self):
        """The B1 gap: today the HandoverEvent is GC'd with the /chat/stream queue.
        B2 buffers it, so a supervisor replaying after the handover sees 'handover'."""
        _need_fastapi()
        import httpx
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        handover_resp = make_mock_response(
            content="transferring",
            tool_calls=[make_mock_tool_call("transfer_to_human", {"reason": "r", "summary": "s"})],
        )
        app = create_app(_config(), client_factory=lambda: MockClient([handover_resp]), enable_cors=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            async with c.stream(
                "POST", "/v1/chat/stream", json={"message": "help", "mode": "act"}, headers={"X-Session-Id": "s2"}
            ) as r:
                await r.aread()
            async with c.stream("GET", "/v1/sessions/s2/stream", headers={"X-Session-Id": "s2"}) as r:
                body = (await r.aread()).decode()
        events = _parse_sse(body)
        types = [e.get("type") if isinstance(e, dict) else e for e in events]
        assert "handover" in types, f"handover event must survive for replay; got {types}"

    async def test_replay_accumulates_across_turns(self):
        _need_fastapi()
        import httpx
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response

        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="turn one"), make_mock_response(content="turn two")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            for msg in ("first", "second"):
                async with c.stream("POST", "/v1/chat/stream", json={"message": msg}, headers={"X-Session-Id": "s3"}) as r:
                    await r.aread()
            async with c.stream("GET", "/v1/sessions/s3/stream", headers={"X-Session-Id": "s3"}) as r:
                body = (await r.aread()).decode()
        events = _parse_sse(body)
        # Both turns' text content appears in the replayed text_delta payloads.
        contents = " ".join(e.get("content", "") for e in events if isinstance(e, dict))
        assert "turn one" in contents
        assert "turn two" in contents

    async def test_delete_clears_buffer(self):
        _need_fastapi()
        import httpx
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response

        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hello")]),
            enable_cors=False,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            async with c.stream("POST", "/v1/chat/stream", json={"message": "hi"}, headers={"X-Session-Id": "s4"}) as r:
                await r.aread()
            assert app.state.session_events.get_events("s4")  # buffered
            await c.delete("/v1/sessions/s4", headers={"X-Session-Id": "s4"})
            assert app.state.session_events.get_events("s4") == []  # forgotten
