"""Unit tests for koboi/server/sse.py (pure encoder, no FastAPI)."""

from __future__ import annotations

import json


from koboi.events import CompleteEvent, TextDeltaEvent, ToolCallEvent
from koboi.server.sse import DONE_FRAME, _frame, sse_stream
from koboi.types import AgentResponse, TokenUsage


async def _gen(events):
    for e in events:
        yield e


def _parse(data: bytes) -> list:
    out = []
    for line in data.decode().split("\n"):
        if line.startswith("data: "):
            payload = line[6:]
            out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


class TestSseEncoder:
    async def test_frames_each_event_then_done(self):
        events = [TextDeltaEvent(content="hi"), ToolCallEvent("calc", "tc1", "{}")]
        data = b""
        async for chunk in sse_stream(_gen(events)):
            data += chunk
        parsed = _parse(data)
        assert parsed[0] == {"type": "text_delta", "content": "hi"}
        assert parsed[1]["type"] == "tool_call"
        assert parsed[-1] == "[DONE]"
        assert data.endswith(DONE_FRAME)

    async def test_done_on_empty_generator(self):
        data = b""
        async for chunk in sse_stream(_gen([])):
            data += chunk
        assert _parse(data) == ["[DONE]"]

    async def test_complete_event_has_token_usage(self):
        resp = AgentResponse(content="answer", usage=TokenUsage(prompt_tokens=10, completion_tokens=5))
        data = b""
        async for chunk in sse_stream(_gen([CompleteEvent(response=resp, content="answer")])):
            data += chunk
        complete = _parse(data)[0]
        assert complete["type"] == "complete"
        assert complete["token_usage"]["total_tokens"] == 15

    async def test_error_emits_error_frame_then_done(self):
        async def boom():
            yield TextDeltaEvent(content="x")
            raise RuntimeError("kaboom")

        data = b""
        async for chunk in sse_stream(boom()):
            data += chunk
        parsed = _parse(data)
        assert parsed[0] == {"type": "text_delta", "content": "x"}
        assert parsed[1]["type"] == "error"
        assert parsed[1]["error"] == "kaboom"
        assert parsed[1]["code"] == "internal_error"
        assert parsed[1]["retriable"] is False
        assert parsed[-1] == "[DONE]"

    def test_frame_compact_json(self):
        assert _frame({"type": "text_delta", "content": "hi"}) == b'data: {"type":"text_delta","content":"hi"}\n\n'

    def test_frame_escapes_crlf_in_content(self):
        # M14: a bare \r/\n in event content must not break the SSE frame boundary
        # (json.dumps escapes control chars; invariant locked vs future regressions).
        framed = _frame({"type": "text_delta", "content": "line1\r\nline2"})
        assert framed.count(b"\n\n") == 1  # only the frame terminator
        assert b"\r" not in framed  # raw CR is escaped, never literal in the frame
        assert b"\\r\\n" in framed  # JSON-escaped form present
