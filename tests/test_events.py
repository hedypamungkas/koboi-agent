"""Tests for koboi/events.py -- StreamEvent types."""
from __future__ import annotations

from koboi.events import (
    CompleteEvent,
    ErrorEvent,
    IterationEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class TestStreamEvents:
    def test_text_delta(self):
        e = TextDeltaEvent(content="hello")
        assert e.content == "hello"

    def test_tool_call(self):
        e = ToolCallEvent(tool_name="read", tool_call_id="tc1", arguments='{"path":"f.py"}')
        assert e.tool_name == "read"
        assert e.tool_call_id == "tc1"

    def test_tool_result(self):
        e = ToolResultEvent(tool_name="read", tool_call_id="tc1", result="file content")
        assert e.result == "file content"

    def test_iteration(self):
        e = IterationEvent(iteration=3, messages_count=5, tokens_estimated=1200)
        assert e.iteration == 3
        assert e.messages_count == 5
        assert e.tokens_estimated == 1200

    def test_iteration_defaults(self):
        e = IterationEvent(iteration=1)
        assert e.messages_count == 0
        assert e.tokens_estimated == 0

    def test_complete_with_content(self):
        e = CompleteEvent(content="done")
        assert e.content == "done"
        assert e.response is None

    def test_complete_with_response(self):
        from koboi.types import AgentResponse
        resp = AgentResponse(content="ok", tool_calls=[], usage=None)
        e = CompleteEvent(response=resp)
        assert e.response.content == "ok"

    def test_error(self):
        err = ValueError("bad")
        e = ErrorEvent(error=err)
        assert e.error is err


class TestCompleteEventEnrichment:
    def test_elapsed_seconds_default(self):
        e = CompleteEvent(content="done")
        assert e.elapsed_seconds == 0.0

    def test_elapsed_seconds_set(self):
        e = CompleteEvent(content="done", elapsed_seconds=2.5)
        assert e.elapsed_seconds == 2.5

    def test_iterations_used_default(self):
        e = CompleteEvent(content="done")
        assert e.iterations_used == 0

    def test_iterations_used_set(self):
        e = CompleteEvent(content="done", iterations_used=3)
        assert e.iterations_used == 3

    def test_tools_used_default(self):
        e = CompleteEvent(content="done")
        assert e.tools_used == []

    def test_tools_used_set(self):
        e = CompleteEvent(content="done", tools_used=["web_search", "calculator"])
        assert e.tools_used == ["web_search", "calculator"]

    def test_all_enriched_fields(self):
        from koboi.types import AgentResponse
        resp = AgentResponse(content="result", tool_calls=[], usage=None)
        e = CompleteEvent(
            response=resp,
            content="result",
            elapsed_seconds=1.23,
            iterations_used=2,
            tools_used=["read_file"],
        )
        assert e.response.content == "result"
        assert e.elapsed_seconds == 1.23
        assert e.iterations_used == 2
        assert e.tools_used == ["read_file"]
