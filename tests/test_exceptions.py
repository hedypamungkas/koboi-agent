"""Tests for koboi/exceptions.py -- Exception hierarchy."""

from __future__ import annotations

from koboi.exceptions import (
    AgentAbortedError,
    AgentError,
    AgentGuardrailError,
    AgentMaxIterationsError,
    AgentStreamError,
    AgentTimeoutError,
    AgentToolError,
)


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(AgentMaxIterationsError, AgentError)
        assert issubclass(AgentGuardrailError, AgentError)
        assert issubclass(AgentToolError, AgentError)
        assert issubclass(AgentTimeoutError, AgentError)
        assert issubclass(AgentStreamError, AgentError)
        assert issubclass(AgentAbortedError, AgentError)

    def test_agent_error_is_exception(self):
        assert issubclass(AgentError, Exception)

    def test_max_iterations(self):
        e = AgentMaxIterationsError(10)
        assert "10" in str(e)

    def test_guardrail_error(self):
        e = AgentGuardrailError("blocked")
        assert "blocked" in str(e)

    def test_tool_error(self):
        e = AgentToolError("my_tool", "tool failed")
        assert "tool failed" in str(e)
        assert e.tool_name == "my_tool"

    def test_timeout_error(self):
        e = AgentTimeoutError("timed out")
        assert "timed out" in str(e)

    def test_stream_error(self):
        e = AgentStreamError("stream broke")
        assert "stream broke" in str(e)

    def test_aborted_error(self):
        e = AgentAbortedError("user cancelled")
        assert "user cancelled" in str(e)
