"""tests/test_self_healing_p3.py -- graceful degrade on max_iterations (self-healing P3).

When the loop exhausts max_iterations, an opted-in agent returns a side-LLM summary
of partial progress (RunResult.success=True, metadata max_iter_degraded) instead of
raising AgentMaxIterationsError. Fail-soft fallback to the last assistant message /
a generic notice when the summary call fails. Both run() and run_stream().
"""

from __future__ import annotations

import pytest

from koboi.events import CompleteEvent, ErrorEvent
from koboi.exceptions import AgentMaxIterationsError
from koboi.guardrails.base import BaseGuardrail
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.types import GuardrailResult
from tests.conftest import make_mock_response, make_mock_tool_call, make_tool_registry


def _looping_client(mock_client, n_iter: int, summary_resp=None):
    """A MockClient that returns a tool_call response for n_iter iterations, then summary_resp."""
    tc = make_mock_response(None, [make_mock_tool_call("calculate", {"expression": "1+1"})])
    responses = [tc] * n_iter
    if summary_resp is not None:
        responses.append(summary_resp)
    return mock_client(responses=responses)


class TestGracefulMaxIter:
    async def test_graceful_returns_summary(self, mock_client):
        client = _looping_client(mock_client, 2, make_mock_response("Summary of progress"))
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=make_tool_registry(),
            max_iterations=2,
            graceful_max_iter=True,
        )
        result = await agent.run("do a big task")
        assert result.success is True
        assert result.content == "Summary of progress"
        assert result.iterations_used == 2
        assert result.metadata["max_iter_degraded"] is True

    async def test_graceful_falls_back_when_summary_empty(self, mock_client):
        # Summary call returns empty content -> fall back (no non-empty assistant msg -> generic).
        client = _looping_client(mock_client, 2, make_mock_response(None))
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=make_tool_registry(),
            max_iterations=2,
            graceful_max_iter=True,
        )
        result = await agent.run("do a big task")
        assert result.success is True
        assert result.metadata["max_iter_degraded"] is True
        assert "unable to complete" in result.content  # generic fallback notice

    async def test_non_graceful_still_raises(self, mock_client):
        client = _looping_client(mock_client, 2)  # no summary response; loop exhausts -> raise
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=make_tool_registry(),
            max_iterations=2,
            graceful_max_iter=False,
        )
        with pytest.raises(AgentMaxIterationsError):
            await agent.run("do a big task")

    async def test_graceful_run_stream_yields_complete(self, mock_client):
        client = _looping_client(mock_client, 2, make_mock_response("Stream summary"))
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=make_tool_registry(),
            max_iterations=2,
            graceful_max_iter=True,
        )
        events = [e async for e in agent.run_stream("do a big task")]
        # The only CompleteEvent yielded to the consumer is the graceful summary (no ErrorEvent).
        assert not any(isinstance(e, ErrorEvent) for e in events)
        completes = [e for e in events if isinstance(e, CompleteEvent)]
        assert completes and completes[-1].content == "Stream summary"
        assert completes[-1].metadata["max_iter_degraded"] is True

    async def test_graceful_summary_respects_blocking_guardrail(self, mock_client):
        # A blocking output guardrail must run on the summary (P3 fix) -> generic notice.
        class _BlockGuardrail(BaseGuardrail):
            async def check(self, content, context=None):
                return GuardrailResult(passed=False, reason="blocked", action="block")

        client = _looping_client(mock_client, 2, make_mock_response("leaked secret"))
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=make_tool_registry(),
            max_iterations=2,
            graceful_max_iter=True,
            output_guardrails=[_BlockGuardrail()],
        )
        result = await agent.run("do a big task")
        assert result.success is True
        assert result.metadata["max_iter_degraded"] is True
        assert "leaked secret" not in result.content  # guardrail blocked the raw summary
        assert "unable to complete" in result.content  # fell back to the generic notice
