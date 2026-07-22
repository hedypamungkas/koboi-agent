"""Wave 2 item 4: per-run token/cost budget (agent.max_total_tokens / max_cost_usd)."""

from __future__ import annotations

import pytest

from koboi.exceptions import AgentBudgetExceededError
from koboi.events import CompleteEvent, ErrorEvent
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.types import TokenUsage

from tests.conftest import MockClient, make_mock_response, make_mock_tool_call, make_tool_registry


def _core(responses, **kw) -> AgentCore:
    return AgentCore(
        client=MockClient(responses),
        memory=ConversationMemory(),
        tools=make_tool_registry(),
        max_iterations=8,
        **kw,
    )


def _tool_loop_responses(n_tool_iters: int = 3):
    """n tool-call iterations (10+20 tokens each via make_mock_response) then a final."""
    responses = [
        make_mock_response(tool_calls=[make_mock_tool_call("get_weather", {"city": "Jakarta"})])
        for _ in range(n_tool_iters)
    ]
    responses.append(make_mock_response("done"))
    return responses


class TestBudgetExceeded:
    async def test_token_budget_raises(self):
        # Each mock response spends 30 tokens; ceiling 30 -> exceeded at iteration-2 top.
        core = _core(_tool_loop_responses(), max_total_tokens=30)
        with pytest.raises(AgentBudgetExceededError) as ei:
            await core.run("do work")
        assert ei.value.spent_tokens == 30
        assert "max_total_tokens=30" in str(ei.value)

    async def test_cost_budget_raises(self):
        # 30 tokens/iter -> $0.00035/iter at default prices; ceiling below that.
        core = _core(_tool_loop_responses(), max_cost_usd=0.0003)
        with pytest.raises(AgentBudgetExceededError) as ei:
            await core.run("do work")
        assert "max_cost_usd=0.0003" in ei.value.limit

    async def test_graceful_returns_degraded_success(self):
        core = _core(
            _tool_loop_responses(),
            max_total_tokens=30,
            graceful_max_iter=True,
        )
        result = await core.run("do work")
        assert result.success is True
        assert result.metadata["budget_degraded"] is True
        assert result.metadata["budget_spent_tokens"] == 30
        assert result.metadata["budget_limit"] == "max_total_tokens=30"
        assert result.iterations_used == 1

    async def test_completed_run_within_budget_unaffected(self):
        # Completes on iteration 1: the check never fires (usage accrues after).
        core = _core([make_mock_response("hi")], max_total_tokens=30)
        result = await core.run("hello")
        assert result.success is True
        assert result.content == "hi"

    async def test_no_budget_configured_is_unbounded(self):
        core = _core(_tool_loop_responses(5))
        result = await core.run("do work")
        assert result.success is True
        assert result.token_usage.total_tokens == 30 * 6  # 5 tool iters + final


class TestBudgetMath:
    def test_budget_info_none_when_unconfigured(self):
        core = _core([])
        assert core._budget_exceeded_info(TokenUsage(prompt_tokens=10**9)) is None

    def test_budget_info_none_when_no_usage(self):
        core = _core([], max_total_tokens=1)
        assert core._budget_exceeded_info(None) is None

    def test_cost_math_uses_prices(self):
        core = _core(
            [],
            max_cost_usd=0.5,
            token_prices={"input_per_1k": 1.0, "output_per_1k": 2.0},
        )
        usage = TokenUsage(prompt_tokens=100, completion_tokens=200)
        info = core._budget_exceeded_info(usage)
        # 100*1.0/1000 + 200*2.0/1000 = 0.5 >= 0.5
        assert info is not None
        assert info["budget_spent_usd"] == 0.5
        assert info["budget_limit"] == "max_cost_usd=0.5"

    def test_token_limit_checked_before_cost(self):
        core = _core([], max_total_tokens=10, max_cost_usd=10.0)
        info = core._budget_exceeded_info(TokenUsage(prompt_tokens=10, completion_tokens=5))
        assert info["budget_limit"] == "max_total_tokens=10"

    def test_cost_math_includes_reasoning_tokens(self):
        # Reasoning tokens are priced (default = output rate) so a reasoning-heavy
        # run trips the cost ceiling, not just prompt+completion.
        core = _core(
            [],
            max_cost_usd=0.01,
            token_prices={"input_per_1k": 1.0, "output_per_1k": 1.0, "reasoning_per_1k": 1.0},
        )
        # 20 reasoning tokens * 1.0/1k = 0.02 >= 0.01 ceiling -> exceeded.
        info = core._budget_exceeded_info(TokenUsage(prompt_tokens=0, completion_tokens=0, reasoning_tokens=20))
        assert info is not None
        assert "max_cost_usd" in info["budget_limit"]

    def test_usage_none_with_budget_falls_back_to_estimate(self):
        # A provider returning usage=None must NOT silently disable a configured
        # ceiling -- _update_usage estimates from the content.
        core = _core([], max_total_tokens=5)
        from koboi.types import AgentResponse

        resp = AgentResponse(content="x" * 100, usage=None)
        total = core._update_usage(resp, None)
        assert total is not None and total.completion_tokens > 0

    def test_usage_none_estimates_prompt_tokens_too(self):
        # I-2: counting completion-only left the ceiling ~10x too high for
        # usage-omitting providers (prompt dominates a coding turn). The caller
        # passes the current message-list estimate; _update_usage must count it.
        core = _core([], max_total_tokens=5)
        from koboi.types import AgentResponse

        resp = AgentResponse(content="x" * 100, usage=None)
        total = core._update_usage(resp, None, estimated_prompt_tokens=2000)
        assert total is not None
        assert total.prompt_tokens == 2000  # the dominant cost side is now counted
        assert total.completion_tokens > 0


class TestBudgetStream:
    async def test_stream_yields_budget_error(self):
        core = _core(_tool_loop_responses(), max_total_tokens=30)
        events = [ev async for ev in core.run_stream("do work")]
        errors = [ev for ev in events if isinstance(ev, ErrorEvent)]
        assert errors and isinstance(errors[-1].error, AgentBudgetExceededError)

    async def test_stream_graceful_completes_with_metadata(self):
        core = _core(_tool_loop_responses(), max_total_tokens=30, graceful_max_iter=True)
        events = [ev async for ev in core.run_stream("do work")]
        completes = [ev for ev in events if isinstance(ev, CompleteEvent)]
        assert completes
        assert completes[-1].metadata["budget_degraded"] is True
        assert completes[-1].metadata["budget_spent_tokens"] == 30
