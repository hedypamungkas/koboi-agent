"""tests/test_self_healing_p2.py -- self-healing P2a (escalation ladder).

Unit tests for RecoveryBudget / FailureClassifierHook / LadderRouterHook, the
DoomLoopHook SESSION_START reset fix, and integration tests for the full ladder
(reflect-with-budget -> complete; budget exhausted -> handover; handover suppressed
while reflect budget remains).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from koboi.exceptions import AgentHandoverError
from koboi.guardrails.base import BaseGuardrail
from koboi.harness.doom_loop import DoomLoopConfig
from koboi.harness.recovery_budget import RecoveryBudget
from koboi.hooks.chain import HookChain, HookContext, HookEvent
from koboi.hooks.doom_loop_hook import DoomLoopHook
from koboi.hooks.failure_classifier_hook import FailureClassifierHook
from koboi.hooks.handover_detection_hook import HandoverDetectionHook
from koboi.hooks.ladder_router_hook import LadderRouterHook
from koboi.hooks.reflection_hook import ReflectionHook
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import AgentResponse, GuardrailResult
from tests.conftest import make_mock_response


class _CriticClient:
    def __init__(self, critique="fix it"):
        self.critique = critique
        self.calls = 0

    async def complete(self, messages, tools=None, response_format=None):
        self.calls += 1
        return AgentResponse(content=self.critique)


class _FakeGroundingGuardrail(BaseGuardrail):
    def __init__(self, coverages):
        self._coverages = list(coverages)
        self._n = 0
        self.last_coverage = None

    async def check(self, content, context=None):
        cov = self._coverages[min(self._n, len(self._coverages) - 1)]
        self._n += 1
        self.last_coverage = cov
        if cov >= 0.8:
            return GuardrailResult(passed=True)
        return GuardrailResult(
            passed=False, reason=f"coverage {cov}", action="abstain", sanitized_content="I cannot answer confidently."
        )


# --------------------------------------------------------------------------- RecoveryBudget


class TestRecoveryBudget:
    def test_consume_until_exhausted(self):
        b = RecoveryBudget(max_turns=2)
        assert b.can_consume()
        b.consume()
        assert b.can_consume()
        b.consume()
        assert not b.can_consume()
        assert b.used == 2

    def test_reset(self):
        b = RecoveryBudget(max_turns=1)
        b.consume()
        assert not b.can_consume()
        b.reset()
        assert b.can_consume()
        assert b.used == 0


# --------------------------------------------------------------------------- FailureClassifierHook


class TestFailureClassifierHook:
    async def test_post_tool_use_classifies_error_kind(self):
        hook = FailureClassifierHook()
        for kind, expected in [("invalid_args", "schema"), ("rate_limit", "transient"), ("policy_denied", "policy")]:
            ctx = HookContext(HookEvent.POST_TOOL_USE)
            ctx.metadata["tool_error_kind"] = kind
            await hook.execute(ctx)
            assert ctx.metadata.get("failure_class") == expected

    async def test_post_tool_use_no_kind_no_class(self):
        hook = FailureClassifierHook()
        ctx = HookContext(HookEvent.POST_TOOL_USE)
        await hook.execute(ctx)
        assert "failure_class" not in ctx.metadata

    async def test_post_output_low_coverage_is_grounding(self):
        hook = FailureClassifierHook(grounding=SimpleNamespace(last_coverage=0.4))
        ctx = HookContext(HookEvent.POST_OUTPUT)
        await hook.execute(ctx)
        assert ctx.metadata.get("failure_class") == "grounding"

    async def test_post_output_high_coverage_no_class(self):
        hook = FailureClassifierHook(grounding=SimpleNamespace(last_coverage=0.9))
        ctx = HookContext(HookEvent.POST_OUTPUT)
        await hook.execute(ctx)
        assert "failure_class" not in ctx.metadata


# --------------------------------------------------------------------------- LadderRouterHook


class TestLadderRouterHook:
    async def test_grounding_with_budget_picks_reflect(self):
        # The router ALLOWs reflect while budget remains but does NOT consume -- the
        # ReflectionHook consumes on an actual fire (tested in the integration suite).
        budget = RecoveryBudget(max_turns=2)
        hook = LadderRouterHook(budget=budget)
        ctx = HookContext(HookEvent.POST_OUTPUT)
        ctx.metadata["failure_class"] = "grounding"
        await hook.execute(ctx)
        assert ctx.metadata["recovery_plan"] == {"class": "grounding", "rung": "reflect"}
        assert budget.used == 0  # router does not consume; reflect does on fire

    async def test_grounding_budget_exhausted_picks_handover(self):
        budget = RecoveryBudget(max_turns=1)
        budget.consume()  # exhaust
        hook = LadderRouterHook(budget=budget)
        ctx = HookContext(HookEvent.POST_OUTPUT)
        ctx.metadata["failure_class"] = "grounding"
        await hook.execute(ctx)
        assert ctx.metadata["recovery_plan"]["rung"] == "handover"

    async def test_policy_class_skips_reflect(self):
        budget = RecoveryBudget(max_turns=3)
        hook = LadderRouterHook(budget=budget)
        ctx = HookContext(HookEvent.POST_OUTPUT)
        ctx.metadata["failure_class"] = "policy"
        await hook.execute(ctx)
        assert ctx.metadata["recovery_plan"]["rung"] == "handover"
        assert budget.used == 0  # reflect not consumed

    async def test_no_failure_class_no_plan(self):
        budget = RecoveryBudget(max_turns=3)
        hook = LadderRouterHook(budget=budget)
        ctx = HookContext(HookEvent.POST_OUTPUT)
        await hook.execute(ctx)
        assert "recovery_plan" not in ctx.metadata

    async def test_session_start_resets_budget(self):
        budget = RecoveryBudget(max_turns=1)
        budget.consume()
        hook = LadderRouterHook(budget=budget)
        await hook.execute(HookContext(HookEvent.SESSION_START))
        assert budget.used == 0


# --------------------------------------------------------------------------- DoomLoopHook SESSION_START fix


class TestDoomLoopSessionStartReset:
    async def test_session_start_clears_detector_history(self):
        hook = DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=3))
        # Accumulate some history via POST_TOOL_USE
        for _ in range(2):
            ctx = HookContext(HookEvent.POST_TOOL_USE, tool_name="t", tool_arguments="{}", tool_result="Error: x")
            await hook.execute(ctx)
        assert len(hook.detector._history) > 0
        # SESSION_START must clear it (per-run isolation)
        await hook.execute(HookContext(HookEvent.SESSION_START))
        assert len(hook.detector._history) == 0


# --------------------------------------------------------------------------- full-ladder integration


def _full_ladder_chain(grounding, critic, budget):
    return HookChain(
        [
            FailureClassifierHook(grounding=grounding),
            LadderRouterHook(budget=budget),
            HandoverDetectionHook(grounding=grounding, coverage_threshold=0.5),
            ReflectionHook(client=critic, grounding=grounding, max_turns=99, grounding_threshold=0.6, budget=budget),
        ]
    )


class TestFullLadderIntegration:
    async def test_reflect_then_complete_handover_suppressed(self, mock_client):
        # coverage low once (reflect retries), then grounded -> complete. Handover
        # must NOT fire while reflect budget remains.
        agent_client = mock_client(responses=[make_mock_response("a1"), make_mock_response("a2")])
        critic = _CriticClient("ungrounded")
        grounding = _FakeGroundingGuardrail([0.4, 0.9])
        budget = RecoveryBudget(max_turns=3)
        agent = AgentCore(
            client=agent_client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=5,
            hook_chain=_full_ladder_chain(grounding, critic, budget),
            output_guardrails=[grounding],
        )
        result = await agent.run("q")
        assert result.content == "a2"
        assert result.metadata["reflection_retries"] == 1
        assert budget.used == 1

    async def test_reflect_budget_exhausted_then_handover(self, mock_client):
        # coverage always low (<0.5 so handover can fire); budget=2 -> 2 reflects then handover.
        agent_client = mock_client(
            responses=[make_mock_response("a1"), make_mock_response("a2"), make_mock_response("a3")]
        )
        critic = _CriticClient("ungrounded")
        grounding = _FakeGroundingGuardrail([0.4, 0.4, 0.4])
        budget = RecoveryBudget(max_turns=2)
        agent = AgentCore(
            client=agent_client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=5,
            hook_chain=_full_ladder_chain(grounding, critic, budget),
            output_guardrails=[grounding],
        )
        with pytest.raises(AgentHandoverError):
            await agent.run("q")
        assert critic.calls == 2  # two reflect critiques before escalation
        assert budget.used == 2

    async def test_no_ladder_no_router_plan_standalone_reflect(self, mock_client):
        # Without the router in the chain, ReflectionHook uses its own budget (P1 path).
        agent_client = mock_client(responses=[make_mock_response("a1"), make_mock_response("a2")])
        critic = _CriticClient("ungrounded")
        grounding = _FakeGroundingGuardrail([0.4, 0.9])
        agent = AgentCore(
            client=agent_client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=5,
            hook_chain=HookChain(
                [ReflectionHook(client=critic, grounding=grounding, max_turns=3, grounding_threshold=0.6)]
            ),
            output_guardrails=[grounding],
        )
        result = await agent.run("q")
        assert result.content == "a2"
        assert result.metadata["reflection_retries"] == 1
