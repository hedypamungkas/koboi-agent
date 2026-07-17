"""tests/test_self_healing_e2e.py -- end-to-end chained integration tests.

Tests the FULL self-healing flow as ONE continuous chain:
  grounding abstain → FailureClassifierHook tags "grounding" → LadderRouterHook
  picks "reflect" → ReflectionHook fires (with optional CRITIC tool-verify) →
  budget exhausts → router escalates to "handover" → HandoverDetectionHook fires
  → AgentHandoverError.

Unlike the unit tests (which test each hook in isolation), these exercise the
priority-ordered HookChain emit where all 4 hooks process the SAME POST_OUTPUT
event in sequence (classifier@5 → router@6 → handover@50 → reflect@60).
"""

from __future__ import annotations

import pytest

from koboi.exceptions import AgentHandoverError
from koboi.guardrails.base import BaseGuardrail
from koboi.harness.recovery_budget import RecoveryBudget
from koboi.hooks.chain import HookChain
from koboi.hooks.failure_classifier_hook import FailureClassifierHook
from koboi.hooks.handover_detection_hook import HandoverDetectionHook
from koboi.hooks.ladder_router_hook import LadderRouterHook
from koboi.hooks.reflection_hook import ReflectionHook
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import AgentResponse, GuardrailResult
from tests.conftest import make_mock_response, make_tool_registry


class _CriticClient:
    """Side-LLM double returning a fixed critique or decompose JSON."""

    def __init__(self, critique="ungrounded claim X"):
        self.critique = critique
        self.calls = 0

    async def complete(self, messages, tools=None, response_format=None):
        self.calls += 1
        return AgentResponse(content=self.critique)


class _SequenceCritic:
    """Side-LLM double returning scripted responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self._i = 0
        self.calls = 0

    async def complete(self, messages, tools=None, response_format=None):
        self.calls += 1
        r = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return AgentResponse(content=r)


class _FakeGroundingGuardrail(BaseGuardrail):
    """Scripted grounding coverage for deterministic testing."""

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
            passed=False,
            reason=f"coverage {cov}",
            action="abstain",
            sanitized_content="I cannot answer confidently.",
        )


def _full_chain(grounding, critic, budget, tools=None, verifier_tools=None):
    """Build the complete 4-hook chain (classifier → router → handover → reflect)."""
    return HookChain(
        [
            FailureClassifierHook(grounding=grounding),
            LadderRouterHook(budget=budget),
            HandoverDetectionHook(grounding=grounding, coverage_threshold=0.5),
            ReflectionHook(
                client=critic,
                grounding=grounding,
                max_turns=99,
                grounding_threshold=0.6,
                budget=budget,
                tools=tools,
                verifier_tools=verifier_tools,
            ),
        ]
    )


class TestEndToEndChain:
    """Full chain: grounding → classifier → router → reflect → (complete | handover)."""

    async def test_reflect_then_complete(self, mock_client):
        """Coverage low once → reflect → coverage recovers → complete. Handover suppressed."""
        client = mock_client(responses=[make_mock_response("vague guess"), make_mock_response("grounded answer")])
        critic = _CriticClient("ungrounded claims")
        grounding = _FakeGroundingGuardrail([0.4, 0.9])
        budget = RecoveryBudget(max_turns=3)
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=5,
            hook_chain=_full_chain(grounding, critic, budget),
            output_guardrails=[grounding],
        )
        result = await agent.run("q")
        assert result.content == "grounded answer"
        assert result.metadata["reflection_retries"] == 1
        assert budget.used == 1

    async def test_budget_exhausted_then_handover(self, mock_client):
        """Coverage always low → 2 reflects → budget exhausts → handover fires."""
        client = mock_client(responses=[make_mock_response("a1"), make_mock_response("a2"), make_mock_response("a3")])
        critic = _CriticClient("ungrounded")
        grounding = _FakeGroundingGuardrail([0.4, 0.4, 0.4])
        budget = RecoveryBudget(max_turns=2)
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=5,
            hook_chain=_full_chain(grounding, critic, budget),
            output_guardrails=[grounding],
        )
        with pytest.raises(AgentHandoverError):
            await agent.run("q")
        assert critic.calls == 2
        assert budget.used == 2

    async def test_critic_math_verification_in_chain(self, mock_client):
        """Agent says '2+2=5' → CRITIC catches via calculate → tool-grounded critique → retry → '2+2=4' → complete."""
        client = mock_client(
            responses=[make_mock_response("The answer is 2+2=5."), make_mock_response("The answer is 2+2=4.")]
        )
        # Critic returns typed-decompose JSON with a math claim on call 1.
        critic = _SequenceCritic(['[{"claim":"2+2=5","kind":"math","hint":"2+2"}]'])
        grounding = _FakeGroundingGuardrail([0.4, 0.9])
        budget = RecoveryBudget(max_turns=3)
        reg = make_tool_registry()  # has calculate
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=reg,
            max_iterations=5,
            hook_chain=_full_chain(grounding, critic, budget, tools=reg, verifier_tools=["calculate"]),
            output_guardrails=[grounding],
        )
        result = await agent.run("q")
        assert "2+2=4" in result.content
        assert result.metadata["reflection_retries"] == 1
        assert budget.used == 1
        assert critic.calls == 1  # only the typed-decompose call (tool-grounded critique built from calculate result)

    async def test_critic_supported_claim_no_critique_fallback_to_intrinsic(self, mock_client):
        """Agent says correct math → CRITIC verifies OK (no mismatch) → falls back to intrinsic critique → reflect."""
        client = mock_client(
            responses=[make_mock_response("The answer is 2+2=4."), make_mock_response("Grounded answer.")]
        )
        # Call 1: typed-decompose (math claim 2+2=4, verifies OK → None). Call 2: intrinsic critique.
        critic = _SequenceCritic(
            [
                '[{"claim":"2+2=4","kind":"math","hint":"2+2"}]',
                "ungrounded claim about the answer",
            ]
        )
        grounding = _FakeGroundingGuardrail([0.4, 0.9])
        budget = RecoveryBudget(max_turns=3)
        reg = make_tool_registry()
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=reg,
            max_iterations=5,
            hook_chain=_full_chain(grounding, critic, budget, tools=reg, verifier_tools=["calculate"]),
            output_guardrails=[grounding],
        )
        result = await agent.run("q")
        assert result.metadata["reflection_retries"] == 1
        assert budget.used == 1
        assert critic.calls == 2  # decompose (CRITIC, no mismatch) + intrinsic critique (fallback)
