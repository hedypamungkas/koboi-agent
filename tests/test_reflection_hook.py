"""tests/test_reflection_hook.py -- self-healing P1 (ReflectionHook).

Unit tests for the hook (POST_TOOL_USE critique, POST_OUTPUT reground, budget,
fail-soft, SESSION_START reset) + integration tests for the loop retry-seam
(_run_loop honors reflection_retry; run_stream skips it).
"""

from __future__ import annotations

from types import SimpleNamespace

from koboi.events import CompleteEvent
from koboi.guardrails.base import BaseGuardrail
from koboi.hooks.chain import HookChain, HookContext, HookEvent
from koboi.hooks.reflection_hook import ReflectionHook
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import AgentResponse, GuardrailResult
from tests.conftest import make_mock_response


class _CriticClient:
    """Minimal side-LLM double for the reflection critic."""

    def __init__(self, critique: str = "fix it", raise_on_call: bool = False):
        self.critique = critique
        self.raise_on_call = raise_on_call
        self.calls = 0

    async def complete(self, messages, tools=None, response_format=None):
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("critic down")
        return AgentResponse(content=self.critique)


class _FakeGroundingGuardrail(BaseGuardrail):
    """Output guardrail that reports a scripted coverage sequence + abstains low."""

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


# --------------------------------------------------------------------------- POST_TOOL_USE


class TestPostToolUseCritique:
    async def test_critique_after_repeated_identical_failure(self):
        critic = _CriticClient("check the divisor before dividing")
        hook = ReflectionHook(client=critic, tool_error_threshold=2)
        args = '{"a": 1}'
        # 1st identical error -> below threshold -> no critique
        ctx1 = HookContext(
            HookEvent.POST_TOOL_USE, tool_name="calc", tool_arguments=args, tool_result="Error: division by zero"
        )
        await hook.execute(ctx1)
        assert ctx1.inject_messages == []
        # 2nd identical error -> at threshold -> critique injected
        ctx2 = HookContext(
            HookEvent.POST_TOOL_USE, tool_name="calc", tool_arguments=args, tool_result="Error: division by zero"
        )
        await hook.execute(ctx2)
        assert len(ctx2.inject_messages) == 1
        assert "calc" in ctx2.inject_messages[0]
        assert "Do not repeat" in ctx2.inject_messages[0]
        assert critic.calls == 1

    async def test_no_critique_on_one_off_error(self):
        hook = ReflectionHook(client=_CriticClient(), tool_error_threshold=2)
        ctx = HookContext(HookEvent.POST_TOOL_USE, tool_name="x", tool_arguments="{}", tool_result="Error: boom")
        await hook.execute(ctx)
        assert ctx.inject_messages == []

    async def test_success_resets_consecutive_counter(self):
        hook = ReflectionHook(client=_CriticClient(), tool_error_threshold=2)
        args = '{"a": 1}'
        for _ in range(2):  # reach threshold (2nd injects)
            await hook.execute(
                HookContext(HookEvent.POST_TOOL_USE, tool_name="x", tool_arguments=args, tool_result="Error: boom")
            )
        # a success resets the consecutive counter
        await hook.execute(HookContext(HookEvent.POST_TOOL_USE, tool_name="x", tool_arguments=args, tool_result="ok"))
        ctx = HookContext(HookEvent.POST_TOOL_USE, tool_name="x", tool_arguments=args, tool_result="Error: boom")
        await hook.execute(ctx)
        assert ctx.inject_messages == []  # counter reset -> below threshold again

    async def test_no_critique_when_budget_exhausted(self):
        critic = _CriticClient()
        hook = ReflectionHook(client=critic, tool_error_threshold=2, max_turns=1)
        hook._turns_used = 1  # budget exhausted
        ctx = HookContext(HookEvent.POST_TOOL_USE, tool_name="x", tool_arguments="{}", tool_result="Error: boom")
        await hook.execute(ctx)
        await hook.execute(ctx)  # 2nd identical
        assert ctx.inject_messages == []
        assert critic.calls == 0


# --------------------------------------------------------------------------- POST_OUTPUT


class TestPostOutputReground:
    async def test_low_grounding_sets_retry_with_critique(self):
        grounding = SimpleNamespace(last_coverage=0.4)
        critic = _CriticClient("claim X is not in the context")
        hook = ReflectionHook(client=critic, grounding=grounding, grounding_threshold=0.6)
        ctx = HookContext(HookEvent.POST_OUTPUT, llm_response=AgentResponse(content="some answer"))
        await hook.execute(ctx)
        retry = ctx.metadata.get("reflection_retry")
        assert retry is not None
        assert retry["coverage"] == 0.4
        assert "claim X" in retry["critique"]

    async def test_high_grounding_no_retry(self):
        grounding = SimpleNamespace(last_coverage=0.9)
        hook = ReflectionHook(client=_CriticClient(), grounding=grounding, grounding_threshold=0.6)
        ctx = HookContext(HookEvent.POST_OUTPUT, llm_response=AgentResponse(content="ans"))
        await hook.execute(ctx)
        assert ctx.metadata.get("reflection_retry") is None

    async def test_no_grounding_signal_no_retry(self):
        grounding = SimpleNamespace(last_coverage=None)
        hook = ReflectionHook(client=_CriticClient(), grounding=grounding, grounding_threshold=0.6)
        ctx = HookContext(HookEvent.POST_OUTPUT, llm_response=AgentResponse(content="ans"))
        await hook.execute(ctx)
        assert ctx.metadata.get("reflection_retry") is None


# --------------------------------------------------------------------------- fail-soft / budget


class TestFailSoftAndBudget:
    async def test_critic_error_does_not_break_run(self):
        grounding = SimpleNamespace(last_coverage=0.4)
        hook = ReflectionHook(client=_CriticClient(raise_on_call=True), grounding=grounding, grounding_threshold=0.6)
        ctx = HookContext(HookEvent.POST_OUTPUT, llm_response=AgentResponse(content="ans"))
        await hook.execute(ctx)  # must not raise
        assert ctx.metadata.get("reflection_retry") is None  # fail-soft skip

    async def test_no_client_is_inert(self):
        grounding = SimpleNamespace(last_coverage=0.4)
        hook = ReflectionHook(client=None, grounding=grounding, grounding_threshold=0.6)
        ctx = HookContext(HookEvent.POST_OUTPUT, llm_response=AgentResponse(content="ans"))
        await hook.execute(ctx)
        assert ctx.metadata.get("reflection_retry") is None

    async def test_budget_exhausted_skips_retry(self):
        grounding = SimpleNamespace(last_coverage=0.4)
        hook = ReflectionHook(client=_CriticClient(), grounding=grounding, max_turns=1, grounding_threshold=0.6)
        hook._turns_used = 1
        ctx = HookContext(HookEvent.POST_OUTPUT, llm_response=AgentResponse(content="ans"))
        await hook.execute(ctx)
        assert ctx.metadata.get("reflection_retry") is None

    async def test_session_start_resets_state(self):
        hook = ReflectionHook(client=_CriticClient(), tool_error_threshold=2)
        hook._turns_used = 5
        hook._tool_error_counts = {"x:abc": 9}
        await hook.execute(HookContext(HookEvent.SESSION_START))
        assert hook._turns_used == 0
        assert hook._tool_error_counts == {}


# --------------------------------------------------------------------------- loop seam integration


class TestLoopSeam:
    async def test_run_retries_on_low_grounding_then_completes(self, mock_client):
        agent_client = mock_client(responses=[make_mock_response("wild guess"), make_mock_response("grounded answer")])
        critic = _CriticClient("claims not in context")
        grounding = _FakeGroundingGuardrail([0.4, 0.9])  # 1st low -> retry, 2nd grounded
        hook = ReflectionHook(client=critic, grounding=grounding, max_turns=3, grounding_threshold=0.6)
        agent = AgentCore(
            client=agent_client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=5,
            hook_chain=HookChain([hook]),
            output_guardrails=[grounding],
        )
        result = await agent.run("q")
        assert result.content == "grounded answer"
        assert result.metadata["reflection_retries"] == 1
        assert agent_client.call_count == 2

    async def test_budget_exhausted_falls_back_to_abstain(self, mock_client):
        # always-low grounding, max_turns=1 -> one retry, then abstain refusal stands
        agent_client = mock_client(responses=[make_mock_response("a1"), make_mock_response("a2")])
        critic = _CriticClient("ungrounded")
        grounding = _FakeGroundingGuardrail([0.4, 0.4])  # never grounded
        hook = ReflectionHook(client=critic, grounding=grounding, max_turns=1, grounding_threshold=0.6)
        agent = AgentCore(
            client=agent_client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=5,
            hook_chain=HookChain([hook]),
            output_guardrails=[grounding],
        )
        result = await agent.run("q")
        assert result.metadata["reflection_retries"] == 1
        assert "confidently" in result.content  # abstain refusal (no more budget)
        assert agent_client.call_count == 2

    async def test_stream_skips_reflection(self, mock_client):
        agent_client = mock_client(responses=[make_mock_response("wild guess")])
        critic = _CriticClient("ungrounded")
        grounding = _FakeGroundingGuardrail([0.4])
        hook = ReflectionHook(client=critic, grounding=grounding, max_turns=3, grounding_threshold=0.6)
        agent = AgentCore(
            client=agent_client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=5,
            hook_chain=HookChain([hook]),
            output_guardrails=[grounding],
        )
        events = [e async for e in agent.run_stream("q")]
        completes = [e for e in events if isinstance(e, CompleteEvent)]
        assert completes
        # streaming defers reflection: no retry, abstain refusal stands
        assert completes[-1].metadata.get("reflection_retries") == 0
        assert "confidently" in completes[-1].content
        assert agent_client.call_count == 1
