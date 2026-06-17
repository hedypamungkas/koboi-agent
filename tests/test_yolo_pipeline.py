"""Tests for YOLO mode bypass behavior in ToolExecutionPipeline."""

from __future__ import annotations

import pytest

from koboi.hooks.chain import Hook, HookChain, HookContext, HookEvent
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.modes import AgentMode, ModeManager
from koboi.tools.registry import ToolRegistry
from koboi.types import RiskLevel, ToolCall


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="test_tool",
        description="A test tool",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": [],
        },
        fn=lambda message="ok": f"result: {message}",
    )
    return registry


@pytest.fixture
def registry():
    return _make_registry()


@pytest.fixture
def memory():
    return ConversationMemory()


def _tc(name: str = "test_tool", args: str = '{"message": "hi"}') -> ToolCall:
    return ToolCall(id="1", name=name, arguments=args)


class _AbortHook(Hook):
    """Hook that always aborts PRE_TOOL_USE (simulates PolicyHook deny)."""

    def handles(self):
        return [HookEvent.PRE_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        ctx.abort = True
        ctx.inject_message = "Blocked by policy"
        return ctx


class TestYoloPipelineBypass:
    async def test_yolo_skips_rate_limit(self, registry, memory):
        from koboi.guardrails.rate_limiter import RateLimiter
        from koboi.types import RateLimitConfig

        rl = RateLimiter(RateLimitConfig(max_tool_calls_per_session=0))
        mgr = ModeManager(AgentMode.YOLO)
        pipeline = ToolExecutionPipeline(
            tools=registry,
            memory=memory,
            rate_limiter=rl,
            mode_manager=mgr,
        )
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert not result.skipped
        assert "result: hi" in result.result

    async def test_yolo_skips_approval(self, registry, memory):
        from koboi.guardrails.approval import ApprovalHandler

        class AlwaysDeny(ApprovalHandler):
            def should_approve(self, tool_name, arguments, risk_level):
                return False

        mgr = ModeManager(AgentMode.YOLO)
        pipeline = ToolExecutionPipeline(
            tools=registry,
            memory=memory,
            approval_handler=AlwaysDeny(),
            mode_manager=mgr,
        )
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert not result.skipped
        assert "result: hi" in result.result

    async def test_yolo_respects_policy_abort(self, registry, memory):
        chain = HookChain()
        chain.add(_AbortHook())
        mgr = ModeManager(AgentMode.YOLO)
        pipeline = ToolExecutionPipeline(
            tools=registry,
            memory=memory,
            hook_chain=chain,
            mode_manager=mgr,
        )
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert result.skipped
        assert result.skip_reason == "policy_denied"
        assert "Blocked by policy" in result.result

    async def test_non_yolo_still_rate_limits(self, registry, memory):
        from koboi.guardrails.rate_limiter import RateLimiter
        from koboi.types import RateLimitConfig

        rl = RateLimiter(RateLimitConfig(max_tool_calls_per_session=0))
        mgr = ModeManager(AgentMode.AUTO)
        pipeline = ToolExecutionPipeline(
            tools=registry,
            memory=memory,
            rate_limiter=rl,
            mode_manager=mgr,
        )
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert result.skipped
        assert result.skip_reason == "rate_limit"

    async def test_no_mode_manager_no_bypass(self, registry, memory):
        from koboi.guardrails.approval import ApprovalHandler

        class AlwaysDeny(ApprovalHandler):
            def should_approve(self, tool_name, arguments, risk_level):
                return False

        pipeline = ToolExecutionPipeline(
            tools=registry,
            memory=memory,
            approval_handler=AlwaysDeny(),
        )
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert result.skipped
        assert result.skip_reason == "denied"

    async def test_policy_abort_works_without_yolo(self, registry, memory):
        chain = HookChain()
        chain.add(_AbortHook())
        pipeline = ToolExecutionPipeline(
            tools=registry,
            memory=memory,
            hook_chain=chain,
        )
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert result.skipped
        assert result.skip_reason == "policy_denied"
