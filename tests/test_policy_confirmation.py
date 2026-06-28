"""Tests for policy_needs_confirmation consumption in ToolExecutionPipeline (M0 16.2).

Covers Q1 (inert when no handler), Q2 Option A (single prompt; no double-prompt),
and the deny path.
"""

from __future__ import annotations

import pytest

from koboi.guardrails.approval import ApprovalHandler
from koboi.hooks.chain import Hook, HookChain, HookContext, HookEvent
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import ToolCall


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=lambda **kw: "ok",
    )
    return registry


@pytest.fixture
def registry():
    return _make_registry()


@pytest.fixture
def memory():
    return ConversationMemory()


def _tc() -> ToolCall:
    return ToolCall(id="1", name="test_tool", arguments="{}")


class _ConfirmHook(Hook):
    """Sets policy_needs_confirmation (simulates PolicyHook CONFIRM)."""

    def handles(self):
        return [HookEvent.PRE_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        ctx.metadata["policy_needs_confirmation"] = True
        ctx.metadata["policy_reason"] = "policy test"
        return ctx


class _CountingHandler(ApprovalHandler):
    """Records how many times should_approve was called."""

    def __init__(self, approved: bool = True):
        self.calls = 0
        self._approved = approved

    def should_approve(self, tool_name, arguments, risk_level):
        self.calls += 1
        return self._approved


class TestPolicyConfirmation:
    async def test_inert_without_handler(self, registry, memory):
        # Q1: policy_needs_confirmation with no approval handler -> tool proceeds.
        chain = HookChain()
        chain.add(_ConfirmHook())
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, hook_chain=chain)
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert not result.skipped

    async def test_handler_consulted_once_with_policy_flag(self, registry, memory):
        # Q2 Option A: single consultation; policy confirm does not re-prompt.
        chain = HookChain()
        chain.add(_ConfirmHook())
        handler = _CountingHandler(approved=True)
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, hook_chain=chain, approval_handler=handler)
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert not result.skipped
        assert handler.calls == 1  # risk path consulted once; 4c skipped (already prompted)

    async def test_no_double_prompt(self, registry, memory):
        chain = HookChain()
        chain.add(_ConfirmHook())
        handler = _CountingHandler(approved=True)
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, hook_chain=chain, approval_handler=handler)
        await pipeline.execute_tool_call(_tc(), iteration=0)
        assert handler.calls == 1

    async def test_handler_denial_blocks_execution(self, registry, memory):
        chain = HookChain()
        chain.add(_ConfirmHook())
        handler = _CountingHandler(approved=False)
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, hook_chain=chain, approval_handler=handler)
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert result.skipped
        assert result.skip_reason == "denied"

    async def test_policy_flag_ignored_in_yolo(self, registry, memory):
        from koboi.modes import AgentMode, ModeManager

        chain = HookChain()
        chain.add(_ConfirmHook())
        mgr = ModeManager(AgentMode.YOLO)
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, hook_chain=chain, mode_manager=mgr)
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert not result.skipped  # YOLO bypasses approval + policy confirm
