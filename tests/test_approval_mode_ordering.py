"""tests/test_approval_mode_ordering.py -- #3 approval-before-ModeHook fix.

Verifies the tool-execution pipeline gates in the correct order:
  PRE_TOOL_USE emit -> policy-abort -> mode-block -> approval -> policy-confirm.

Previously approval ran BEFORE the mode-block check, so (a) a DESTRUCTIVE tool
in chat/plan mode could prompt the user and then be mode-blocked (wasted prompt),
and (b) a trust-DB allow rule bypassed chat/plan mode-blocking entirely.
"""

from __future__ import annotations

from koboi.hooks.chain import HookChain
from koboi.hooks.mode_hook import ModeHook
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.modes import AgentMode, ModeManager
from koboi.tools.registry import ToolRegistry
from koboi.types import RiskLevel, ToolCall


class _RecordingApproval:
    """Approval handler that records whether ``should_approve`` was invoked."""

    def __init__(self):
        self.called = False

    def should_approve(self, tool_name, arguments, risk_level):  # sync handler
        self.called = True
        return True  # would approve


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="write_file",
        description="destructive write",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        fn=lambda path: "ok",
        risk_level=RiskLevel.DESTRUCTIVE,
    )
    return registry


def _tc() -> ToolCall:
    return ToolCall(id="1", name="write_file", arguments='{"path": "/x"}')


async def test_chat_mode_blocks_destructive_without_prompting():
    """CHAT mode blocks a destructive tool BEFORE approval is prompted."""
    mgr = ModeManager(AgentMode.CHAT)
    chain = HookChain([ModeHook(mgr)])
    handler = _RecordingApproval()
    pipeline = ToolExecutionPipeline(
        tools=_registry(),
        memory=ConversationMemory(),
        approval_handler=handler,
        hook_chain=chain,
        mode_manager=mgr,
    )

    result = await pipeline.execute_tool_call(_tc(), iteration=0)

    assert result.skipped
    assert result.skip_reason == "mode_blocked"
    # The fix: approval must NOT be prompted for a tool that is mode-blocked.
    assert handler.called is False


async def test_act_mode_destructive_prompts_approval_and_executes():
    """ACT mode allows writes -> approval IS prompted; on approve, the tool runs."""
    mgr = ModeManager(AgentMode.ACT)
    chain = HookChain([ModeHook(mgr)])
    handler = _RecordingApproval()
    pipeline = ToolExecutionPipeline(
        tools=_registry(),
        memory=ConversationMemory(),
        approval_handler=handler,
        hook_chain=chain,
        mode_manager=mgr,
    )

    result = await pipeline.execute_tool_call(_tc(), iteration=0)

    assert handler.called is True  # approval prompted (mode allows writes)
    assert not result.skipped  # approved -> executed
