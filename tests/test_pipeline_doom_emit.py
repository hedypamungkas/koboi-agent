"""Tests that ToolExecutionPipeline fans out DOOM_LOOP_DETECTED (Issue 1).

DoomLoopHook sets a ``doom_loop_detected`` metadata flag on POST_TOOL_USE but cannot
re-emit the event itself (hooks react; the emitter emits). The pipeline must fan the
event out so Audit/Telemetry/Langfuse/Notification subscribers actually fire.
"""

from __future__ import annotations

from koboi.hooks.chain import Hook, HookChain, HookContext, HookEvent
from koboi.hooks.doom_loop_hook import DoomLoopHook
from koboi.harness.doom_loop import DoomLoopConfig
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import ToolCall


class _DoomSubscriber(Hook):
    """Counts DOOM_LOOP_DETECTED emissions (stands in for Audit/Telemetry/etc)."""

    def __init__(self):
        self.fired = 0

    def handles(self):
        return [HookEvent.DOOM_LOOP_DETECTED]

    async def execute(self, ctx: HookContext) -> HookContext:
        self.fired += 1
        return ctx


def _registry_with_flaky_tool() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        name="flaky",
        description="always errors",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=lambda: "Error: connection refused",
    )
    return reg


class TestPipelineEmitsDoomLoopDetected:
    async def test_doom_detection_fires_subscriber(self):
        subscriber = _DoomSubscriber()
        chain = HookChain([DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=2)), subscriber])
        pipeline = ToolExecutionPipeline(
            tools=_registry_with_flaky_tool(),
            memory=ConversationMemory(),
            hook_chain=chain,
        )
        tc = ToolCall(id="1", name="flaky", arguments="{}")
        for _ in range(3):  # 3 identical failing calls -> doom triggers
            await pipeline.execute_tool_call(tc, iteration=0)

        # Before the fix this was 0 (event never emitted); now subscribers fire.
        assert subscriber.fired >= 1

    async def test_no_doom_no_spurious_emit(self):
        subscriber = _DoomSubscriber()
        chain = HookChain([DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=99)), subscriber])
        pipeline = ToolExecutionPipeline(
            tools=_registry_with_flaky_tool(), memory=ConversationMemory(), hook_chain=chain
        )
        await pipeline.execute_tool_call(ToolCall(id="1", name="flaky", arguments="{}"), iteration=0)
        assert subscriber.fired == 0
