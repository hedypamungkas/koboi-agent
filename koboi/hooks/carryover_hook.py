"""koboi/hooks/carryover_hook.py -- Hook that manages carryover state across events.

Preserves and updates metadata across context compaction boundaries.
"""

from __future__ import annotations

from koboi.hooks.chain import Hook, HookContext, HookEvent
from koboi.harness.carryover import CarryoverState
from koboi.harness.utils import is_tool_error


class CarryoverHook(Hook):
    """Hook that manages carryover state across events.

    Subscribes to POST_TOOL_USE, POST_COMPACT, SESSION_START, and SESSION_END
    to maintain and update the CarryoverState, ensuring metadata persists
    across context compaction.
    """

    def __init__(self, state: CarryoverState | None = None):
        self.state = state or CarryoverState()

    def handles(self) -> list[HookEvent]:
        return [
            HookEvent.SESSION_START,
            HookEvent.SESSION_END,
            HookEvent.POST_TOOL_USE,
            HookEvent.POST_COMPACT,
        ]

    async def execute(self, ctx: HookContext) -> HookContext:
        # Ensure carryover is attached to context
        if ctx.carryover is None:
            ctx.carryover = self.state

        handler = {
            HookEvent.SESSION_START: self._on_session_start,
            HookEvent.SESSION_END: self._on_session_end,
            HookEvent.POST_TOOL_USE: self._on_post_tool_use,
            HookEvent.POST_COMPACT: self._on_post_compact,
        }.get(ctx.event)
        if handler:
            handler(ctx)
        return ctx

    def _on_session_start(self, ctx: HookContext) -> None:
        # Initialize carryover state
        ctx.carryover = self.state

    def _on_session_end(self, ctx: HookContext) -> None:
        # Serialize final state to metadata
        ctx.metadata["carryover_summary"] = self.state.summary()

    def _on_post_tool_use(self, ctx: HookContext) -> None:
        if not ctx.tool_name:
            return

        # Record tool usage in carryover
        success = not is_tool_error(ctx.tool_result)
        self.state.record_tool_use(
            tool_name=ctx.tool_name,
            arguments=ctx.tool_arguments or "",
            result=ctx.tool_result or "",
            iteration=ctx.iteration,
            success=success,
        )

    def _on_post_compact(self, ctx: HookContext) -> None:
        # Inject carryover context into messages if compaction happened
        carryover_msg = self.state.to_context_message()
        if carryover_msg and ctx.messages:
            # Check if carryover is already in the messages
            for msg in ctx.messages:
                if msg.get("role") == "system" and "<harness-carryover>" in (msg.get("content", "")):
                    return
            # Inject as a system message
            ctx.messages.insert(
                1,
                {
                    "role": "system",
                    "content": carryover_msg,
                },
            )
