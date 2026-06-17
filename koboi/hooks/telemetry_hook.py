"""koboi/hooks/telemetry_hook.py -- Hook that records telemetry at all events.

Collects metrics throughout the agent session lifecycle for health reporting.
"""

from __future__ import annotations

from koboi.hooks.chain import Hook, HookContext, HookEvent
from koboi.harness.telemetry import TelemetryCollector
from koboi.harness.utils import is_tool_error


class TelemetryHook(Hook):
    """Hook that records telemetry at all events.

    Subscribes to all hook events and records relevant metrics
    into the TelemetryCollector for session analysis and reporting.
    """

    def __init__(self, telemetry: TelemetryCollector):
        self.telemetry = telemetry

    def handles(self) -> list[HookEvent]:
        return list(HookEvent)

    async def execute(self, ctx: HookContext) -> HookContext:
        handler = {
            HookEvent.SESSION_START: self._on_session_start,
            HookEvent.SESSION_END: self._on_session_end,
            HookEvent.PRE_TOOL_USE: self._on_pre_tool_use,
            HookEvent.POST_TOOL_USE: self._on_post_tool_use,
            HookEvent.PRE_LLM_CALL: self._on_pre_llm_call,
            HookEvent.POST_LLM_CALL: self._on_post_llm_call,
            HookEvent.DOOM_LOOP_DETECTED: self._on_doom_loop,
            HookEvent.PRE_INPUT: self._on_pre_input,
            HookEvent.POST_OUTPUT: self._on_post_output,
            HookEvent.PRE_COMPACT: self._on_pre_compact,
            HookEvent.POST_COMPACT: self._on_post_compact,
        }.get(ctx.event)
        if handler:
            handler(ctx)
        return ctx

    def _on_session_start(self, ctx: HookContext) -> None:
        self.telemetry.session_start()

    def _on_session_end(self, ctx: HookContext) -> None:
        self.telemetry.session_end()

    def _on_pre_tool_use(self, ctx: HookContext) -> None:
        if ctx.tool_name:
            self.telemetry.record_tool_call(ctx.tool_name)

    def _on_post_tool_use(self, ctx: HookContext) -> None:
        # Determine success/failure from tool result
        if is_tool_error(ctx.tool_result):
            self.telemetry.record_tool_failure()
        else:
            self.telemetry.record_tool_success()

        # Record permission decision if available
        policy = ctx.metadata.get("policy_decision")
        if policy:
            self.telemetry.record_permission(
                tool_name=ctx.tool_name or "",
                action=policy.get("action", "allowed"),
                rule_name=policy.get("matched_rule"),
            )

    def _on_pre_llm_call(self, ctx: HookContext) -> None:
        # Track iteration start for timing
        tokens = 0
        if ctx.messages:
            tokens = sum(len(m.get("content", "")) for m in ctx.messages) // 4
        self.telemetry.iteration_start(tokens_current=tokens)

    def _on_post_llm_call(self, ctx: HookContext) -> None:
        tokens_after = 0
        if ctx.llm_response and hasattr(ctx.llm_response, "usage") and ctx.llm_response.usage:
            tokens_after = ctx.llm_response.usage.total_tokens

        self.telemetry.iteration_end(
            iteration=ctx.iteration,
            tokens_after=tokens_after,
        )

    def _on_doom_loop(self, ctx: HookContext) -> None:
        self.telemetry.record_doom_loop()

    def _on_pre_input(self, ctx: HookContext) -> None:
        pass

    def _on_post_output(self, ctx: HookContext) -> None:
        pass

    def _on_pre_compact(self, ctx: HookContext) -> None:
        pass

    def _on_post_compact(self, ctx: HookContext) -> None:
        pass
