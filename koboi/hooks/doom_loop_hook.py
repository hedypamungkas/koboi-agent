"""koboi/hooks/doom_loop_hook.py -- Hook for doom loop detection at POST_TOOL_USE.

Records tool calls and checks for doom loop patterns after each tool execution.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from koboi.hooks.chain import Hook, HookContext, HookEvent
from koboi.harness.doom_loop import DoomLoopDetector, DoomLoopConfig, DoomLoopResult
from koboi.harness.utils import is_tool_error


class DoomLoopHook(Hook):
    """Hook for doom loop detection at POST_TOOL_USE.

    Records each tool call into the DoomLoopDetector and checks for
    repeating unproductive patterns. When detected, injects a recovery
    message into the context.
    """

    def __init__(
        self,
        config: DoomLoopConfig | None = None,
        on_doom_loop: Callable[[DoomLoopResult, HookContext], None] | None = None,
    ):
        self.detector = DoomLoopDetector(config)
        self._on_doom_loop = on_doom_loop

    def handles(self) -> list[HookEvent]:
        return [HookEvent.SESSION_START, HookEvent.POST_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        # P2a: reset per-run so detector state never contaminates the next run (the
        # detector is per-session; without this its history accumulates across run()
        # calls within a session). Fixes a latent multi-run contamination bug.
        if ctx.event == HookEvent.SESSION_START:
            self.detector.reset()
            return ctx
        if not ctx.tool_name:
            return ctx

        # Determine if the tool result was an error
        is_error = is_tool_error(ctx.tool_result)

        # Record the tool call, with an output fingerprint so the detector can
        # tell "stuck" (identical output every time) from "progressing"
        # (identical call, changing output -- e.g. a shrinking test-failure list).
        fingerprint = None
        if ctx.tool_result:
            fingerprint = hashlib.sha1(ctx.tool_result.encode("utf-8", "replace")).hexdigest()  # nosec B324
        self.detector.record(
            tool_name=ctx.tool_name,
            arguments=ctx.tool_arguments or "",
            is_error=is_error,
            result_fingerprint=fingerprint,
        )

        # Check for doom loop patterns
        result: DoomLoopResult = self.detector.check()

        if result.detected:
            ctx.metadata["doom_loop"] = {
                "detected": True,
                "loop_type": result.loop_type,
                "pattern": result.pattern_description,
                "recovery_hint": result.recovery_hint,
                "iterations_wasted": result.iterations_wasted,
            }

            # Inject recovery message
            recovery_msg = DoomLoopDetector.build_recovery_message(result)
            ctx.inject_message = recovery_msg

            # Callback for external handling
            if self._on_doom_loop:
                self._on_doom_loop(result, ctx)

            # Emit DOOM_LOOP_DETECTED event via metadata flag
            ctx.metadata["doom_loop_detected"] = True

        return ctx
