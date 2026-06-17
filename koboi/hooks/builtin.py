"""koboi/hooks/builtin.py -- Built-in hooks for logging and audit trail.

Adapted from agent/hooks.py LoggingHook and AuditHook.
"""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.logger import AgentLogger
    from koboi.guardrails.audit import AuditTrail


class LoggingHook(Hook):
    _log_console = None
    priority = 0

    def __init__(self, logger: AgentLogger | None = None, verbose: bool = False):
        self.logger = logger
        self.verbose = verbose

    def handles(self) -> list[HookEvent]:
        return list(HookEvent)

    async def execute(self, ctx: HookContext) -> HookContext:
        parts = [f"[HOOK] {ctx.event.value}"]
        if ctx.tool_name:
            parts.append(f"tool={ctx.tool_name}")
        if ctx.iteration:
            parts.append(f"iter={ctx.iteration}")
        msg = " ".join(parts)
        if self.logger:
            self.logger.log(msg)
        return ctx


class AuditHook(Hook):
    priority = 80

    def __init__(self, audit_trail: AuditTrail):
        self.audit_trail = audit_trail

    def handles(self) -> list[HookEvent]:
        return [HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE, HookEvent.DOOM_LOOP_DETECTED]

    async def execute(self, ctx: HookContext) -> HookContext:
        from koboi.types import AuditEntry

        self.audit_trail.record(
            AuditEntry(
                timestamp=_time.time(),
                event_type=f"harness_{ctx.event.value}",
                tool_name=ctx.tool_name,
                arguments=(ctx.tool_arguments or "")[:200],
                result=(ctx.tool_result or "")[:200],
                details=f"iteration={ctx.iteration}",
            )
        )
        return ctx
