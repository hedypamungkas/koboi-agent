"""koboi/hooks/ladder_router_hook.py -- declarative escalation-ladder router (P2a).

The arbitration core of the single-agent escalation ladder. Runs after the
``FailureClassifierHook`` (priority 5) at priority 6, BEFORE the recovery rungs
(doom 50 / handover 50 / reflect 60). Given the tagged ``failure_class`` and the
shared ``RecoveryBudget``, it chooses ONE rung for the turn and stamps
``ctx.metadata["recovery_plan"] = {"class", "rung"}``. Each recovery rung then
guards on its rung name (one line each).

P2a scope: arbitrates the conflicting POST_OUTPUT axis (reflect vs handover). On a
grounding failure with budget remaining -> ``reflect`` (let the model re-attempt,
grounded); when the reflect budget is exhausted -> ``handover`` (escalate). This
makes the previously-implicit ladder EXPLICIT and ordered: today handover silently
wins by control flow at coverage < 0.5, discarding reflection; the router lets
reflect *try first*, escalating only when its budget burns out.

POST_TOOL_USE rungs (doom nudge, tool-error reflect) keep P1 behavior in P2a --
folding doom into the router needs its detection (priority 50) moved earlier
(deferred to P2b). The budget is reset on SESSION_START (per run).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.harness.recovery_budget import RecoveryBudget

_logger = logging.getLogger(__name__)

# Default failure-class -> ordered recovery rungs. Rungs in P2a: reflect, handover
# (doom_nudge is P2b). ``handover`` is terminal and always available.
DEFAULT_LADDER: dict[str, list[str]] = {
    "grounding": ["reflect", "handover"],
    "schema": ["reflect", "handover"],
    "transient": ["reflect", "handover"],
    "loop": ["reflect", "handover"],
    "policy": ["handover"],
    "budget": ["handover"],
    "default": ["reflect", "handover"],
}


class LadderRouterHook(Hook):
    """Pick one recovery rung per turn from the declarative ladder (P2a)."""

    priority = 6

    def __init__(self, budget: RecoveryBudget, ladder: dict[str, list[str]] | None = None) -> None:
        self._budget = budget
        self._ladder = {**DEFAULT_LADDER, **(ladder or {})}

    def handles(self) -> list[HookEvent]:
        return [HookEvent.SESSION_START, HookEvent.POST_OUTPUT]

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.event == HookEvent.SESSION_START:
            self._budget.reset()
            return ctx
        if ctx.event != HookEvent.POST_OUTPUT:
            return ctx
        try:
            fclass = ctx.metadata.get("failure_class")
            if not fclass:
                return ctx  # no classified failure this turn -> no arbitration
            rungs = self._ladder.get(fclass) or self._ladder.get("default") or []
            chosen = self._choose(rungs)
            if chosen is None and rungs:
                # A custom ladder listed rungs for this class but none were actionable
                # (e.g. typo'd rung name, or reflect budget exhausted with no handover
                # rung). Log so misconfiguration isn't silent.
                _logger.warning("LadderRouterHook: no actionable rung for class=%s rungs=%s", fclass, rungs)
            ctx.metadata["recovery_plan"] = {"class": fclass, "rung": chosen}
        except Exception as exc:  # fail-soft: routing metadata must never break the run
            _logger.warning("LadderRouterHook fail-soft: %s", exc)
        return ctx

    def _choose(self, rungs: list[str]) -> str | None:
        for rung in rungs:
            if rung == "handover":
                return "handover"  # terminal -- always available
            # Reflect is allowed while the shared budget has room; ReflectionHook
            # CONSUMES the budget only when it actually fires (after a successful
            # critique), so budget.used == real reflect turns (not declined attempts).
            if rung == "reflect" and self._budget.can_consume():
                return "reflect"
            # doom_nudge folding into the router is deferred to P2b.
        return None
