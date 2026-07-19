"""koboi/hooks/failure_classifier_hook.py -- tag failure_class on recovery events (P2a).

Infra-band (priority 5): runs before the ``LadderRouterHook`` (6) and all recovery
rungs (doom 50 / handover 50 / reflect 60). Stamps ``ctx.metadata["failure_class"]``
from existing signals so the router can route by class. Fail-soft -- classification
is observability/routing metadata only; it must never break a run.

Classes (taxonomy): schema / transient / policy / grounding / loop / budget / handover.
P2a classifies tool errors (POST_TOOL_USE, via P0-D's ``error_kind`` surfaced onto the
ctx by the pipeline) and grounding failures (POST_OUTPUT, via
``GroundingGuardrail.last_coverage``). Loop detection lives in DoomLoopHook (priority
50) and is NOT folded into the router in P2a (deferred to P2b -- needs detection moved
ahead of the router).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.guardrails.grounding import GroundingGuardrail

_logger = logging.getLogger(__name__)

# Below the default grounding abstain threshold (0.8) = a grounding failure worth
# routing. Mirrors GroundingGuardrail's default; kept as a constant for P2a.
_GROUNDING_FAIL_COVERAGE = 0.8

# P0-D error_kind -> failure class.
_TOOL_CLASS: dict[str, str] = {
    "tool_not_found": "schema",
    "invalid_args": "schema",
    "timeout": "transient",
    "execution_error": "transient",
    "rate_limit": "transient",
    # Wave 2: non-zero shell exit lifted into the structured signal by the
    # pipeline ("[exit code: N]" parse). Transient: a failing build/test is a
    # normal, retryable step of an edit->test loop, not a schema/policy fault.
    "command_failed": "transient",
    "policy_denied": "policy",
    "mode_blocked": "policy",
    "denied": "policy",
    "approval_denied": "policy",
}


class FailureClassifierHook(Hook):
    """Tag ``failure_class`` onto ``ctx.metadata`` for the ladder router (P2a)."""

    priority = 5

    def __init__(self, grounding: GroundingGuardrail | None = None) -> None:
        self._grounding = grounding

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_TOOL_USE, HookEvent.POST_OUTPUT]

    async def execute(self, ctx: HookContext) -> HookContext:
        try:
            if ctx.event == HookEvent.POST_TOOL_USE:
                kind = ctx.metadata.get("tool_error_kind")
                if kind:
                    ctx.metadata["failure_class"] = _TOOL_CLASS.get(kind, "transient")
            elif ctx.event == HookEvent.POST_OUTPUT:
                if self._grounding is not None:
                    cov = getattr(self._grounding, "last_coverage", None)
                    # Use the LIVE guardrail abstain threshold (not a hardcoded 0.8) so
                    # the classifier tags grounding exactly where the guardrail abstains.
                    threshold = getattr(self._grounding, "_threshold", _GROUNDING_FAIL_COVERAGE)
                    if cov is not None and cov < threshold:
                        ctx.metadata["failure_class"] = "grounding"
        except Exception as exc:  # fail-soft: classification is observability-only
            _logger.warning("FailureClassifierHook fail-soft: %s", exc)
        return ctx
