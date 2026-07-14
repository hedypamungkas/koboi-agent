"""koboi/hooks/handover_detection_hook.py -- structural handover detection (Wave 2 B1.5).

B1 ships LLM-INITIATED handover (the bot calls ``transfer_to_human`` when IT
decides). But the LLM does not always self-escalate -- it can be confidently-wrong
(A3 catches -> abstain) or simply not realize it is stuck. B1.5 is the
SYSTEM-INITIATED layer: this hook fires handover STRUCTURALLY, even when the LLM
did not call the tool. Triggers:

  - PRE_INPUT  -- explicit user ask ("talk to a human"). Fires before the LLM
    runs (no wasted call).
  - POST_OUTPUT -- A3 low grounding coverage (the bot answered but it is
    ungrounded -> escalate). Reads the injected ``GroundingGuardrail.last_coverage``
    (fresh: A3's ``check()`` runs in the output-guardrail loop before POST_OUTPUT).

Mechanism: the hook sets ``ctx.metadata["handover_requested"] = {reason, summary}``
and does NOT raise (``HookChain.emit`` swallows hook exceptions into ``ctx.abort`` ->
``AgentAbortedError``, the wrong class). The emit-site (``loop.py`` _validate_input /
_process_output) reads the flag and raises ``AgentHandoverError`` -> B1's HandoverEvent
machinery. Mirrors the DoomLoopHook flag pattern (``doom_loop_hook.py`` /
``loop_pipeline.py``).

Opt-in (``handover.detection.enabled``); default off (no behavior change). The
coverage trigger is the A3-fed differentiator (no OSS peer ships structural
handover detection). Confidence ladder: answer (>=A3 abstain 0.8) -> abstain
(0.5-0.8) -> handover (<coverage_threshold 0.5).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.guardrails.grounding import GroundingGuardrail

_logger = logging.getLogger(__name__)

DEFAULT_ASK_PATTERNS = [
    r"talk\s+to\s+(a\s+)?(human|agent|person)",
    r"speak\s+to\s+(a\s+)?(human|agent)",
    r"(human|live)\s+agent",
    r"transfer\s+(me\s+)?to\s+(a\s+)?(human|agent|manager)",
    r"connect\s+me\s+to\s+(a\s+)?(human|agent|manager|person)",
    r"\bagent\b.*\bplease\b",
]


class HandoverDetectionHook(Hook):
    """Structural handover detection (B1.5). See module docstring."""

    def __init__(
        self,
        grounding: GroundingGuardrail | None = None,
        coverage_threshold: float = 0.5,
        ask_patterns: list[str] | None = None,
    ) -> None:
        self._grounding = grounding
        self._coverage_threshold = float(coverage_threshold)
        patterns = ask_patterns if ask_patterns else DEFAULT_ASK_PATTERNS
        self._ask_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def handles(self) -> list[HookEvent]:
        return [HookEvent.PRE_INPUT, HookEvent.POST_OUTPUT]

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.event == HookEvent.PRE_INPUT:
            msg = ctx.user_message or ""
            if any(p.search(msg) for p in self._ask_patterns):
                ctx.metadata["handover_requested"] = {
                    "reason": "user requested a human",
                    "summary": msg[:200],
                }
        elif ctx.event == HookEvent.POST_OUTPUT and self._grounding is not None:
            cov: Any = getattr(self._grounding, "last_coverage", None)
            if cov is not None and cov < self._coverage_threshold:
                ctx.metadata["handover_requested"] = {
                    "reason": f"low grounding coverage ({cov:.2f} < {self._coverage_threshold})",
                    "summary": "",
                }
            elif cov is None:
                _logger.debug(
                    "HandoverDetectionHook: grounding signal absent (last_coverage=None) "
                    "-- coverage-based handover skipped this turn"
                )
        # NOTE: deliberately does NOT set ctx.abort -- the emit-site reads the flag
        # and raises AgentHandoverError (setting abort would raise AgentAbortedError
        # first, the wrong exception). See module docstring.
        return ctx
