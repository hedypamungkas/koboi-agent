"""koboi/hooks/proactive_extraction_hook.py -- Auto-extract durable facts at SESSION_END (D).

Fires once when a run completes (not on intermediate tool-call iterations) and
asks the LLM to extract durable user facts/preferences from the conversation,
then stores them (redacted) in the KV memory store so future runs can recall
them. Best-effort: never aborts the hook chain on failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.proactive_memory import ProactiveMemory

_logger = logging.getLogger(__name__)


class ProactiveExtractionHook(Hook):
    """Extract durable facts at SESSION_END and store them (issue D).

    Priority 65: post-processing band (runs after business hooks; no ordering
    dependency on the skill/task persistence hooks at 45/46).
    """

    priority = 65

    def __init__(self, proactive: ProactiveMemory) -> None:
        self._proactive = proactive

    def handles(self) -> list[HookEvent]:
        return [HookEvent.SESSION_END]

    async def execute(self, ctx: HookContext) -> HookContext:
        try:
            await self._proactive.extract_and_store()
        except Exception as exc:  # nosec - best-effort; must not break the run
            _logger.warning("Proactive extraction hook failed: %s", exc)
        return ctx
