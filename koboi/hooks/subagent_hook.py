"""koboi/hooks/subagent_hook.py -- Hook that bridges subagent events to the TUI.

Listens for AGENT_DISPATCHED and AGENT_COMPLETED events from SubAgentManager
and posts Textual messages so the TUI shows subagent progress.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    pass


class SubagentUIHook(Hook):
    """Translates subagent lifecycle events into TUI messages."""

    def __init__(self, app: Any = None) -> None:
        self._app = app

    def set_app(self, app: Any) -> None:
        self._app = app

    def handles(self) -> list[HookEvent]:
        return [HookEvent.AGENT_DISPATCHED, HookEvent.AGENT_COMPLETED]

    async def execute(self, ctx: HookContext) -> HookContext:
        if self._app is None:
            return ctx

        meta = ctx.metadata
        # Only handle subagent events (not orchestration events)
        if "subagent_label" not in meta:
            return ctx

        label = meta.get("subagent_label", "unknown")
        index = meta.get("subagent_index", 0)
        total = meta.get("subagent_total", 1)

        if ctx.event == HookEvent.AGENT_DISPATCHED:
            self._app.post_message(_SubagentDispatch(label, index, total))
        elif ctx.event == HookEvent.AGENT_COMPLETED:
            elapsed = meta.get("subagent_elapsed", 0.0)
            success = meta.get("subagent_success", True)
            error = meta.get("subagent_error")
            self._app.post_message(_SubagentResult(label, elapsed, success, error))

        return ctx


# Textual Messages (guarded -- textual is an optional TUI dependency)

try:
    from textual.message import Message

    class _SubagentDispatch(Message):
        """A subagent is starting."""

        def __init__(self, label: str, index: int, total: int) -> None:
            super().__init__()
            self.label = label
            self.index = index
            self.total = total

    class _SubagentResult(Message):
        """A subagent finished."""

        def __init__(self, label: str, elapsed: float, success: bool, error: str | None = None) -> None:
            super().__init__()
            self.label = label
            self.elapsed = elapsed
            self.success = success
            self.error = error

except ImportError:
    _SubagentDispatch = None  # type: ignore[misc,assignment]
    _SubagentResult = None  # type: ignore[misc,assignment]
