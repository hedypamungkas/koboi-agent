"""koboi/hooks/task_hook.py -- Inject task reminders into conversation context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.task import TaskManager


class TaskHook(Hook):
    """Periodically reminds the LLM about pending tasks."""

    def __init__(self, reminder_interval: int = 3, manager: TaskManager | None = None) -> None:
        self._reminder_interval = reminder_interval
        self._calls_since_reminder = 0
        self.manager = manager

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_LLM_CALL]

    async def execute(self, ctx: HookContext) -> HookContext:
        mgr = self.manager
        if mgr is None:
            return ctx

        self._calls_since_reminder += 1
        summary = mgr.summary()

        if not summary:
            self._calls_since_reminder = 0
            return ctx

        if self._calls_since_reminder >= self._reminder_interval:
            ctx.inject_message = (
                f"[Task Reminder]\n{summary}\nUse task_update to mark tasks as in_progress or completed."
            )
            self._calls_since_reminder = 0

        return ctx
