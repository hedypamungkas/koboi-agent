"""koboi/hooks/task_hook.py -- Inject task reminders into conversation context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.task import TaskManager


class TaskHook(Hook):
    """Nudges the LLM to use task tools once, then periodically reminds about tasks."""

    def __init__(self, reminder_interval: int = 3, manager: TaskManager | None = None) -> None:
        self._reminder_interval = reminder_interval
        self._calls_since_reminder = 0
        self._nudged = False
        self.manager = manager

    def handles(self) -> list[HookEvent]:
        return [HookEvent.PRE_LLM_CALL, HookEvent.POST_LLM_CALL]

    async def execute(self, ctx: HookContext) -> HookContext:
        mgr = self.manager
        if mgr is None:
            return ctx

        if ctx.event == HookEvent.PRE_LLM_CALL:
            # #6: one-time nudge so task tools are actually invoked (validated 0%->100%
            # with a nudge). Reaches the model from the next iteration via memory.
            if not self._nudged:
                self._nudged = True
                ctx.inject_message = (
                    "[Task Tools] For non-trivial multi-step work, use task_create to break the "
                    "goal into tracked tasks (blocked_by for dependencies), task_update to mark "
                    "progress, and task_complete when done."
                )
            return ctx

        # POST_LLM_CALL: periodic reminder (existing behavior).
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
