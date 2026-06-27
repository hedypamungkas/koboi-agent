"""koboi/hooks/task_persistence_hook.py -- Re-inject active todo list after compaction.

Mirrors SkillPersistenceHook: when context compaction may have dropped the
task summary from the message history, this hook re-appends the current
active-task summary so the agent retains todo state across long conversations.

Re-injection is idempotent (re-adding the same summary text is harmless) and
runs on every POST_COMPACT emission, matching SkillPersistenceHook's behavior.
No compaction gate is needed -- unlike resetting read-tracking, injecting a
duplicate task list is non-destructive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.task import TaskManager


class TaskPersistenceHook(Hook):
    """Re-inject the active-task summary after context compaction.

    Priority 46: runs just after SkillPersistenceHook (45) so skill bodies and
    task state are re-injected together within the persistence band.
    """

    priority = 46

    def __init__(self, manager: TaskManager):
        self.manager = manager

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_COMPACT]

    async def execute(self, ctx: HookContext) -> HookContext:
        summary = self.manager.summary()
        if summary:
            ctx.inject_messages.append(f"[Active Task State]\n{summary}")
        return ctx
