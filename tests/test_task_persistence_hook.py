"""Tests for TaskPersistenceHook (P3b)."""

from __future__ import annotations

from koboi.task import TaskManager
from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.task_persistence_hook import TaskPersistenceHook


class TestTaskPersistenceHook:
    def test_priority_is_46(self):
        assert TaskPersistenceHook(TaskManager()).priority == 46

    def test_handles_post_compact(self):
        assert TaskPersistenceHook(TaskManager()).handles() == [HookEvent.POST_COMPACT]

    async def test_injects_summary_when_active_tasks(self):
        mgr = TaskManager()
        mgr.create("Fix login bug", description="the OAuth flow")
        ctx = await TaskPersistenceHook(mgr).execute(HookContext(event=HookEvent.POST_COMPACT))
        assert len(ctx.inject_messages) == 1
        assert "[Active Task State]" in ctx.inject_messages[0]
        assert "Fix login bug" in ctx.inject_messages[0]

    async def test_no_injection_when_no_tasks(self):
        ctx = await TaskPersistenceHook(TaskManager()).execute(HookContext(event=HookEvent.POST_COMPACT))
        assert ctx.inject_messages == []

    async def test_no_injection_when_all_completed(self):
        mgr = TaskManager()
        t = mgr.create("Already done")
        mgr.mark_completed(t.id)
        ctx = await TaskPersistenceHook(mgr).execute(HookContext(event=HookEvent.POST_COMPACT))
        assert ctx.inject_messages == []

    async def test_idempotent_re_injection(self):
        """Re-injecting on every POST_COMPACT is harmless (duplicate appends retained)."""
        mgr = TaskManager()
        mgr.create("Task A")
        hook = TaskPersistenceHook(mgr)
        ctx = HookContext(event=HookEvent.POST_COMPACT)
        await hook.execute(ctx)
        await hook.execute(ctx)
        assert len(ctx.inject_messages) == 2
