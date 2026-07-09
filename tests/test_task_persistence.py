"""tests/test_task_persistence.py -- #6 task state survives --resume + nudge."""

from __future__ import annotations

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.task_hook import TaskHook
from koboi.task import TaskManager


def test_tasks_persist_and_reload(tmp_path):
    db = str(tmp_path / "tasks.db")
    sid = "sess-1"

    mgr1 = TaskManager(db_path=db, session_id=sid)
    a = mgr1.create("task A")
    b = mgr1.create("task B", blocked_by=[a.id])  # B blocked by A (a NOT completed)

    # A fresh manager on the same db+session rehydrates (simulates --resume).
    mgr2 = TaskManager(db_path=db, session_id=sid)
    tasks = {t.id: t for t in mgr2.list_tasks()}
    assert set(tasks) == {a.id, b.id}
    assert tasks[a.id].status == "pending"
    assert tasks[b.id].status == "blocked"
    assert tasks[b.id].blocked_by == [a.id]  # deps round-trip while still blocked
    # counter restored -> new task_id doesn't collide with reloaded ones
    c = mgr2.create("task C")
    assert c.id not in {a.id, b.id}


def test_mark_completed_persists_and_unblocks(tmp_path):
    db = str(tmp_path / "tasks.db")
    sid = "sess-1"

    mgr1 = TaskManager(db_path=db, session_id=sid)
    a = mgr1.create("task A")
    mgr1.create("task B", blocked_by=[a.id])
    mgr1.mark_completed(a.id)  # A done -> B unblocks (blocked_by cleared by design)

    mgr2 = TaskManager(db_path=db, session_id=sid)
    tasks = {t.id: t for t in mgr2.list_tasks()}
    assert tasks[a.id].status == "completed"
    b_task = next(t for t in tasks.values() if t.subject == "task B")
    assert b_task.status == "pending"  # unblocked
    assert b_task.blocked_by == []  # _try_unblock clears deps on unblock


def test_no_persistence_without_db_is_back_compat():
    mgr = TaskManager()  # in-memory only (pre-#6 behavior)
    mgr.create("x")
    assert len(mgr.list_tasks()) == 1


def test_different_sessions_are_isolated(tmp_path):
    db = str(tmp_path / "tasks.db")
    TaskManager(db_path=db, session_id="s1").create("s1-task")
    mgr2 = TaskManager(db_path=db, session_id="s2")
    assert mgr2.list_tasks() == []  # s2 sees no s1 tasks


def test_clear_persists(tmp_path):
    db = str(tmp_path / "tasks.db")
    sid = "sess-clear"
    mgr1 = TaskManager(db_path=db, session_id=sid)
    mgr1.create("a")
    mgr1.clear()
    mgr2 = TaskManager(db_path=db, session_id=sid)
    assert mgr2.list_tasks() == []


async def test_task_hook_nudges_once_then_stops():
    hook = TaskHook(manager=TaskManager())

    ctx1 = HookContext(event=HookEvent.PRE_LLM_CALL)
    await hook.execute(ctx1)
    assert ctx1.inject_message is not None
    assert "task_create" in ctx1.inject_message

    ctx2 = HookContext(event=HookEvent.PRE_LLM_CALL)
    await hook.execute(ctx2)
    assert ctx2.inject_message is None  # nudged only once


async def test_task_hook_no_nudge_without_manager():
    hook = TaskHook(manager=None)
    ctx = HookContext(event=HookEvent.PRE_LLM_CALL)
    await hook.execute(ctx)
    assert ctx.inject_message is None
