"""tests/test_task.py -- Tests for TaskManager, task tools, and TaskHook."""

from __future__ import annotations

import json

import pytest

from koboi.task import TaskManager
from koboi.tools.registry import ToolRegistry, register_decorated
from koboi.hooks.chain import HookContext, HookEvent


# ---------------------------------------------------------------------------
# TaskManager tests
# ---------------------------------------------------------------------------


class TestTaskManager:
    def test_create_task(self):
        mgr = TaskManager()
        task = mgr.create("Fix login bug", "The login form returns 500")
        assert task.id == "t_0001"
        assert task.subject == "Fix login bug"
        assert task.description == "The login form returns 500"
        assert task.status == "pending"
        assert task.created_at > 0

    def test_create_multiple_tasks(self):
        mgr = TaskManager()
        t1 = mgr.create("Task 1")
        t2 = mgr.create("Task 2")
        t3 = mgr.create("Task 3")
        assert t1.id == "t_0001"
        assert t2.id == "t_0002"
        assert t3.id == "t_0003"

    def test_get_existing_task(self):
        mgr = TaskManager()
        created = mgr.create("Test task")
        retrieved = mgr.get("t_0001")
        assert retrieved is created

    def test_get_nonexistent_task(self):
        mgr = TaskManager()
        assert mgr.get("t_9999") is None

    def test_list_all_tasks(self):
        mgr = TaskManager()
        mgr.create("Task 1")
        mgr.create("Task 2")
        mgr.create("Task 3")
        tasks = mgr.list_tasks()
        assert len(tasks) == 3

    def test_list_filter_by_status(self):
        mgr = TaskManager()
        mgr.create("Task 1")
        mgr.create("Task 2")
        mgr.update("t_0001", status="completed")
        pending = mgr.list_tasks(status_filter="pending")
        completed = mgr.list_tasks(status_filter="completed")
        assert len(pending) == 1
        assert len(completed) == 1

    def test_update_status(self):
        mgr = TaskManager()
        mgr.create("Test task")
        updated = mgr.update("t_0001", status="in_progress")
        assert updated is not None
        assert updated.status == "in_progress"

    def test_update_subject(self):
        mgr = TaskManager()
        mgr.create("Old subject")
        updated = mgr.update("t_0001", subject="New subject")
        assert updated is not None
        assert updated.subject == "New subject"

    def test_update_description(self):
        mgr = TaskManager()
        mgr.create("Task", "Old description")
        updated = mgr.update("t_0001", description="New description")
        assert updated is not None
        assert updated.description == "New description"

    def test_update_nonexistent_task(self):
        mgr = TaskManager()
        result = mgr.update("t_9999", status="completed")
        assert result is None

    def test_summary_with_active_tasks(self):
        mgr = TaskManager()
        mgr.create("Task 1")
        mgr.create("Task 2")
        mgr.update("t_0001", status="in_progress")
        summary = mgr.summary()
        assert "t_0001" in summary
        assert "t_0002" in summary
        assert "in_progress" in summary
        assert "pending" in summary

    def test_summary_no_active_tasks(self):
        mgr = TaskManager()
        mgr.create("Task 1")
        mgr.update("t_0001", status="completed")
        assert mgr.summary() == ""

    def test_summary_empty(self):
        mgr = TaskManager()
        assert mgr.summary() == ""

    def test_summary_short(self):
        mgr = TaskManager()
        mgr.create("Task 1")
        mgr.create("Task 2")
        mgr.create("Task 3")
        mgr.update("t_0001", status="in_progress")
        mgr.update("t_0002", status="completed")
        short = mgr.summary_short()
        assert "1 pending" in short
        assert "1 in progress" in short
        assert "1 done" in short

    def test_summary_short_empty(self):
        mgr = TaskManager()
        assert mgr.summary_short() == ""

    def test_clear(self):
        mgr = TaskManager()
        mgr.create("Task 1")
        mgr.create("Task 2")
        mgr.clear()
        assert mgr.list_tasks() == []
        # Counter resets, so next task gets t_0001 again
        task = mgr.create("New task")
        assert task.id == "t_0001"

    # -- Dependency tests --

    def test_create_with_blocked_by(self):
        mgr = TaskManager()
        mgr.create("Step 1")
        task = mgr.create("Step 2", blocked_by=["t_0001"])
        assert task.status == "blocked"
        assert task.blocked_by == ["t_0001"]

    def test_create_without_blocked_by(self):
        mgr = TaskManager()
        task = mgr.create("Independent task")
        assert task.status == "pending"
        assert task.blocked_by == []

    def test_update_blocked_task_fails(self):
        mgr = TaskManager()
        mgr.create("Step 1")
        mgr.create("Step 2", blocked_by=["t_0001"])
        result = mgr.update("t_0002", status="in_progress")
        assert isinstance(result, tuple)
        task, reason = result
        assert task.status == "blocked"
        assert "Blocked by" in reason

    def test_mark_completed_unblocks_dependents(self):
        mgr = TaskManager()
        mgr.create("Step 1")
        mgr.create("Step 2", blocked_by=["t_0001"])
        task, unblocked = mgr.mark_completed("t_0001")
        assert task.status == "completed"
        assert "t_0002" in unblocked
        assert mgr.get("t_0002").status == "pending"

    def test_mark_completed_partial_unblock(self):
        mgr = TaskManager()
        mgr.create("Step 1")
        mgr.create("Step 2")
        mgr.create("Step 3", blocked_by=["t_0001", "t_0002"])
        # Complete only step 1 -- step 3 stays blocked
        _, unblocked = mgr.mark_completed("t_0001")
        assert "t_0003" not in unblocked
        assert mgr.get("t_0003").status == "blocked"
        # Complete step 2 -- now step 3 unblocks
        _, unblocked = mgr.mark_completed("t_0002")
        assert "t_0003" in unblocked
        assert mgr.get("t_0003").status == "pending"

    def test_add_dependency(self):
        mgr = TaskManager()
        mgr.create("Step 1")
        mgr.create("Step 2")
        success, msg = mgr.add_dependency("t_0002", "t_0001")
        assert success is True
        assert "t_0001" in msg
        assert mgr.get("t_0002").blocked_by == ["t_0001"]
        assert mgr.get("t_0002").status == "blocked"

    def test_add_dependency_already_exists(self):
        mgr = TaskManager()
        mgr.create("Step 1")
        mgr.create("Step 2", blocked_by=["t_0001"])
        success, msg = mgr.add_dependency("t_0002", "t_0001")
        assert success is True
        assert "Already" in msg

    def test_add_dependency_self(self):
        mgr = TaskManager()
        mgr.create("Task")
        success, msg = mgr.add_dependency("t_0001", "t_0001")
        assert success is False
        assert "itself" in msg

    def test_add_dependency_circular(self):
        mgr = TaskManager()
        mgr.create("Step 1")
        mgr.create("Step 2", blocked_by=["t_0001"])
        # t_0001 depends on t_0002 would create cycle
        success, msg = mgr.add_dependency("t_0001", "t_0002")
        assert success is False
        assert "Circular" in msg

    def test_add_dependency_nonexistent_task(self):
        mgr = TaskManager()
        mgr.create("Task")
        success, msg = mgr.add_dependency("t_9999", "t_0001")
        assert success is False
        assert "not found" in msg

    def test_add_dependency_nonexistent_dep(self):
        mgr = TaskManager()
        mgr.create("Task")
        success, msg = mgr.add_dependency("t_0001", "t_9999")
        assert success is False
        assert "not found" in msg

    def test_would_cycle_no_cycle(self):
        mgr = TaskManager()
        mgr.create("A")
        mgr.create("B", blocked_by=["t_0001"])
        mgr.create("C", blocked_by=["t_0002"])
        # Adding dep from C to A is fine (no cycle)
        assert mgr._would_cycle("t_0003", "t_0001") is False

    def test_would_cycle_detects(self):
        mgr = TaskManager()
        mgr.create("A")
        mgr.create("B", blocked_by=["t_0001"])
        # A -> B -> A would be a cycle
        assert mgr._would_cycle("t_0001", "t_0002") is True

    def test_update_after_deps_completed(self):
        """Once deps are completed, blocked task can be started."""
        mgr = TaskManager()
        mgr.create("Step 1")
        mgr.create("Step 2", blocked_by=["t_0001"])
        mgr.mark_completed("t_0001")
        # Now should be able to start
        result = mgr.update("t_0002", status="in_progress")
        assert not isinstance(result, tuple)
        assert result.status == "in_progress"

    def test_summary_shows_blocked(self):
        mgr = TaskManager()
        mgr.create("Step 1")
        mgr.create("Step 2", blocked_by=["t_0001"])
        summary = mgr.summary()
        assert "blocked" in summary
        assert "t_0001" in summary

    def test_list_filter_blocked(self):
        mgr = TaskManager()
        mgr.create("Step 1")
        mgr.create("Step 2", blocked_by=["t_0001"])
        blocked = mgr.list_tasks(status_filter="blocked")
        assert len(blocked) == 1
        assert blocked[0].id == "t_0002"


# ---------------------------------------------------------------------------
# Task tool tests
# ---------------------------------------------------------------------------


class TestTaskTools:
    @pytest.fixture(autouse=True)
    def _setup_manager(self):
        """Set up a fresh TaskManager for each test."""
        mgr = TaskManager()
        self._mgr = mgr
        yield

    def _register_tools(self) -> ToolRegistry:
        from koboi.tools.builtin import task as task_mod

        registry = ToolRegistry()
        registry.set_dep("task_manager", self._mgr)
        register_decorated(registry, task_mod)
        return registry

    @pytest.mark.asyncio
    async def test_task_create(self):
        registry = self._register_tools()
        result = await registry.execute("task_create", json.dumps({"subject": "Test task"}))
        assert "t_0001" in result
        assert "Test task" in result

    @pytest.mark.asyncio
    async def test_task_create_with_description(self):
        registry = self._register_tools()
        result = await registry.execute(
            "task_create",
            json.dumps(
                {
                    "subject": "Fix bug",
                    "description": "The login page crashes",
                }
            ),
        )
        assert "t_0001" in result
        assert "Fix bug" in result

    @pytest.mark.asyncio
    async def test_task_list_empty(self):
        registry = self._register_tools()
        result = await registry.execute("task_list", json.dumps({}))
        assert "No tasks" in result

    @pytest.mark.asyncio
    async def test_task_list_with_tasks(self):
        self._mgr.create("Task 1")
        self._mgr.create("Task 2")
        registry = self._register_tools()
        result = await registry.execute("task_list", json.dumps({}))
        assert "Task 1" in result
        assert "Task 2" in result

    @pytest.mark.asyncio
    async def test_task_list_filter(self):
        self._mgr.create("Task 1")
        self._mgr.create("Task 2")
        self._mgr.update("t_0001", status="completed")
        registry = self._register_tools()
        result = await registry.execute("task_list", json.dumps({"status": "pending"}))
        assert "Task 2" in result
        assert "Task 1" not in result

    @pytest.mark.asyncio
    async def test_task_get(self):
        self._mgr.create("My task", "Description here")
        registry = self._register_tools()
        result = await registry.execute("task_get", json.dumps({"task_id": "t_0001"}))
        assert "My task" in result
        assert "Description here" in result

    @pytest.mark.asyncio
    async def test_task_get_not_found(self):
        registry = self._register_tools()
        result = await registry.execute("task_get", json.dumps({"task_id": "t_9999"}))
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_task_update_status(self):
        self._mgr.create("Test task")
        registry = self._register_tools()
        result = await registry.execute(
            "task_update",
            json.dumps(
                {
                    "task_id": "t_0001",
                    "status": "completed",
                }
            ),
        )
        assert "completed" in result
        assert self._mgr.get("t_0001").status == "completed"

    @pytest.mark.asyncio
    async def test_task_update_not_found(self):
        registry = self._register_tools()
        result = await registry.execute(
            "task_update",
            json.dumps(
                {
                    "task_id": "t_9999",
                    "status": "completed",
                }
            ),
        )
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_tools_without_manager(self):
        """Tools should return error strings when manager is not set."""
        from koboi.tools.builtin import task as task_mod

        registry = ToolRegistry()
        # Don't set dep "task_manager" -- simulates uninitialized state
        register_decorated(registry, task_mod)
        result = await registry.execute("task_create", json.dumps({"subject": "Test"}))
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_task_create_with_blocked_by(self):
        registry = self._register_tools()
        await registry.execute("task_create", json.dumps({"subject": "Step 1"}))
        result = await registry.execute(
            "task_create",
            json.dumps(
                {
                    "subject": "Step 2",
                    "blocked_by": ["t_0001"],
                }
            ),
        )
        assert "t_0002" in result
        assert "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_task_update_blocked_fails(self):
        registry = self._register_tools()
        await registry.execute("task_create", json.dumps({"subject": "Step 1"}))
        await registry.execute(
            "task_create",
            json.dumps(
                {
                    "subject": "Step 2",
                    "blocked_by": ["t_0001"],
                }
            ),
        )
        result = await registry.execute(
            "task_update",
            json.dumps(
                {
                    "task_id": "t_0002",
                    "status": "in_progress",
                }
            ),
        )
        assert "Cannot start" in result or "Blocked" in result

    @pytest.mark.asyncio
    async def test_task_complete_unblocks(self):
        registry = self._register_tools()
        await registry.execute("task_create", json.dumps({"subject": "Step 1"}))
        await registry.execute(
            "task_create",
            json.dumps(
                {
                    "subject": "Step 2",
                    "blocked_by": ["t_0001"],
                }
            ),
        )
        result = await registry.execute(
            "task_update",
            json.dumps(
                {
                    "task_id": "t_0001",
                    "status": "completed",
                }
            ),
        )
        assert "Unblocked" in result or "unblocked" in result.lower()

    @pytest.mark.asyncio
    async def test_task_add_dependency(self):
        registry = self._register_tools()
        await registry.execute("task_create", json.dumps({"subject": "Step 1"}))
        await registry.execute("task_create", json.dumps({"subject": "Step 2"}))
        result = await registry.execute(
            "task_add_dependency",
            json.dumps(
                {
                    "task_id": "t_0002",
                    "depends_on": "t_0001",
                }
            ),
        )
        assert "t_0001" in result
        assert self._mgr.get("t_0002").status == "blocked"

    @pytest.mark.asyncio
    async def test_task_add_dependency_circular(self):
        registry = self._register_tools()
        await registry.execute("task_create", json.dumps({"subject": "Step 1"}))
        await registry.execute(
            "task_create",
            json.dumps(
                {
                    "subject": "Step 2",
                    "blocked_by": ["t_0001"],
                }
            ),
        )
        result = await registry.execute(
            "task_add_dependency",
            json.dumps(
                {
                    "task_id": "t_0001",
                    "depends_on": "t_0002",
                }
            ),
        )
        assert "Circular" in result

    @pytest.mark.asyncio
    async def test_task_list_shows_blocked(self):
        registry = self._register_tools()
        await registry.execute("task_create", json.dumps({"subject": "Step 1"}))
        await registry.execute(
            "task_create",
            json.dumps(
                {
                    "subject": "Step 2",
                    "blocked_by": ["t_0001"],
                }
            ),
        )
        result = await registry.execute("task_list", json.dumps({}))
        assert "blocked" in result
        assert "t_0001" in result


# ---------------------------------------------------------------------------
# TaskHook tests
# ---------------------------------------------------------------------------


class TestTaskHook:
    @pytest.fixture(autouse=True)
    def _setup_manager(self):
        mgr = TaskManager()
        self._mgr = mgr
        yield

    @pytest.mark.asyncio
    async def test_hook_no_tasks_no_injection(self):
        from koboi.hooks.task_hook import TaskHook

        hook = TaskHook(reminder_interval=1, manager=self._mgr)
        ctx = HookContext(event=HookEvent.POST_LLM_CALL)
        result = await hook.execute(ctx)
        assert result.inject_message is None

    @pytest.mark.asyncio
    async def test_hook_injects_after_interval(self):
        from koboi.hooks.task_hook import TaskHook

        hook = TaskHook(reminder_interval=2, manager=self._mgr)
        self._mgr.create("Pending task")

        # First call -- should not inject (counter=1 < interval=2)
        ctx1 = HookContext(event=HookEvent.POST_LLM_CALL)
        result1 = await hook.execute(ctx1)
        assert result1.inject_message is None

        # Second call -- should inject (counter=2 >= interval=2)
        ctx2 = HookContext(event=HookEvent.POST_LLM_CALL)
        result2 = await hook.execute(ctx2)
        assert result2.inject_message is not None
        assert "t_0001" in result2.inject_message

    @pytest.mark.asyncio
    async def test_hook_resets_counter_on_injection(self):
        from koboi.hooks.task_hook import TaskHook

        hook = TaskHook(reminder_interval=1, manager=self._mgr)
        self._mgr.create("Task")

        ctx = HookContext(event=HookEvent.POST_LLM_CALL)
        await hook.execute(ctx)
        assert hook._calls_since_reminder == 0

    @pytest.mark.asyncio
    async def test_hook_resets_when_no_active_tasks(self):
        from koboi.hooks.task_hook import TaskHook

        hook = TaskHook(reminder_interval=5, manager=self._mgr)
        self._mgr.create("Task")
        self._mgr.update("t_0001", status="completed")

        ctx = HookContext(event=HookEvent.POST_LLM_CALL)
        result = await hook.execute(ctx)
        assert result.inject_message is None
        assert hook._calls_since_reminder == 0

    @pytest.mark.asyncio
    async def test_hook_handles_no_manager(self):
        from koboi.hooks.task_hook import TaskHook

        hook = TaskHook(reminder_interval=1)  # no manager
        ctx = HookContext(event=HookEvent.POST_LLM_CALL)
        result = await hook.execute(ctx)
        assert result.inject_message is None
