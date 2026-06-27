"""Regression test: SubAgentManager and TaskManager use distinct tool-dep keys.

Previously both ``_setup_subagent`` and ``_setup_tasks`` called
``set_dep("manager", ...)``, so registering both ``delegate_tasks`` and the
task tools clobbered one manager with the other. The keys are now
``"subagent_manager"`` and ``"task_manager"``.
"""

from __future__ import annotations

from koboi.task import TaskManager
from koboi.tools.builtin import register_all
from koboi.tools.registry import ToolRegistry


class TestManagerDepKeys:
    def test_distinct_keys_coexist_on_one_registry(self):
        """Both managers must be retrievable simultaneously (no clobber)."""
        registry = ToolRegistry()
        register_all(registry)
        sub_mgr, task_mgr = object(), object()
        registry.set_dep("subagent_manager", sub_mgr)
        registry.set_dep("task_manager", task_mgr)

        assert registry.get_dep("subagent_manager") is sub_mgr
        assert registry.get_dep("task_manager") is task_mgr
        # The bug: both used "manager" so one overwrote the other.
        assert registry.get_dep("subagent_manager") is not registry.get_dep("task_manager")

    async def test_task_tool_reads_task_manager_key(self):
        """task_list must receive the TaskManager via the "task_manager" dep."""
        registry = ToolRegistry()
        register_all(registry)
        registry.set_dep("task_manager", TaskManager())

        result = await registry.execute("task_list", '{"status": ""}')
        assert isinstance(result, str)
        assert "not initialized" not in result.lower() and "error" not in result.lower()

    def test_old_manager_key_is_gone(self):
        """The deprecated overloaded "manager" key must no longer be set by the tools."""
        registry = ToolRegistry()
        register_all(registry)
        # Neither builtin path sets the old key anymore.
        assert registry.get_dep("manager") is None
