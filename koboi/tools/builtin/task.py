"""koboi/tools/builtin/task.py -- Task management tools with dependency support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from koboi.tools.registry import tool
from koboi.types import RiskLevel

if TYPE_CHECKING:
    from koboi.task import Task


def get_manager() -> Any:
    """Backward-compat accessor. Prefer _deps['manager'] in tool functions."""
    raise RuntimeError("TaskManager not initialized — use _deps['manager'] instead")


@tool(
    name="task_create",
    group="task",
    description=(
        "Create a new task to track work progress. Optionally specify dependencies "
        "on other tasks -- a task with dependencies is 'blocked' until all dependencies "
        "are completed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Brief title for the task (imperative form, e.g. 'Fix login bug')",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of what needs to be done",
            },
            "blocked_by": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of task IDs this task depends on (e.g. ['t_0001', 't_0002']). Task won't start until all dependencies complete.",
            },
        },
        "required": ["subject"],
    },
    risk_level=RiskLevel.SAFE,
    deps=["manager"],
)
def task_create(
    subject: str, description: str = "", blocked_by: list[str] | None = None, _deps: dict | None = None
) -> str:
    try:
        mgr = _deps.get("manager") if _deps else None
        if mgr is None:
            return "Error: TaskManager not initialized"
        task = mgr.create(subject, description, blocked_by=blocked_by)
        dep_info = f" (blocked by: {', '.join(blocked_by)})" if blocked_by else ""
        return f"Created task {task.id}: {task.subject} [{task.status}]{dep_info}"
    except RuntimeError as e:
        return f"Error: {e}"


@tool(
    name="task_list",
    group="task",
    description=(
        "List all tasks, optionally filtered by status. Shows dependency info "
        "for blocked tasks. Use this to check remaining work and track progress."
    ),
    parameters={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "Filter by status: 'pending', 'in_progress', 'completed', or 'blocked'",
                "enum": ["pending", "in_progress", "completed", "blocked"],
            },
        },
        "required": [],
    },
    risk_level=RiskLevel.SAFE,
    deps=["manager"],
)
def task_list(status: str = "", _deps: dict | None = None) -> str:
    try:
        mgr = _deps.get("manager") if _deps else None
        if mgr is None:
            return "Error: TaskManager not initialized"
        tasks = mgr.list_tasks(status_filter=status or None)
        if not tasks:
            return "No tasks found."
        lines = [f"Tasks ({len(tasks)}):"]
        for t in tasks:
            desc = f" - {t.description[:60]}" if t.description else ""
            dep_info = ""
            if t.blocked_by:
                dep_info = f" [blocked by: {', '.join(t.blocked_by)}]"
            lines.append(f"  [{t.status}] {t.id}: {t.subject}{desc}{dep_info}")
        return "\n".join(lines)
    except RuntimeError as e:
        return f"Error: {e}"


@tool(
    name="task_get",
    group="task",
    description="Get details of a specific task by its ID, including dependencies.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID (e.g. 't_0001')",
            },
        },
        "required": ["task_id"],
    },
    risk_level=RiskLevel.SAFE,
    deps=["manager"],
)
def task_get(task_id: str, _deps: dict | None = None) -> str:
    try:
        mgr = _deps.get("manager") if _deps else None
        if mgr is None:
            return "Error: TaskManager not initialized"
        task = mgr.get(task_id)
        if task is None:
            return f"Task not found: {task_id}"
        lines = [
            f"Task {task.id}:",
            f"  Subject: {task.subject}",
            f"  Status: {task.status}",
        ]
        if task.description:
            lines.append(f"  Description: {task.description}")
        if task.blocked_by:
            lines.append(f"  Blocked by: {', '.join(task.blocked_by)}")
        return "\n".join(lines)
    except RuntimeError as e:
        return f"Error: {e}"


@tool(
    name="task_update",
    group="task",
    description=(
        "Update a task's status, subject, or description. Use this to mark "
        "tasks as in_progress when starting work, or completed when done. "
        "Note: setting status to 'in_progress' will fail if the task has "
        "unfinished dependencies."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID (e.g. 't_0001')",
            },
            "status": {
                "type": "string",
                "description": "New status: 'pending', 'in_progress', or 'completed'",
                "enum": ["pending", "in_progress", "completed"],
            },
            "subject": {
                "type": "string",
                "description": "New subject/title for the task",
            },
            "description": {
                "type": "string",
                "description": "New description for the task",
            },
        },
        "required": ["task_id"],
    },
    risk_level=RiskLevel.SAFE,
    deps=["manager"],
)
def task_update(
    task_id: str, status: str = "", subject: str = "", description: str = "", _deps: dict | None = None
) -> str:
    try:
        mgr = _deps.get("manager") if _deps else None
        if mgr is None:
            return "Error: TaskManager not initialized"

        # Use mark_completed for completed status to trigger auto-unblock
        if status == "completed":
            task, unblocked = mgr.mark_completed(task_id)
            if task is None:
                return f"Task not found: {task_id}"
            msg = f"Updated task {task.id}: [{task.status}] {task.subject}"
            if unblocked:
                msg += f"\nUnblocked: {', '.join(unblocked)}"
            return msg

        result: Task | tuple[Task, str] | None = mgr.update(
            task_id,
            status=status or None,
            subject=subject or None,
            description=description or None,
        )
        if result is None:
            return f"Task not found: {task_id}"
        # Check if blocked (tuple return)
        if isinstance(result, tuple):
            task, reason = result
            return f"Cannot start {task.id}: {reason}"
        return f"Updated task {result.id}: [{result.status}] {result.subject}"
    except RuntimeError as e:
        return f"Error: {e}"


@tool(
    name="task_add_dependency",
    description=(
        "Add a dependency between tasks. The target task will be blocked until the dependency task is completed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task that should be blocked (e.g. 't_0002')",
            },
            "depends_on": {
                "type": "string",
                "description": "The task that must complete first (e.g. 't_0001')",
            },
        },
        "required": ["task_id", "depends_on"],
    },
    risk_level=RiskLevel.SAFE,
    deps=["manager"],
)
def task_add_dependency(task_id: str, depends_on: str, _deps: dict | None = None) -> str:
    try:
        mgr = _deps.get("manager") if _deps else None
        if mgr is None:
            return "Error: TaskManager not initialized"
        success, message = mgr.add_dependency(task_id, depends_on)
        return message
    except RuntimeError as e:
        return f"Error: {e}"
