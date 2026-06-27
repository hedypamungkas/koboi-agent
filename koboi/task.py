"""koboi/task.py -- Task data model and in-memory TaskManager."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Task:
    id: str
    subject: str
    description: str = ""
    status: Literal["pending", "in_progress", "completed", "blocked"] = "pending"
    blocked_by: list[str] = field(default_factory=list)
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)


class TaskManager:
    """Session-scoped, in-memory task store with dependency support."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._counter: int = 0

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[str] | None = None,
    ) -> Task:
        self._counter += 1
        task_id = f"t_{self._counter:04x}"
        deps = list(blocked_by) if blocked_by else []
        # Determine initial status: blocked if has deps, else pending
        initial_status: Literal["pending", "blocked"] = "blocked" if deps else "pending"
        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            status=initial_status,
            blocked_by=deps,
            created_at=time.time(),
        )
        self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self, status_filter: str | None = None) -> list[Task]:
        tasks = list(self._tasks.values())
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        return tasks

    def update(
        self,
        task_id: str,
        status: str | None = None,
        subject: str | None = None,
        description: str | None = None,
    ) -> Task | tuple[Task, str] | None:
        """Update a task. Returns Task on success, (Task, reason) if blocked, or None if not found."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if status is not None:
            if status == "in_progress" and task.blocked_by:
                # Check if all dependencies are completed
                unfinished = [
                    dep for dep in task.blocked_by if dep in self._tasks and self._tasks[dep].status != "completed"
                ]
                if unfinished:
                    dep_subjects = [
                        f"{dep} ({self._tasks[dep].subject})" if dep in self._tasks else dep for dep in unfinished
                    ]
                    reason = f"Blocked by: {', '.join(dep_subjects)}"
                    return task, reason
                # All deps done -- clear blocked_by and allow transition
                task.blocked_by = []
            task.status = status  # type: ignore[assignment]
        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        return task

    def add_dependency(self, task_id: str, depends_on: str) -> tuple[bool, str]:
        """Add a dependency. Returns (success, message)."""
        task = self._tasks.get(task_id)
        if task is None:
            return False, f"Task not found: {task_id}"
        dep = self._tasks.get(depends_on)
        if dep is None:
            return False, f"Dependency task not found: {depends_on}"
        if task_id == depends_on:
            return False, "Task cannot depend on itself"
        if depends_on in task.blocked_by:
            return True, f"Already depends on {depends_on}"
        # Check for circular dependency
        if self._would_cycle(task_id, depends_on):
            return False, f"Circular dependency: {depends_on} already depends on {task_id}"
        task.blocked_by.append(depends_on)
        # If task is pending and now has deps, mark blocked
        if task.status == "pending" and task.blocked_by:
            task.status = "blocked"
        return True, f"{task_id} now depends on {depends_on}"

    def _would_cycle(self, task_id: str, new_dep: str) -> bool:
        """Check if adding new_dep to task_id would create a cycle."""
        visited: set[str] = set()
        stack = [new_dep]
        while stack:
            current = stack.pop()
            if current == task_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            t = self._tasks.get(current)
            if t:
                stack.extend(t.blocked_by)
        return False

    def _try_unblock(self) -> list[str]:
        """Check blocked tasks and unblock those whose deps are all done. Returns unblocked IDs."""
        unblocked: list[str] = []
        for task in self._tasks.values():
            if task.status != "blocked":
                continue
            unfinished = [
                dep for dep in task.blocked_by if dep in self._tasks and self._tasks[dep].status != "completed"
            ]
            if not unfinished:
                task.blocked_by = []
                task.status = "pending"
                unblocked.append(task.id)
        return unblocked

    def mark_completed(self, task_id: str) -> tuple[Task | None, list[str]]:
        """Mark task completed and auto-unblock dependents. Returns (task, unblocked_ids)."""
        task = self._tasks.get(task_id)
        if task is None:
            return None, []
        task.status = "completed"
        unblocked = self._try_unblock()
        return task, unblocked

    def summary(self) -> str:
        """Human-readable summary of active tasks, for context injection."""
        active = [t for t in self._tasks.values() if t.status != "completed"]
        if not active:
            return ""
        lines = ["Active tasks:"]
        for t in active:
            dep_info = ""
            if t.blocked_by:
                dep_info = f" [blocked by: {', '.join(t.blocked_by)}]"
            lines.append(f"  [{t.status}] {t.id}: {t.subject}{dep_info}")
        return "\n".join(lines)

    def summary_short(self) -> str:
        """One-line summary for status bar display."""
        pending = sum(1 for t in self._tasks.values() if t.status == "pending")
        in_progress = sum(1 for t in self._tasks.values() if t.status == "in_progress")
        completed = sum(1 for t in self._tasks.values() if t.status == "completed")
        blocked = sum(1 for t in self._tasks.values() if t.status == "blocked")
        parts: list[str] = []
        if pending:
            parts.append(f"{pending} pending")
        if in_progress:
            parts.append(f"{in_progress} in progress")
        if blocked:
            parts.append(f"{blocked} blocked")
        if completed:
            parts.append(f"{completed} done")
        return ", ".join(parts) if parts else ""

    def clear(self) -> None:
        self._tasks.clear()
        self._counter = 0
