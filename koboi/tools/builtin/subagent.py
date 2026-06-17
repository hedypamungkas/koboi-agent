"""koboi/tools/builtin/subagent.py -- delegate_tasks tool for spawning subagents."""
from __future__ import annotations


from koboi.tools.registry import tool
from koboi.types import RiskLevel


@tool(
    name="delegate_tasks",
    description=(
        "Delegate one or more tasks to subagents that run in parallel. "
        "Each subagent works independently on its task and returns a result. "
        "Use this to break complex work into parallel subtasks for faster completion."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "List of tasks to delegate to subagents.",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Clear description of what the subagent should do.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Short label for this task (e.g. 'research', 'analyze'). Used for tracking.",
                        },
                    },
                    "required": ["task"],
                },
                "minItems": 1,
                "maxItems": 10,
            },
        },
        "required": ["tasks"],
    },
    risk_level=RiskLevel.SAFE,
    deps=["manager"],
)
async def delegate_tasks(tasks: list[dict], _deps: dict | None = None) -> str:
    _mgr = _deps.get("manager") if _deps else None
    if _mgr is None:
        return "Error: subagent system not initialized. Cannot delegate tasks."

    from koboi.subagent import SubagentTask

    subagent_tasks = [
        SubagentTask(
            task=t["task"],
            label=t.get("label", f"task_{i}"),
        )
        for i, t in enumerate(tasks)
    ]

    # Get parent messages for context sharing
    parent_messages = None
    parent_memory = getattr(_mgr, "_parent_memory", None)
    if parent_memory is not None:
        try:
            parent_messages = parent_memory.get_messages()
        except Exception:
            pass

    results = await _mgr.run_tasks(subagent_tasks, parent_messages=parent_messages)

    # Format results as a structured string
    parts = []
    for r in results:
        status = "OK" if r.success else f"FAILED: {r.error}"
        parts.append(
            f"[{r.label}] ({status}, {r.elapsed_seconds:.1f}s)\n"
            f"Task: {r.task}\n"
            f"Answer: {r.answer}"
        )

    return "\n\n---\n\n".join(parts)
