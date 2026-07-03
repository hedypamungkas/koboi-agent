"""Example 26: Task Management.

Demonstrates:
- task_create, task_list, task_get, task_update tools
- RichTaskHook for real-time console output of task activity
- TaskHook auto-reminder injection (every N LLM calls)
- Status bar task progress in TUI
- /tasks slash command in TUI
- Dual mode: automatic (3 multi-step questions) and interactive (free chat)

Run:
    python examples/26_task_management.py                  # automatic mode
    python examples/26_task_management.py -m interactive   # interactive mode
"""

from __future__ import annotations


import click
from rich.panel import Panel
from rich.table import Table

from conftest import (
    console,
    ensure_path,
    load_env,
    setup_example,
    dual_mode_options,
    create_agent,
    automatic_batch,
    interactive_loop,
)

ensure_path()
load_env()

QUESTIONS = [
    "I need to plan a team lunch for 15 people. "
    "Create tasks for: (1) finding a restaurant, (2) collecting dietary restrictions, "
    "(3) sending invitations, (4) making a reservation. "
    "Mark each task as completed as you finish it.",
    "Help me write a Python script that reads a CSV, filters rows by date, "
    "and outputs a summary. Create tasks for each step and mark them done as you go.",
    # Dependency demo: tasks with ordering constraints
    "I want to deploy a web app. Create these tasks WITH dependencies:\n"
    "  1. Set up database (no dependencies)\n"
    "  2. Write API endpoints (depends on #1)\n"
    "  3. Build frontend (depends on #2)\n"
    "  4. Run tests (depends on #2 and #3)\n"
    "  5. Deploy to production (depends on #4)\n"
    "Use blocked_by to set up the dependencies. "
    "Then work through them in order, completing each one.",
]


def _show_task_tools_info() -> None:
    """Explain the task management tools."""
    console.print(
        Panel(
            "[bold]Task Management Tools[/bold]\n\n"
            "  [cyan]task_create[/cyan]         -- Create a new task with subject, description, and optional blocked_by\n"
            "  [cyan]task_list[/cyan]           -- List all tasks, optionally filtered by status\n"
            "  [cyan]task_get[/cyan]            -- Get details of a specific task by ID\n"
            "  [cyan]task_update[/cyan]         -- Update task status (pending -> in_progress -> completed)\n"
            "  [cyan]task_add_dependency[/cyan] -- Add a dependency between tasks\n\n"
            "[bold]Dependency Support:[/bold]\n"
            "  Tasks can depend on other tasks via [cyan]blocked_by[/cyan].\n"
            "  A blocked task cannot start until all dependencies are completed.\n"
            "  Completing a task auto-unblocks its dependents.\n\n"
            "[bold]What you'll see below:[/bold]\n"
            "  1. [yellow]+[/yellow] Task created     -- when the agent creates a task\n"
            "  2. [yellow]>[/yellow] Task started      -- when the agent marks a task in_progress\n"
            "  3. [green]✓[/green] Task completed    -- when the agent marks a task done\n"
            "  4. [red]⊘[/red] Task blocked      -- when a task can't start (deps not met)\n"
            "  5. [magenta]⤳[/magenta] Dependency added -- when a dependency is declared\n"
            "  6. A summary table after each question showing all task states\n\n"
            "[bold]Auto-Reminder Hook[/bold]\n"
            "  Injects a reminder every 3rd LLM call when tasks are active.\n"
            "  Keeps the agent on track without manual prompting.\n\n"
            "[dim]Config: tools.builtin includes task tools, harness.tasks.enabled = true[/dim]",
            title="Task Management",
        )
    )


# Set in main() once the agent is built; the helpers below read it (the
# TaskManager lives on the agent's tool registry, not as a process global).
_task_mgr = None


def _show_task_state(label: str = "Current Tasks") -> None:
    """Display current task state from the TaskManager."""
    mgr = _task_mgr
    if not mgr:
        console.print("[dim]TaskManager not initialized.[/dim]")
        return

    tasks = mgr.list_tasks()
    if not tasks:
        console.print("[dim]No tasks.[/dim]")
        return

    table = Table(title=label, show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Subject")
    for t in tasks:
        status_color = {"pending": "yellow", "in_progress": "blue", "completed": "green", "blocked": "red"}.get(
            t.status, "white"
        )
        status_icon = {"pending": "○", "in_progress": "►", "completed": "✓", "blocked": "⊘"}.get(t.status, "?")
        dep_info = f" [blocked by: {', '.join(t.blocked_by)}]" if t.blocked_by else ""
        table.add_row(t.id, f"[{status_color}]{status_icon} {t.status}[/{status_color}]", f"{t.subject}{dep_info}")
    console.print(table)


def _clear_tasks() -> None:
    """Clear all tasks between questions."""
    if _task_mgr:
        _task_mgr.clear()


def _post_answer_hook(result, q, i, total) -> None:
    """Show task state after each answer, then clear for next question."""
    console.print()
    _show_task_state(f"Tasks after Question {i}")
    _clear_tasks()
    console.print()


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 26: Task Management."""
    setup_example(
        "Example 26: Task Management",
        "Track multi-step work with task_create, task_list, task_update.\n\n"
        "[dim]Run with -m interactive for chat mode with task tracking.[/dim]",
    )

    _show_task_tools_info()
    console.print()

    agent = create_agent("26_task_management", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]")

    # Expose the TaskManager to the module-level helper functions below.
    global _task_mgr
    _task_mgr = agent.core.tools.get_dep("task_manager") if agent.core else None

    # Register RichTaskHook so task activity is visible in console
    from koboi.hooks.rich_task_hook import RichTaskHook

    if agent.core is not None:
        agent.core.hooks.add(RichTaskHook(console=console))
        console.print("[dim]RichTaskHook registered -- task activity will be printed below.[/dim]")
    console.print()

    if mode == "interactive":
        interactive_loop(agent)
    else:
        automatic_batch(agent, QUESTIONS, post_answer=_post_answer_hook)
        console.print()
        _show_task_state("Final Task State")


if __name__ == "__main__":
    main()
