"""Example 25: Subagent Delegation.

Demonstrates:
- delegate_tasks tool for parallel subagent execution
- Subagent lifecycle: timeout, cancel, list running
- Context sharing between parent and child agents
- YAML config for subagent limits (timeout, max_iterations)
- RichSubagentHook for console progress output
- Dual mode: automatic (3 multi-part questions) and interactive (free chat)

Run:
    python examples/25_subagent_delegation.py                  # automatic mode
    python examples/25_subagent_delegation.py -m interactive   # interactive mode
"""

from __future__ import annotations


import click
from rich.markdown import Markdown
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
    "Compare the pros and cons of Python vs Go for building web APIs. "
    "You MUST use the delegate_tasks tool to analyze performance, ecosystem, "
    "and developer experience as 3 separate parallel subtasks.",
    "I need a market analysis for a new coffee shop. "
    "You MUST use the delegate_tasks tool to research target demographics, "
    "pricing strategy, and location factors as 3 separate parallel subtasks.",
    "Help me plan a 2-week trip to Japan. "
    "You MUST use the delegate_tasks tool to research the best season to visit, "
    "must-see cities, and budget estimate as 3 separate parallel subtasks.",
]


def _show_subagent_config(agent) -> None:
    """Display subagent configuration."""
    sub_conf = agent.config.subagent
    if not sub_conf:
        console.print("[dim]No subagent config in YAML (using defaults).[/dim]")
        return

    table = Table(title="Subagent Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    for key, val in sub_conf.items():
        table.add_row(key, str(val))
    console.print(table)
    console.print()


def _show_lifecycle_info() -> None:
    """Explain the lifecycle management features."""
    console.print(
        Panel(
            "[bold]Subagent Lifecycle Management[/bold]\n\n"
            "Subagents are managed automatically:\n"
            "  - [cyan]Timeout[/cyan]: each subagent has a configurable time limit\n"
            "  - [cyan]Auto-cleanup[/cyan]: resources freed after completion, timeout, or cancel\n"
            "  - [cyan]Cancel[/cyan]: use /kill or Ctrl+K in TUI to cancel running subagents\n"
            "  - [cyan]Status[/cyan]: use /subagents in TUI to list running subagents\n\n"
            "[dim]Config: subagent.timeout, subagent.max_iterations in YAML[/dim]",
            title="Lifecycle",
        )
    )


def _run_with_monitoring_demo() -> None:
    """Show how to monitor subagents programmatically."""
    from koboi.tools.builtin.subagent import get_manager

    console.print("\n[bold cyan]Programmatic Subagent Control[/bold cyan]\n")

    agent = create_agent("25_subagent_delegation")

    # Show that manager is wired up
    manager = get_manager()
    if manager:
        console.print("  SubAgentManager: [green]active[/green]")
        console.print(f"  Timeout: [cyan]{manager.timeout}s[/cyan]")
        console.print(f"  Max iterations: [cyan]{manager.max_iterations}[/cyan]")
    else:
        console.print("  [yellow]SubAgentManager not initialized[/yellow]")
        return

    # Run a task that exercises subagents
    console.print("\n  Running a query that triggers parallel subagents...\n")
    result = agent.run_sync(
        "What are 3 benefits of exercise? "
        "You MUST use the delegate_tasks tool to analyze physical, mental, "
        "and social benefits as 3 separate parallel subtasks."
    )
    console.print(Panel(Markdown(str(result)), title="Result", border_style="green"))

    # Show running state after completion
    running = manager.list_running()
    console.print(f"\n  Running subagents after completion: [cyan]{len(running)}[/cyan]")


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 25: Subagent Delegation."""
    setup_example(
        "Example 25: Subagent Delegation",
        "Parallel task execution via delegate_tasks tool.\n\n"
        "[dim]Run with -m interactive for chat mode with subagent support.[/dim]",
    )

    _show_lifecycle_info()
    console.print()

    agent = create_agent("25_subagent_delegation", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]")
    _show_subagent_config(agent)

    # Register RichSubagentHook so subagent activity is visible in console
    from koboi.hooks.rich_subagent_hook import RichSubagentHook

    if agent.core is not None:
        agent.core.hooks.add(RichSubagentHook(console=console))
        console.print("[dim]RichSubagentHook registered -- subagent activity will be printed below.[/dim]\n")

    if mode == "interactive":
        interactive_loop(agent)
    else:
        automatic_batch(agent, QUESTIONS)


if __name__ == "__main__":
    main()
