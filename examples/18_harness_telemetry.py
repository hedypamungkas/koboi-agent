"""Example 18: Harness telemetry -- track agent performance metrics.

Demonstrates:
- Agent with telemetry tracking via YAML config
- Batch conversation with 4 turns about Acme Corp products
- Accessing TelemetryCollector after each turn
- Dual mode: automatic (batch with telemetry) and interactive (chat with live metrics)

Run:
    python examples/17_harness_telemetry.py                  # automatic mode
    python examples/17_harness_telemetry.py -m interactive   # interactive mode
"""

from __future__ import annotations

import time

import click
from rich.panel import Panel
from rich.table import Table

from conftest import (
    console,
    setup_example,
    dual_mode_options,
    create_agent,
    automatic_batch,
    interactive_loop,
)

CONVERSATIONS = [
    "What products are available?",
    "How much does AcmeERP cost?",
    "Calculate the total for 20 AcmeCRM users for one year",
    "Compare the Starter and Professional plans",
]


def _find_telemetry_collector(agent):
    """Find the TelemetryCollector from the agent's hook chain."""
    hook = agent.core.hooks.find_hook(lambda h: hasattr(h, "telemetry") and hasattr(h.telemetry, "snapshot"))
    return hook.telemetry if hook else None
    return None


def _print_telemetry(collector, turn: int):
    """Print a telemetry snapshot as a Rich table."""
    snap = collector.snapshot

    table = Table(title=f"Telemetry -- Turn {turn}", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=28)
    table.add_column("Value", style="green")

    table.add_row("Total Iterations", str(snap.total_iterations))
    table.add_row("Total Tool Calls", str(snap.total_tool_calls))
    table.add_row("Unique Tools", ", ".join(sorted(snap.unique_tools_used)) or "-")
    table.add_row("Tools Succeeded", str(snap.tools_succeeded))
    table.add_row("Tools Failed", str(snap.tools_failed))
    table.add_row("Doom Loops Detected", str(snap.doom_loops_detected))
    table.add_row("Health Score", f"{collector.health_score()}/100")
    table.add_row("Loop Health", f"{collector.loop_health():.2f}")
    table.add_row("Tool Success Rate", f"{collector.tool_success_rate():.2f}")
    table.add_row("Context Efficiency", f"{collector.context_efficiency():.2f}")

    console.print(table)
    console.print()


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 18: Harness telemetry tracking."""
    setup_example(
        "Example 18: Harness Telemetry",
        "Running a batch conversation and displaying telemetry after each turn.\n\n"
        "[dim]Run with -m interactive for chat with live metrics.[/dim]",
    )

    agent = create_agent("18_harness_telemetry", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]\n")

    telemetry = _find_telemetry_collector(agent)
    if telemetry is None:
        console.print("[yellow]Warning: TelemetryHook not found in hook chain.[/yellow]")
        console.print("[dim]Make sure config has harness.telemetry: true[/dim]\n")

    if mode == "interactive":

        def _post_receive(result, a):
            t = _find_telemetry_collector(a)
            if t:
                _print_telemetry(t, a.core.memory.get_messages().__len__() // 2)

        interactive_loop(agent, post_receive=_post_receive)
    else:
        overall_start = time.time()

        def _post_answer(result, q, i, total):
            if telemetry is not None:
                _print_telemetry(telemetry, i)

        automatic_batch(agent, CONVERSATIONS, post_answer=_post_answer)

        # Final summary
        overall_duration = time.time() - overall_start
        console.print(
            Panel(
                f"[bold]Session Summary[/bold]\n\n"
                f"Total Turns: {len(CONVERSATIONS)}\n"
                f"Total Duration: {overall_duration:.1f}s\n"
                + (f"Health Score: {telemetry.health_score()}/100\n" if telemetry else "")
                + (f"Total Tool Calls: {telemetry.snapshot.total_tool_calls}\n" if telemetry else "")
                + (f"Doom Loops: {telemetry.snapshot.doom_loops_detected}" if telemetry else ""),
                title="Final Report",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
