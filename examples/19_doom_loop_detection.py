"""Example 19: Doom loop detection -- simulate detection of repetitive patterns.

Demonstrates:
- DoomLoopDetector with aggressive thresholds
- 3 scenarios: consecutive identical, repeating pattern, error retry
- Dual mode: automatic (3 scenarios) and interactive (type tool calls, see detection live)

Run:
    python examples/18_doom_loop_detection.py                  # automatic mode
    python examples/18_doom_loop_detection.py -m interactive   # interactive mode
"""

from __future__ import annotations

import click
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from conftest import (
    console,
    ensure_path,
    load_env,
    setup_example,
    dual_mode_options,
)

ensure_path()
load_env()


def _escape_rich(text: str) -> str:
    """Escape square brackets for Rich markup rendering."""
    return text.replace("[", "\\[")


def _scenario_1():
    """Scenario 1: Consecutive identical calls."""
    console.print(
        Panel(
            "[bold]Scenario 1: Consecutive Identical Calls[/bold]\n"
            "The same tool with the same arguments is called repeatedly.\n"
            "Threshold: consecutive_identical_threshold=3",
            border_style="red",
        )
    )

    from koboi.harness.doom_loop import DoomLoopConfig, DoomLoopDetector

    config = DoomLoopConfig(
        consecutive_identical_threshold=3,
        repeating_pattern_window=4,
        repeating_pattern_threshold=2,
        error_retry_threshold=3,
    )
    detector = DoomLoopDetector(config)

    tool_name = "web_search"
    arguments = '{"query": "AcmeERP pricing"}'

    for step in range(1, 5):
        detector.record(tool_name, arguments)
        result = detector.check()
        status = "[red]DETECTED[/red]" if result.detected else "[green]OK[/green]"
        console.print(f"  Step {step}: {tool_name}({arguments[:30]}...) -> {status}")
        if result.detected:
            console.print(
                Panel(
                    f"[bold red]Doom Loop Detected![/bold red]\n\n"
                    f"Type: {result.loop_type}\nPattern: {_escape_rich(result.pattern_description)}\n"
                    f"Recovery: {result.recovery_hint}\nWasted Iterations: {result.iterations_wasted}",
                    title="Detection Result",
                    border_style="red",
                )
            )
            break
    console.print()


def _scenario_2():
    """Scenario 2: Repeating pattern (A, B, A, B)."""
    console.print(
        Panel(
            "[bold]Scenario 2: Repeating Pattern (A, B, A, B)[/bold]\n"
            "Two tools alternating repeatedly.\n"
            "Threshold: repeating_pattern_threshold=2, window=6",
            border_style="yellow",
        )
    )

    from koboi.harness.doom_loop import DoomLoopConfig, DoomLoopDetector

    config = DoomLoopConfig(
        consecutive_identical_threshold=3,
        repeating_pattern_window=6,
        repeating_pattern_threshold=2,
        error_retry_threshold=3,
    )
    detector = DoomLoopDetector(config)

    pattern = [("web_search", '{"query": "AcmeERP"}'), ("memory_recall", '{"key": "pricing"}')]
    calls = pattern * 4

    for step, (tool, args) in enumerate(calls, 1):
        detector.record(tool, args)
        result = detector.check()
        status = "[red]DETECTED[/red]" if result.detected else "[green]OK[/green]"
        console.print(f"  Step {step}: {tool} -> {status}")
        if result.detected:
            console.print(
                Panel(
                    f"[bold red]Doom Loop Detected![/bold red]\n\n"
                    f"Type: {result.loop_type}\nPattern: {_escape_rich(result.pattern_description)}\n"
                    f"Recovery: {result.recovery_hint}\nWasted Iterations: {result.iterations_wasted}",
                    title="Detection Result",
                    border_style="red",
                )
            )
            break
    console.print()


def _scenario_3():
    """Scenario 3: Error retry."""
    console.print(
        Panel(
            "[bold]Scenario 3: Error Retry[/bold]\n"
            "The same tool fails repeatedly (is_error=True).\n"
            "Threshold: error_retry_threshold=3",
            border_style="magenta",
        )
    )

    from koboi.harness.doom_loop import DoomLoopConfig, DoomLoopDetector

    config = DoomLoopConfig(
        consecutive_identical_threshold=3,
        repeating_pattern_window=4,
        repeating_pattern_threshold=2,
        error_retry_threshold=3,
    )
    detector = DoomLoopDetector(config)

    tool_name = "run_shell"
    arguments = '{"command": "curl http://api.example.com/data"}'

    for step in range(1, 5):
        detector.record(tool_name, arguments, is_error=True)
        result = detector.check()
        status = "[red]DETECTED[/red]" if result.detected else "[green]OK[/green]"
        console.print(f"  Step {step}: {tool_name} (error=True) -> {status}")
        if result.detected:
            console.print(
                Panel(
                    f"[bold red]Doom Loop Detected![/bold red]\n\n"
                    f"Type: {result.loop_type}\nPattern: {_escape_rich(result.pattern_description)}\n"
                    f"Recovery: {result.recovery_hint}\nWasted Iterations: {result.iterations_wasted}",
                    title="Detection Result",
                    border_style="red",
                )
            )
            break
    console.print()


def run_automatic():
    """Run all 3 scenarios automatically."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Scenario", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Threshold", style="yellow")
    table.add_row("1", "Consecutive identical calls", "3x same tool+args")
    table.add_row("2", "Repeating pattern (A,B,A,B)", "2x pattern in window 6")
    table.add_row("3", "Error retry (same failing call)", "3x error on same call")
    console.print(table)
    console.print()

    _scenario_1()
    _scenario_2()
    _scenario_3()

    console.print(
        Panel(
            "[bold green]All scenarios completed.[/bold green]\n\n"
            "Doom loop detection helps prevent agents from getting stuck\n"
            "in unproductive tool call patterns.",
            title="Summary",
            border_style="green",
        )
    )


def run_interactive():
    """Type tool calls and see doom loop detection in real-time."""
    from koboi.harness.doom_loop import DoomLoopConfig, DoomLoopDetector

    config = DoomLoopConfig(
        consecutive_identical_threshold=3,
        repeating_pattern_window=6,
        repeating_pattern_threshold=2,
        error_retry_threshold=3,
    )
    detector = DoomLoopDetector(config)
    step = 0

    console.print("[dim]Enter tool calls as: tool_name arguments[/dim]")
    console.print("[dim]Prefix with '!' for error calls: ! tool_name arguments[/dim]")
    console.print("[dim]Commands: 'status', 'reset', 'quit'[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]Tool call[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        stripped = user_input.strip().lower()
        if stripped in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break

        if stripped == "reset":
            detector = DoomLoopDetector(config)
            step = 0
            console.print("[yellow]Detector reset.[/yellow]\n")
            continue

        if stripped == "status":
            result = detector.check()
            console.print(
                f"  Detected: {'[red]YES[/red]' if result.detected else '[green]NO[/green]'} | Type: {result.loop_type or '-'}\n"
            )
            continue

        if not user_input.strip():
            continue

        is_error = False
        line = user_input.strip()
        if line.startswith("!"):
            is_error = True
            line = line[1:].strip()

        parts = line.split(maxsplit=1)
        tool_name = parts[0]
        arguments = parts[1] if len(parts) > 1 else "{}"

        step += 1
        detector.record(tool_name, arguments, is_error=is_error)
        result = detector.check()
        status = "[red]DETECTED[/red]" if result.detected else "[green]OK[/green]"
        error_tag = " (error)" if is_error else ""
        console.print(f"  Step {step}: {tool_name}{error_tag} -> {status}")

        if result.detected:
            console.print(
                Panel(
                    f"Type: {result.loop_type}\nPattern: {_escape_rich(result.pattern_description)}\n"
                    f"Recovery: {result.recovery_hint}\nWasted Iterations: {result.iterations_wasted}",
                    title="Doom Loop Detected!",
                    border_style="red",
                )
            )
            # Reset after detection
            detector = DoomLoopDetector(config)
            step = 0
            console.print("[yellow]Detector auto-reset.[/yellow]")
        console.print()


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 19: Doom loop detection simulation."""
    setup_example(
        "Example 19: Doom Loop Detection",
        "Simulating 3 scenarios to detect unproductive tool call patterns.\n\n"
        "[dim]Run with -m interactive to type tool calls and see detection live.[/dim]",
    )

    if mode == "interactive":
        run_interactive()
    else:
        run_automatic()


if __name__ == "__main__":
    main()
