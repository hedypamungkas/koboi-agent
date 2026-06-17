"""Example 20: Carryover state -- metadata that persists across context compaction.

Demonstrates:
- CarryoverState: goals, artifacts, tool usage tracking
- Serialize / deserialize via to_context_message() / from_context_message()
- Round-trip verification
- Dual mode: automatic (full demo) and interactive (build state step by step)

Run:
    python examples/19_carryover_state.py                  # automatic mode
    python examples/19_carryover_state.py -m interactive   # interactive mode
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


def run_automatic():
    """Full carryover state demo automatically."""
    from koboi.harness.carryover import CarryoverState

    # Build state
    console.print("\n[bold cyan]1. Building CarryoverState[/bold cyan]\n")

    state = CarryoverState()
    state.add_goal("Find AcmeERP product information")
    state.add_goal("Calculate pricing for 10 users")
    console.print(f"  Goals added: {state.user_goals}")

    state.record_tool_use(tool_name="web_search", arguments='{"query": "AcmeERP"}', result="Found: AcmeERP Enterprise - $15,000", iteration=1)
    state.record_tool_use(tool_name="calculate", arguments='{"expression": "15000 * 10"}', result="150000", iteration=2)
    console.print(f"  Tool calls recorded: {dict(state.invoked_tools)}")

    state.add_artifact("AcmeERP pricing", "Price: $15,000/year, 10 users = $150,000")
    console.print(f"  Artifact added: {list(state.active_artifacts.keys())}")

    state.complete_goal("Find AcmeERP product information")
    console.print(f"  Goal completed: completed={len(state.completed_goals)}, remaining={len(state.user_goals)}")

    state.mark_verified("AcmeERP pricing verified from catalog")
    console.print(f"  Verified work: {len(state.verified_work)} item(s)")

    # Serialize
    console.print("\n[bold cyan]2. Serialize to Context Message[/bold cyan]\n")
    context_msg = state.to_context_message()
    console.print(Panel(context_msg, title="Serialized State", border_style="green"))

    summary = state.summary()
    table = Table(title="State Summary", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)

    # Deserialize
    console.print("\n[bold cyan]3. Deserialize (Round-Trip)[/bold cyan]\n")
    state2 = CarryoverState.from_context_message(context_msg)

    console.print(f"  Goals: {len(state2.user_goals)}")
    console.print(f"    -> {state2.user_goals}")
    console.print(f"  Artifacts: {len(state2.active_artifacts)}")
    console.print(f"    -> {list(state2.active_artifacts.keys())}")
    console.print(f"  Tools used: {len(state2.invoked_tools)}")
    console.print(f"    -> {dict(state2.invoked_tools)}")
    console.print(f"  Verified work: {len(state2.verified_work)}")
    console.print(f"    -> {state2.verified_work}")

    # Verify round-trip
    console.print("\n[bold cyan]4. Round-Trip Verification[/bold cyan]\n")
    checks = [
        ("Goals match", len(state2.user_goals) == len(state.user_goals)),
        ("Artifacts match", len(state2.active_artifacts) == len(state.active_artifacts)),
        ("Tool counts match", state2.invoked_tools == state.invoked_tools),
        ("Verified preserved", len(state2.verified_work) > 0),
    ]

    all_pass = True
    for label, passed in checks:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {label}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        console.print("\n[bold green]All round-trip checks passed![/bold green]")
    else:
        console.print("\n[bold yellow]Some checks failed -- inspect serialization.[/bold yellow]")

    console.print(Panel(
        "[bold]Carryover State[/bold] enables agents to preserve important metadata\n"
        "(goals, artifacts, verified work) that is not lost when context is truncated.",
        title="Conclusion", border_style="blue",
    ))


def run_interactive():
    """Build carryover state step by step."""
    from koboi.harness.carryover import CarryoverState

    state = CarryoverState()

    console.print("[dim]Commands:[/dim]")
    console.print("[dim]  goal <text>         - Add a goal[/dim]")
    console.print("[dim]  done <text>        - Complete a goal[/dim]")
    console.print("[dim]  tool <name> <args> - Record a tool call[/dim]")
    console.print("[dim]  artifact <k> <v>   - Add an artifact[/dim]")
    console.print("[dim]  verify <text>      - Mark verified work[/dim]")
    console.print("[dim]  show               - Display current state[/dim]")
    console.print("[dim]  serialize          - Show serialized state[/dim]")
    console.print("[dim]  reset              - Clear state[/dim]")
    console.print("[dim]  quit               - Exit[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]Command[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        stripped = user_input.strip().lower()
        if stripped in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break

        if not user_input.strip():
            continue

        parts = user_input.strip().split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd == "goal" and len(parts) >= 2:
            text = " ".join(parts[1:])
            state.add_goal(text)
            console.print(f"  [green]Goal added:[/green] {text}")
        elif cmd == "done" and len(parts) >= 2:
            text = " ".join(parts[1:])
            state.complete_goal(text)
            console.print(f"  [green]Goal completed:[/green] {text}")
        elif cmd == "tool" and len(parts) >= 2:
            tool_name = parts[1]
            args = parts[2] if len(parts) > 2 else "{}"
            state.record_tool_use(tool_name=tool_name, arguments=args, result="(manual)", iteration=1)
            console.print(f"  [green]Tool recorded:[/green] {tool_name}")
        elif cmd == "artifact" and len(parts) >= 3:
            state.add_artifact(parts[1], parts[2])
            console.print(f"  [green]Artifact added:[/green] {parts[1]}")
        elif cmd == "verify" and len(parts) >= 2:
            text = " ".join(parts[1:])
            state.mark_verified(text)
            console.print(f"  [green]Verified:[/green] {text}")
        elif cmd == "show":
            console.print(f"  Goals: {state.user_goals}")
            console.print(f"  Completed: {state.completed_goals}")
            console.print(f"  Artifacts: {list(state.active_artifacts.keys())}")
            console.print(f"  Tools: {dict(state.invoked_tools)}")
            console.print(f"  Verified: {state.verified_work}")
        elif cmd == "serialize":
            msg = state.to_context_message()
            console.print(Panel(msg, title="Serialized State", border_style="green"))
        elif cmd == "reset":
            state = CarryoverState()
            console.print("[yellow]State cleared.[/yellow]")
        else:
            console.print("[yellow]Unknown command. Type 'goal', 'done', 'tool', 'artifact', 'verify', 'show', 'serialize', 'reset', or 'quit'.[/yellow]")
        console.print()


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 20: Carryover state demonstration."""
    setup_example(
        "Example 20: Carryover State",
        "Metadata that persists across context compaction.\n\n"
        "[dim]Run with -m interactive to build state step by step.[/dim]",
    )

    if mode == "interactive":
        run_interactive()
    else:
        run_automatic()


if __name__ == "__main__":
    main()
