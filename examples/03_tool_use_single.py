"""Example 03: Single tool use.

Demonstrates:
- KoboiAgent with calculate tool only
- Dual mode: automatic (3 math questions) and interactive (free chat)
- Shows tool call flow in memory

Run:
    python examples/02_tool_use_single.py                  # automatic mode
    python examples/02_tool_use_single.py -m interactive   # interactive mode
"""

from __future__ import annotations

import click
from rich.table import Table

from conftest import (
    console,
    setup_example,
    dual_mode_options,
    create_agent,
    automatic_batch,
    interactive_loop,
)

QUESTIONS = [
    "What is the result of 15 * 34 + 127?",
    "Calculate the square root of 1024",
    "What is 999 divided by 37?",
]


def print_memory_summary(agent) -> None:
    """Display memory contents as a Rich table."""
    messages = agent.core.memory.get_messages()

    table = Table(title="Message History", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", style="bold", width=12)
    table.add_column("Content", max_width=80)
    table.add_column("Tool Calls", width=20)

    for idx, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if content and len(content) > 120:
            content = content[:117] + "..."
        tool_calls = msg.get("tool_calls")
        tc_str = ""
        if tool_calls:
            tc_str = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
        table.add_row(str(idx), role, content or "(empty)", tc_str)

    console.print(table)

    tool_call_count = sum(1 for m in messages if m.get("tool_calls"))
    tool_result_count = sum(1 for m in messages if m.get("role") == "tool")
    console.print(
        f"\n[dim]Total messages: {len(messages)} | "
        f"Tool calls: {tool_call_count} | "
        f"Tool results: {tool_result_count}[/dim]"
    )


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 03: Single tool use."""
    setup_example(
        "Example 03: Single Tool Use",
        "Agent with calculate tool. Sends 3 math questions\n"
        "and shows the tool call flow.\n\n"
        "[dim]Run with -m interactive for chat mode.[/dim]",
    )

    agent = create_agent("03_tool_use_single", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]\n")

    if mode == "interactive":
        interactive_loop(agent)
    else:
        automatic_batch(
            agent,
            QUESTIONS,
            final_summary=lambda a, _: print_memory_summary(a),
        )


if __name__ == "__main__":
    main()
