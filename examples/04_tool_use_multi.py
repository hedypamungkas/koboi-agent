"""Example 04: Multi-tool use.

Demonstrates:
- Agent with 3 tools: calculate, memory_store, memory_recall
- Dual mode: automatic (batch) and interactive (free chat)
- Commands in interactive: quit, tools

Run:
    python examples/03_tool_use_multi.py                  # automatic mode
    python examples/03_tool_use_multi.py -m interactive   # interactive mode
"""

from __future__ import annotations

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

QUESTIONS = [
    "Calculate 25 * 4 and store the result as 'product_25x4'",
    "What is the square root of 144? Save it as 'sqrt_144'",
    "What did you store earlier? Recall all values.",
    "Calculate the total of all stored values.",
    "What is 17% of 350?",
]


def _show_tools(agent) -> None:
    """Display list of registered tools."""
    tools = agent.core.tools
    table = Table(title="Registered Tools", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Description", max_width=60)
    table.add_column("Parameters", max_width=40)

    for name, td in tools._tools.items():
        params = td.parameters.get("properties", {})
        param_str = ", ".join(params.keys()) if params else "(none)"
        table.add_row(name, td.description, param_str)

    console.print(table)


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 04: Multi-tool chat."""
    setup_example(
        "Example 04: Multi-Tool Chat",
        "Agent with calculator, memory_store, and memory_recall.\n"
        "Acme Corp sales assistant.\n\n"
        "[dim]Interactive commands: quit, tools[/dim]",
    )

    agent = create_agent("04_tool_use_multi", verbose=verbose)

    tools = agent.core.tools
    tool_names = list(tools._tools.keys()) if tools._tools else []
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]")
    console.print(f"[dim]Tools: {', '.join(tool_names) if tool_names else '(none)'}[/dim]\n")

    if mode == "interactive":
        interactive_loop(
            agent,
            extra_commands={"tools": _show_tools},
        )
    else:
        automatic_batch(agent, QUESTIONS)


if __name__ == "__main__":
    main()
