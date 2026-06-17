"""Example 14: Custom Tools.

Demonstrates:
- @tool decorator for registering new tools
- Manual ToolRegistry.register() for custom handlers
- register_decorated() for loading tools from a module
- KoboiAgent.from_config() with builtin + custom tools
- Dual mode: automatic (demo + batch) and interactive (free chat)

Run:
    python examples/14_custom_tools.py                  # automatic mode
    python examples/14_custom_tools.py -m interactive   # interactive mode
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
    run_async,
)

ensure_path()
load_env()

from pathlib import Path

QUESTIONS = [
    "What is the weather in Jakarta?",
    "Translate good morning to Indonesian",
    "What is 100 * 50?",
    "What is the weather in Tokyo?",
    "Translate thank you to Japanese",
]


def _run_tool_registration_demo():
    """Demonstrates 3 ways to register custom tools."""
    console.print(
        Panel(
            "[bold]Part 1: Three Ways to Register Custom Tools[/bold]\n\n"
            "1. [cyan]@tool decorator[/cyan] -- declarative\n"
            "2. [cyan]Manual registry.register()[/cyan] -- flexible\n"
            "3. [cyan]register_decorated() from module[/cyan] -- modular",
            title="Custom Tools",
        )
    )

    from koboi.tools.registry import ToolRegistry, tool, register_decorated
    from koboi.types import RiskLevel

    registry = ToolRegistry()

    # Approach 1: @tool decorator
    console.print("\n[bold cyan]Approach 1: @tool decorator[/bold cyan]")

    @tool(
        name="reverse_text",
        description="Reverse the character order of text",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to reverse"}},
            "required": ["text"],
        },
    )
    def reverse_text(text: str) -> str:
        return text[::-1]

    registry.register(
        name=reverse_text._tool_def.name,
        description=reverse_text._tool_def.description,
        parameters=reverse_text._tool_def.parameters,
        fn=reverse_text,
        risk_level=reverse_text._tool_def.risk_level,
    )
    console.print(f"  Registered: [cyan]reverse_text[/cyan]")
    result = run_async(registry.execute("reverse_text", '{"text": "koboi"}'))
    console.print(f"  Test: reverse_text('koboi') = [green]{result}[/green]")

    # Approach 2: Manual register
    console.print("\n[bold cyan]Approach 2: Manual registry.register()[/bold cyan]")
    registry.register(
        name="uppercase",
        description="Convert text to uppercase",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to convert"}},
            "required": ["text"],
        },
        fn=lambda text: text.upper(),
        risk_level=RiskLevel.SAFE,
    )
    console.print(f"  Registered: [cyan]uppercase[/cyan]")
    result = run_async(registry.execute("uppercase", '{"text": "hello koboi"}'))
    console.print(f"  Test: uppercase('hello koboi') = [green]{result}[/green]")

    # Approach 3: register_decorated from module
    console.print("\n[bold cyan]Approach 3: register_decorated() from module[/bold cyan]")
    from examples.data.custom_tools import weather, translate

    register_decorated(registry, weather)
    console.print(f"  Registered from weather: [cyan]get_weather[/cyan]")
    register_decorated(registry, translate)
    console.print(f"  Registered from translate: [cyan]translate_text[/cyan]")

    result = run_async(registry.execute("get_weather", '{"city": "jakarta"}'))
    console.print(f"  Test: get_weather('jakarta') = [green]{result}[/green]")
    result = run_async(registry.execute("translate_text", '{"text": "good morning", "target_lang": "id"}'))
    console.print(f"  Test: translate_text('good morning', 'id') = [green]{result}[/green]")

    # Summary
    console.print("\n[bold]All registered tools:[/bold]")
    defs = registry.get_definitions()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="white")
    for d in defs:
        table.add_row(d["function"]["name"], d["function"]["description"][:60])
    console.print(table)


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 14: Custom Tools registration."""
    setup_example(
        "Example 14: Custom Tools",
        "Demonstrates 3 ways to register custom tools.\n\n"
        "[dim]Run with -m interactive for chat mode with custom tools.[/dim]",
    )

    # Part 1 always runs
    _run_tool_registration_demo()
    console.print()

    # Part 2: Agent
    agent = create_agent("14_custom_tools", verbose=verbose)

    tool_defs = agent.core.tools.get_definitions()
    if tool_defs:
        tool_names = [t["function"]["name"] for t in tool_defs]
        console.print(f"[dim]Tools loaded: {tool_names}[/dim]\n")
    else:
        console.print("[yellow]Warning: No tools loaded[/yellow]\n")

    if mode == "interactive":
        interactive_loop(agent)
    else:
        automatic_batch(agent, QUESTIONS)


if __name__ == "__main__":
    main()
