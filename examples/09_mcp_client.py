"""Example 09: MCP Client -- Model Context Protocol demonstration.

Demonstrates:
- Part 1: Direct MCPClient usage (connect, discover, call, disconnect)
- Part 2: Via KoboiAgent facade using YAML config
- Dual mode: automatic (batch queries) and interactive (free chat)

Run:
    python examples/08_mcp_client.py                  # automatic mode
    python examples/08_mcp_client.py -m interactive   # interactive mode
"""
from __future__ import annotations

import sys

import click
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from conftest import (
    console,
    setup_example,
    dual_mode_options,
    create_agent,
    automatic_batch,
    interactive_loop,
    run_async,
)

QUERIES = [
    "What time is it now?",
    "How many days until 2026-12-31?",
    "What day of the week is Christmas 2026?",
]


def run_mcp_direct_demo():
    """Part 1: Direct MCPClient usage."""
    console.print("\n[bold cyan]=== Part 1: Direct MCPClient ===[/bold cyan]")

    from koboi.mcp.client import MCPClient

    server_cmd = [sys.executable, "examples/data/mcp_servers/time_server.py"]
    console.print(f"[dim]Connecting to MCP server: {' '.join(server_cmd)}[/dim]")

    client = MCPClient(server_command=server_cmd)

    try:
        server_info = client.connect()
        console.print(f"[green]Connected![/green]")
        console.print(f"  Server: {server_info.get('serverInfo', {}).get('name', 'unknown')}")
        console.print(f"  Version: {server_info.get('serverInfo', {}).get('version', '?')}")

        # Discover tools
        console.print("\n[bold]Discovering tools...[/bold]")
        tools = client.discover_tools()

        tool_table = Table(title="Discovered MCP Tools")
        tool_table.add_column("Name", style="cyan")
        tool_table.add_column("Description", style="white")
        tool_table.add_column("Parameters", style="dim")

        for tool_info in tools:
            params = tool_info.input_schema.get("properties", {})
            param_str = ", ".join(f"{k}({v.get('type', '?')})" for k, v in params.items())
            tool_table.add_row(tool_info.name, tool_info.description, param_str)

        console.print(tool_table)

        # Call tools
        console.print("\n[bold]Calling get_current_time...[/bold]")
        result = run_async(client.call_tool("get_current_time", {}))
        console.print(f"  Result: {result}")

        timezone_tools = [t for t in tools if "timezone" in t.name]
        if timezone_tools:
            console.print(f"\n[bold]Calling {timezone_tools[0].name}...[/bold]")
            result_tz = run_async(client.call_tool(timezone_tools[0].name, {"timezone": "Asia/Jakarta"}))
            console.print(f"  Result: {result_tz}")

        client.close()
        console.print("[green]Disconnected from MCP server.[/green]")
        return True

    except Exception as e:
        console.print(f"[red]Part 1 failed: {e}[/red]")
        client.close()
        console.print("[yellow]Skipping Part 2 because MCP server is not available.[/yellow]")
        return False


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 09: MCP Client."""
    setup_example(
        "Example 09: MCP Client",
        "Part 1: Direct MCPClient usage\n"
        "Part 2: Via KoboiAgent facade (YAML config)\n\n"
        "[dim]Run with -m interactive for chat mode.[/dim]",
    )

    # Part 1 always runs
    mcp_ok = run_mcp_direct_demo()

    if not mcp_ok:
        return

    # Part 2: Via agent
    console.print("\n[bold cyan]=== Part 2: Via KoboiAgent Facade ===[/bold cyan]")

    agent = create_agent("09_mcp_client", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]\n")

    if mode == "interactive":
        interactive_loop(agent)
    else:
        for query in QUERIES:
            console.print(f"[bold green]User:[/bold green] {query}")
            try:
                result = agent.run_sync(query)
                console.print(Panel(str(result), title="Agent"))
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            agent.reset()
            console.print()


if __name__ == "__main__":
    main()
