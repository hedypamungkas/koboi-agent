"""Example 24: Config-driven Orchestration.

Demonstrates:
- YAML-based multi-agent orchestration (no code changes needed)
- KoboiAgent.from_config() with orchestration: section
- Agent definitions with custom system prompts, tools, and RAG
- Automatic routing via KeywordRouter from config
- Dual mode: automatic (batch) and interactive (free chat)

Run:
    python examples/24_config_driven_orchestration.py                  # automatic mode
    python examples/24_config_driven_orchestration.py -m interactive   # interactive mode
"""

from __future__ import annotations

import sys

import click
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from conftest import (
    console,
    ensure_path,
    load_env,
    setup_example,
    dual_mode_options,
    run_async,
)

ensure_path()
load_env()

from pathlib import Path

from koboi.facade import KoboiAgent

QUERIES = [
    "How much annual leave do I have?",
    "How much does the Enterprise package cost?",
    "What is the company policy on remote work?",
    "Tell me about the Starter package features",
    "What are the working hours?",
]


def create_orchestrated_agent(verbose: bool = False) -> KoboiAgent:
    """Create a KoboiAgent with orchestration from YAML config."""
    config_path = Path(__file__).parent / "24_config_driven_orchestration.yaml"
    try:
        return KoboiAgent.from_config(str(config_path), verbose=verbose)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Make sure OPENAI_API_KEY is set in .env[/dim]")
        sys.exit(1)


def run_automatic(verbose: bool):
    """Run queries automatically with routing display."""
    agent = create_orchestrated_agent(verbose=verbose)
    orch = agent.orchestrator

    if not orch:
        console.print("[red]Error: Orchestration not enabled in config[/red]")
        sys.exit(1)

    # Show agent configuration
    console.print("\n[bold]Configured Agents:[/bold]")
    agent_table = Table(show_header=True, header_style="bold magenta")
    agent_table.add_column("Agent", style="cyan")
    agent_table.add_column("Description")
    agent_table.add_column("Keywords", style="dim")

    for name, ag in orch._agents_map.items():
        # Get keywords from router if available
        keywords = []
        if hasattr(orch.router, "keyword_map"):
            keywords = orch.router.keyword_map.get(name, [])
        kw_str = ", ".join(keywords[:5]) + ("..." if len(keywords) > 5 else "")
        agent_table.add_row(name, getattr(ag, "system_prompt", "")[:50] + "...", kw_str)

    console.print(agent_table)

    # Execute queries
    console.print("\n[bold]Executing Queries:[/bold]\n")
    for i, q in enumerate(QUERIES, 1):
        console.rule(f"[bold]Query {i}: {q}[/bold]")
        try:
            result = run_async(agent.run(q))

            # Show routing info from metadata
            meta = result.metadata or {}
            routing_method = meta.get("routing_method", "unknown")
            routing_confidence = meta.get("routing_confidence", 0)
            agents_used = meta.get("agents_used", [])

            console.print(
                f"  Routing: {routing_method} -> "
                f"[cyan]{', '.join(agents_used)}[/cyan] "
                f"(confidence: {routing_confidence:.2f})"
            )
            console.print(
                f"  Agents: {len(agents_used)} | "
                f"Time: {result.elapsed_seconds:.2f}s | "
                f"Tokens: {meta.get('total_tokens', 'N/A')}"
            )
            console.print(
                Panel(
                    Markdown(result.content),
                    title=f"Answer (mode: {meta.get('execution_mode', 'unknown')})",
                )
            )
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
        console.print()


def run_interactive(verbose: bool):
    """Free chat with routing display."""
    agent = create_orchestrated_agent(verbose=verbose)
    orch = agent.orchestrator

    if not orch:
        console.print("[red]Error: Orchestration not enabled in config[/red]")
        sys.exit(1)

    console.print("[dim]Type any question. Routing will be displayed automatically.[/dim]")
    console.print("[dim]Agents: sales (products/pricing), hr (policies/benefits), general[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break
        if not user_input.strip():
            continue

        try:
            result = run_async(agent.run(user_input))

            # Show routing info
            meta = result.metadata or {}
            routing_method = meta.get("routing_method", "unknown")
            agents_used = meta.get("agents_used", [])

            console.print(f"[dim]Routed: {routing_method} -> [cyan]{', '.join(agents_used)}[/cyan][/dim]")
            console.print(
                Panel(
                    Markdown(result.content),
                    title="Answer",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 24: Config-driven Orchestration."""
    setup_example(
        "Example 24: Config-driven Orchestration",
        "YAML-based multi-agent orchestration with specialist routing.\n\n"
        "[dim]Agents: sales (products/pricing), hr (policies/benefits), general[/dim]\n"
        "[dim]Run with -m interactive for free chat with routing.[/dim]",
    )

    if mode == "interactive":
        run_interactive(verbose)
    else:
        run_automatic(verbose)


if __name__ == "__main__":
    main()
