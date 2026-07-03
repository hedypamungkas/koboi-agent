"""Example 15: Orchestration with KeywordRouter.

Demonstrates:
- KeywordRouter: routing based on keyword mapping
- Orchestrator: multi-agent coordination (hr, sales, finance)
- Dual mode: automatic (4 queries) and interactive (free chat with routing)

Run:
    python examples/14_orchestration_keyword.py                  # automatic mode
    python examples/14_orchestration_keyword.py -m interactive   # interactive mode
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


QUERIES = [
    "How much annual leave do I have?",
    "How much does the Enterprise package cost?",
    "When is the invoice due?",
    "Leave and package for a team of 10",
]

DOMAIN_KEYWORDS = {
    "hr": ["leave", "vacation", "salary", "employee", "working hours", "remote", "time off"],
    "sales": [
        "price",
        "product",
        "package",
        "promo",
        "discount",
        "buy",
        "order",
        "enterprise",
        "cost",
        "pricing",
        "subscription",
    ],
    "finance": ["invoice", "payment", "billing", "due date", "refund"],
}


def _build_orchestrator():
    """Create orchestrator with KeywordRouter."""
    from koboi.client import Client
    from koboi.orchestration.orchestrator import Orchestrator
    from koboi.orchestration.router import KeywordRouter
    from koboi.logger import AgentLogger

    client = Client(provider="anthropic")
    logger = AgentLogger(session_id="ex14_keyword_orch")
    router = KeywordRouter()
    router.KEYWORD_MAP = DOMAIN_KEYWORDS
    return Orchestrator(client=client, router=router, logger=logger), router


def run_automatic():
    """Run 4 queries automatically."""
    try:
        orchestrator, router = _build_orchestrator()
    except Exception as e:
        console.print(f"[red]Error creating Client: {e}[/red]")
        console.print("[dim]Make sure OPENAI_API_KEY is set in .env[/dim]")
        sys.exit(1)

    # Routing table
    console.print("\n[bold]Routing Decisions:[/bold]")
    route_table = Table(show_header=True, header_style="bold magenta")
    route_table.add_column("Query", style="green")
    route_table.add_column("Domain", style="cyan")
    route_table.add_column("Confidence", style="yellow")
    route_table.add_column("Method", style="dim")

    for q in QUERIES:
        decision = run_async(router.route(q))
        route_table.add_row(q[:50], ", ".join(decision.agents), f"{decision.confidence:.2f}", decision.method)
    console.print(route_table)

    # Execute queries
    console.print("\n[bold]Executing Queries:[/bold]\n")
    for i, q in enumerate(QUERIES, 1):
        console.rule(f"[bold]Query {i}: {q}[/bold]")
        try:
            result = run_async(orchestrator.run(q, mode="sequential"))
            console.print(
                f"  Routing: {result.routing.method} -> [cyan]{result.routing.agents}[/cyan] (confidence: {result.routing.confidence:.2f})"
            )
            for ar in result.agent_results:
                console.print(
                    f"  Agent [cyan]{ar.agent_name.upper()}[/cyan]: {ar.elapsed_seconds:.2f}s, {ar.tokens_used} tokens"
                )
            console.print(
                Panel(
                    Markdown(result.final_answer),
                    title=f"Answer (mode: {result.execution_mode}, total: {result.total_elapsed_seconds:.2f}s)",
                )
            )
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
        console.print()


def run_interactive():
    """Free chat with routing display."""
    orchestrator, router = _build_orchestrator()

    console.print("[dim]Type any question. Routing will be displayed automatically.[/dim]")
    console.print("[dim]Domains: hr, sales, finance[/dim]\n")

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

        # Show routing
        decision = run_async(router.route(user_input))
        console.print(
            f"[dim]Routed: {decision.method} -> [cyan]{', '.join(decision.agents)}[/cyan] (confidence: {decision.confidence:.2f})[/dim]"
        )

        try:
            result = run_async(orchestrator.run(user_input, mode="sequential"))
            console.print(Panel(Markdown(result.final_answer), title="Answer", border_style="green"))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 15: Orchestration with KeywordRouter."""
    setup_example(
        "Example 15: Orchestration - KeywordRouter",
        "Routing based on keyword mapping.\n\n[dim]Run with -m interactive for free chat with routing.[/dim]",
    )

    if mode == "interactive":
        run_interactive()
    else:
        run_automatic()


if __name__ == "__main__":
    main()
