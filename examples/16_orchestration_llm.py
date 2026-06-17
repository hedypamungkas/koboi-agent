"""Example 16: Orchestration - Router Comparison.

Demonstrates a comparison of 3 routers:
- KeywordRouter: routing based on keyword matching
- LLMRouter: routing using LLM with confidence scores
- HybridRouter: keyword first, LLM fallback

Dual mode: automatic (compare all routers) and interactive (pick router, chat freely)

Run:
    python examples/15_orchestration_llm.py                  # automatic mode
    python examples/15_orchestration_llm.py -m interactive   # interactive mode
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

TEST_QUERIES = [
    ("How much leave do I have left?", "hr"),
    ("Which package is suitable for a startup?", "sales"),
    ("What is the cybersecurity incident procedure?", "dynamic/new domain"),
    ("What is the salary raise procedure and enterprise package for a team of 50?", "hr + sales (multi-domain)"),
]


def _build_router(router_name: str):
    """Create a router instance by name."""
    from koboi.client import Client
    from koboi.orchestration.router import KeywordRouter, LLMRouter, HybridRouter

    client = Client(provider="anthropic")
    if router_name == "KeywordRouter":
        return KeywordRouter()
    elif router_name == "LLMRouter":
        return LLMRouter(client=client, enable_dynamic=True)
    elif router_name == "HybridRouter":
        return HybridRouter(client=client, confidence_threshold=0.5, enable_dynamic=True)
    return KeywordRouter()


def _run_router_automatic(router_name: str, router, client, logger):
    """Run all test queries with a single router."""
    from koboi.orchestration.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        client=client,
        router=router,
        logger=logger,
        enable_dynamic=(router_name in ("LLMRouter", "HybridRouter")),
    )

    console.rule(f"[bold]{router_name}[/bold]")

    route_table = Table(show_header=True, header_style="bold magenta")
    route_table.add_column("Query", style="green", max_width=45)
    route_table.add_column("Expected", style="dim", max_width=20)
    route_table.add_column("Routed To", style="cyan", max_width=25)
    route_table.add_column("Confidence", style="yellow")
    route_table.add_column("Method", style="dim")

    results_summary = []
    for query, expected in TEST_QUERIES:
        try:
            decision = run_async(router.route(query))
            route_table.add_row(
                query[:45], expected[:20], ", ".join(decision.agents), f"{decision.confidence:.2f}", decision.method
            )
            results_summary.append((query, expected, decision))
        except Exception as e:
            route_table.add_row(query[:45], expected[:20], f"[red]Error: {e}[/red]", "-", "-")

    console.print(route_table)

    # Execute first 3 queries
    console.print(f"\n[bold]Executing queries with {router_name}:[/bold]\n")
    for query, expected in TEST_QUERIES[:3]:
        try:
            result = run_async(orchestrator.run(query, mode="sequential"))
            console.print(
                f"  [green]{query[:50]}[/green]\n"
                f"    -> Routed: [cyan]{result.routing.agents}[/cyan] "
                f"| Confidence: {result.routing.confidence:.2f} "
                f"| Time: {result.total_elapsed_seconds:.2f}s"
            )
            answer_preview = result.final_answer[:200]
            if len(result.final_answer) > 200:
                answer_preview += "..."
            console.print(f"    Answer: {answer_preview}\n")
        except Exception as e:
            console.print(f"  [red]Error for '{query[:40]}...': {e}[/red]\n")

    return results_summary


def _compare_routers(results_map: dict):
    """Display routing decision comparison across all routers."""
    console.rule("[bold]Routing Decisions Comparison[/bold]")

    comp_table = Table(show_header=True, header_style="bold magenta")
    comp_table.add_column("Query", style="green", max_width=40)
    comp_table.add_column("Expected", style="dim", max_width=15)

    for name in results_map:
        comp_table.add_column(name, style="cyan", max_width=20)

    for i in range(len(TEST_QUERIES)):
        query, expected = TEST_QUERIES[i]
        row = [query[:40], expected[:15]]
        for name in results_map:
            results = results_map[name]
            if i < len(results):
                decision = results[i][2]
                row.append(f"{', '.join(decision.agents)}\n({decision.confidence:.2f})")
            else:
                row.append("-")
        comp_table.add_row(*row)

    console.print(comp_table)


def run_automatic():
    """Compare all 3 routers automatically."""
    from koboi.client import Client
    from koboi.orchestration.router import KeywordRouter, LLMRouter, HybridRouter
    from koboi.logger import AgentLogger

    try:
        client = Client()
    except Exception as e:
        console.print(f"[red]Error creating Client: {e}[/red]")
        console.print("[dim]Make sure OPENAI_API_KEY is set in .env[/dim]")
        sys.exit(1)

    logger = AgentLogger(session_id="ex15_router_comparison")
    all_results: dict[str, list] = {}

    for router_name, router_cls in [
        ("KeywordRouter", KeywordRouter),
        ("LLMRouter", LLMRouter),
        ("HybridRouter", HybridRouter),
    ]:
        router = _build_router(router_name)
        results = _run_router_automatic(router_name, router, client, logger)
        all_results[router_name] = results
        console.print()

    if len(all_results) > 1:
        _compare_routers(all_results)

    console.print("\n[bold green]Done![/bold green]")


def run_interactive():
    """Pick a router, then chat freely."""
    from koboi.client import Client
    from koboi.orchestration.orchestrator import Orchestrator
    from koboi.logger import AgentLogger

    console.print("[bold]Available routers:[/bold]")
    console.print("  1. KeywordRouter -- keyword matching")
    console.print("  2. LLMRouter -- LLM-based routing")
    console.print("  3. HybridRouter -- keyword + LLM fallback")

    choice = Prompt.ask("Pick a router number", choices=["1", "2", "3"])
    router_names = ["KeywordRouter", "LLMRouter", "HybridRouter"]
    selected = router_names[int(choice) - 1]

    console.print(f"\n[bold cyan]Using: {selected}[/bold]\n")

    try:
        client = Client()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    router = _build_router(selected)
    logger = AgentLogger(session_id=f"ex15_interactive_{selected}")
    orchestrator = Orchestrator(
        client=client,
        router=router,
        logger=logger,
        enable_dynamic=(selected in ("LLMRouter", "HybridRouter")),
    )

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
    """Example 16: Orchestration - Router Comparison."""
    setup_example(
        "Example 16: Orchestration - Router Comparison",
        "Comparing 3 routing strategies:\n"
        "  1. [cyan]KeywordRouter[/cyan] -- keyword matching\n"
        "  2. [cyan]LLMRouter[/cyan] -- LLM-based routing + confidence\n"
        "  3. [cyan]HybridRouter[/cyan] -- keyword + LLM fallback\n\n"
        "[dim]Run with -m interactive to pick a router and chat freely.[/dim]",
    )

    if mode == "interactive":
        run_interactive()
    else:
        run_automatic()


if __name__ == "__main__":
    main()
