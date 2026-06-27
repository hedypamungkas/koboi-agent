"""Example 21: Eval suite -- evaluate agent quality with automatic scoring.

Demonstrates:
- EvalRunner with custom scorers
- 5 EvalCases: pricing, calculation, recommendation, consultation, SaaS plans
- KeywordPresenceScorer, ToolUsageScorer, OutputLengthScorer, IterationEfficiencyScorer
- Dual mode: automatic (full eval suite) and interactive (type queries, see scoring)

Run:
    python examples/21_eval_suite.py                  # automatic mode
    python examples/21_eval_suite.py -m interactive   # interactive mode
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
    interactive_loop,
    run_async,
)


def _build_cases():
    """Build the 5 evaluation test cases."""
    from koboi.types import EvalCase

    return [
        EvalCase(
            name="Product pricing",
            user_message="How much does AcmeERP Enterprise cost?",
            expected_keywords=["15000", "perpetual"],
            expected_tools=[],
            max_iterations=10,
        ),
        EvalCase(
            name="Calculate total",
            user_message="What is the total for 15 AcmeCRM users for one year?",
            expected_keywords=["4500"],
            expected_tools=["calculate"],
            max_iterations=10,
        ),
        EvalCase(
            name="Recommendation",
            user_message="Which product is suitable for a retail business with 5 outlets?",
            expected_keywords=["AcmePOS"],
            expected_tools=[],
            max_iterations=10,
        ),
        EvalCase(
            name="Consultation",
            user_message="How much does 2 days of IT consultation cost?",
            expected_keywords=["200", "consultation"],
            expected_tools=["calculate"],
            max_iterations=10,
        ),
        EvalCase(
            name="SaaS plans",
            user_message="Explain the available SaaS plans",
            expected_keywords=["Starter", "Professional", "Enterprise"],
            expected_tools=[],
            max_iterations=10,
        ),
    ]


def _print_results(results):
    """Print eval results as a Rich table."""
    table = Table(title="Evaluation Results", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Case", style="cyan", width=20)
    table.add_column("Status", width=8)
    table.add_column("Score", style="green", width=8)
    table.add_column("Duration", width=10)
    table.add_column("Details", style="dim", ratio=1)

    for i, r in enumerate(results, 1):
        status = "[green]PASS[/green]" if r.overall_score >= 0.6 else "[red]FAIL[/red]"
        score_str = f"{r.overall_score:.1%}"
        duration_str = f"{r.duration_seconds:.1f}s"
        details_parts = [f"{s.name}={s.value:.2f}" for s in r.scores]
        details = ", ".join(details_parts)
        table.add_row(str(i), r.case_name, status, score_str, duration_str, details)

    console.print(table)

    # Per-case detail
    console.print()
    for r in results:
        status = "PASS" if r.overall_score >= 0.6 else "FAIL"

        detail_table = Table(show_header=True, header_style="bold", title=f"[{status}] {r.case_name}")
        detail_table.add_column("Scorer", style="cyan", width=25)
        detail_table.add_column("Score", width=8)
        detail_table.add_column("Reason", style="dim", ratio=1)

        for s in r.scores:
            bar_len = 10
            filled = int(s.value * bar_len)
            bar = "+" * filled + "-" * (bar_len - filled)
            detail_table.add_row(s.name, f"[{bar}] {s.value:.2f}", s.reason)

        console.print(detail_table)
        console.print()


def run_automatic(verbose: bool):
    """Run full eval suite automatically."""
    from koboi.eval.runner import EvalRunner
    from koboi.eval.scorers import ToolUsageScorer, KeywordPresenceScorer, OutputLengthScorer, IterationEfficiencyScorer

    agent = create_agent("21_eval_suite", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]\n")

    cases = _build_cases()

    # Show test cases
    case_table = Table(title="Test Cases", show_header=True, header_style="bold cyan")
    case_table.add_column("#", width=3)
    case_table.add_column("Name", style="cyan", width=20)
    case_table.add_column("Query", ratio=1)
    case_table.add_column("Keywords", style="yellow", width=30)
    case_table.add_column("Tools", style="magenta", width=15)

    for i, c in enumerate(cases, 1):
        case_table.add_row(
            str(i), c.name, c.user_message, ", ".join(c.expected_keywords), ", ".join(c.expected_tools) or "-"
        )

    console.print(case_table)
    console.print()

    scorers = [KeywordPresenceScorer(), ToolUsageScorer(), OutputLengthScorer(), IterationEfficiencyScorer()]
    runner = EvalRunner(
        harness_factory=lambda: create_agent("21_eval_suite", verbose=verbose),
        scorers=scorers,
    )

    console.print("[bold]Running evaluation...[/bold]\n")
    results = run_async(runner.run_suite(cases))

    console.print()
    _print_results(results)

    total = len(results)
    passed = sum(1 for r in results if r.overall_score >= 0.6)
    avg_score = sum(r.overall_score for r in results) / total if total else 0
    total_duration = sum(r.duration_seconds for r in results)

    console.print(
        Panel(
            f"[bold]Summary[/bold]\n\n"
            f"Passed: {passed}/{total}\n"
            f"Average Score: {avg_score:.1%}\n"
            f"Total Duration: {total_duration:.1f}s",
            title="Evaluation Summary",
            border_style="green" if passed == total else "yellow",
        )
    )


def run_interactive(verbose: bool):
    """Chat freely and see basic quality metrics per response."""
    agent = create_agent("21_eval_suite", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]")
    console.print("[dim]After each response, a basic quality summary is shown.[/dim]\n")

    turn = 0

    def _post_receive(result, a):
        nonlocal turn
        turn += 1
        # Basic metrics
        msg_count = len(a.core.memory.get_messages())
        console.print(f"  [dim]Turn {turn}: {len(str(result))} chars, {msg_count} messages in memory[/dim]\n")

    interactive_loop(agent, post_receive=_post_receive)


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 21: Eval suite for agent quality evaluation."""
    setup_example(
        "Example 21: Evaluation Suite",
        "Running 5 test cases and scoring agent output.\n\n"
        "[dim]Run with -m interactive for chat mode with quality metrics.[/dim]",
    )

    if mode == "interactive":
        run_interactive(verbose)
    else:
        run_automatic(verbose)


if __name__ == "__main__":
    main()
