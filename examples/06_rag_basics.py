"""Example 06: RAG basics.

Demonstrates:
- Agent with RAG augmentation (on_the_fly)
- Document Q&A using company_policy.md and product_catalog.md
- Dual mode: automatic (3 demo questions) and interactive (free chat)

Run:
    python examples/05_rag_basics.py                  # automatic mode
    python examples/05_rag_basics.py -m interactive   # interactive mode
"""
from __future__ import annotations

import click
from rich.markdown import Markdown
from rich.panel import Panel

from conftest import (
    console,
    setup_example,
    dual_mode_options,
    create_agent,
    automatic_batch,
    interactive_loop,
    run_async,
)

QUESTIONS = [
    "How many annual leave days do permanent employees get?",
    "What SaaS packages are available?",
    "What is the remote work policy?",
]


def _show_retrieved_context(augmentation, question: str):
    """Show retrieved context panel for a question."""
    if not augmentation:
        return
    try:
        context, _ = run_async(augmentation._retrieve_and_format(question))
        if context:
            console.print(Panel(
                context[:500] + ("..." if len(context) > 500 else ""),
                title="[dim]Retrieved Context[/dim]",
                border_style="dim",
                style="dim",
            ))
    except Exception:
        pass


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 06: RAG basics."""
    setup_example(
        "Example 06: RAG Basics",
        "Agent with RAG (Retrieval-Augmented Generation).\n"
        "Documents: company_policy.md, product_catalog.md\n\n"
        "[dim]Run with -m interactive for chat mode.[/dim]",
    )

    agent = create_agent("06_rag_basics", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]")

    augmentation = agent.core.augmentation
    if augmentation:
        console.print(f"[dim]RAG: enabled | Retriever: {type(augmentation.retriever).__name__} | Top-K: {augmentation.top_k}[/dim]\n")
    else:
        console.print("[dim]RAG: not loaded (documents not found?)[/dim]\n")

    if mode == "interactive":
        interactive_loop(
            agent,
            pre_send=lambda q: _show_retrieved_context(augmentation, q),
        )
    else:
        automatic_batch(
            agent, QUESTIONS,
            pre_question=lambda q, i, total: _show_retrieved_context(
                augmentation, q if isinstance(q, str) else q.get("input", "")
            ),
        )


if __name__ == "__main__":
    main()
