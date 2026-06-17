"""Example 01: Simple chat without tools.

Demonstrates:
- KoboiAgent.from_config() facade pattern
- Dual mode: automatic (batch) and interactive (free chat)

Run:
    python examples/01_simple_chat.py                  # automatic mode
    python examples/01_simple_chat.py -m interactive   # interactive mode
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
)

QUESTIONS = [
    "What can you help me with?",
    "Explain what an AI agent is in simple terms.",
    "Tell me a short fun fact about technology.",
]


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 01: Simple chat without tools."""
    setup_example(
        "Example 01: Simple Chat",
        "Basic chat without tools. Agent uses LLM only.\n\n[dim]Run with -m interactive for chat mode.[/dim]",
    )

    agent = create_agent("01_simple_chat", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]\n")

    if mode == "interactive":
        interactive_loop(agent)
    else:
        automatic_batch(agent, QUESTIONS)


if __name__ == "__main__":
    main()
