"""Example 17: Multi-provider LLM -- OpenAI vs Anthropic side by side.

Demonstrates:
- create_client() factory for both providers
- Side-by-side comparison using the same query
- Dual mode: automatic (side-by-side comparison) and interactive (chat with chosen provider)

Run:
    python examples/17_anthropic_provider.py                  # automatic mode (side-by-side)
    python examples/17_anthropic_provider.py -m interactive   # interactive mode
"""
from __future__ import annotations

import os

import click
from rich.console import Console
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
    interactive_loop,
    run_async,
)

ensure_path()
load_env()

from pathlib import Path

QUESTIONS = [
    "Explain what AI is in 2 sentences.",
    "What is the difference between a list and a tuple in Python?",
    "Write a haiku about programming.",
]


def _check_key(provider: str) -> str | None:
    """Return the API key for a provider, or None if missing."""
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
    return os.environ.get("OPENAI_API_KEY", "").strip() or None


def _make_client(provider: str):
    """Create an LLM client for the given provider."""
    from koboi.llm.factory import create_client

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        auth_token = ""

    return create_client(
        provider=provider, api_key=api_key, base_url=base_url,
        model=model, auth_token=auth_token,
    )


def run_side_by_side(query: str):
    """Send the same query to both providers."""
    console.print(Panel(f"[bold]Query:[/bold] {query}", title="Side-by-Side Comparison"))

    results: dict[str, str] = {}
    for provider in ("openai", "anthropic"):
        key = _check_key(provider)
        if not key:
            console.print(f"[yellow]  {provider.upper()}: SKIP -- API key not found[/yellow]")
            results[provider] = "(no API key)"
            continue

        try:
            client = _make_client(provider)
            response = run_async(client.complete(
                messages=[{"role": "system", "content": "Answer concisely and clearly."}, {"role": "user", "content": query}]
            ))
            results[provider] = response.content or "(empty)"
        except Exception as e:
            results[provider] = f"[ERROR] {e}"

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan", width=12)
    table.add_column("Response", ratio=1)
    for provider in ("openai", "anthropic"):
        table.add_row(provider, results[provider])
    console.print(table)
    console.print()


def run_automatic():
    """Run side-by-side comparison for predefined questions."""
    for q in QUESTIONS:
        run_side_by_side(q)


def run_interactive():
    """Pick a provider and chat freely."""
    console.print("[bold]Available providers:[/bold]")
    console.print("  1. OpenAI")
    console.print("  2. Anthropic")

    choice = Prompt.ask("Pick a provider", choices=["1", "2"])
    provider = "openai" if choice == "1" else "anthropic"

    key = _check_key(provider)
    if not key:
        env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        console.print(Panel(
            f"[red]API key for {provider.upper()} not found.[/red]\n\n"
            f"Set environment variable:\n  export {env_var}=sk-...",
            title=f"[red]Missing {env_var}[/red]",
        ))
        return

    if provider == "anthropic":
        config_file = "17_anthropic_provider.yaml"
    else:
        config_file = "01_simple_chat.yaml"

    config_path = Path(__file__).parent / config_file
    if not config_path.exists():
        console.print(f"[yellow]Config file {config_file} not found.[/yellow]")
        return

    from koboi.facade import KoboiAgent

    try:
        agent = KoboiAgent.from_config(str(config_path))
    except Exception as e:
        console.print(f"[red]Failed to create agent: {e}[/red]")
        return

    console.print(Panel(
        f"[bold]Provider: {provider.upper()}[/bold]\n"
        f"Agent: {agent.config.agent_name}\n"
        f"Model: {agent.config.model}\n\n"
        "Type 'quit' to exit.",
        title="Interactive Chat",
    ))

    interactive_loop(agent, title=f"Agent ({provider})")


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 17: Multi-provider LLM."""
    setup_example(
        "Example 17: Multi-Provider LLM",
        "Demonstrates OpenAI and Anthropic provider usage.\n\n"
        "[dim]Automatic mode: side-by-side comparison.\n"
        "Interactive mode: pick a provider and chat.[/dim]",
    )

    if mode == "interactive":
        run_interactive()
    else:
        run_automatic()


if __name__ == "__main__":
    main()
