"""Shared helpers for koboi-agent examples.

Provides dual-mode utilities:
- run_async: persistent event loop wrapper for calling async methods
- dual_mode_options: click decorator for --mode automatic/interactive
- setup_example: common boilerplate (sys.path, .env, header)
- create_agent: KoboiAgent factory with error handling
- automatic_batch: predefined questions loop with Rich UI
- interactive_loop: free chat loop with Prompt.ask (wraps koboi.tui.loop)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

import click

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA = PROJECT_ROOT / "data" / "sample"
EXAMPLE_DATA = Path(__file__).resolve().parent / "data"

EXIT_WORDS = ("quit", "exit", "q")

# Persistent event loop for sync examples calling async methods.
# asyncio.run() closes the loop after each call, which breaks httpx.AsyncClient.
_persistent_loop: asyncio.AbstractEventLoop | None = None


def run_async(coro):
    """Run an async coroutine using a persistent event loop.

    Unlike asyncio.run(), this reuses the same event loop across calls,
    preventing 'Event loop is closed' errors with httpx.AsyncClient.
    """
    global _persistent_loop
    if _persistent_loop is None or _persistent_loop.is_closed():
        _persistent_loop = asyncio.new_event_loop()
    return _persistent_loop.run_until_complete(coro)


def load_env():
    """Load .env from project root."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def check_api_key(provider: str = "openai") -> str:
    """Check that API key is set, exit with help message if not."""
    load_env()
    env_var = f"{provider.upper()}_API_KEY"
    key = os.environ.get(env_var, "")
    if key in ("", "your-api-key-here", "sk-xxx"):
        console.print(f"[red]Error: Set {env_var} in .env file[/red]")
        console.print("[dim]Copy .env.example to .env and add your key.[/dim]")
        sys.exit(1)
    return key


def ensure_path():
    """Ensure project root is on sys.path so 'koboi' is importable."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def print_header(title: str, subtitle: str = ""):
    """Print a styled example header."""
    text = f"[bold]{title}[/bold]"
    if subtitle:
        text += f"\n{subtitle}"
    console.print(Panel(text, title="Koboi Agent Example"))


def sample_path(filename: str) -> Path:
    """Return absolute path to a file in data/sample/."""
    return SAMPLE_DATA / filename


# ---------------------------------------------------------------------------
# Dual-mode helpers
# ---------------------------------------------------------------------------

def dual_mode_options(func):
    """Decorator: adds --mode and --verbose flags to a click command.

    Usage::

        @click.command()
        @dual_mode_options
        def main(mode: str, verbose: bool):
            ...
    """
    func = click.option(
        "--verbose", "-v", is_flag=True, help="Show debug output",
    )(func)
    func = click.option(
        "--mode", "-m",
        type=click.Choice(["automatic", "interactive"]),
        default="automatic",
        help="Run mode (default: automatic)",
    )(func)
    return func


def setup_example(title: str, subtitle: str = "") -> Console:
    """Common setup for all examples: sys.path, .env, header.

    Returns the shared console object.
    """
    ensure_path()
    load_env()
    print_header(title, subtitle)
    return console


def create_agent(example_name: str, verbose: bool = False):
    """Load KoboiAgent from the YAML config matching *example_name*.

    Args:
        example_name: e.g. "02_tool_use_single" (matches the .yaml file)
        verbose: pass through to from_config

    Exits with an error message if config or API key is missing.
    """
    from koboi.facade import KoboiAgent

    config_path = Path(__file__).parent / f"{example_name}.yaml"
    try:
        return KoboiAgent.from_config(str(config_path), verbose=verbose)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Make sure OPENAI_API_KEY is set in .env[/dim]")
        sys.exit(1)


def interactive_loop(
    agent,
    *,
    title: str = "Agent",
    extra_commands: dict[str, Callable] | None = None,
    pre_send: Callable | None = None,
    post_receive: Callable | None = None,
):
    """Run an interactive chat loop with an agent.

    Wraps koboi.tui.loop.interactive_loop with backward-compatible
    signature (accepts pre_send/post_receive and old-style extra_commands).
    Falls back to local implementation if koboi.tui.loop is unavailable.
    """
    try:
        from koboi.tui.loop import interactive_loop as _tui_loop

        # Adapt old-style extra_commands: old callable(agent) -> new callable(agent, console)
        adapted = None
        if extra_commands:
            adapted = {}
            for key, fn in extra_commands.items():
                def _wrap(f=fn):
                    def wrapper(a, c):
                        return f(a)
                    return wrapper
                adapted[key] = _wrap(fn)

        run_async(_tui_loop(agent, console, title=title, extra_commands=adapted))
        return
    except ImportError:
        pass

    # Local fallback
    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        stripped = user_input.strip().lower()
        if stripped in EXIT_WORDS:
            console.print("[dim]Bye![/dim]")
            break

        if not user_input.strip():
            continue

        # Check extra commands
        if extra_commands and stripped in extra_commands:
            extra_commands[stripped](agent)
            continue

        if pre_send:
            pre_send(user_input)

        try:
            result = agent.run_sync(user_input)
            console.print(Panel(Markdown(str(result)), title=title, border_style="green"))
            if post_receive:
                post_receive(result, agent)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def automatic_batch(
    agent,
    questions: list[str | dict],
    *,
    title: str = "Agent",
    pre_question: Callable | None = None,
    post_answer: Callable | None = None,
    final_summary: Callable | None = None,
) -> list[dict]:
    """Run questions in automatic batch mode with Rich UI.

    Each question gets a rule separator, the agent answer in a Panel,
    and optional callbacks for per-question and final summary logic.

    Args:
        agent: KoboiAgent instance
        questions: list of strings or dicts with 'label' and 'input' keys
        title: Panel title for answers
        pre_question: callable(q, i, total) called before each question
        post_answer: callable(result, q, i, total) called after each answer
        final_summary: callable(agent, results) called after all questions

    Returns:
        list of result dicts with 'input', 'result', 'status' keys
    """
    results = []

    for i, q in enumerate(questions, 1):
        if isinstance(q, dict):
            label = q.get("label", f"Question {i}")
            user_input = q.get("input", "")
        else:
            label = f"Question {i}"
            user_input = q

        console.rule(f"[bold cyan]{label} ({i}/{len(questions)})[/bold cyan]")

        if pre_question:
            pre_question(q, i, len(questions))

        console.print(f"[bold yellow]Q:[/bold yellow] {user_input}")

        try:
            result = agent.run_sync(user_input)
            console.print(Panel(Markdown(str(result)), title=title, border_style="green"))
            results.append({"input": user_input, "result": result, "status": "ok"})
            if post_answer:
                post_answer(result, q, i, len(questions))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            results.append({"input": user_input, "result": str(e), "status": "error"})

        console.print()

    if final_summary:
        final_summary(agent, results)

    return results
