"""koboi/tui/app.py -- Interactive chat surface (Textual + legacy Rich loop).

The console-script dispatch lives in :mod:`koboi.cli`; this module owns only the
*interactive* chat experience, exposed via :func:`run_chat_interactive`, which
:func:`koboi.cli._run_chat` lazy-imports when the user runs ``koboi chat``
without ``--print``. ``koboi chat --print`` (JSON-line output) is handled
core-only in :mod:`koboi.cli_commands` and does NOT touch this module.

Requires the ``[tui]`` extra (``rich`` for the legacy ``--no-tui`` loop and
``textual`` for the default Textual app). Importing this module therefore
preserves the rich import at module top; cli.py gates the import behind a
graceful ``ImportError`` so a bare install gets a clear install hint.
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel

console = Console(no_color=not sys.stdout.isatty())


def _build_welcome_panel(agent, title: str = "Welcome") -> Panel:
    """Build an information-rich welcome panel for an agent session."""
    config = agent.config
    core = agent.core

    lines = [f"[bold]{config.agent_name}[/bold]"]
    lines.append(f"Model: {config.provider}/{config.model}")
    lines.append(f"Max iterations: {config.max_iterations}")

    # Tools
    tool_names = list(core.tools.list_tools().keys())
    if tool_names:
        display = ", ".join(sorted(tool_names)[:8])
        if len(tool_names) > 8:
            display += f" ... (+{len(tool_names) - 8})"
        lines.append(f"Tools ({len(tool_names)}): {display}")

    # Guardrails
    guards = []
    if core.input_guardrail:
        guards.append("input")
    if core.output_guardrail:
        guards.append("output")
    if core.rate_limiter:
        guards.append("rate_limit")
    if core.approval_handler:
        guards.append("approval")
    if guards:
        lines.append(f"Guardrails: {', '.join(guards)}")

    # RAG
    if config.rag_enabled:
        lines.append("RAG: [green]enabled[/green]")

    # Hooks
    hooks = core.hooks.list_hooks()
    if hooks:
        hook_names = [h["name"] for h in hooks]
        display = ", ".join(hook_names[:6])
        if len(hook_names) > 6:
            display += f" ... (+{len(hook_names) - 6})"
        lines.append(f"Hooks ({len(hooks)}): {display}")

    lines.append("\n[dim]Type 'quit' to exit. Use '/help' for commands.[/dim]")

    return Panel("\n".join(lines), title=title, border_style="cyan")


async def _run_interactive(agent, stream: bool = True):
    """Legacy Rich-based interactive chat loop (``chat --no-tui``)."""
    from koboi.tui.commands import build_registry
    from koboi.tui.loop import interactive_loop

    console.print(_build_welcome_panel(agent, title="Chat Session"))
    registry = build_registry()
    await interactive_loop(agent, console, title="Agent", command_registry=registry, stream=stream)


def run_chat_interactive(
    config_path: str, *, verbose: bool = False, no_tui: bool = False, no_stream: bool = False
) -> int:
    """Interactive ``koboi chat`` (default Textual app, or legacy Rich via ``--no-tui``).

    Returns an exit code (0 success, 1 agent-load error). The caller
    (:func:`koboi.cli._run_chat`) is responsible for the ``[tui]``-missing
    fallback; this function assumes the TUI extra is importable.
    """
    from koboi.facade import KoboiAgent

    try:
        agent = KoboiAgent.from_config(config_path, verbose=verbose)
    except Exception as e:
        console.print(f"[red bold]Error loading agent:[/red bold] {e}")
        console.print("[dim]Check your config file and API key settings.[/dim]")
        return 1

    if no_tui:
        asyncio.run(_run_interactive(agent, stream=not no_stream))
    else:
        from koboi.tui.textual_app import KoboiApp

        KoboiApp(agent).run()
    return 0
