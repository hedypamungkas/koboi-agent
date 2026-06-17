"""koboi/tui/loop.py -- Async interactive loop with streaming support."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from koboi.events import (
    CompleteEvent, ErrorEvent, IterationEvent,
    TextDeltaEvent, ToolCallEvent, ToolResultEvent,
)

if TYPE_CHECKING:
    from koboi.tui.commands import SlashCommandRegistry

EXIT_WORDS = ("quit", "exit", "q")


async def interactive_loop(
    agent,
    console: Console,
    *,
    title: str = "Agent",
    extra_commands: dict[str, Callable] | None = None,
    command_registry: SlashCommandRegistry | None = None,
    stream: bool = True,
):
    """Run an async interactive chat loop with streaming support."""
    turn_count = 0
    start_time = time.time()

    while True:
        try:
            user_input = await asyncio.to_thread(
                console.input, "[bold green]You[/bold green] > "
            )
        except (KeyboardInterrupt, EOFError):
            _print_summary(console, turn_count, start_time, agent)
            break

        stripped = user_input.strip().lower()
        if stripped in EXIT_WORDS:
            _print_summary(console, turn_count, start_time, agent)
            break

        if not user_input.strip():
            continue

        # Slash command dispatch
        if stripped.startswith("/"):
            handled = False
            if command_registry is not None:
                from koboi.tui.commands import CommandContext
                parts = stripped.split(maxsplit=1)
                cmd_word = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                ctx = CommandContext(
                    agent=agent,
                    output=lambda text: console.print(f"[dim]{text}[/dim]"),
                    args=args,
                )
                result = await command_registry.dispatch(cmd_word, ctx)
                if result is not None:
                    if result.message:
                        user_input = result.message
                    else:
                        continue
                    handled = True
            elif extra_commands:
                cmd_word = stripped.split()[0]
                if cmd_word in extra_commands:
                    args = stripped[len(cmd_word):].strip()
                    try:
                        result = extra_commands[cmd_word](agent, console, args)
                    except TypeError:
                        result = extra_commands[cmd_word](agent, console)
                    import asyncio as _aio
                    if _aio.iscoroutine(result):
                        result = await result
                    if result and isinstance(result, str):
                        user_input = result
                    else:
                        continue
                    handled = True
            if not handled and stripped.startswith("/"):
                console.print(f"[dim]Unknown command: {stripped.split()[0]}. Type /help.[/dim]")
                continue

        try:
            if stream:
                await _stream_response(agent, user_input, console, title)
            else:
                result = await agent.run(user_input)
                console.print(Panel(Markdown(str(result)), title=title, border_style="green"))
            turn_count += 1
        except Exception as e:
            console.print(f"[red bold]Error:[/red bold] {e}")
            console.print("[dim]Session preserved. Try again or type 'quit'.[/dim]")


async def _stream_response(agent, user_input: str, console: Console, title: str) -> None:
    """Stream agent response with Rich Live display."""
    text_buffer = ""

    with Live(console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        async for event in agent.run_stream(user_input):
            if isinstance(event, TextDeltaEvent):
                text_buffer += event.content
                live.update(Panel(Markdown(text_buffer), title=title, border_style="green"))

            elif isinstance(event, ToolCallEvent):
                live.update(Panel(
                    Markdown(text_buffer) if text_buffer else "",
                    subtitle=f"[dim]{event.tool_name}[/dim]",
                    title=title,
                    border_style="green",
                ))

            elif isinstance(event, ToolResultEvent):
                pass

            elif isinstance(event, IterationEvent):
                pass

            elif isinstance(event, CompleteEvent):
                if event.content:
                    text_buffer = event.content
                live.update(Panel(Markdown(text_buffer), title=title, border_style="green"))

            elif isinstance(event, ErrorEvent):
                live.update(Panel(
                    f"[red]{event.error}[/red]",
                    title=title,
                    border_style="red",
                ))


def _print_summary(console: Console, turns: int, start_time: float, agent=None) -> None:
    """Print session summary on exit."""
    elapsed = time.time() - start_time
    parts = [f"{turns} message(s)", f"{elapsed:.1f}s"]

    if agent:
        try:
            mem_len = len(agent.core.memory)
            parts.append(f"{mem_len} in memory")
        except Exception:
            pass

    console.print(f"\n[dim]Goodbye! {' | '.join(parts)}[/dim]")


def build_slash_commands(agent) -> dict[str, Callable]:
    """Build slash command handlers for the interactive loop."""
    from koboi.tui.app import _build_welcome_panel

    def cmd_reset(agent, console):
        agent.reset()
        console.print("[green]Conversation reset.[/green]")

    def cmd_info(agent, console):
        console.print(_build_welcome_panel(agent, title="Agent Info"))

    def cmd_history(agent, console):
        messages = agent.core.memory.get_messages()
        if not messages:
            console.print("[dim]No messages in history.[/dim]")
            return
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            role_style = {"user": "green", "assistant": "blue", "system": "dim"}.get(role, "white")
            preview = content[:120] + "..." if len(content) > 120 else content
            console.print(f"  [{role_style}][{i}] {role}:[/{role_style}] {preview}")

    def cmd_tools(agent, console):
        tools_dict = agent.core.tools.list_tools()
        if not tools_dict:
            console.print("[dim]No tools registered.[/dim]")
            return
        table = Table(title="Registered Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Risk Level")
        for name in sorted(tools_dict.keys()):
            td = tools_dict[name]
            risk = str(td.risk_level.value) if hasattr(td, 'risk_level') else "safe"
            risk_style = {"safe": "green", "moderate": "yellow", "destructive": "red"}.get(risk, "white")
            table.add_row(name, f"[{risk_style}]{risk}[/{risk_style}]")
        console.print(table)

    def cmd_run(agent, console, args=""):  # noqa: ARG001
        """Hot-load a new config and optionally send a message."""
        if not args:
            console.print("[yellow]Usage: /run <config_path> [message][/yellow]")
            return
        parts = args.split(maxsplit=1)
        config_path = parts[0]
        run_message = parts[1] if len(parts) > 1 else ""
        from pathlib import Path
        if not Path(config_path).exists():
            console.print(f"[red]Config not found: {config_path}[/red]")
            return
        try:
            from koboi.facade import KoboiAgent
            new_agent = KoboiAgent.from_config(config_path)
            agent.replace_from(new_agent)
            console.print(f"[green]Loaded config: {config_path}[/green]")
            if run_message:
                # Return the message to be processed by the caller
                return run_message
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")

    async def cmd_compact_async(agent, console):
        core = agent.core
        if not core or not core.context_manager:
            console.print("[yellow]No context strategy configured.[/yellow]")
            return
        from koboi.tokens import estimate_tokens
        messages = core.memory.get_messages()
        tokens_before = estimate_tokens(messages)
        compacted = await core.context_manager.manage(messages, max_tokens=0)
        tokens_after = estimate_tokens(compacted)
        removed = len(messages) - len(compacted)
        core.memory.replace_messages(compacted)
        console.print(f"[green]Compacted: {len(messages)} -> {len(compacted)} messages "
                       f"({tokens_before} -> {tokens_after} tokens, {removed} removed)[/green]")

    def cmd_model(agent, console, args=""):
        # Resolve client from core or orchestrator
        if agent.core is not None:
            client = agent.core.client
            client_attr = "core"
        elif agent._orchestrator is not None:
            client = agent._orchestrator.client
            client_attr = "orchestrator"
        else:
            console.print("[yellow]No agent core or orchestrator available.[/yellow]")
            return
        if not args:
            console.print(f"Current model: {client.provider}/{client.model}")
            return
        raw = args.strip()
        try:
            from koboi.client import Client
            old_client = client
            # Parse "provider/model" format
            if "/" in raw:
                new_provider, new_model = raw.split("/", 1)
            else:
                new_provider = old_client.provider
                new_model = raw
            if new_provider != old_client.provider:
                # Provider changed — let Client re-resolve api_key/base_url for new provider
                new_client = Client(
                    model=new_model,
                    logger=old_client.logger,
                    provider=new_provider,
                    temperature=old_client.temperature,
                )
            else:
                new_client = Client(
                    api_key=old_client.api_key,
                    base_url=old_client.base_url,
                    model=new_model,
                    logger=old_client.logger,
                    provider=old_client.provider,
                    temperature=old_client.temperature,
                )
            if client_attr == "core":
                agent.core.client = new_client
            else:
                agent._orchestrator.client = new_client
                # Also update all specialist agents' clients
                for specialist in agent._orchestrator._agents_map.values():
                    specialist.client = new_client
            agent.config.raw.setdefault("llm", {})["model"] = new_model
            agent.config.raw.setdefault("llm", {})["provider"] = new_provider
            console.print(f"[green]Switched model to: {new_provider}/{new_model}[/green]")
        except Exception as e:
            console.print(f"[red]Error switching model: {e}[/red]")

    def cmd_editor(agent, console):
        import os
        import subprocess
        import tempfile
        editor = os.environ.get("EDITOR", "vim")
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("")
            tmp_path = f.name
        try:
            result = subprocess.run([editor, tmp_path])
            if result.returncode == 0:
                content = open(tmp_path).read().strip()
                if content:
                    return content
                else:
                    console.print("[dim]Editor closed with empty content.[/dim]")
            else:
                console.print("[yellow]Editor exited with error.[/yellow]")
        finally:
            os.unlink(tmp_path)

    def cmd_undo(agent, console, args=""):
        import subprocess as _sp
        n = 1
        if args:
            try:
                n = int(args.strip())
            except ValueError:
                console.print("[yellow]Usage: /undo [n][/yellow]")
                return
        if n < 1 or n > 10:
            console.print("[yellow]Can only revert 1-10 commits.[/yellow]")
            return
        try:
            log_result = _sp.run(
                ["git", "log", f"-{n}", "--oneline"],
                capture_output=True, text=True, timeout=10
            )
            if log_result.returncode != 0:
                console.print(f"[red]Git error: {log_result.stderr.strip()}[/red]")
                return
            console.print(f"Reverting last {n} commit(s):\n{log_result.stdout.strip()}")
        except Exception as e:
            console.print(f"[red]Git error: {e}[/red]")
            return
        reverted = 0
        for i in range(n):
            try:
                result = _sp.run(
                    ["git", "revert", "HEAD", "--no-edit"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    reverted += 1
                else:
                    console.print(f"[red]Revert failed at commit {i+1}: {result.stderr.strip()}[/red]")
                    break
            except Exception as e:
                console.print(f"[red]Revert error: {e}[/red]")
                break
        console.print(f"[green]Reverted {reverted}/{n} commit(s).[/green]")

    def cmd_copy(agent, console):
        messages = agent.core.memory.get_messages()
        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = msg["content"]
                break
        if not last_assistant:
            console.print("[dim]No assistant message to copy.[/dim]")
            return
        copied = False
        try:
            import pyperclip
            pyperclip.copy(last_assistant)
            copied = True
        except ImportError:
            pass
        if not copied:
            import shutil
            import subprocess as _sp
            if shutil.which("pbcopy"):
                _sp.run(["pbcopy"], input=last_assistant.encode(), check=True)
                copied = True
            elif shutil.which("xclip"):
                _sp.run(["xclip", "-selection", "clipboard"], input=last_assistant.encode(), check=True)
                copied = True
        if copied:
            preview = last_assistant[:80] + "..." if len(last_assistant) > 80 else last_assistant
            console.print(f"[green]Copied to clipboard: {preview}[/green]")
        else:
            console.print("[yellow]No clipboard backend found. Install pyperclip or xclip.[/yellow]")

    def cmd_diagnostics(agent, console):
        from datetime import datetime
        from pathlib import Path
        from koboi.diagnostics import collect_diagnostics
        try:
            data = collect_diagnostics(agent)
            filename = f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            Path(filename).write_bytes(data)
            size_kb = len(data) / 1024
            console.print(f"[green]Diagnostics exported to {filename} ({size_kb:.1f} KB)[/green]")
        except Exception as e:
            console.print(f"[red]Error generating diagnostics: {e}[/red]")

    def cmd_help(agent, console):
        console.print(Panel(
            "[bold]Commands:[/bold]\n"
            "  /reset   -- Clear conversation memory\n"
            "  /info    -- Show agent configuration\n"
            "  /history -- Show conversation history\n"
            "  /tools   -- List registered tools\n"
            "  /run <config> [msg] -- Hot-load a config\n"
            "  /compact -- Manually compact context window\n"
            "  /model <name> -- Switch LLM model mid-session\n"
            "  /editor  -- Open $EDITOR for long messages\n"
            "  /undo [n] -- Revert last AI commit(s)\n"
            "  /copy    -- Copy last response to clipboard\n"
            "  /diagnostics -- Export session diagnostics ZIP\n"
            "  /help    -- Show this help\n"
            "  quit     -- Exit the session",
            title="Help",
        ))

    return {
        "/reset": cmd_reset,
        "/info": cmd_info,
        "/history": cmd_history,
        "/tools": cmd_tools,
        "/run": cmd_run,
        "/compact": cmd_compact_async,
        "/model": cmd_model,
        "/editor": cmd_editor,
        "/undo": cmd_undo,
        "/copy": cmd_copy,
        "/diagnostics": cmd_diagnostics,
        "/help": cmd_help,
    }
