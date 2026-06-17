"""koboi/tui/app.py — CLI entry point for koboi-agent."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

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
    """Async helper to run interactive chat loop."""
    from koboi.tui.loop import interactive_loop
    from koboi.tui.commands import build_registry

    console.print(_build_welcome_panel(agent, title="Chat Session"))
    registry = build_registry()
    await interactive_loop(agent, console, title="Agent", command_registry=registry, stream=stream)


async def _run_print_mode(agent, message: str) -> None:
    """Stream agent output as JSON lines (for piping/CI)."""
    from koboi.events import event_to_dict

    async for event in agent.run_stream(message):
        print(json.dumps(event_to_dict(event)), flush=True)


async def _chat_print_mode(agent) -> None:
    """Interactive chat with JSON line output (no TUI)."""
    from koboi.events import event_to_dict

    print(json.dumps({"type": "session_start", "agent": agent.config.agent_name,
                       "model": f"{agent.config.provider}/{agent.config.model}"}), flush=True)

    while True:
        try:
            message = await asyncio.get_event_loop().run_in_executor(None, lambda: input())
        except (EOFError, KeyboardInterrupt):
            break
        message = message.strip()
        if not message:
            continue
        if message.lower() in ("quit", "exit", "/quit", "/exit"):
            break

        async for event in agent.run_stream(message):
            print(json.dumps(event_to_dict(event)), flush=True)

    print(json.dumps({"type": "session_end"}), flush=True)


@click.group()
def main():
    """Koboi Agent — Universal configurable AI agent framework."""
    load_dotenv()


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--message", "-m", help="Message to send (required for non-interactive)")
@click.option("--verbose", "-v", is_flag=True, help="Show debug output")
@click.option("--print", "print_mode", is_flag=True, help="Output streaming JSON lines (pipe-friendly)")
def run(config_path: str, message: str | None, verbose: bool, print_mode: bool):
    """Run a single agent query (non-interactive or one-shot)."""
    from koboi.facade import KoboiAgent

    try:
        agent = KoboiAgent.from_config(config_path, verbose=verbose)
    except Exception as e:
        if print_mode:
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)
        else:
            console.print(f"[red bold]Error loading agent:[/red bold] {e}")
            console.print("[dim]Check your config file and API key settings.[/dim]")
        raise SystemExit(1)

    if not message:
        if print_mode:
            # Read from stdin in print mode
            message = sys.stdin.read().strip()
            if not message:
                print(json.dumps({"type": "error", "error": "No message provided"}), flush=True)
                raise SystemExit(1)
        else:
            message = Prompt.ask("[bold green]Query[/bold green]")

    if print_mode:
        asyncio.run(_run_print_mode(agent, message))
    else:
        console.print(Panel(f"[bold]{message}[/bold]", title="Input"))
        try:
            with console.status("[bold cyan]Processing...[/bold cyan]", spinner="dots"):
                result = asyncio.run(agent.run(message))
        except Exception as e:
            console.print(f"[red bold]Agent error:[/red bold] {e}")
            raise SystemExit(1)
        console.print(Panel(Markdown(str(result)), title="Output"))


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show debug output")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
@click.option("--no-tui", is_flag=True, help="Use legacy Rich-based interface instead of Textual TUI")
@click.option("--print", "print_mode", is_flag=True, help="Output streaming JSON lines (pipe-friendly, no TUI)")
def chat(config_path: str, verbose: bool, no_stream: bool, no_tui: bool, print_mode: bool):
    """Start an interactive chat session with an agent."""
    from koboi.facade import KoboiAgent

    try:
        agent = KoboiAgent.from_config(config_path, verbose=verbose)
    except Exception as e:
        if print_mode:
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)
        else:
            console.print(f"[red bold]Error loading agent:[/red bold] {e}")
            console.print("[dim]Check your config file and API key settings.[/dim]")
        raise SystemExit(1)

    if print_mode:
        asyncio.run(_chat_print_mode(agent))
    elif no_tui:
        asyncio.run(_run_interactive(agent, stream=not no_stream))
    else:
        from koboi.tui.textual_app import KoboiApp
        KoboiApp(agent).run()


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--cases", type=click.Path(), help="Eval cases YAML file")
def eval(config_path: str, cases: str | None):
    """Run evaluation suite against an agent config."""
    try:
        from koboi.facade import KoboiAgent
        from koboi.eval.runner import EvalRunner
        from koboi.types import EvalCase
        from koboi.eval.scorers import (
            ToolUsageScorer, KeywordPresenceScorer,
            OutputLengthScorer, IterationEfficiencyScorer,
        )
    except ImportError as e:
        console.print(f"[red bold]Import error:[/red bold] {e}")
        console.print("[dim]Make sure koboi-agent is installed correctly.[/dim]")
        raise SystemExit(1)

    console.print(f"[bold]Running eval for: {config_path}[/bold]")

    scorers = [
        ToolUsageScorer(),
        KeywordPresenceScorer(),
        OutputLengthScorer(),
        IterationEfficiencyScorer(),
    ]

    # Default eval cases if no file provided
    eval_cases = []
    if cases and Path(cases).exists():
        import yaml
        with open(cases) as f:
            data = yaml.safe_load(f) or {}
        for case_data in data.get("cases", []):
            eval_cases.append(EvalCase(**case_data))

    if not eval_cases:
        console.print("[yellow]No eval cases found. Provide --cases file.[/yellow]")
        return

    def factory():
        try:
            return KoboiAgent.from_config(config_path)
        except Exception as e:
            console.print(f"[red bold]Error creating agent for eval:[/red bold] {e}")
            raise SystemExit(1)

    try:
        runner = EvalRunner(
            harness_factory=factory,
            scorers=scorers,
            console=console,
        )
        results = asyncio.run(runner.run_suite(eval_cases))
        console.print(runner.format_results(results))
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red bold]Eval runner error:[/red bold] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str):
    """Validate a YAML config file without running the agent."""
    import os
    from koboi.config import Config

    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        console.print(f"[red bold]Config parse error:[/red bold] {e}")
        raise SystemExit(1)

    issues = []
    if not config.agent_name:
        issues.append("agent.name is missing")
    if not config.model:
        issues.append("llm.model is missing")
    if config.provider not in ("openai", "anthropic", "cloudflare"):
        issues.append(f"Unknown provider: {config.provider}")

    # Check API key availability
    api_key = config.api_key
    if not api_key or api_key in ("", "your-api-key-here", "sk-xxx"):
        env_var = f"{config.provider.upper()}_API_KEY"
        if not os.environ.get(env_var, ""):
            issues.append(f"API key not set ({env_var})")

    if issues:
        console.print("[red bold]Validation failed:[/red bold]")
        for issue in issues:
            console.print(f"  [red]-[/red] {issue}")
        raise SystemExit(1)
    else:
        console.print(Panel(
            f"[green bold]Config is valid[/green bold]\n\n"
            f"Agent: {config.agent_name}\n"
            f"Model: {config.provider}/{config.model}\n"
            f"RAG: {'enabled' if config.rag_enabled else 'disabled'}\n"
            f"Max iterations: {config.max_iterations}",
            title="Validation Result",
            border_style="green",
        ))


@main.command()
@click.option("--target", type=click.Path(), help="Custom plugin install directory")
def init_zsh(target: str | None):
    """Install the ZSH plugin for :koboi prefix command."""
    import shutil
    from pathlib import Path

    plugin_src = Path(__file__).parent.parent.parent / "shell" / "koboi.plugin.zsh"
    if not plugin_src.exists():
        console.print("[red bold]Plugin source not found.[/red bold] Reinstall koboi-agent.")
        raise SystemExit(1)

    # Determine target directory
    if target:
        dest_dir = Path(target)
    else:
        # Try oh-my-zsh custom plugins
        zsh_custom = os.environ.get("ZSH_CUSTOM", "")
        if zsh_custom and Path(zsh_custom).is_dir():
            dest_dir = Path(zsh_custom) / "plugins" / "koboi"
        else:
            # Fallback to ~/.zsh/koboi
            dest_dir = Path.home() / ".zsh" / "koboi"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "koboi.plugin.zsh"
    shutil.copy2(plugin_src, dest_file)

    console.print(f"[green bold]Plugin installed to:[/green bold] {dest_file}")
    console.print("")
    console.print("To activate, add to your [bold].zshrc[/bold]:")
    console.print("")
    if "oh-my-zsh" in str(dest_dir):
        console.print("  plugins=(... koboi)")
    else:
        console.print(f"  source {dest_file}")
    console.print("")
    console.print("Then set your default config (optional):")
    console.print("  export KOBOI_CONFIG=configs/simple_chat.yaml")
    console.print("")
    console.print("Usage: [bold]:koboi your question here[/bold]")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path (default: diagnostics_<timestamp>.zip)")
def diagnostics(config_path: str, output: str | None):
    """Export session diagnostics as a ZIP bundle."""
    from datetime import datetime
    from pathlib import Path
    from koboi.facade import KoboiAgent
    from koboi.diagnostics import collect_diagnostics

    try:
        agent = KoboiAgent.from_config(config_path)
    except Exception as e:
        console.print(f"[red bold]Error loading agent:[/red bold] {e}")
        raise SystemExit(1)

    try:
        data = collect_diagnostics(agent)
        filename = output or f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        Path(filename).write_bytes(data)
        size_kb = len(data) / 1024
        console.print(f"[green]Diagnostics exported to {filename} ({size_kb:.1f} KB)[/green]")
    except Exception as e:
        console.print(f"[red bold]Error generating diagnostics:[/red bold] {e}")
        raise SystemExit(1)
    finally:
        asyncio.run(agent.close())


if __name__ == "__main__":
    main()
