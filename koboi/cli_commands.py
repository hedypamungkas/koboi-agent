"""koboi/cli_commands -- Core command logic for the ``koboi`` console script.

These handlers are the single source of truth for every no-TUI subcommand. They
use only stdlib I/O (``print`` / ``input``) so the console script
(:mod:`koboi.cli`) works on a bare ``pip install koboi-agent`` (no extras) for
all non-interactive commands. The interactive ``chat`` surface (Textual / legacy
Rich loop) stays in :mod:`koboi.tui.app` and is lazy-imported with a graceful
fallback when the ``[tui]`` extra is absent.

Each ``cmd_*`` function returns an int exit code (0 = success, 1 = error) so the
dispatcher in :mod:`koboi.cli` can ``sys.exit()`` with it. ``--print`` modes emit
streaming JSON lines (pipe-friendly); errors in print mode are emitted as
``{"type": "error", "error": ...}`` JSON lines on stdout.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Streaming helpers (pipe-friendly JSON-lines output)
# --------------------------------------------------------------------------- #
async def _run_print_mode(agent: Any, message: str) -> None:
    """Stream agent output as JSON lines (for piping/CI)."""
    from koboi.events import event_to_dict

    async for event in agent.run_stream(message):
        print(json.dumps(event_to_dict(event)), flush=True)


async def _chat_print_mode(agent: Any) -> None:
    """Interactive chat with JSON-line output (no TUI)."""
    from koboi.events import event_to_dict

    print(
        json.dumps(
            {
                "type": "session_start",
                "agent": agent.config.agent_name,
                "model": f"{agent.config.provider}/{agent.config.model}",
            }
        ),
        flush=True,
    )

    while True:
        try:
            message = await asyncio.get_event_loop().run_in_executor(None, input)
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


def _print_error(message: str, *, print_mode: bool = False) -> None:
    """Emit an error in the right shape for the active mode."""
    if print_mode:
        print(json.dumps({"type": "error", "error": message}), flush=True)
    else:
        print(f"Error: {message}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def cmd_graph(config_path: str, fmt: str = "mermaid") -> int:
    """Render the orchestration agent DAG (depends_on edges) as Mermaid or JSON.

    Reads ``orchestration.agents[*].depends_on`` from config without running any
    agent. Useful for inspecting a workflow graph and (Phase 3) feeding visualizers.
    """
    from koboi.config import Config
    from koboi.facade import _parse_agent_defs

    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        print(f"Config parse error: {e}", file=sys.stderr)
        return 1

    try:
        agent_defs = _parse_agent_defs(config)
    except ValueError:
        print("No orchestration agents found (orchestration.agents is empty).", file=sys.stderr)
        return 1

    nodes = [ad.name for ad in agent_defs]
    edges = [{"from": dep, "to": ad.name} for ad in agent_defs for dep in ad.depends_on]

    if fmt == "json":
        print(json.dumps({"nodes": nodes, "edges": edges}, indent=2))
    else:
        lines = ["graph TD"]
        for n in nodes:
            lines.append(f'  {n}["{n}"]')
        for edge in edges:
            lines.append(f"  {edge['from']} --> {edge['to']}")
        print("\n".join(lines))
    return 0


def cmd_validate(config_path: str) -> int:
    """Validate a YAML config file without running the agent."""
    from koboi.config import Config

    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        print(f"Config parse error: {e}", file=sys.stderr)
        return 1

    issues: list[str] = []
    if not config.agent_name:
        issues.append("agent.name is missing")
    if not config.model:
        issues.append("llm.model is missing")
    if config.provider not in ("openai", "anthropic", "cloudflare"):
        issues.append(f"Unknown provider: {config.provider}")

    api_key = config.api_key
    if not api_key or api_key in ("", "your-api-key-here", "sk-xxx"):
        env_var = f"{config.provider.upper()}_API_KEY"
        if not os.environ.get(env_var, ""):
            issues.append(f"API key not set ({env_var})")

    if issues:
        print("Validation failed:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    print("Config is valid")
    print(f"  Agent: {config.agent_name}")
    print(f"  Model: {config.provider}/{config.model}")
    print(f"  RAG: {'enabled' if config.rag_enabled else 'disabled'}")
    print(f"  Max iterations: {config.max_iterations}")
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(
    config_path: str,
    message: str | None,
    verbose: bool,
    print_mode: bool,
    resume_session: str | None,
) -> int:
    """Run a single agent query (non-interactive or one-shot)."""
    from koboi.facade import KoboiAgent

    try:
        agent = KoboiAgent.from_config(config_path, verbose=verbose, resume_session=resume_session)
    except Exception as e:
        _print_error(f"loading agent: {e}", print_mode=print_mode)
        return 1

    # --resume: rehydrate-and-continue an interrupted session.
    if resume_session:
        try:
            result = asyncio.run(agent.resume())
        except Exception as e:
            print(f"Resume error: {e}", file=sys.stderr)
            return 1
        print(f"[Resumed ({resume_session[:8]})]")
        print(result)
        return 0

    if not message:
        if print_mode:
            message = sys.stdin.read().strip()
            if not message:
                _print_error("No message provided", print_mode=True)
                return 1
        else:
            try:
                message = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                return 1
            if not message:
                print("No message provided.", file=sys.stderr)
                return 1

    if print_mode:
        asyncio.run(_run_print_mode(agent, message))
        return 0

    print(f"[Input] {message}")
    try:
        result = asyncio.run(agent.run(message))
    except Exception as e:
        print(f"Agent error: {e}", file=sys.stderr)
        return 1
    print("[Output]")
    print(result)
    return 0


# --------------------------------------------------------------------------- #
# chat (print mode only lives here; interactive lives in koboi.tui.app)
# --------------------------------------------------------------------------- #
def cmd_chat_print(config_path: str, verbose: bool) -> int:
    """``chat --print``: interactive chat with JSON-line output (no TUI deps)."""
    from koboi.facade import KoboiAgent

    try:
        agent = KoboiAgent.from_config(config_path, verbose=verbose)
    except Exception as e:
        _print_error(str(e), print_mode=True)
        return 1
    asyncio.run(_chat_print_mode(agent))
    return 0


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #
def cmd_sessions(config_path: str, limit: int) -> int:
    """List persisted sessions for an agent's SQLite database (P2-A)."""
    from koboi.config import Config
    from koboi.memory_sqlite import SQLiteMemory

    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    db_path = config.get("memory", "db_path", default="koboi_memory.db")
    if config.get("memory", "backend", default="sqlite") != "sqlite":
        print("Memory backend is not sqlite -- no persisted sessions.")
        return 0

    rows = SQLiteMemory.list_sessions(db_path, limit=limit)
    if not rows:
        print(f"No sessions found in {db_path}.")
        return 0

    print(f"Sessions in {db_path}")
    print(f"{'Session ID':<14} {'Title':<40} {'Msgs':>5}  {'Updated':<18} First message")
    for row in rows:
        sid = (row.get("session_id") or "")[:12]
        title = (row.get("title") or "")[:40]
        msgs = str(row.get("message_count", 0))
        updated = str(row.get("updated_at"))[:18]
        first = (row.get("first_message") or "").strip().replace("\n", " ")[:60]
        print(f"{sid:<14} {title:<40} {msgs:>5}  {updated:<18} {first}")
    print(f"\nResume with: koboi run {config_path} --resume <session-id>")
    return 0


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def cmd_eval(config_path: str, cases: str | None) -> int:
    """Run evaluation suite against an agent config."""
    from koboi.eval.runner import EvalRunner
    from koboi.eval.scorers import (
        IterationEfficiencyScorer,
        KeywordPresenceScorer,
        OutputLengthScorer,
        ToolUsageScorer,
    )
    from koboi.facade import KoboiAgent
    from koboi.types import EvalCase

    print(f"Running eval for: {config_path}")

    scorers = [
        ToolUsageScorer(),
        KeywordPresenceScorer(),
        OutputLengthScorer(),
        IterationEfficiencyScorer(),
    ]

    eval_cases: list[EvalCase] = []
    if cases and Path(cases).exists():
        import yaml

        with open(cases) as f:
            data = yaml.safe_load(f) or {}
        for case_data in data.get("cases", []):
            eval_cases.append(EvalCase(**case_data))

    if not eval_cases:
        print("No eval cases found. Provide --cases file.")
        return 0

    def factory() -> KoboiAgent:
        try:
            return KoboiAgent.from_config(config_path)
        except Exception as e:
            print(f"Error creating agent for eval: {e}", file=sys.stderr)
            raise SystemExit(1) from None

    try:
        runner = EvalRunner(harness_factory=factory, scorers=scorers, console=None)
        results = asyncio.run(runner.run_suite(eval_cases))
        print(runner.format_results(results))
    except SystemExit:
        raise
    except Exception as e:
        print(f"Eval runner error: {e}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# eval-test (eve-style ``t`` evals)
# --------------------------------------------------------------------------- #
def cmd_eval_test(
    path: str,
    config: str | None,
    mock: bool | None,
    strict: bool,
    threshold: float,
    parallel: bool,
    max_concurrency: int,
    tags: str | None,
) -> int:
    """Run eve-style ``t`` eval tests (``*.eval.py`` files).

    With ``--strict`` exits non-zero if any test fails (gate failure or below
    threshold) -- suitable for CI.
    """
    from koboi.eval.runner import EvalRunner
    from koboi.eval.t import run_tests_sync

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    try:
        results = run_tests_sync(
            path,
            threshold=threshold,
            parallel=parallel,
            max_concurrency=max_concurrency,
            tags=tag_list,
            config=config,
            mock=mock,
        )
    except Exception as exc:  # discovery/import/config errors
        print(f"eval-test error: {exc}", file=sys.stderr)
        return 2

    if not results:
        print("No tests found.")
        return 2

    print(EvalRunner.format_results(results, threshold))

    failed = [r for r in results if not r.passed]
    if strict and failed:
        print(f"\n{len(failed)} test(s) failed (gate failure or below threshold).", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def cmd_diagnostics(config_path: str, output: str | None) -> int:
    """Export session diagnostics as a ZIP bundle."""
    from koboi.diagnostics import collect_diagnostics
    from koboi.facade import KoboiAgent

    try:
        agent = KoboiAgent.from_config(config_path)
    except Exception as e:
        print(f"Error loading agent: {e}", file=sys.stderr)
        return 1

    try:
        from datetime import datetime

        data = collect_diagnostics(agent)
        filename = output or f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        Path(filename).write_bytes(data)
        print(f"Diagnostics exported to {filename} ({len(data) / 1024:.1f} KB)")
        return 0
    except Exception as e:
        print(f"Error generating diagnostics: {e}", file=sys.stderr)
        return 1
    finally:
        asyncio.run(agent.close())


# --------------------------------------------------------------------------- #
# init-zsh
# --------------------------------------------------------------------------- #
def cmd_init_zsh(target: str | None) -> int:
    """Install the ZSH plugin for ``:koboi`` prefix command."""
    plugin_src = Path(__file__).parent.parent / "shell" / "koboi.plugin.zsh"
    if not plugin_src.exists():
        print("Plugin source not found. Reinstall koboi-agent.", file=sys.stderr)
        return 1

    if target:
        dest_dir = Path(target)
    else:
        zsh_custom = os.environ.get("ZSH_CUSTOM", "")
        if zsh_custom and Path(zsh_custom).is_dir():
            dest_dir = Path(zsh_custom) / "plugins" / "koboi"
        else:
            dest_dir = Path.home() / ".zsh" / "koboi"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "koboi.plugin.zsh"
    shutil.copy2(plugin_src, dest_file)

    print(f"Plugin installed to: {dest_file}")
    print()
    print("To activate, add to your .zshrc:")
    print()
    if "oh-my-zsh" in str(dest_dir):
        print("  plugins=(... koboi)")
    else:
        print(f"  source {dest_file}")
    print()
    print("Then set your default config (optional):")
    print("  export KOBOI_CONFIG=configs/simple_chat.yaml")
    print()
    print("Usage: :koboi your question here")
    return 0
