"""Example 22: Full production agent -- capstone example with all features.

Demonstrates:
- Production agent with all features active (RAG, guardrails, policy, telemetry, carryover, doom loop)
- Custom ProfilingHook added to the hook chain for per-call timing
- Hook chain inspection via public list_hooks() / find_hook() API
- Dual mode: automatic (batch with telemetry + profiling) and interactive (chat with live metrics)
- Commands in interactive: quit, reset, info, hooks

Run:
    python examples/22_full_production.py                  # automatic mode
    python examples/22_full_production.py -m interactive   # interactive mode
    python examples/22_full_production.py -v               # automatic + verbose
"""

from __future__ import annotations

import time

import click
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from conftest import (
    console,
    setup_example,
    dual_mode_options,
    create_agent,
    automatic_batch,
    interactive_loop,
)

ensure_path = __import__("conftest").ensure_path
ensure_path()
from koboi.hooks.chain import Hook, HookContext, HookEvent  # noqa: E402


# ---------------------------------------------------------------------------
# Custom hook: ProfilingHook
# ---------------------------------------------------------------------------


class ProfilingHook(Hook):
    """Production profiling hook: tracks per-tool and per-LLM-call timing."""

    def __init__(self):
        self._timings: dict[str, list[float]] = {}
        self._starts: dict[str, float] = {}

    def handles(self) -> list[HookEvent]:
        return [
            HookEvent.PRE_TOOL_USE,
            HookEvent.POST_TOOL_USE,
            HookEvent.PRE_LLM_CALL,
            HookEvent.POST_LLM_CALL,
            HookEvent.SESSION_START,
            HookEvent.SESSION_END,
        ]

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.event in (HookEvent.PRE_TOOL_USE, HookEvent.PRE_LLM_CALL):
            key = ctx.tool_name if ctx.tool_name else "llm_call"
            self._starts[key] = time.time()
        elif ctx.event in (HookEvent.POST_TOOL_USE, HookEvent.POST_LLM_CALL):
            key = ctx.tool_name if ctx.tool_name else "llm_call"
            start = self._starts.pop(key, 0)
            if start > 0:
                delta = time.time() - start
                self._timings.setdefault(key, []).append(delta)
        elif ctx.event == HookEvent.SESSION_END:
            ctx.metadata["profiling_summary"] = self.summary
        return ctx

    @property
    def summary(self) -> dict[str, dict]:
        result = {}
        for key, deltas in self._timings.items():
            result[key] = {
                "calls": len(deltas),
                "total": sum(deltas),
                "avg": sum(deltas) / len(deltas) if deltas else 0,
                "max": max(deltas) if deltas else 0,
            }
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What products does Acme Corp offer?",
    "How much does AcmeERP Enterprise cost?",
    "What is the remote work policy?",
    "Calculate the total for 10 AcmeCRM Professional users for one year",
]


def _find_telemetry_collector(agent):
    """Find the TelemetryCollector from the agent's hook chain."""
    hook = agent.core.hooks.find_hook(lambda h: hasattr(h, "telemetry") and hasattr(h.telemetry, "snapshot"))
    return hook.telemetry if hook else None


def _print_hooks(agent) -> None:
    """Print registered hooks via the public API."""
    hooks_info = agent.core.hooks.list_hooks()
    table = Table(title="Registered Hooks", show_header=True, header_style="bold cyan")
    table.add_column("Hook", style="cyan")
    table.add_column("Events", style="green", max_width=60)
    for info in hooks_info:
        table.add_row(info["name"], ", ".join(info["events"]))
    console.print(table)
    console.print()


def _print_agent_info(agent) -> None:
    """Print agent configuration details."""
    config = agent.config

    info_table = Table(title="Agent Configuration", show_header=True, header_style="bold cyan")
    info_table.add_column("Setting", style="cyan", width=25)
    info_table.add_column("Value", style="green")

    info_table.add_row("Name", config.agent_name)
    info_table.add_row("Provider", config.provider)
    info_table.add_row("Model", config.model)
    info_table.add_row("Max Iterations", str(config.max_iterations))

    builtin_tools = config.get("tools", "builtin", default=[])
    if builtin_tools:
        info_table.add_row("Builtin Tools", ", ".join(builtin_tools))

    ctx_strategy = config.get("context", "strategy", default="noop")
    info_table.add_row("Context Strategy", ctx_strategy)

    if config.rag_enabled:
        info_table.add_row("RAG", "enabled")

    input_grd = config.get("guardrails", "input", default={})
    output_grd = config.get("guardrails", "output", default={})
    rate_limit = config.get("guardrails", "rate_limit", default={})
    if input_grd:
        info_table.add_row("Input Guardrail", "active")
    if output_grd:
        info_table.add_row("Output Guardrail", "active")
    if rate_limit:
        info_table.add_row("Rate Limit", f"{rate_limit.get('max_calls_per_session', '-')}/session")

    harness_conf = config.get("harness", default={})
    harness_features = []
    for feat in ("telemetry", "carryover", "doom_loop"):
        if harness_conf.get(feat):
            harness_features.append(feat)
    if harness_features:
        info_table.add_row("Harness", ", ".join(harness_features))

    hooks_count = len(agent.core.hooks.list_hooks())
    info_table.add_row("Registered Hooks", str(hooks_count))

    console.print(info_table)
    console.print()


def _print_telemetry_line(agent, profiling: ProfilingHook | None = None) -> None:
    """Print a dim one-line telemetry summary."""
    telemetry = _find_telemetry_collector(agent)
    if telemetry is None:
        return

    snap = telemetry.snapshot
    parts = [
        f"iters={snap.total_iterations}",
        f"tools={snap.total_tool_calls}",
        f"health={telemetry.health_score():.0f}",
    ]
    if snap.unique_tools_used:
        parts.append(f"unique={','.join(sorted(snap.unique_tools_used))}")
    console.print(f"  [dim]telemetry: {' | '.join(parts)}[/dim]")

    if profiling and profiling.summary:
        prof_parts = [f"{k}: {v['avg']:.3f}s avg" for k, v in profiling.summary.items()]
        console.print(f"  [dim]profiling: {' | '.join(prof_parts)}[/dim]")
    console.print()


def run_automatic(agent, profiling: ProfilingHook):
    """Run 4 predefined questions with telemetry summary."""
    start = time.time()

    def _post_answer(result, q, i, total):
        _print_telemetry_line(agent, profiling)

    automatic_batch(agent, QUESTIONS, post_answer=_post_answer)

    # Final summary
    elapsed = time.time() - start
    telemetry = _find_telemetry_collector(agent)

    summary_lines = [
        f"Total Questions: {len(QUESTIONS)}",
        f"Duration: {elapsed:.1f}s",
    ]
    if telemetry:
        summary_lines.append(f"Health Score: {telemetry.health_score()}/100")
        summary_lines.append(f"Total Tool Calls: {telemetry.snapshot.total_tool_calls}")
        summary_lines.append(f"Doom Loops: {telemetry.snapshot.doom_loops_detected}")

    prof_summary = profiling.summary
    if prof_summary:
        summary_lines.append("Profiling:")
        for key, stats in prof_summary.items():
            summary_lines.append(f"  {key}: {stats['calls']} calls, {stats['avg']:.3f}s avg")

    console.print(Panel("\n".join(summary_lines), title="Session Summary", border_style="green"))


def run_interactive(agent, profiling: ProfilingHook):
    """Interactive chat with live telemetry and session summary."""
    start = time.time()
    messages = 0
    tools_used = set()

    def _post_receive(result, a):
        nonlocal messages
        messages += 1
        _print_telemetry_line(a, profiling)
        telemetry = _find_telemetry_collector(a)
        if telemetry:
            tools_used.update(telemetry.snapshot.unique_tools_used)

    extra_commands = {
        "info": lambda a: _print_agent_info(a),
        "hooks": lambda a: _print_hooks(a),
        "reset": lambda a: (
            a.reset(),
            console.print("[yellow]Conversation reset.[/yellow]\n"),
        ),
    }

    interactive_loop(
        agent,
        extra_commands=extra_commands,
        post_receive=_post_receive,
    )

    # Session summary on exit
    elapsed = time.time() - start
    summary_lines = [
        f"Messages: {messages}",
        f"Tools Used: {', '.join(sorted(tools_used)) or '-'}",
        f"Duration: {elapsed:.1f}s",
    ]
    telemetry = _find_telemetry_collector(agent)
    if telemetry:
        summary_lines.append(f"Health Score: {telemetry.health_score()}/100")
        summary_lines.append(f"Total Tool Calls: {telemetry.snapshot.total_tool_calls}")

    prof_summary = profiling.summary
    if prof_summary:
        summary_lines.append("Profiling:")
        for key, stats in prof_summary.items():
            summary_lines.append(f"  {key}: {stats['calls']} calls, {stats['avg']:.3f}s avg")

    console.print(Panel("\n".join(summary_lines), title="Session Summary", border_style="green"))


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 22: Full production agent with all features."""
    setup_example(
        "Example 22: Full Production Agent",
        "Complete production agent with RAG, guardrails, policy, telemetry,\n"
        "carryover, doom loop detection, and a custom ProfilingHook.\n\n"
        "[dim]Interactive commands: quit, reset, info, hooks[/dim]",
    )

    agent = create_agent("22_full_production", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]\n")

    # Add custom profiling hook to the chain
    profiling = ProfilingHook()
    agent.core.hooks.add(profiling)
    console.print("[dim]Added ProfilingHook to the hook chain[/dim]\n")

    # Show registered hooks
    _print_hooks(agent)

    if mode == "interactive":
        run_interactive(agent, profiling)
    else:
        run_automatic(agent, profiling)


if __name__ == "__main__":
    main()
