"""examples/38_self_healing_demo.py -- self-healing initiative demo (P0-P4).

Demonstrates the full self-healing stack:
  - P0-C: empty-response re-ask (always-on)
  - P0-D: structured tool-error feedback (always-on)
  - P3:   graceful degrade on max_iterations (opt-in)
  - P1/P2a/P4: reflection + escalation ladder + CRITIC (real-LLM mode only --
               these require the grounding guardrail to fire, which needs a real API)

Two modes:
  --mock   : offline scripted demo (no API key needed). Showcases P0-C/P0-D/P3.
  (default): real-LLM mode. Showcases the full stack (P1 reflection, P2a ladder,
             P4 CRITIC). Requires OPENAI_API_KEY + RAG docs.

Usage:
  python examples/38_self_healing_demo.py --mock
  OPENAI_API_KEY=... python examples/38_self_healing_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from koboi.facade import KoboiAgent
from koboi.exceptions import AgentHandoverError, AgentMaxIterationsError
from koboi.types import AgentResponse, TokenUsage

CONFIG_PATH = PROJECT_ROOT / "configs" / "self_healing_demo.yaml"
console = Console()


class _MockClient:
    """Scripted mock LLM client for offline demo."""

    def __init__(self, responses: list[AgentResponse]):
        self._responses = responses
        self._index = 0
        self._model = "mock-model"
        self.call_count = 0

    @property
    def model(self):
        return self._model

    async def complete(self, messages, tools=None, response_format=None):
        self.call_count += 1
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        return AgentResponse(content="No more responses", tool_calls=[])

    async def complete_stream(self, messages, tools=None, response_format=None):
        from koboi.events import TextDeltaEvent, CompleteEvent

        resp = await self.complete(messages, tools, response_format=response_format)
        if resp.content:
            yield TextDeltaEvent(content=resp.content)
        yield CompleteEvent(response=resp, content=resp.content or "")

    async def get_embeddings(self, text):
        return None

    async def close(self):
        pass


def _make_response(content=None, tool_calls=None):
    return AgentResponse(
        content=content, tool_calls=tool_calls or [], usage=TokenUsage(prompt_tokens=10, completion_tokens=20)
    )


def _print_result(title: str, result):
    """Print a run result with self-healing metadata."""
    table = Table(title=title, show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Content", (result.content or "")[:200])
    table.add_row("Success", str(result.success))
    table.add_row("Iterations", str(result.iterations_used))
    meta = result.metadata or {}
    table.add_row("Reflection retries", str(meta.get("reflection_retries", 0)))
    table.add_row("Empty re-asks", str(meta.get("empty_response_reasked", 0)))
    table.add_row("Self-consistency", str(meta.get("self_consistency")))
    table.add_row("Max-iter degraded", str(meta.get("max_iter_degraded", False)))
    table.add_row("Guardrail outcomes", str(meta.get("guardrail_outcomes")))
    console.print(table)


async def run_mock_scenarios(agent: KoboiAgent):
    """Offline scripted scenarios (no real API needed)."""

    # --- Scenario 1: P0-C empty-response re-ask ---
    console.print(
        Panel(
            "[bold]Scenario 1: Empty-Response Re-Ask (P0-C)[/bold]\nThe agent returns an empty first response → nudged → re-asks."
        )
    )
    agent._core.client = _MockClient([_make_response(None), _make_response("Hello! I'm here to help.")])
    result = await agent.run("Hi")
    _print_result("Empty Re-Ask Result", result)
    console.print()

    # --- Scenario 2: P3 graceful degrade ---
    console.print(
        Panel(
            "[bold]Scenario 2: Graceful Degrade (P3)[/bold]\nThe agent loops until max_iterations → graceful summary."
        )
    )
    from koboi.types import ToolCall

    tc = ToolCall(id="tc1", name="calculate", arguments='{"expression": "1+1"}')
    agent._core.client = _MockClient([_make_response(None, [tc])] * 10 + [_make_response("Partial work summary.")])
    try:
        result = await agent.run("Do a very complex task")
        _print_result("Graceful Degrade Result", result)
    except AgentMaxIterationsError:
        console.print("[yellow]AgentMaxIterationsError (graceful_max_iter not wired in this run)[/yellow]")
    console.print()

    # --- Scenario 3: P0-D structured tool-error ---
    console.print(
        Panel(
            "[bold]Scenario 3: Structured Tool-Error Feedback (P0-D)[/bold]\nA tool error produces an actionable, structured message."
        )
    )
    from koboi.tools.registry import ToolRegistry

    reg = ToolRegistry()

    def _boom():
        raise ValueError("kaboom")

    reg.register("boom", "boom", {"type": "object", "properties": {}}, _boom)
    tc_err = ToolCall(id="tc2", name="boom", arguments="{}")
    agent._core.tools = reg
    agent._core.client = _MockClient([_make_response(None, [tc_err]), _make_response("Handled the error.")])
    result = await agent.run("Use the boom tool")
    _print_result("Tool-Error Result", result)
    console.print()


async def run_real_llm_scenarios(agent: KoboiAgent):
    """Real-LLM scenarios (requires API key + RAG docs)."""

    queries = [
        "What is the company's return policy?",
        "How many vacation days do employees get?",
        "What products does Acme Corp sell?",
    ]

    for query in queries:
        console.print(Panel(f"[bold]Query:[/bold] {query}"))
        try:
            result = await agent.run(query)
            _print_result(f"Result: {query}", result)
        except AgentHandoverError as he:
            console.print(f"[red]Handover triggered:[/red] {he.reason}")
            if he.summary:
                console.print(f"  Summary: {he.summary[:200]}")
        except AgentMaxIterationsError:
            console.print("[yellow]Max iterations reached (graceful degrade may or may not be active)[/yellow]")
        console.print()


@click.command()
@click.option("--mock", is_flag=True, help="Run offline scripted demo (no API key needed).")
def main(mock: bool):
    """Self-healing initiative demo."""
    if mock:
        os.environ.setdefault("OPENAI_API_KEY", "dummy")

    console.print(
        Panel.fit(
            "[bold]Self-Healing Initiative Demo (P0-P4)[/bold]\n"
            "Mode: " + ("MOCK (offline)" if mock else "REAL LLM") + "\n"
            "Config: " + str(CONFIG_PATH),
            border_style="blue",
        )
    )

    agent = KoboiAgent.from_config(str(CONFIG_PATH))

    if mock:
        asyncio.run(run_mock_scenarios(agent))
    else:
        asyncio.run(run_real_llm_scenarios(agent))


if __name__ == "__main__":
    main()
