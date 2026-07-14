"""Example 35: Confidence-aware CS with human handover.

Demonstrates the full confidence ladder (Wave 1+2+3: A1-A3, B1/B1.5, B4) in a
real-world customer-service scenario:

  Case 1 (+ answerable):    RAG retrieves → bot answers "12 days"
  Case 2 (− abstain):       OOS query → empty retrieval → A2 marker → bot refuses
  Case 3 (edge handover):   "speak to a human" → B1.5 structural (LLM NOT called)
  Case 4 (edge handover):   complex case → B1 tool (transfer_to_human)
  Case 5 (+ answerable):    normal operation resumes

Run:
    python examples/35_confidence_handover_demo.py --mock     # deterministic (no API key)
    python examples/35_confidence_handover_demo.py            # live (needs OPENAI_API_KEY)

In mock mode the retriever runs for REAL (BM25 over the Acme corpus) so the
retrieval_confidence + A2 abstention marker are genuine signals; only the LLM text
is scripted. A3's faithfulness judge + B4's digest need a live judge → pass-through
in mock (documented). B1 + B1.5 fire for real (they don't need a judge).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from koboi.exceptions import AgentHandoverError
from koboi.facade import KoboiAgent
from koboi.types import AgentResponse, TokenUsage, ToolCall

load_dotenv(PROJECT_ROOT / ".env")
console = Console()
CONFIG_PATH = PROJECT_ROOT / "configs" / "cs_confidence_handover.yaml"

# The 5 demo queries (+ / − / edge / edge / +).
QUERIES = [
    ("How many days of annual leave for permanent employees?", "+ answer"),
    ("What is the mating ritual of deep-sea anglerfish?", "− abstain (empty retrieval)"),
    ("I want to speak to a human agent please.", "edge handover (B1.5 structural)"),
    ("I need a refund for order #1234 but it's been 45 days and the item is damaged.", "edge handover (B1 tool)"),
    ("What are the working hours?", "+ answer"),
]


# ---------------------------------------------------------------------------
# Inline mock client (deterministic; no API key needed)
# ---------------------------------------------------------------------------


class _MockClient:
    """Returns scripted AgentResponses in order. Mirrors tests/conftest.MockClient."""

    def __init__(self, responses: list[AgentResponse]):
        self._responses = responses
        self._index = 0
        self.call_count = 0

    @property
    def model(self) -> str:
        return "mock-model"

    async def complete(self, messages, tools=None, response_format=None):
        self.call_count += 1
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        return AgentResponse(content="No more responses", tool_calls=[], usage=TokenUsage(0, 0))

    async def complete_stream(self, messages, tools=None, response_format=None):
        from koboi.events import CompleteEvent, TextDeltaEvent

        resp = await self.complete(messages, tools, response_format)
        if resp.content:
            yield TextDeltaEvent(content=resp.content)
        yield CompleteEvent(response=resp, content=resp.content or "")

    async def get_embeddings(self, text):
        return None

    async def close(self):
        pass


def _mock_responses() -> list[AgentResponse]:
    """The 4 scripted responses (case 3 doesn't reach the LLM — B1.5 fires first)."""
    return [
        # Case 1: answerable → grounded answer.
        AgentResponse(
            content="Permanent employees receive 12 days of annual leave per year.",
            tool_calls=[],
            usage=TokenUsage(10, 20),
        ),
        # Case 2: OOS → refusal (A2 marker cued the model to refuse).
        AgentResponse(
            content="I don't have information about deep-sea anglerfish in the provided context.",
            tool_calls=[],
            usage=TokenUsage(10, 20),
        ),
        # Case 3: SKIPPED — B1.5 fires at PRE_INPUT before the LLM is called.
        # Case 4: complex → LLM calls transfer_to_human.
        AgentResponse(
            content="Let me transfer you to a specialist who can help with this.",
            tool_calls=[
                ToolCall(
                    id="tc_handover",
                    name="transfer_to_human",
                    arguments=json.dumps(
                        {"reason": "refund out-of-window + damaged item", "summary": "Cust wants refund for #1234, 45 days past window, item damaged"}
                    ),
                )
            ],
            usage=TokenUsage(10, 20),
        ),
        # Case 5: answerable → grounded answer.
        AgentResponse(
            content="Working hours are Monday to Friday, 08:00 to 17:00.",
            tool_calls=[],
            usage=TokenUsage(10, 20),
        ),
    ]


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


async def run_demo(mock: bool) -> None:
    """Build the agent + run the 5 queries, pretty-printing each result."""
    console.print(
        Panel(
            f"[bold]Confidence-Aware CS with Handover[/bold]\nMode: {'mock (deterministic)' if mock else 'live (real LLM)'}",
            style="cyan",
        )
    )

    if mock:
        import os
        os.environ.setdefault("OPENAI_API_KEY", "dummy")
    agent = KoboiAgent.from_config(str(CONFIG_PATH))

    if mock:
        agent._core.client = _MockClient(_mock_responses())  # type: ignore[attr-defined]
        console.print("[dim]Mock client injected. Retriever runs for real (BM25 over Acme corpus).[/dim]\n")

    results_table = Table(title="Confidence Ladder", show_header=True, header_style="bold")
    results_table.add_column("#", width=3)
    results_table.add_column("Case", width=28)
    results_table.add_column("Status", width=16)
    results_table.add_column("Detail")

    for i, (query, label) in enumerate(QUERIES, 1):
        console.print(f"\n[bold cyan]═══ Query {i}/5: {query} ═══[/bold cyan]")
        console.print(f"[dim]Case: {label}[/dim]")
        try:
            result = await agent.run(query)
            rc = (result.metadata or {}).get("retrieval_confidence", {})
            guardrail = (result.metadata or {}).get("guardrail_outcomes")
            count = rc.get("count", "?") if isinstance(rc, dict) else "?"
            method = rc.get("method", "?") if isinstance(rc, dict) else "?"
            console.print(f"  [dim][retrieval_confidence][/dim] count={count} method={method}")
            if guardrail:
                console.print(f"  [dim][guardrail][/dim] {guardrail[0].get('action', '?')}: {guardrail[0].get('reason', '')[:80]}")
            # Classify the outcome.
            text = result.content.strip()
            if count == 0:
                status, emoji = "ABSTAINED", "⚠️ "
            elif guardrail and guardrail[0].get("action") == "abstain":
                status, emoji = "ABSTAINED (A3)", "⚠️ "
            else:
                status, emoji = "ANSWERED", "✅ "
            console.print(f"  {emoji}[bold]{status}[/bold]")
            console.print(f"  [green]{text[:200]}[/green]")
            results_table.add_row(str(i), label, f"{emoji} {status}", text[:80])
        except AgentHandoverError as he:
            source = "B1.5 structural" if "user requested" in he.reason else "B1 tool"
            console.print(f"  [dim][handover][/dim] {he.reason}")
            console.print(f"  🔁 [bold yellow]HANDED OVER[/bold yellow] ({source})")
            results_table.add_row(str(i), label, "🔁 HANDED OVER", f"{source}: {he.reason[:60]}")
        except Exception as exc:
            console.print(f"  [red]ERROR: {exc}[/red]")
            results_table.add_row(str(i), label, "❌ ERROR", str(exc)[:80])

    console.print()
    console.print(results_table)
    if mock:
        console.print(
            "\n[dim]Note: A3 (grounding NLI) + B4 (warm digest) need a live judge LLM → "
            "pass-through in mock. Run without --mock to see the full stack fire.[/dim]"
        )


@click.command()
@click.option("--mock", is_flag=True, help="Run with scripted mock LLM (no API key needed).")
def main(mock: bool):
    """Example 35: Confidence-aware CS with handover — the confidence ladder visualized."""
    asyncio.run(run_demo(mock=mock))


if __name__ == "__main__":
    main()
