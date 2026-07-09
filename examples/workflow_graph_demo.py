"""examples/workflow_graph_demo.py -- programmatic workflow graph (#7).

Shows the WorkflowGraph ergonomic API for building + running a workflow graph in
pure Python (no YAML, no LLM planner). Delegates to the DagScheduler + Orchestrator
for execution with edge data flow + conditional branching.

Usage (needs an LLM client):
    python examples/workflow_graph_demo.py

Loads OPENAI_* from .env.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from koboi.client import Client
from koboi.orchestration.workflow_graph import WorkflowGraph

load_dotenv()


async def main() -> int:
    client = Client(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        provider="openai",
    )
    print(f"[setup] model={os.environ.get('OPENAI_MODEL')}")

    # --- Example 1: Linear pipeline with edge data flow ---
    print("\n===== Example 1: Linear pipeline (research -> draft -> review) =====")
    g1 = WorkflowGraph()
    g1.add_node("research", "Reply with ONE fact about the topic, starting 'FACT:'.")
    g1.add_node("draft", "Reply in ONE sentence that builds on the upstream fact.")
    g1.add_node("review", "Reply in ONE sentence reviewing the draft.")
    g1.add_edge("research", "draft")
    g1.add_edge("draft", "review")

    graph1 = g1.compile()
    result1 = await graph1.invoke("topic: the invention of the printing press", client)
    print(f"  result: {(result1 or '')[:150]}")

    # --- Example 2: Conditional branching ---
    print("\n===== Example 2: Conditional branching (classify -> praise OR critique) =====")
    g2 = WorkflowGraph()
    g2.add_node("classify", "Classify the sentiment as POSITIVE or NEGATIVE. Reply with one word only.")
    g2.add_node("praise", "In one sentence, explain why this is a great review.")
    g2.add_node("critique", "In one sentence, explain why this is a critical review.")
    g2.add_conditional_edges("classify", {"POSITIVE": "praise", "NEGATIVE": "critique"})

    graph2 = g2.compile()
    result2 = await graph2.invoke("Review: 'This product exceeded all expectations! Amazing quality.'", client)
    print(f"  result: {(result2 or '')[:150]}")

    # --- Example 3: Diamond with parallel middle wave ---
    print("\n===== Example 3: Diamond (research -> {draft_a, draft_b} -> combine) =====")
    g3 = WorkflowGraph()
    g3.add_node("research", "Reply with ONE fact about the topic.")
    g3.add_node("draft_a", "Reply in ONE sentence drafting angle A.")
    g3.add_node("draft_b", "Reply in ONE sentence drafting angle B.")
    g3.add_node("combine", "Reply in ONE sentence combining the two drafts.")
    g3.add_edge("research", "draft_a")
    g3.add_edge("research", "draft_b")
    g3.add_edge("draft_a", "combine")
    g3.add_edge("draft_b", "combine")

    graph3 = g3.compile()
    result3 = await graph3.invoke("topic: renewable energy", client)
    print(f"  result: {(result3 or '')[:150]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
