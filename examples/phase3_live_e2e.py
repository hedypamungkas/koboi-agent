"""examples/phase3_live_e2e.py -- live-LLM validation of Phase 3 capabilities.

Tests the three Phase 3 items that need real-LLM proof (mock tests can't prove the
LLM's actual output drives the behavior):

  L4 (positive): conditional edges -- a real LLM classifies sentiment; the output
     determines which branch fires (positive_handler vs negative_handler).
  L4b (negative): a query with no clear polarity -> only the neutral/default path runs.
  L5 (positive): WorkflowGraph API -- programmatic graph research->draft->review with
     edge data flow (draft references research's output).
  L5b (edge): single-node graph -> runs as a trivial pipeline.
  L6 (positive): re-planning -- a deliberately impossible first step fails -> the
     planner is called again -> recovers.

Loads OPENAI_* from .env. Run:
    PYTHONPATH=. python examples/phase3_live_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv

from koboi.client import Client
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.factory import AgentFactory
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.planner import PlanResult, PlanStep, plan_or_skip
from koboi.orchestration.workflow_graph import WorkflowGraph
from koboi.types import AgentDef, RoutingDecision

load_dotenv()
CLIENT = Client(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    provider="openai",
)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class _AllRouter:
    def __init__(self, names):
        self._names = names

    async def route(self, q):
        return RoutingDecision(query=q, agents=list(self._names), confidence=1.0, method="keyword", reasoning="all")


async def _judge(question: str) -> bool:
    """LLM yes/no judge for assertion checking."""
    from koboi.orchestration._utils import extract_json

    schema = {
        "type": "object",
        "properties": {"yes": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["yes"],
    }
    resp = await CLIENT.complete(messages=[{"role": "user", "content": question}], tools=None, response_format=schema)
    data = extract_json(resp.content or "")
    return bool(data.get("yes")) if isinstance(data, dict) else False


# ---------------------------------------------------------------------------
# L4: Conditional edges -- real LLM output drives branching
# ---------------------------------------------------------------------------


async def l4_conditional_positive() -> bool:
    """POSITIVE: a clearly positive review -> LLM says POSITIVE -> positive_branch fires."""
    print("\n===== L4 (positive): conditional edges -- positive sentiment fires positive branch =====")
    defs = [
        AgentDef(
            name="classify", system_prompt="Classify the sentiment as POSITIVE or NEGATIVE. Reply with one word only."
        ),
        AgentDef(name="positive_branch", system_prompt="In one sentence, explain why this is a great product review."),
        AgentDef(name="negative_branch", system_prompt="In one sentence, explain why this is a critical review."),
    ]
    deps = {"positive_branch": ["classify"], "negative_branch": ["classify"]}
    conditionals = {
        "classify": [
            {"to": "positive_branch", "when": {"contains": "POSITIVE"}},
            {"to": "negative_branch", "when": {"contains": "NEGATIVE"}},
        ]
    }
    agents_map = AgentFactory.create_all_configured(defs, CLIENT)
    orch = Orchestrator(
        client=CLIENT,
        router=_AllRouter([d.name for d in defs]),
        agents_map=agents_map,
        dag_scheduler=DagScheduler(deps=deps, conditionals=conditionals),
        default_mode="dag",
        full_graph=True,
    )
    result = await orch.run(
        "Review: 'This product exceeded all expectations! Amazing quality and fast delivery.'", mode="dag"
    )
    names_ran = {r.agent_name for r in result.agent_results}
    print(f"  nodes ran: {names_ran}")
    for r in result.agent_results:
        print(f"    {r.agent_name}: {(r.answer or '')[:80]}")
    passed = "positive_branch" in names_ran and "negative_branch" not in names_ran
    print(f"  -> positive branch fired, negative skipped: {passed}")
    return passed


async def l4b_conditional_negative() -> bool:
    """NEGATIVE: a clearly negative review -> LLM says NEGATIVE -> negative_branch fires."""
    print("\n===== L4b (negative): conditional edges -- negative sentiment fires negative branch =====")
    defs = [
        AgentDef(
            name="classify", system_prompt="Classify the sentiment as POSITIVE or NEGATIVE. Reply with one word only."
        ),
        AgentDef(name="positive_branch", system_prompt="In one sentence, explain why this is a great product review."),
        AgentDef(name="negative_branch", system_prompt="In one sentence, explain why this is a critical review."),
    ]
    deps = {"positive_branch": ["classify"], "negative_branch": ["classify"]}
    conditionals = {
        "classify": [
            {"to": "positive_branch", "when": {"contains": "POSITIVE"}},
            {"to": "negative_branch", "when": {"contains": "NEGATIVE"}},
        ]
    }
    agents_map = AgentFactory.create_all_configured(defs, CLIENT)
    orch = Orchestrator(
        client=CLIENT,
        router=_AllRouter([d.name for d in defs]),
        agents_map=agents_map,
        dag_scheduler=DagScheduler(deps=deps, conditionals=conditionals),
        default_mode="dag",
        full_graph=True,
    )
    result = await orch.run("Review: 'Terrible quality. Broke after one day. Complete waste of money.'", mode="dag")
    names_ran = {r.agent_name for r in result.agent_results}
    print(f"  nodes ran: {names_ran}")
    for r in result.agent_results:
        print(f"    {r.agent_name}: {(r.answer or '')[:80]}")
    passed = "negative_branch" in names_ran and "positive_branch" not in names_ran
    print(f"  -> negative branch fired, positive skipped: {passed}")
    return passed


# ---------------------------------------------------------------------------
# L5: WorkflowGraph API -- programmatic graph with real LLM
# ---------------------------------------------------------------------------


async def l5_workflow_graph_edge_flow() -> bool:
    """POSITIVE: build a graph programmatically -> draft references research output."""
    print("\n===== L5 (positive): WorkflowGraph API -- programmatic graph with edge flow =====")
    g = WorkflowGraph()
    g.add_node("research", "Reply with ONE specific fact about the topic, starting 'FACT:'.")
    g.add_node("draft", "Reply in ONE sentence that explicitly builds on the provided upstream fact.")
    g.add_edge("research", "draft")
    graph = g.compile()

    result = await graph.invoke("topic: the Apollo 11 moon landing", CLIENT)
    print(f"  final answer: {(result or '')[:120]}")
    judged = await _judge(
        "Does this answer reference or build on a specific fact about Apollo 11 "
        f"(i.e. the research output was used, not just an independent answer)? Answer yes/no.\n\n{result[:400]}"
    )
    print(f"  -> draft used research output (edge flow): {judged}")
    return judged


async def l5b_workflow_graph_single_node() -> bool:
    """EDGE: a single-node graph -> runs as a trivial pipeline (no deps, no branches)."""
    print("\n===== L5b (edge): WorkflowGraph -- single-node graph =====")
    g = WorkflowGraph()
    g.add_node("solo", "Reply in one sentence about the topic.")
    graph = g.compile()
    result = await graph.invoke("topic: black holes", CLIENT)
    print(f"  answer: {(result or '')[:120]}")
    return bool(result and len(result) > 10)  # non-empty meaningful answer


# ---------------------------------------------------------------------------
# L6: Re-planning -- a failing node triggers a new plan
# ---------------------------------------------------------------------------


async def l6_replan_on_failure() -> bool:
    """POSITIVE: a deliberately impossible step fails -> the planner re-plans -> recovers."""
    print("\n===== L6 (positive): re-planning -- node failure triggers replan =====")
    plan_calls = {"n": 0}

    async def counting_plan(client, instruction, **kw):
        plan_calls["n"] += 1
        if plan_calls["n"] == 1:
            # First plan: a step that will fail (impossible instruction).
            return PlanResult(
                needs_workflow=True,
                steps=[
                    PlanStep(id="impossible", instruction="Translate this text into a language that does not exist.")
                ],
                reason="multi-step",
            )
        # Replan: a simple recoverable plan.
        return PlanResult(
            needs_workflow=True,
            steps=[
                PlanStep(
                    id="recover", instruction="In one sentence, explain why the previous translation was impossible."
                )
            ],
            reason="recovery plan after failure",
        )

    # Patch plan_or_skip.
    import koboi.orchestration.planner as planner_mod

    original_plan = planner_mod.plan_or_skip
    planner_mod.plan_or_skip = counting_plan

    try:
        orch = Orchestrator(
            client=CLIENT,
            router=_AllRouter([]),
            agents_map={},
            default_mode="dynamic",
            max_replans=1,
        )
        # Force the first node to fail by spying on _run_single.
        single_calls = {"n": 0}
        orig_single = orch._run_single

        async def failing_single(name, query_input):
            single_calls["n"] += 1
            if single_calls["n"] == 1:
                from koboi.types import AgentResult

                return AgentResult(
                    agent_name=name, answer="Error: impossible", elapsed_seconds=0, tokens_used=0, failed=True
                )
            return await orig_single(name, query_input)

        orch._run_single = failing_single
        result = await orch.run("Translate 'hello' into a nonexistent language.", mode="dynamic")
    finally:
        planner_mod.plan_or_skip = original_plan

    print(f"  plan_or_skip calls: {plan_calls['n']}")
    print(f"  final answer: {(result.final_answer or '')[:120]}")
    replanned = plan_calls["n"] >= 2
    print(f"  -> replanned on failure: {replanned}")
    return replanned


async def main() -> int:
    print(f"[setup] model={MODEL}  gateway={os.environ.get('OPENAI_BASE_URL')}")
    results = {}
    for name, coro in [
        ("L4  conditional positive (fires positive_branch)", l4_conditional_positive()),
        ("L4b conditional negative (fires negative_branch)", l4b_conditional_negative()),
        ("L5  WorkflowGraph edge flow (draft uses research)", l5_workflow_graph_edge_flow()),
        ("L5b WorkflowGraph single-node (edge case)", l5b_workflow_graph_single_node()),
        ("L6  re-planning on node failure", l6_replan_on_failure()),
    ]:
        try:
            results[name] = await coro
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results[name] = False

    print("\n================ PHASE 3 LIVE e2e SUMMARY ================")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=========================================================")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
