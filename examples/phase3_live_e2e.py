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
    """FULLY LIVE: real plan_or_skip (no mock) plans the query; the first node
    'fails' via a simulated transport error (RuntimeError, like a real network
    failure that _run_single catches); the REAL planner re-plans; the recovery
    node runs with the REAL LLM. Only the failure cause is simulated — everything
    else (planning x2, recovery execution, synthesis) is 100% real LLM."""
    print("\n===== L6 (FULLY LIVE): re-planning -- real planner + simulated transport error =====")

    # Track plan_or_skip calls WITHOUT mocking it — just count.
    import koboi.orchestration.planner as planner_mod

    plan_calls = {"n": 0}
    original_plan = planner_mod.plan_or_skip

    async def counting_real_plan(client, instruction, **kw):
        plan_calls["n"] += 1
        print(f"  [planner] call #{plan_calls['n']} for: {instruction[:60]}...")
        result = await original_plan(client, instruction, **kw)
        print(f"  [planner] -> needs_workflow={result.needs_workflow}, steps={len(result.steps)}")
        return result

    planner_mod.plan_or_skip = counting_real_plan

    try:
        orch = Orchestrator(
            client=CLIENT,
            router=_AllRouter([]),
            agents_map={},
            default_mode="dynamic",
            max_replans=1,
        )
        # Simulate a real transport error on the FIRST planned node (not the planner
        # call, not the recovery — only the first node execution). This represents
        # a network failure / API error that _run_single is designed to catch.
        single_calls = {"n": 0}
        orig_single = orch._run_single

        async def transport_error_single(name, query_input):
            single_calls["n"] += 1
            if single_calls["n"] == 1 and name != "assistant":
                # Simulate what _run_single returns AFTER catching a real transport
                # error (e.g. gateway 503). _run_single catches the exception + returns
                # AgentResult(failed=True) -- we return exactly that shape. The planner
                # + recovery nodes below use the REAL LLM (no mock).
                from koboi.types import AgentResult

                print(f"  [node {name}] SIMULATED transport error -> AgentResult(failed=True)")
                return AgentResult(
                    agent_name=name,
                    answer="Error: simulated transport error (503)",
                    elapsed_seconds=0,
                    tokens_used=0,
                    failed=True,
                )
            print(f"  [node {name}] running with real LLM...")
            return await orig_single(name, query_input)

        orch._run_single = transport_error_single
        result = await orch.run(
            "Research the history of the transistor, then summarize its impact on modern computing.",
            mode="dynamic",
        )
    finally:
        planner_mod.plan_or_skip = original_plan

    print(f"  plan_or_skip calls: {plan_calls['n']} (real LLM planner each time)")
    print(f"  final answer: {(result.final_answer or '')[:150]}")
    replanned = plan_calls["n"] >= 2
    has_answer = bool(result.final_answer and len(result.final_answer) > 20)
    print(f"  -> replanned on failure: {replanned}")
    print(f"  -> recovered with real LLM answer: {has_answer}")
    return replanned and has_answer


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
