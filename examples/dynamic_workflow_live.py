"""examples/dynamic_workflow_live.py -- live-LLM e2e for dynamic + dag workflows.

Validates the dynamic-workflow feature end-to-end with a real LLM:
  L1 (positive): a complex multi-step instruction -> dynamic mode -> the LLM plans a
     graph + executes it with edge data flow (downstream nodes reference upstream).
  L2 (negative): a simple query -> dynamic mode -> triage SKIPS the planner -> direct.
  L3 (positive): static dag mode (now with edge flow) -> draft references research.

Loads OPENAI_* from .env (load_dotenv). Run:
    python examples/dynamic_workflow_live.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from koboi.client import Client
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.factory import AgentFactory
from koboi.orchestration.orchestrator import Orchestrator
from koboi.types import AgentDef, RoutingDecision

load_dotenv()
CLIENT = Client(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    provider="openai",
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"yes": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["yes"],
}


class _AllRouter:
    def __init__(self, names):
        self._names = names

    async def route(self, q):
        return RoutingDecision(query=q, agents=list(self._names), confidence=1.0, method="keyword", reasoning="all")


async def _judge(question: str) -> bool:
    from koboi.orchestration._utils import extract_json

    resp = await CLIENT.complete(
        messages=[{"role": "user", "content": question}], tools=None, response_format=JUDGE_SCHEMA
    )
    data = extract_json(resp.content or "")
    return bool(data.get("yes")) if isinstance(data, dict) else False


BATTERY = (
    "I want a short report on solid-state batteries. First note one key advantage, "
    "then write a one-sentence summary. From that summary, BOTH translate it to "
    "Indonesian AND write a catchy social-media post. Then combine them into a final line."
)


async def l1_dynamic_complex() -> bool:
    print("\n===== L1 (positive): dynamic mode, complex multi-step instruction =====")
    orch = Orchestrator(client=CLIENT, router=_AllRouter([]), agents_map={}, default_mode="dynamic")
    result = await orch.run(BATTERY, mode="dynamic")
    nodes = [(r.agent_name, (r.answer or "")[:60]) for r in result.agent_results]
    print(f"  planned nodes: {[n for n, _ in nodes]}")
    for n, a in nodes:
        print(f"    {n}: {a}")
    judged = await _judge(
        "Does this final answer reflect a coherent multi-step result that synthesizes "
        f"research, a summary, a translation, and a social post? Answer yes/no.\n\n{result.final_answer[:600]}"
    )
    print(f"  -> multi-step synthesis coherent? {judged} (nodes={len(nodes)})")
    return judged and len(nodes) > 1


async def l2_dynamic_simple() -> bool:
    print("\n===== L2 (negative): dynamic mode, simple query -> triage skips planner =====")
    orch = Orchestrator(client=CLIENT, router=_AllRouter([]), agents_map={}, default_mode="dynamic")
    result = await orch.run("What is the capital of France?", mode="dynamic")
    n = len(result.agent_results)
    print(f"  nodes run: {n}  (expected 1 = direct, no workflow)")
    print(f"  answer: {(result.final_answer or '')[:120]}")
    return n == 1  # triage skipped the planner -> single direct agent


async def l3_static_dag_edge_flow() -> bool:
    print("\n===== L3 (positive): static dag mode with edge data flow =====")
    defs = [
        AgentDef(name="research", system_prompt="Reply with ONE concrete fact about the topic, starting 'FACT:'."),
        AgentDef(
            name="draft",
            system_prompt="Reply in ONE sentence that builds on the provided upstream fact.",
            depends_on=["research"],
        ),
    ]
    agents_map = AgentFactory.create_all_configured(defs, CLIENT)
    orch = Orchestrator(
        client=CLIENT,
        router=_AllRouter([d.name for d in defs]),
        agents_map=agents_map,
        dag_scheduler=DagScheduler(deps={"draft": ["research"]}),
        default_mode="dag",
    )
    result = await orch.run("topic: the James Webb Space Telescope", mode="dag")
    by = {r.agent_name: r.answer for r in result.agent_results}
    print(f"  research: {(by.get('research') or '')[:100]}")
    print(f"  draft:    {(by.get('draft') or '')[:100]}")
    judged = await _judge(
        "Does the DRAFT build on / reference the RESEARCH fact (i.e. the draft used the "
        f"upstream output, not just answered independently)? Answer yes/no.\n\nRESEARCH: {by.get('research', '')}\nDRAFT: {by.get('draft', '')}"
    )
    print(f"  -> draft used research output? {judged}")
    return judged


async def main() -> int:
    print(f"[setup] model={os.environ.get('OPENAI_MODEL')} gateway={os.environ.get('OPENAI_BASE_URL')}")
    results = {
        "L1 dynamic complex (multi-step + edge flow)": await l1_dynamic_complex(),
        "L2 dynamic simple (triage skips planner)": await l2_dynamic_simple(),
        "L3 static dag edge flow (draft uses research)": await l3_static_dag_edge_flow(),
    }
    print("\n================ LIVE e2e SUMMARY ================")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("==================================================")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
