"""tests/test_dynamic_workflow.py -- WS4/WS5: execution.mode: dynamic.

Verifies the planner→graph→dag-with-flow path (complex) + the triage-skip path (simple).
Edge flow within the dynamic path is asserted by spying on _run_single's inputs.
"""

from __future__ import annotations

import json

from koboi.orchestration.orchestrator import Orchestrator
from koboi.types import AgentResponse, RoutingDecision
from tests.conftest import make_mock_response


class _AllRouter:
    def __init__(self, names):
        self._names = names

    async def route(self, q):
        return RoutingDecision(query=q, agents=list(self._names), confidence=1.0, method="keyword", reasoning="all")


PLAN_COMPLEX = json.dumps(
    {
        "needs_workflow": True,
        "reason": "multi-step",
        "steps": [
            {"id": "a", "instruction": "do step a", "depends_on": []},
            {"id": "b", "instruction": "do step b from a", "depends_on": ["a"]},
        ],
    }
)
PLAN_SIMPLE = json.dumps({"needs_workflow": False, "reason": "simple", "steps": []})


async def test_dynamic_complex_runs_planned_workflow_with_edge_flow(mock_client):
    # responses in call order: planner(plan) -> a -> b -> synthesis
    client = mock_client(
        responses=[
            AgentResponse(content=PLAN_COMPLEX),
            AgentResponse(content="a-output"),
            AgentResponse(content="b-output"),
            make_mock_response("synthesis"),
        ]
    )
    orch = Orchestrator(client=client, router=_AllRouter([]), agents_map={}, default_mode="dynamic")

    # Spy on _run_single to capture each node's input (verifies edge flow).
    inputs: dict = {}
    orig_run_single = orch._run_single

    async def spy(name, query):
        inputs[name] = query
        return await orig_run_single(name, query)

    orch._run_single = spy

    result = await orch.run("the task", mode="dynamic")

    assert result.execution_mode == "dynamic"
    assert len(result.agent_results) == 2
    assert {r.agent_name for r in result.agent_results} == {"a", "b"}
    # Edge flow: b's input contained a's output (the dynamic graph flows data).
    assert "a-output" in inputs["b"]


async def test_dynamic_simple_answers_directly(mock_client):
    # responses: planner(simple plan) -> direct answer
    client = mock_client(responses=[AgentResponse(content=PLAN_SIMPLE), make_mock_response("direct-answer")])
    orch = Orchestrator(client=client, router=_AllRouter([]), agents_map={}, default_mode="dynamic")

    result = await orch.run("what is 2+2", mode="dynamic")

    assert result.execution_mode == "dynamic"
    assert len(result.agent_results) == 1  # single direct agent, no workflow graph
    assert "direct" in result.final_answer or "direct" in result.agent_results[0].answer


async def test_dynamic_planner_fallback_answers_directly(mock_client):
    # Planner returns malformed -> plan_or_skip falls back to needs_workflow=False -> direct.
    client = mock_client(responses=[AgentResponse(content="not json"), make_mock_response("direct-answer")])
    orch = Orchestrator(client=client, router=_AllRouter([]), agents_map={}, default_mode="dynamic")

    result = await orch.run("anything", mode="dynamic")

    assert result.execution_mode == "dynamic"
    assert len(result.agent_results) == 1  # fell back to a single direct agent
