"""tests/test_dynamic_workflow.py -- WS4/WS5: execution.mode: dynamic.

Verifies the planner→graph→dag-with-flow path (complex) + the triage-skip path (simple).
Edge flow within the dynamic path is asserted by spying on _run_single's inputs.
"""

from __future__ import annotations

import json

from koboi.hooks.chain import Hook, HookChain, HookEvent
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


async def test_dynamic_agents_receive_hook_chain(mock_client):
    """#5: dynamic-mode agents get the parent hook_chain (was missing in WS4)."""
    fired: list = []

    class _RecordingHook(Hook):
        def handles(self):
            return [HookEvent.PRE_LLM_CALL]

        async def execute(self, ctx):
            fired.append(ctx.event)
            return ctx

    chain = HookChain([_RecordingHook()])
    client = mock_client(
        responses=[
            AgentResponse(content=PLAN_SIMPLE),  # planner -> simple
            make_mock_response("direct-answer"),  # assistant run
        ]
    )
    orch = Orchestrator(
        client=client,
        router=_AllRouter([]),
        agents_map={},
        default_mode="dynamic",
        hook_chain=chain,
    )

    await orch.run("hi", mode="dynamic")

    # The hook fired -> the dynamic "assistant" agent received + used the hook_chain.
    # Without the #5 fix (empty HookChain), fired would be empty.
    assert len(fired) > 0


async def test_dynamic_replans_on_node_failure(mock_client, monkeypatch):
    """#3: if a node fails, the planner is called again (bounded by max_replans)."""
    from koboi.orchestration import planner
    from koboi.orchestration.planner import PlanResult, PlanStep
    from koboi.types import AgentResult

    plan_calls = {"n": 0}

    async def counting_plan(client, instruction, **kw):
        plan_calls["n"] += 1
        if plan_calls["n"] == 1:
            return PlanResult(needs_workflow=True, steps=[PlanStep(id="x", instruction="do x")], reason="multi")
        return PlanResult(needs_workflow=False, reason="give up after failure", steps=[])

    monkeypatch.setattr(planner, "plan_or_skip", counting_plan)

    client = mock_client(responses=[make_mock_response("recovered")])
    orch = Orchestrator(
        client=client,
        router=_AllRouter([]),
        agents_map={},
        default_mode="dynamic",
        max_replans=1,
    )

    single_calls = {"n": 0}
    orig = orch._run_single

    async def spy_single(name, query_input):
        single_calls["n"] += 1
        if single_calls["n"] == 1:  # first node ("x") fails
            return AgentResult(agent_name=name, answer="Error", elapsed_seconds=0, tokens_used=0, failed=True)
        return await orig(name, query_input)

    orch._run_single = spy_single

    result = await orch.run("task", mode="dynamic")

    assert plan_calls["n"] == 2  # initial plan + replan on failure
    assert len(result.agent_results) == 1  # fell back to direct after replan


async def test_dynamic_no_replan_when_max_is_zero(mock_client, monkeypatch):
    """Edge: max_replans=0 (default) -> node fails but NO replan (called once)."""
    from koboi.orchestration import planner
    from koboi.orchestration.planner import PlanResult, PlanStep
    from koboi.types import AgentResult

    plan_calls = {"n": 0}

    async def counting_plan(client, instruction, **kw):
        plan_calls["n"] += 1
        return PlanResult(needs_workflow=True, steps=[PlanStep(id="x", instruction="do x")], reason="multi")

    monkeypatch.setattr(planner, "plan_or_skip", counting_plan)

    client = mock_client(responses=[make_mock_response("recovered")])
    orch = Orchestrator(
        client=client,
        router=_AllRouter([]),
        agents_map={},
        default_mode="dynamic",
        max_replans=0,  # default — no replanning
    )

    single_calls = {"n": 0}
    orig = orch._run_single

    async def failing_single(name, query_input):
        single_calls["n"] += 1
        if single_calls["n"] == 1:
            return AgentResult(agent_name=name, answer="Error", elapsed_seconds=0, tokens_used=0, failed=True)
        return await orig(name, query_input)

    orch._run_single = failing_single
    result = await orch.run("task", mode="dynamic")

    assert plan_calls["n"] == 1  # NO replan (max_replans=0)
