"""tests/test_dag_edge_flow.py -- WS1/WS2: edge data flow between DAG nodes.

A downstream node's INPUT must contain its dependencies' OUTPUTS (previously every
node got the raw query -> combine never saw research/drafts).
"""

from __future__ import annotations

from types import SimpleNamespace

from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.types import RoutingDecision
from tests.conftest import make_mock_response


class _RecordingAgent:
    """Records the input each node receives; returns '<name>-output' as the answer."""

    def __init__(self, name: str, inputs: dict):
        self.name = name
        self._inputs = inputs
        self.memory = SimpleNamespace(get_messages=lambda: [])

    async def run(self, query):
        self._inputs[self.name] = query
        return SimpleNamespace(content=f"{self.name}-output")


class _AllRouter:
    def __init__(self, names):
        self._names = names

    async def route(self, q):
        return RoutingDecision(query=q, agents=list(self._names), confidence=1.0, method="keyword", reasoning="all")


async def test_edge_flow_diamond_downstream_receives_upstream(mock_client):
    # diamond A -> {B, C} -> D
    names = ["A", "B", "C", "D"]
    deps = {"B": ["A"], "C": ["A"], "D": ["B", "C"]}
    inputs: dict = {}
    agents_map = {n: _RecordingAgent(n, inputs) for n in names}
    orch = Orchestrator(
        client=mock_client(responses=[make_mock_response("synthesis")]),
        router=_AllRouter(names),
        agents_map=agents_map,
        dag_scheduler=DagScheduler(deps=deps),
        default_mode="dag",
    )

    await orch.run("the-task", mode="dag")

    assert inputs["A"] == "the-task"  # wave 0: no upstream -> raw query
    assert "A-output" in inputs["B"]  # wave 1: B sees A's output
    assert "A-output" in inputs["C"]  # wave 1: C sees A's output
    assert "B-output" in inputs["D"] and "C-output" in inputs["D"]  # wave 2: D sees both


async def test_edge_flow_linear_chain(mock_client):
    names = ["a", "b", "c"]
    deps = {"b": ["a"], "c": ["b"]}
    inputs: dict = {}
    agents_map = {n: _RecordingAgent(n, inputs) for n in names}
    orch = Orchestrator(
        client=mock_client(responses=[make_mock_response("synthesis")]),
        router=_AllRouter(names),
        agents_map=agents_map,
        dag_scheduler=DagScheduler(deps=deps),
        default_mode="dag",
    )

    await orch.run("Q", mode="dag")

    assert inputs["a"] == "Q"
    assert "a-output" in inputs["b"]
    assert "b-output" in inputs["c"]


class _FixedAgent:
    """Returns a fixed answer (or empty) and records its input."""

    def __init__(self, name: str, inputs: dict, answer: str):
        self.name = name
        self._inputs = inputs
        self._answer = answer
        self.memory = SimpleNamespace(get_messages=lambda: [])

    async def run(self, query):
        self._inputs[self.name] = query
        return SimpleNamespace(content=self._answer)


async def test_edge_flow_empty_upstream_does_not_crash_downstream(mock_client):
    """N3: a node returning empty output -> downstream still runs (empty context), no crash."""
    names = ["A", "B"]
    deps = {"B": ["A"]}
    inputs: dict = {}
    agents_map = {"A": _FixedAgent("A", inputs, ""), "B": _FixedAgent("B", inputs, "b-out")}
    orch = Orchestrator(
        client=mock_client(responses=[make_mock_response("synthesis")]),
        router=_AllRouter(names),
        agents_map=agents_map,
        dag_scheduler=DagScheduler(deps=deps),
        default_mode="dag",
    )

    await orch.run("Q", mode="dag")  # must not raise

    assert set(inputs) == {"A", "B"}  # both ran despite A's empty output
    assert "Q" in inputs["B"]  # B's input still carries the original query
