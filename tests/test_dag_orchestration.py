"""tests/test_dag_orchestration.py -- #2 execution.mode: dag wiring + integration.

Verifies the full path: AgentDef.depends_on -> _parse_agent_deps -> DagScheduler ->
Orchestrator._execute_pipeline "dag" branch -> wave-ordered execution with the
real Orchestrator (router + agents_map + synthesis) and execution_mode == "dag".
"""

from __future__ import annotations

from types import SimpleNamespace

from koboi.config import Config
from koboi.facade import _parse_agent_defs
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.types import RoutingDecision
from tests.conftest import make_mock_response


class _OrderAgent:
    """Stub agent that records its execution order; returns a content-bearing result."""

    def __init__(self, name: str, order: list[str]):
        self.name = name
        self._order = order
        self.memory = SimpleNamespace(get_messages=lambda: [])

    async def run(self, query):
        self._order.append(self.name)
        return SimpleNamespace(content=f"{self.name}-answer")


class _AllRouter:
    """Router that selects every configured agent (DAG runs the full graph)."""

    def __init__(self, names: list[str]):
        self._names = names

    async def route(self, query: str) -> RoutingDecision:
        return RoutingDecision(
            query=query,
            agents=list(self._names),
            confidence=1.0,
            method="keyword",
            reasoning="all",
        )


def test_parse_agent_defs_reads_depends_on():
    config = Config(
        {
            "orchestration": {
                "agents": [
                    {"name": "research"},
                    {"name": "draft", "depends_on": ["research"]},
                    {"name": "publish", "depends_on": ["draft"]},
                ]
            }
        }
    )
    defs = _parse_agent_defs(config)
    by_name = {d.name: d for d in defs}
    assert by_name["research"].depends_on == []
    assert by_name["draft"].depends_on == ["research"]
    assert by_name["publish"].depends_on == ["draft"]


async def test_dag_orchestration_runs_in_dependency_waves(mock_client):
    names = ["A", "B", "C", "D"]
    order: list[str] = []
    agents_map = {n: _OrderAgent(n, order) for n in names}
    deps = {"B": ["A"], "C": ["A"], "D": ["B", "C"]}  # diamond A -> {B,C} -> D

    orchestrator = Orchestrator(
        client=mock_client(responses=[make_mock_response("combined-answer")]),
        router=_AllRouter(names),
        agents_map=agents_map,
        dag_scheduler=DagScheduler(deps=deps),
        default_mode="dag",
    )

    result = await orchestrator.run("go", mode="dag")

    assert result.execution_mode == "dag"
    assert len(result.agent_results) == 4
    assert set(order) == {"A", "B", "C", "D"}
    # Wave order enforced: A before B and C; B and C before D.
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


async def test_dag_without_scheduler_falls_back_to_sequential(mock_client):
    """mode=dag with no DagScheduler warns and runs sequentially (no silent mis-execution)."""
    names = ["A", "B"]
    order: list[str] = []
    agents_map = {n: _OrderAgent(n, order) for n in names}

    orchestrator = Orchestrator(
        client=mock_client(responses=[make_mock_response("combined")]),
        router=_AllRouter(names),
        agents_map=agents_map,
        dag_scheduler=None,  # not configured
        default_mode="dag",
    )

    result = await orchestrator.run("go", mode="dag")

    # Still runs all agents (sequential fallback), execution_mode reported as dag.
    assert set(order) == {"A", "B"}
    assert len(result.agent_results) == 2


class _SubsetRouter:
    """Routes to a single agent (subset) -- to test that full_graph overrides routing."""

    async def route(self, query: str) -> RoutingDecision:
        return RoutingDecision(query=query, agents=["A"], confidence=1.0, method="keyword", reasoning="subset")


async def test_dag_full_graph_runs_all_nodes_despite_subset_routing(mock_client):
    """#4: full_graph=True -> dag runs the ENTIRE configured graph, ignoring the routed subset."""
    names = ["A", "B", "C", "D"]
    order: list[str] = []
    agents_map = {n: _OrderAgent(n, order) for n in names}
    deps = {"B": ["A"], "C": ["A"], "D": ["B", "C"]}

    orchestrator = Orchestrator(
        client=mock_client(responses=[make_mock_response("combined")]),
        router=_SubsetRouter(),  # routes to only "A"
        agents_map=agents_map,
        dag_scheduler=DagScheduler(deps=deps),
        default_mode="dag",
        full_graph=True,
    )

    await orchestrator.run("go", mode="dag")

    # full_graph overrides routing -> ALL 4 nodes ran (not just the routed "A").
    assert set(order) == {"A", "B", "C", "D"}
