"""tests/test_conditional_edges.py -- #1: conditional (branching) edges.

A -> if output contains 'YES' then B, if contains 'NO' then C.
Only the fired branch runs; the unfired branch is skipped.
"""

from __future__ import annotations

from types import SimpleNamespace

from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.types import RoutingDecision
from tests.conftest import make_mock_response


class _Agent:
    """Returns a fixed answer; records whether it ran."""

    def __init__(self, name: str, ran: set, answer: str):
        self.name = name
        self._ran = ran
        self._answer = answer
        self.memory = SimpleNamespace(get_messages=lambda: [])

    async def run(self, query):
        self._ran.add(self.name)
        return SimpleNamespace(content=self._answer)


class _AllRouter:
    def __init__(self, names):
        self._names = names

    async def route(self, q):
        return RoutingDecision(query=q, agents=list(self._names), confidence=1.0, method="keyword", reasoning="all")


DEPS = {"B": ["A"], "C": ["A"]}
CONDS = {"A": [{"to": "B", "when": {"contains": "YES"}}, {"to": "C", "when": {"contains": "NO"}}]}


def _build(ran, a_answer):
    names = ["A", "B", "C"]
    agents_map = {
        "A": _Agent("A", ran, a_answer),
        "B": _Agent("B", ran, "B-ran"),
        "C": _Agent("C", ran, "C-ran"),
    }
    return Orchestrator(
        client=__import__("tests.conftest", fromlist=["mock_client"]).mock_client(
            responses=[make_mock_response("syn")]
        ),
        router=_AllRouter(names),
        agents_map=agents_map,
        dag_scheduler=DagScheduler(deps=DEPS, conditionals=CONDS),
        default_mode="dag",
    )


async def test_conditional_yes_fires_b_not_c(mock_client):
    ran: set = set()
    orch = Orchestrator(
        client=mock_client(responses=[make_mock_response("syn")]),
        router=_AllRouter(["A", "B", "C"]),
        agents_map={
            "A": _Agent("A", ran, "YES"),
            "B": _Agent("B", ran, "B-ran"),
            "C": _Agent("C", ran, "C-ran"),
        },
        dag_scheduler=DagScheduler(deps=DEPS, conditionals=CONDS),
        default_mode="dag",
    )

    await orch.run("go", mode="dag")

    assert "A" in ran  # A always runs (no deps)
    assert "B" in ran  # A said "YES" -> B's predicate fired
    assert "C" not in ran  # A didn't say "NO" -> C skipped


async def test_conditional_no_fires_c_not_b(mock_client):
    ran: set = set()
    orch = Orchestrator(
        client=mock_client(responses=[make_mock_response("syn")]),
        router=_AllRouter(["A", "B", "C"]),
        agents_map={
            "A": _Agent("A", ran, "NO"),
            "B": _Agent("B", ran, "B-ran"),
            "C": _Agent("C", ran, "C-ran"),
        },
        dag_scheduler=DagScheduler(deps=DEPS, conditionals=CONDS),
        default_mode="dag",
    )

    await orch.run("go", mode="dag")

    assert "A" in ran
    assert "C" in ran  # A said "NO" -> C's predicate fired
    assert "B" not in ran  # A didn't say "YES" -> B skipped


async def test_conditional_json_field_predicate(mock_client):
    """Predicate on typed JSON output: {field, op, value}."""
    ran: set = set()
    orch = Orchestrator(
        client=mock_client(responses=[make_mock_response("syn")]),
        router=_AllRouter(["A", "B"]),
        agents_map={
            "A": _Agent("A", ran, '{"score": 0.9}'),
            "B": _Agent("B", ran, "B-ran"),
        },
        dag_scheduler=DagScheduler(
            deps={"B": ["A"]},
            conditionals={"A": [{"to": "B", "when": {"field": "score", "op": ">", "value": 0.8}}]},
        ),
        default_mode="dag",
    )

    await orch.run("go", mode="dag")

    assert "B" in ran  # score 0.9 > 0.8 -> B enabled
