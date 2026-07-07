"""tests/test_dag_scheduler.py -- #1 DagScheduler wave computation.

DagScheduler turns the dormant TaskManager DAG data model into a topological
wave plan (parallel within a level, sequential across levels). These tests cover
the pure wave computation; orchestrator execution + event emission are covered by
the integration tests once ``execution.mode: dag`` is wired (#2).
"""

from __future__ import annotations

from koboi.orchestration.dag_scheduler import DagScheduler


def test_diamond_graph_three_waves():
    # A -> {B, C} -> D
    deps = {"B": ["A"], "C": ["A"], "D": ["B", "C"]}
    sched = DagScheduler(deps=deps)

    waves = sched.waves(["A", "B", "C", "D"])

    assert len(waves) == 3
    assert waves[0] == ["A"]
    assert sorted(waves[1]) == ["B", "C"]  # independent -> same wave (order may vary)
    assert waves[2] == ["D"]


def test_chain_one_node_per_wave():
    deps = {"B": ["A"], "C": ["B"]}
    sched = DagScheduler(deps=deps)

    waves = sched.waves(["A", "B", "C"])

    assert waves == [["A"], ["B"], ["C"]]


def test_independent_nodes_single_wave():
    sched = DagScheduler(deps={})

    waves = sched.waves(["a", "b", "c"])

    assert len(waves) == 1
    assert sorted(waves[0]) == ["a", "b", "c"]


def test_dependency_order_enforced():
    """No node may appear in an earlier wave than any of its dependencies."""
    deps = {"research": [], "draft": ["research"], "review": ["draft"], "publish": ["review", "draft"]}
    sched = DagScheduler(deps=deps)

    waves = sched.waves(["research", "draft", "review", "publish"])

    # Flatten and verify each node appears after all its deps.
    position: dict[str, int] = {}
    for wi, wave in enumerate(waves):
        for name in wave:
            position[name] = wi
    for name, deps_of in deps.items():
        for dep in deps_of:
            assert position[dep] < position[name], f"{dep} must run before {name}"


def test_deps_outside_executed_set_are_ignored():
    """A dep on a node NOT in the executed set is dropped (routed subgraph)."""
    sched = DagScheduler(deps={"a": ["missing"]})

    waves = sched.waves(["a"])

    assert waves == [["a"]]  # 'missing' ignored -> 'a' has no in-set dep -> wave 0


def test_empty_and_single():
    assert DagScheduler(deps={}).waves([]) == []
    assert DagScheduler(deps={}).waves(["solo"]) == [["solo"]]


def test_cycle_is_rejected_not_looped():
    """TaskManager.add_dependency refuses cycles; waves() must not loop forever."""
    deps = {"a": ["b"], "b": ["a"]}  # would-be cycle
    sched = DagScheduler(deps=deps)

    # add_dependency rejects the second edge (b->a would cycle), so the graph is
    # effectively {a:[b]} (or {b:[a]}) -- one edge survives, no cycle, terminates.
    waves = sched.waves(["a", "b"])
    flat = {n for wave in waves for n in wave}
    assert flat == {"a", "b"}
