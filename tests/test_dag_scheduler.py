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


def test_graph_run_id_minted_per_call():
    sched = DagScheduler(deps={})
    assert sched.graph_run_id is None  # before any waves() call
    sched.waves(["a"])
    g1 = sched.graph_run_id
    sched.waves(["a"])
    g2 = sched.graph_run_id
    assert g1 is not None and g2 is not None and g1 != g2


def test_persist_plan_writes_graph_rows(tmp_path):
    """#3: persist_plan writes one durable row per node tagged with graph_run_id + wave."""
    import sqlite3

    db = str(tmp_path / "graph.db")
    deps = {"B": ["A"], "C": ["A"], "D": ["B", "C"]}  # diamond A -> {B,C} -> D
    sched = DagScheduler(deps=deps, db_path=db)
    sched.waves(["A", "B", "C", "D"])
    graph_run_id = sched.persist_plan()

    assert graph_run_id is not None
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT node_id, turn_index FROM steps WHERE graph_run_id=? AND status='graph_plan' "
            "ORDER BY turn_index, step_index",
            (graph_run_id,),
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 4
    assert rows[0] == ("A", 0)  # wave 0
    assert sorted(n for n, w in rows if w == 1) == ["B", "C"]  # wave 1 (parallel)
    assert rows[-1] == ("D", 2)  # wave 2


def test_persist_plan_noop_without_db_path():
    sched = DagScheduler(deps={"B": ["A"]})
    sched.waves(["A", "B"])
    assert sched.persist_plan() is None  # no db_path -> durable capture skipped


def test_graph_cursor_resume_round_trip(tmp_path):
    """#2: persist plan + record completions + list completed nodes for resume."""
    db = str(tmp_path / "resume.db")
    deps = {"B": ["A"], "C": ["A"]}
    sched = DagScheduler(deps=deps, db_path=db)
    names = ["A", "B", "C"]
    sched.waves(names)
    sched.persist_plan()
    grid = sched.graph_run_id
    assert grid is not None
    # A + B completed before crash; C did not.
    sched.record_node_completion("A", "A-output")
    sched.record_node_completion("B", "B-output")
    completed = DagScheduler.list_completed_nodes(db, grid)
    assert completed == {"A": "A-output", "B": "B-output"}
    assert "C" not in completed  # incomplete -> needs re-run on resume


def test_record_node_completion_noop_without_db():
    sched = DagScheduler(deps={})
    sched.waves(["x"])
    sched.record_node_completion("x", "output")  # must not crash (no db_path)


def test_list_completed_nodes_empty_for_unknown_graph(tmp_path):
    db = str(tmp_path / "empty.db")
    result = DagScheduler.list_completed_nodes(db, "nonexistent-graph-id")
    assert result == {}
