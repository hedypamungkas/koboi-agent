"""koboi/orchestration/dag_scheduler.py -- dependency-ordered (wave-parallel) agent execution.

Turns koboi's dormant DAG data model (``koboi/task.py``: ``Task.blocked_by`` +
``add_dependency`` + ``_would_cycle`` + ``mark_completed``->``_try_unblock``) into a
topological execution plan. The scheduler is **pure**: it computes the wave
grouping (parallel within a level, sequential across levels) and the
:class:`Orchestrator` owns the actual per-node execution + event emission -- so
this module imports only ``koboi.task`` (no orchestrator/event coupling, no
circular import).

Wave-parallel is safe because the orchestrator's ``agents_map`` gives one distinct
``AgentCore`` per node name (no shared mutable core across concurrent nodes).
Edges come from config (``AgentDef.depends_on``); deps referencing nodes outside
the executed set are ignored (a routed subgraph runs in its induced order).
"""

from __future__ import annotations

import sqlite3
from uuid import uuid4

from koboi.memory_sqlite import ensure_steps_table
from koboi.task import TaskManager


class DagScheduler:
    """Computes topological execution waves for a set of agents + their dependencies.

    Args:
        agents_map: optional ``{name: AgentCore}`` (kept for symmetry with the
            orchestrator; wave computation uses names only).
        deps: ``{agent_name: [depends_on agent names]}`` -- the DAG edges.
        db_path: optional SQLite path; when set, :meth:`persist_plan` writes one
            durable graph-plan row per node (tagged with ``graph_run_id`` +
            ``node_id``) so a Phase-3 graph-cursor resume can recover mid-graph.
    """

    def __init__(
        self,
        agents_map: dict | None = None,
        deps: dict[str, list[str]] | None = None,
        db_path: str | None = None,
        conditionals: dict[str, list[dict]] | None = None,
    ) -> None:
        self._agents_map = agents_map or {}
        self._deps = deps or {}
        self._db_path = db_path
        self._graph_run_id: str | None = None
        self._last_waves: list[list[str]] | None = None
        # #1: conditional edges {source: [{to, when}]} for runtime branching.
        self._conditionals = conditionals or {}

    @property
    def conditionals(self) -> dict[str, list[dict]]:
        return self._conditionals

    @property
    def deps(self) -> dict[str, list[str]]:
        return self._deps

    @property
    def graph_run_id(self) -> str | None:
        """The graph_run_id of the most recent :meth:`waves` call (None before one)."""
        return self._graph_run_id

    def waves(self, agent_names: list[str]) -> list[list[str]]:
        """Return ``agent_names`` grouped into topological execution waves.

        Wave 0 = nodes with no (in-set) dependencies; each subsequent wave = nodes
        whose dependencies all completed in prior waves. Within a wave, nodes are
        independent and may run in parallel. Cycles are prevented at edge-insertion
        (``TaskManager._would_cycle``); a defensive guard bounds the loop. A fresh
        ``graph_run_id`` is minted per call.
        """
        self._graph_run_id = uuid4().hex
        if not agent_names:
            self._last_waves = []
            return []

        node_set = set(agent_names)
        mgr = TaskManager()
        name_to_id = {name: mgr.create(name).id for name in agent_names}

        # Add edges, restricted to the executed set (deps outside the set are ignored).
        for name in agent_names:
            for dep in self._deps.get(name, []):
                if dep in node_set:
                    mgr.add_dependency(name_to_id[name], name_to_id[dep])

        out: list[list[str]] = []
        guard = 0
        ready = [t.id for t in mgr.list_tasks("pending")]
        while ready:
            out.append([mgr.get(rid).subject for rid in ready])  # type: ignore[union-attr]
            for rid in ready:
                mgr.mark_completed(rid)  # cascades via _try_unblock -> next wave
            ready = [t.id for t in mgr.list_tasks("pending")]
            guard += 1
            if guard > len(agent_names) + 1:  # defensive: cycle (shouldn't occur)
                break
        self._last_waves = out
        return out

    def persist_plan(self) -> str | None:
        """Write one durable graph-plan row per node to the ``steps`` table.

        Rows are tagged ``status='graph_plan'`` with ``graph_run_id`` + ``node_id`` +
        ``turn_index``=wave index, so the planned graph (nodes + wave order) is
        recoverable. No-op when no ``db_path`` was configured or :meth:`waves` has
        not run. Returns the ``graph_run_id`` (or None if nothing was written).
        """
        if not self._db_path or not self._last_waves or not self._graph_run_id:
            return None
        conn = sqlite3.connect(self._db_path)
        try:
            ensure_steps_table(conn)
            for wave_idx, wave in enumerate(self._last_waves):
                for node_idx, node in enumerate(wave):
                    conn.execute(
                        "INSERT INTO steps (session_id, turn_index, step_index, status, "
                        "node_id, graph_run_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (self._graph_run_id, wave_idx, node_idx, "graph_plan", node, self._graph_run_id),
                    )
            conn.commit()
        finally:
            conn.close()
        return self._graph_run_id
