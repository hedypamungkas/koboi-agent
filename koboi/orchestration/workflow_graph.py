"""koboi/orchestration/workflow_graph.py -- ergonomic programmatic graph builder (#7).

A thin LangGraph-shaped builder over the existing DagScheduler + Orchestrator
primitives. Lets you compose a workflow graph programmatically (``add_node`` /
``add_edge`` / ``add_conditional_edges`` / ``compile().invoke()``) instead of via
YAML or the LLM planner. Delegates to ``_run_dag_waves_with_flow`` / ``_run_conditional_graph``
for execution with edge data flow.

Usage::

    g = WorkflowGraph()
    g.add_node("research", "Gather facts.")
    g.add_node("draft", "Draft from research.")
    g.add_edge("research", "draft")
    graph = g.compile()
    result = await graph.invoke("topic: Mars", client=my_client)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.factory import AgentFactory
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.router import BaseRouter
from koboi.types import AgentDef, RoutingDecision

if TYPE_CHECKING:
    from koboi.client import Client


class _AllNodesRouter(BaseRouter):
    """Selects every node (the graph runs the full configured graph)."""

    def __init__(self, names: list[str]):
        self._names = names

    async def route(self, query: str) -> RoutingDecision:
        return RoutingDecision(query=query, agents=list(self._names), confidence=1.0, method="keyword", reasoning="all")


class WorkflowGraph:
    """Programmatic workflow graph builder (LangGraph-shaped)."""

    def __init__(self) -> None:
        self._nodes: dict[str, AgentDef] = {}
        self._edges: list[tuple[str, str]] = []
        self._conditionals: dict[str, list[dict]] = {}

    def add_node(self, name: str, instruction: str = "") -> WorkflowGraph:
        """Add a node with a one-sentence instruction (becomes the agent's system_prompt)."""
        self._nodes[name] = AgentDef(name=name, system_prompt=instruction or name)
        return self

    def add_edge(self, src: str, dst: str) -> WorkflowGraph:
        """Add a static dependency edge: dst runs after src completes."""
        self._edges.append((src, dst))
        return self

    def add_conditional_edges(
        self, src: str, mapping: dict[str, str], predicate_key: str = "contains"
    ) -> WorkflowGraph:
        """Add conditional edges: ``{value: target}`` — src's output is matched against
        each value; the matching target is enabled. ``predicate_key``: 'contains' | 'regex'."""
        for value, target in mapping.items():
            self._conditionals.setdefault(src, []).append({"to": target, "when": {predicate_key: value}})
        return self

    def compile(self) -> CompiledGraph:
        """Build a runnable graph from the accumulated nodes + edges."""
        deps: dict[str, list[str]] = {}
        for src, dst in self._edges:
            deps.setdefault(dst, []).append(src)
        # Also wire conditionals' targets as dependents of their source.
        for src, conds in self._conditionals.items():
            for c in conds:
                deps.setdefault(c["to"], []).append(src)
        return CompiledGraph(list(self._nodes.values()), deps, self._conditionals)


class CompiledGraph:
    """A compiled, runnable workflow graph."""

    def __init__(
        self,
        node_defs: list[AgentDef],
        deps: dict[str, list[str]],
        conditionals: dict[str, list[dict]],
    ) -> None:
        self._node_defs = node_defs
        self._deps = deps
        self._conditionals = conditionals

    async def invoke(self, query: str, client: Client) -> str:
        """Run the graph + return the final synthesized answer."""
        if not self._node_defs:
            return "No nodes in graph."
        names = [d.name for d in self._node_defs]
        agents_map = AgentFactory.create_all_configured(self._node_defs, client)
        sched = DagScheduler(agents_map=agents_map, deps=self._deps, conditionals=self._conditionals)
        orch = Orchestrator(
            client=client,
            router=_AllNodesRouter(names),
            agents_map=agents_map,
            dag_scheduler=sched,
            default_mode="dag",
            full_graph=True,
        )
        result = await orch.run(query, mode="dag")
        return result.final_answer
