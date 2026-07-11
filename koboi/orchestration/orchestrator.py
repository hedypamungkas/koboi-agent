"""Orchestrator and QualityEvaluator for multi-agent coordination.

Orchestrator: routes queries to specialized agents, collects and combines results.
QualityEvaluator: LLM-based answer quality evaluation with revision support.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from koboi.tokens import estimate_tokens
from koboi.types import AgentBlueprint, AgentResult, OrchestratorResult, RoutingDecision
from koboi.events import (
    AgentDispatchEvent,
    AgentResultEvent,
    CoverageEvent,
    OrchestrationCompleteEvent,
    RoutingDecisionEvent,
    SourceEvent,
    TextDeltaEvent,
)


@dataclass
class _AgentCompletedEvent:
    """Internal event carrying full AgentResult for run() collection."""

    agent_result: AgentResult


from koboi.orchestration.router import BaseRouter
from koboi.orchestration.factory import AgentFactory, DynamicAgentBuilder

if TYPE_CHECKING:
    from koboi.client import Client
    from koboi.hooks.chain import HookChain
    from koboi.logger import AgentLogger
    from koboi.orchestration.dag_scheduler import DagScheduler
    from koboi.orchestration.research import ResearchContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QualityEvaluator
# ---------------------------------------------------------------------------

# JSON Schema for the evaluator's structured response. Passed as response_format
# so providers enforce JSON (OpenAI native / Anthropic forced-tool), removing the
# need for the previous brittle extract_json + broad-except parsing.
_QUALITY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "feedback": {"type": "string"},
        "needs_revision": {"type": "boolean"},
    },
    "required": ["score", "feedback", "needs_revision"],
}


class QualityEvaluator:
    EVAL_PROMPT = (
        "You are an answer quality evaluator.\n\n"
        "Question: {query}\n"
        "Answer: {answer}\n\n"
        "Evaluate:\n"
        "1. Relevant to the question? (0-1)\n"
        "2. Has specific information (numbers, details)? (0-1)\n"
        "3. Says 'don't know' when it should be answerable? (-1 if yes)\n\n"
        'Answer ONLY JSON: {{"score": 0.8, "feedback": "...", "needs_revision": false}}'
    )

    def __init__(self, client: Client, threshold: float = 0.6):
        self.client = client
        self.threshold = threshold

    async def evaluate(self, query: str, answer: str) -> tuple[float, str, bool]:
        from koboi.orchestration._utils import extract_json as _extract_json

        prompt = self.EVAL_PROMPT.format(query=query, answer=answer)
        try:
            resp = await self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                response_format=_QUALITY_SCHEMA,
            )
            content = resp.content or ""
            # response_format enforces JSON on capable providers; _extract_json
            # stays as a tolerant fallback for providers that ignore it.
            data = _extract_json(content)
            if data:
                score = float(data.get("score", 0.5))
                feedback = data.get("feedback", "")
                needs = bool(data.get("needs_revision", score < self.threshold))
                return score, feedback, needs
        except Exception as e:  # noqa: BLE001 - resilience boundary: the evaluator is
            # embedded in the orchestration revision loop, so any client/transport/parse
            # failure must degrade to the fallback rather than crash the orchestration.
            # JSON reliability comes from response_format; this catch is NOT parsing.
            logger.warning("Quality evaluation failed for query '%s': %s", query[:50], e)
        return 0.5, "evaluation failed", True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    def __init__(
        self,
        client: Client,
        router: BaseRouter,
        logger: AgentLogger | None = None,
        max_revisions: int = 2,
        evaluator: QualityEvaluator | None = None,
        use_revision: bool = False,
        enable_dynamic: bool = False,
        dynamic_builder: DynamicAgentBuilder | None = None,
        agent_context_tokens: int = 8000,
        top_k: int = 3,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
        agents_map: dict | None = None,
        dag_scheduler: DagScheduler | None = None,
        default_mode: str = "sequential",
        hook_chain: HookChain | None = None,
        full_graph: bool = False,
        max_replans: int = 0,
        # W2: deep_research mode -- sandbox for the planned nodes' web tools + research
        # config (caps / threshold / citations). Unused by other modes.
        sandbox: object | None = None,
        research: dict | None = None,
    ):
        self.client = client
        self.router = router
        self.logger = logger
        self.max_revisions = max_revisions
        self.evaluator = evaluator
        self.use_revision = use_revision
        self.enable_dynamic = enable_dynamic
        self._agent_context_tokens = agent_context_tokens
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._dynamic_builder = dynamic_builder
        self._dynamic_blueprints: dict[str, AgentBlueprint] = {}
        self._agents_map: dict = agents_map or {}
        self._dag_scheduler = dag_scheduler
        self.default_mode = default_mode
        # #5: hook chain for dynamic-mode agents (so they get logging/policy/guardrails/
        # telemetry -- restores what WS4 omitted). Facade passes assembler.hook_chain.
        self._hook_chain = hook_chain
        # #4: full_graph -> dag mode runs the entire configured graph, not the routed subset.
        self._full_graph = full_graph
        # #3: max re-plans on node failure in dynamic mode.
        self._max_replans = max_replans
        # W2: deep_research knobs.
        self._sandbox = sandbox
        self._research = research or {}

    def _make_agent_logger(self, agent_name: str) -> AgentLogger | None:
        if not self.logger:
            return None
        from koboi.logger import AgentLogger

        session_id = f"{self.logger.session_id}_{agent_name}"
        return AgentLogger(log_dir=self.logger.log_dir, session_id=session_id)

    async def _resolve_dynamic_agents(self, query: str, decision: RoutingDecision) -> list[str]:
        if not self._dynamic_builder:
            from koboi.orchestration.factory import DynamicAgentBuilder

            self._dynamic_builder = DynamicAgentBuilder(
                client=self.client,
                logger=self.logger,
                top_k=self._top_k,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                max_context_tokens=self._agent_context_tokens,
            )

        blueprint = await self._dynamic_builder.build_blueprint(
            query,
            domain_label=decision.domain_label,
        )
        self._dynamic_blueprints[blueprint.name] = blueprint

        resolved = []
        for name in decision.agents:
            if name == "dynamic":
                resolved.append(blueprint.name)
            else:
                resolved.append(name)
        return resolved

    async def run(self, query: str, mode: str = "sequential") -> OrchestratorResult:
        start = time.time()

        if self.use_revision and self.evaluator:
            return await self._run_with_revision_legacy(query, mode, start)

        decision = None
        results: list[AgentResult] = []
        combined_answer = ""
        execution_mode = mode
        meta: dict = {}

        async for event in self._execute_pipeline(query, mode):
            if isinstance(event, RoutingDecisionEvent):
                decision = RoutingDecision(
                    query=query,
                    agents=event.agents,
                    confidence=event.confidence,
                    method=event.method,  # type: ignore[arg-type]  # event.method is str; constrained to the Literal at runtime by the router
                    reasoning=event.reasoning,
                    domain_label=event.domain_label,
                )
            elif isinstance(event, _AgentCompletedEvent):
                results.append(event.agent_result)
            elif isinstance(event, TextDeltaEvent):
                combined_answer += event.content
            elif isinstance(event, OrchestrationCompleteEvent):
                execution_mode = event.execution_mode
                meta = dict(event.metadata)

        if decision is None:
            decision = RoutingDecision(
                query=query,
                agents=[],
                confidence=0.0,
                method="keyword",
                reasoning="error",
            )

        elapsed = time.time() - start
        orch_result = OrchestratorResult(
            query=query,
            routing=decision,
            agent_results=results,
            final_answer=combined_answer,
            total_elapsed_seconds=elapsed,
            execution_mode=execution_mode,  # type: ignore[arg-type]  # str var; one of the Literal execution modes
            metadata=meta,
        )
        if self.logger:
            self.logger.log_orchestration_summary(orch_result)
        return orch_result

    async def _run_with_revision_legacy(
        self,
        query: str,
        mode: str,
        start: float,
    ) -> OrchestratorResult:
        """Legacy path for revision-enabled runs. Not streamable."""
        decision = await self.router.route(query)
        if self.logger:
            self.logger.log_routing(query, decision)

        agent_names = decision.agents
        if self.enable_dynamic and "dynamic" in agent_names:
            agent_names = await self._resolve_dynamic_agents(query, decision)

        results = await self._execute_with_revision(query, agent_names, mode)
        final = await self._combine_results(results, query)
        elapsed = time.time() - start

        orch_result = OrchestratorResult(
            query=query,
            routing=decision,
            agent_results=results,
            final_answer=final,
            total_elapsed_seconds=elapsed,
            execution_mode=f"{mode}+revision",  # type: ignore[arg-type]  # constructed from a validated mode
        )
        if self.logger:
            self.logger.log_orchestration_summary(orch_result)
        return orch_result

    async def _execute_sequential(self, query: str, agent_names: list[str]) -> list[AgentResult]:
        results: list[AgentResult] = []
        for name in agent_names:
            if self.logger:
                self.logger.log_agent_dispatch(name, query, "sequential")
            result = await self._run_single(name, query)
            results.append(result)
            if self.logger:
                self.logger.log_agent_result(result)
        return results

    async def _execute_parallel(self, query: str, agent_names: list[str]) -> list[AgentResult]:
        if self.logger:
            for name in agent_names:
                self.logger.log_agent_dispatch(name, query, "parallel")

        order = {name: i for i, name in enumerate(agent_names)}
        tasks = [self._run_single(name, query) for name in agent_names]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[AgentResult] = []
        for result in completed:
            if isinstance(result, Exception):
                results.append(
                    AgentResult(
                        agent_name="unknown",
                        answer=f"Error: {result}",
                        elapsed_seconds=0,
                        tokens_used=0,
                    )
                )
            else:
                results.append(result)  # type: ignore[arg-type]  # gather(return_exceptions=True): success branch; Exception handled above
                if self.logger:
                    self.logger.log_agent_result(result)

        results.sort(key=lambda r: order.get(r.agent_name, len(order)))
        return results

    async def _execute_with_revision(
        self,
        query: str,
        agent_names: list[str],
        mode: str = "sequential",
    ) -> list[AgentResult]:
        results: list[AgentResult] = []
        for name in agent_names:
            if self.logger:
                self.logger.log_agent_dispatch(name, query, f"{mode}+revision")

            current_query = query
            best_result: AgentResult | None = None

            for attempt in range(self.max_revisions + 1):
                result = await self._run_single(name, current_query)
                result.revision_count = attempt

                if not self.evaluator:
                    best_result = result
                    break

                score, feedback, needs = await self.evaluator.evaluate(query, result.answer)
                result.quality_score = score

                if not needs or score >= self.evaluator.threshold:
                    best_result = result
                    break

                current_query = (
                    f"{query}\n\n"
                    f"Revision notes (attempt {attempt + 1}): {feedback}. "
                    "Provide a more detailed and specific answer."
                )
                best_result = result

            if best_result:
                results.append(best_result)
                if self.logger:
                    self.logger.log_agent_result(best_result)

        return results

    async def _run_single(self, agent_name: str, query: str) -> AgentResult:
        agent_logger = self._make_agent_logger(agent_name)

        if agent_name in self._agents_map:
            agent = self._agents_map[agent_name]
        elif agent_name in self._dynamic_blueprints:
            blueprint = self._dynamic_blueprints[agent_name]
            agent = AgentFactory.create_dynamic_agent(blueprint, self.client, agent_logger)
        else:
            agent = AgentFactory.create_agent(agent_name, self.client, agent_logger)

        start = time.time()

        try:
            result = await agent.run(query)
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            logger.error("Agent %s failed: %s", agent_name, e, exc_info=True)
            answer = f"Error: {e}"

        elapsed = time.time() - start
        try:
            tokens = estimate_tokens(agent.memory.get_messages())
        except Exception:
            tokens = 0

        is_dynamic = agent_name in self._dynamic_blueprints
        domain_label = self._dynamic_blueprints[agent_name].domain_label if is_dynamic else None

        return AgentResult(
            agent_name=agent_name,
            answer=answer,
            elapsed_seconds=elapsed,
            tokens_used=tokens,
            is_dynamic=is_dynamic,
            domain_label=domain_label,
        )

    async def _run_dag_waves_with_flow(
        self,
        agent_names: list[str],
        query: str,
        deps: dict[str, list[str]],
        ctx: ResearchContext | None = None,
    ) -> AsyncGenerator:
        """Run ``agent_names`` as a dependency graph in topological waves WITH EDGE DATA FLOW.

        Each node's input = the original query + its dependencies' outputs (from prior
        waves), so downstream nodes actually consume upstream results (closes the
        no-data-flow gap). Wave-parallel within a level (each node is a distinct
        AgentCore -> safe), sequential across levels. Yields the
        AgentDispatch/AgentResult/_AgentCompleted event trio per node so the orchestrator
        event stream + downstream synthesis are unchanged. Shared by the static ``dag``
        branch and the ``dynamic`` mode.
        """
        from koboi.orchestration.dag_scheduler import DagScheduler

        total = len(agent_names)
        waves = DagScheduler(deps=deps).waves(agent_names)
        outputs: dict[str, str] = {}

        def _input_for(name: str) -> str:
            node_deps = deps.get(name, [])
            upstream = "\n".join(f"[{d}]: {outputs.get(d, '')}" for d in node_deps if d in outputs)
            if not upstream:
                return query
            return (
                f"Original request:\n{query}\n\nUpstream results:\n{upstream}\n\n"
                "Continue from the upstream results above."
            )

        flat = 0
        for wave in waves:
            for name in wave:
                yield AgentDispatchEvent(agent_name=name, agent_index=flat, total_agents=total, mode="dag")
                flat += 1
            wave_results = await asyncio.gather(*[self._run_single(n, _input_for(n)) for n in wave])
            for result in wave_results:
                outputs[result.agent_name] = result.answer
                # W2: collect the node's findings into the research context (deep_research).
                if ctx is not None:
                    cid = ctx.add_findings(result.agent_name, result.answer)
                    if cid:
                        yield SourceEvent(citation_id=cid, node_id=result.agent_name, preview=result.answer[:160])
                # #2: record durable per-node completion for graph-cursor resume.
                if self._dag_scheduler:
                    self._dag_scheduler.record_node_completion(result.agent_name, result.answer)
                yield AgentResultEvent(
                    agent_name=result.agent_name,
                    answer=result.answer[:200],
                    elapsed_seconds=result.elapsed_seconds,
                    tokens_used=result.tokens_used,
                    is_dynamic=result.is_dynamic,
                    domain_label=result.domain_label,
                    failed=result.failed,
                )
                yield _AgentCompletedEvent(agent_result=result)
                # #6: surface a [NODE_INTERRUPT] marker after an interrupt-flagged node.
                if self._dag_scheduler and result.agent_name in self._dag_scheduler.interrupt_nodes:
                    yield TextDeltaEvent(
                        content=f"[NODE_INTERRUPT] {result.agent_name} completed — awaiting human review"
                    )

    @staticmethod
    def _eval_conditional(when: dict, output: str) -> bool:
        """Evaluate a conditional predicate on a node's output (#1).

        Supports: {contains: "str"}, {regex: "pattern"}, {field, op, value} on JSON.
        """
        import json as _json
        import re as _re

        text = output or ""
        if "contains" in when:
            return str(when["contains"]).lower() in text.lower()
        if "regex" in when:
            return _re.search(str(when["regex"]), text) is not None
        if "field" in when:
            try:
                data = _json.loads(text)
                val = data.get(when["field"])
                op, target = when.get("op"), when.get("value")
                if op == ">":
                    return val is not None and val > target
                if op == ">=":
                    return val is not None and val >= target
                if op == "<":
                    return val is not None and val < target
                if op == "<=":
                    return val is not None and val <= target
                if op in ("==", "="):
                    return val == target
                if op == "!=":
                    return val != target
            except (ValueError, TypeError):
                return False
        return False

    async def _run_conditional_graph(
        self,
        agent_names: list[str],
        query: str,
        deps: dict[str, list[str]],
        conditionals: dict[str, list[dict]],
    ) -> AsyncGenerator:
        """Runtime scheduler for conditional edges (#1).

        Unlike the static wave scheduler (which pre-computes all waves), this evaluates
        predicates on each node's output AS IT COMPLETES -> enables/disables branches.
        A node is READY when its static deps are all completed AND (if it has incoming
        conditionals) at least one source's predicate fired. Wave-parallel within the
        ready set; edge data flow preserved (downstream gets upstream outputs).
        """
        total = len(agent_names)
        node_set = set(agent_names)
        outputs: dict[str, str] = {}
        completed: set[str] = set()
        enabled: set[str] = set()  # nodes enabled by a fired conditional
        remaining = set(agent_names)

        # incoming conditionals: {target: [(source, predicate), ...]}
        incoming: dict[str, list[tuple[str, dict]]] = {}
        for src, conds in conditionals.items():
            for c in conds:
                if c.get("to") in node_set:
                    incoming.setdefault(c["to"], []).append((src, c.get("when", {})))

        def _ready(node: str) -> bool:
            if not all(d in completed for d in deps.get(node, []) if d in node_set):
                return False
            if node in incoming:
                return node in enabled  # must be enabled by a fired conditional
            return True

        def _input_for(name: str) -> str:
            node_deps = deps.get(name, [])
            upstream = "\n".join(f"[{d}]: {outputs.get(d, '')}" for d in node_deps if d in outputs)
            if not upstream:
                return query
            return f"Original request:\n{query}\n\nUpstream results:\n{upstream}\n\nContinue from the upstream results above."

        flat = 0
        while remaining:
            ready = [n for n in sorted(remaining) if _ready(n)]
            if not ready:
                logger.warning("conditional graph: %d nodes could not be reached (no predicate fired)", len(remaining))
                break
            for n in ready:
                yield AgentDispatchEvent(agent_name=n, agent_index=flat, total_agents=total, mode="dag")
                flat += 1
            wave_results = await asyncio.gather(*[self._run_single(n, _input_for(n)) for n in ready])
            for result in wave_results:
                outputs[result.agent_name] = result.answer
                completed.add(result.agent_name)
                remaining.discard(result.agent_name)
                # #2: record durable per-node completion for graph-cursor resume.
                if self._dag_scheduler:
                    self._dag_scheduler.record_node_completion(result.agent_name, result.answer)
                yield AgentResultEvent(
                    agent_name=result.agent_name,
                    answer=result.answer[:200],
                    elapsed_seconds=result.elapsed_seconds,
                    tokens_used=result.tokens_used,
                    is_dynamic=result.is_dynamic,
                    domain_label=result.domain_label,
                    failed=result.failed,
                )
                yield _AgentCompletedEvent(agent_result=result)
                # #6: surface a [NODE_INTERRUPT] marker after an interrupt-flagged node.
                if self._dag_scheduler and result.agent_name in self._dag_scheduler.interrupt_nodes:
                    yield TextDeltaEvent(
                        content=f"[NODE_INTERRUPT] {result.agent_name} completed — awaiting human review"
                    )
                # Evaluate this node's outgoing conditionals -> enable targets.
                for cond in conditionals.get(result.agent_name, []):
                    if cond.get("to") in node_set and self._eval_conditional(cond.get("when", {}), result.answer):
                        enabled.add(cond["to"])

    async def _combine_results(self, results: list[AgentResult], query: str) -> str:
        if not results:
            return "No agent available to answer this question."

        if len(results) == 1:
            return results[0].answer

        # Multi-agent: structured concatenation with headers
        parts = []
        for r in results:
            label = r.agent_name.upper()
            parts.append(f"=== Answer from {label} Agent ===\n{r.answer}")

        combined = "\n\n".join(parts)

        # Use LLM to synthesize if there are multiple agents
        if len(results) > 1:
            try:
                synthesis_prompt = (
                    f"Combine the following answers from several specialist agents "
                    f"into a coherent answer.\n\n"
                    f"Question: {query}\n\n"
                )
                for r in results:
                    synthesis_prompt += f"[{r.agent_name.upper()}]: {r.answer}\n\n"
                synthesis_prompt += "Provide a structured and complete answer."

                resp = await self.client.complete(
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    tools=None,
                )
                if resp.content:
                    return resp.content
            except Exception as e:
                logger.warning("LLM synthesis failed, using concatenation: %s", e)

        return combined

    async def _combine_results_stream(self, results: list[AgentResult], query: str) -> AsyncGenerator:
        """Streaming version of _combine_results -- yields TextDeltaEvent chunks."""
        if not results:
            return

        if len(results) == 1:
            yield TextDeltaEvent(content=results[0].answer)
            return

        # Multi-agent: build synthesis prompt and stream LLM response
        synthesis_prompt = (
            "Combine the following answers from several specialist agents "
            "into a coherent answer.\n\n"
            f"Question: {query}\n\n"
        )
        for r in results:
            synthesis_prompt += f"[{r.agent_name.upper()}]: {r.answer}\n\n"
        synthesis_prompt += "Provide a structured and complete answer."

        try:
            async for event in self.client.complete_stream(
                messages=[{"role": "user", "content": synthesis_prompt}],
                tools=None,
            ):
                yield event
        except Exception as e:
            logger.warning("LLM synthesis streaming failed, using concatenation: %s", e)
            # Fallback to concatenation
            parts = []
            for r in results:
                label = r.agent_name.upper()
                parts.append(f"=== Answer from {label} Agent ===\n{r.answer}")
            yield TextDeltaEvent(content="\n\n".join(parts))

    async def _run_dynamic(self, query: str) -> AsyncGenerator:
        """Dynamic workflow (mode='dynamic'): the LLM plans a graph from the query,
        then the engine executes it with edge data flow. Simple queries skip the
        planner and answer directly with one general agent (no workflow overhead)."""
        from koboi.orchestration.factory import AgentFactory
        from koboi.orchestration.planner import plan_or_skip
        from koboi.types import AgentDef

        start = time.time()
        plan = await plan_or_skip(self.client, query)
        results: list[AgentResult] = []

        if plan.needs_workflow and plan.steps:
            step_ids = [s.id for s in plan.steps]
            yield RoutingDecisionEvent(
                agents=step_ids,
                confidence=1.0,
                method="dynamic",
                reasoning=plan.reason or "dynamic plan",
                domain_label=None,
            )
            # Build per-node agents from the plan (system_prompt = the step instruction).
            # #5: pass the parent hook_chain so dynamic nodes get logging/policy/guardrails.
            self._agents_map = {
                s.id: AgentFactory.create_configured_agent(
                    AgentDef(name=s.id, system_prompt=s.instruction or s.id),
                    self.client,
                    hook_chain=self._hook_chain,
                )
                for s in plan.steps
            }
            async for event in self._run_dag_waves_with_flow(step_ids, query, plan.deps):
                if isinstance(event, _AgentCompletedEvent):
                    results.append(event.agent_result)
                yield event
            routing_agents = step_ids
            # #3: re-plan on node failure (bounded by max_replans).
            replans_left = self._max_replans
            while replans_left > 0 and any(r.failed for r in results):
                replans_left -= 1
                failed_names = [r.agent_name for r in results if r.failed]
                retry_query = f"{query}\n\nNote: steps {failed_names} failed previously. Adjust the plan."
                plan = await plan_or_skip(self.client, retry_query)
                if not plan.needs_workflow or not plan.steps:
                    break
                results = []
                step_ids = [s.id for s in plan.steps]
                self._agents_map = {
                    s.id: AgentFactory.create_configured_agent(
                        AgentDef(name=s.id, system_prompt=s.instruction or s.id),
                        self.client,
                        hook_chain=self._hook_chain,
                    )
                    for s in plan.steps
                }
                async for event in self._run_dag_waves_with_flow(step_ids, query, plan.deps):
                    if isinstance(event, _AgentCompletedEvent):
                        results.append(event.agent_result)
                    yield event
                routing_agents = step_ids
        else:
            # Simple request: answer directly (no workflow). The negative/triage path.
            yield RoutingDecisionEvent(
                agents=["assistant"],
                confidence=1.0,
                method="dynamic",
                reasoning=f"direct: {plan.reason}",
                domain_label=None,
            )
            self._agents_map = {
                "assistant": AgentFactory.create_configured_agent(
                    AgentDef(name="assistant", system_prompt="You are a helpful assistant."),
                    self.client,
                    hook_chain=self._hook_chain,
                )
            }
            yield AgentDispatchEvent(agent_name="assistant", agent_index=0, total_agents=1, mode="dynamic")
            result = await self._run_single("assistant", query)
            results.append(result)
            yield AgentResultEvent(
                agent_name=result.agent_name,
                answer=result.answer[:200],
                elapsed_seconds=result.elapsed_seconds,
                tokens_used=result.tokens_used,
                is_dynamic=result.is_dynamic,
                domain_label=result.domain_label,
                failed=result.failed,
            )
            yield _AgentCompletedEvent(agent_result=result)
            routing_agents = ["assistant"]

        # Synthesis + complete (mirrors _execute_pipeline tail).
        combined_answer = ""
        async for event in self._combine_results_stream(results, query):
            if isinstance(event, TextDeltaEvent):
                combined_answer += event.content
            yield event
        yield OrchestrationCompleteEvent(
            final_answer=combined_answer,
            elapsed_seconds=time.time() - start,
            agent_results=results,
            execution_mode="dynamic",
            routing_agents=routing_agents,
            routing_confidence=1.0,
        )

    async def _run_deep_research(self, query: str) -> AsyncGenerator:
        """W2: deep-research orchestration.

        An iterative, cited research loop: ``plan_research`` -> per-node search/fetch waves
        (each node's findings become a cited source) -> ``CoverageEvaluator`` -> drill deeper
        on gaps -> synthesize a cited report. Bounded by ``max_depth`` + ``ResearchBudget``.
        NOT ``max_replans`` (which rarely fires) -- the coverage score drives iteration.
        Falls back to ``_run_dynamic`` if the planner deems the request simple.
        """
        from uuid import uuid4

        from koboi.orchestration.factory import AgentFactory
        from koboi.orchestration.planner import plan_research
        from koboi.orchestration.research import (
            CoverageEvaluator,
            ResearchBudget,
            ResearchContext,
            RESEARCH_TOOLS_CONFIG,
        )
        from koboi.types import AgentDef

        start = time.time()
        rc = self._research or {}
        budget = ResearchBudget(
            max_searches=int(rc.get("max_searches", 15)),
            max_fetches=int(rc.get("max_fetches", 20)),
            max_depth=int(rc.get("max_depth", 3)),
            max_tokens=int(rc.get("max_tokens", 0)),
        )
        threshold = float(rc.get("coverage_threshold", 0.7))
        tools_config = rc.get("tools") or RESEARCH_TOOLS_CONFIG

        ctx = ResearchContext(budget=budget)
        # Journaling: mint a run id + reuse the scheduler's db_path so the run state is
        # inspectable. Cross-session rehydrate-on-resume is W2.1.
        run_id = str(uuid4())
        db_path = self._dag_scheduler.db_path if self._dag_scheduler else None

        # Initial plan (or answer directly if the request is simple).
        plan = await plan_research(self.client, query)
        if not plan.needs_workflow or not plan.steps:
            async for event in self._run_dynamic(query):  # simple request -> direct answer
                yield event
            return

        ctx.sub_questions = [s.instruction for s in plan.steps]
        results: list[AgentResult] = []
        routing_agents: list[str] = []
        score = 0.0

        while True:
            step_ids = [s.id for s in plan.steps]
            if not routing_agents:
                yield RoutingDecisionEvent(
                    agents=step_ids,
                    confidence=1.0,
                    method="dynamic",
                    reasoning=plan.reason or "deep research plan",
                    domain_label=None,
                )
            # Build per-node agents WITH the research tool bundle + sandbox (closes the
            # blocker: planned nodes currently get no tools).
            self._agents_map = {
                s.id: AgentFactory.create_configured_agent(
                    AgentDef(
                        name=s.id,
                        system_prompt=s.instruction or s.id,
                        tools_config=dict(tools_config),
                    ),
                    self.client,
                    hook_chain=self._hook_chain,
                    sandbox=self._sandbox,
                )
                for s in plan.steps
            }
            # Proxy budget metering: charge one search per node carrying seed queries.
            budget.record_searches(sum(1 for s in plan.steps if s.search_queries))

            async for event in self._run_dag_waves_with_flow(step_ids, query, plan.deps, ctx=ctx):
                if isinstance(event, _AgentCompletedEvent):
                    results.append(event.agent_result)
                yield event
            routing_agents = list({r.agent_name for r in results}) or step_ids

            # Journal the run state after each round (observability; W2.1 adds rehydrate).
            if db_path:
                try:
                    from koboi.orchestration.dag_scheduler import DagScheduler

                    DagScheduler.persist_research_context(db_path, run_id, ctx.to_json())
                except Exception as e:  # noqa: BLE001 - journaling is best-effort
                    logger.warning("research context journal failed: %s", e)

            # Assess coverage (fail-safe -> score 1.0 -> stop).
            score, follow_ups, covmap = await CoverageEvaluator(self.client, threshold).evaluate(ctx)
            ctx.coverage_map = covmap
            yield CoverageEvent(depth=ctx.depth, score=score, gaps=follow_ups)

            ctx.depth += 1
            if score >= threshold or ctx.depth >= budget.max_depth or not budget.remaining():
                break
            if not follow_ups:
                break

            # Re-plan to drill deeper on the gaps.
            drill_query = query + "\n\nRemaining gaps to investigate: " + "; ".join(follow_ups)
            plan = await plan_research(self.client, drill_query)
            if not plan.needs_workflow or not plan.steps:
                break
            ctx.sub_questions.extend(s.instruction for s in plan.steps)

        # Synthesize a cited report from the gathered findings.
        combined_answer = ""
        async for event in self._synthesize_research_stream(query, ctx):
            if isinstance(event, TextDeltaEvent):
                combined_answer += event.content
            yield event

        # W3: persist the gathered findings (best-effort) for cross-session corpus reuse.
        persist_path = self._research.get("persist_findings")
        if persist_path:
            try:
                ctx.source_store.to_corpus_file(persist_path)
            except Exception as e:  # noqa: BLE001 - persistence is best-effort, never fatal
                logger.warning("research findings persistence failed: %s", e)

        yield OrchestrationCompleteEvent(
            final_answer=combined_answer,
            elapsed_seconds=time.time() - start,
            agent_results=results,
            execution_mode="deep_research",
            routing_agents=routing_agents,
            routing_confidence=1.0,
            metadata={
                "research_sources": ctx.source_store.sources_list(),
                "coverage": score,
                "depth": ctx.depth,
                "run_id": run_id,
            },
        )

    async def _synthesize_research_stream(self, query: str, ctx: ResearchContext) -> AsyncGenerator:
        """W2: stream a cited research report built from the gathered findings + Sources footer."""
        from koboi.orchestration.research import build_research_synthesis_prompt

        prompt = build_research_synthesis_prompt(query, ctx)
        try:
            async for event in self.client.complete_stream(messages=[{"role": "user", "content": prompt}], tools=None):
                yield event
        except Exception as e:  # noqa: BLE001 - synthesis is best-effort
            logger.warning("research synthesis streaming failed; using findings concatenation: %s", e)
            yield TextDeltaEvent(content=ctx.source_store.format_for_synthesis())
            return
        # Append a Sources footer listing every citation.
        sources = ctx.source_store.sources_list()
        if sources:
            footer = "\n\n## Sources\n" + "\n".join(f"[{s['citation_id']}] {s['node_id']}" for s in sources)
            yield TextDeltaEvent(content=footer)

    async def _execute_pipeline(
        self,
        query: str,
        mode: str = "sequential",
    ) -> AsyncGenerator:
        """Core orchestration pipeline. Yields events for both run() and run_stream()."""
        if self.use_revision:
            logger.warning("Revision logic is not supported in streaming mode; falling back to direct execution.")

        start = time.time()

        if mode == "dynamic":
            async for event in self._run_dynamic(query):
                yield event
            return

        if mode == "deep_research":
            async for event in self._run_deep_research(query):
                yield event
            return

        decision = await self.router.route(query)
        if self.logger:
            self.logger.log_routing(query, decision)

        yield RoutingDecisionEvent(
            agents=decision.agents,
            confidence=decision.confidence,
            method=decision.method,
            reasoning=decision.reasoning,
            domain_label=decision.domain_label,
        )

        agent_names = decision.agents
        if self.enable_dynamic and "dynamic" in agent_names:
            agent_names = await self._resolve_dynamic_agents(query, decision)

        total = len(agent_names)
        results: list[AgentResult] = []

        if mode == "parallel":
            for i, name in enumerate(agent_names):
                yield AgentDispatchEvent(
                    agent_name=name,
                    agent_index=i,
                    total_agents=total,
                    mode=mode,
                )

            order = {name: i for i, name in enumerate(agent_names)}
            tasks = {name: asyncio.create_task(self._run_single(name, query)) for name in agent_names}
            pending = set(tasks.values())
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        result = task.result()
                    except Exception as e:
                        result = AgentResult(
                            agent_name="unknown",
                            answer=f"Error: {e}",
                            elapsed_seconds=0,
                            tokens_used=0,
                            failed=True,
                        )
                    results.append(result)
                    yield AgentResultEvent(
                        agent_name=result.agent_name,
                        answer=result.answer[:200],
                        elapsed_seconds=result.elapsed_seconds,
                        tokens_used=result.tokens_used,
                        is_dynamic=result.is_dynamic,
                        domain_label=result.domain_label,
                        failed=result.failed,
                    )
                    yield _AgentCompletedEvent(agent_result=result)

            results.sort(key=lambda r: order.get(r.agent_name, len(order)))
        elif mode == "dag" and self._dag_scheduler is not None:
            # #4: full_graph runs the entire configured graph (bypasses the routed subset).
            _dag_names = list(self._agents_map.keys()) if self._full_graph else agent_names
            self._dag_scheduler.waves(_dag_names)  # populate _last_waves for persist
            self._dag_scheduler.persist_plan()
            # #1: if any conditional edges are configured, use the runtime scheduler
            # (evaluates predicates on node outputs to enable/disable branches).
            # Otherwise, the faster pre-computed wave scheduler.
            if self._dag_scheduler.conditionals:
                async for event in self._run_conditional_graph(
                    _dag_names, query, self._dag_scheduler.deps, self._dag_scheduler.conditionals
                ):
                    if isinstance(event, _AgentCompletedEvent):
                        results.append(event.agent_result)
                    yield event
            else:
                async for event in self._run_dag_waves_with_flow(_dag_names, query, self._dag_scheduler.deps):
                    if isinstance(event, _AgentCompletedEvent):
                        results.append(event.agent_result)
                    yield event
        else:
            if mode == "dag":
                logger.warning("execution.mode=dag requested but no DagScheduler configured; running sequentially.")
            for i, name in enumerate(agent_names):
                yield AgentDispatchEvent(
                    agent_name=name,
                    agent_index=i,
                    total_agents=total,
                    mode=mode,
                )

                result = await self._run_single(name, query)
                results.append(result)

                yield AgentResultEvent(
                    agent_name=result.agent_name,
                    answer=result.answer[:200],
                    elapsed_seconds=result.elapsed_seconds,
                    tokens_used=result.tokens_used,
                    is_dynamic=result.is_dynamic,
                    domain_label=result.domain_label,
                    failed=result.failed,
                )
                yield _AgentCompletedEvent(agent_result=result)

        combined_answer = ""
        async for event in self._combine_results_stream(results, query):
            if isinstance(event, TextDeltaEvent):
                combined_answer += event.content
            yield event

        elapsed = time.time() - start

        yield OrchestrationCompleteEvent(
            final_answer=combined_answer,
            elapsed_seconds=elapsed,
            agent_results=results,
            execution_mode=mode if not self.use_revision else f"{mode}+revision",
            routing_agents=decision.agents,
            routing_confidence=decision.confidence,
        )

    async def run_stream(self, query: str, mode: str = "sequential") -> AsyncGenerator:
        """Streaming version of run() -- yields orchestration events."""
        async for event in self._execute_pipeline(query, mode):
            if isinstance(event, _AgentCompletedEvent):
                continue
            yield event
