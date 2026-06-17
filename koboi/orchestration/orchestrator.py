"""Orchestrator and QualityEvaluator for multi-agent coordination.

Orchestrator: routes queries to specialized agents, collects and combines results.
QualityEvaluator: LLM-based answer quality evaluation with revision support.
"""

from __future__ import annotations

import asyncio
import json
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
    OrchestrationCompleteEvent,
    RoutingDecisionEvent,
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
    from koboi.logger import AgentLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QualityEvaluator
# ---------------------------------------------------------------------------


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
            )
            content = resp.content or ""
            data = _extract_json(content)
            if data:
                score = float(data.get("score", 0.5))
                feedback = data.get("feedback", "")
                needs = bool(data.get("needs_revision", score < self.threshold))
                return score, feedback, needs
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            logger.warning("Quality evaluation failed for query '%s': %s", query[:50], e)
        except Exception as e:
            logger.error("Unexpected error in quality evaluation: %s", e, exc_info=True)
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

        async for event in self._execute_pipeline(query, mode):
            if isinstance(event, RoutingDecisionEvent):
                decision = RoutingDecision(
                    query=query,
                    agents=event.agents,
                    confidence=event.confidence,
                    method=event.method,
                    reasoning=event.reasoning,
                    domain_label=event.domain_label,
                )
            elif isinstance(event, _AgentCompletedEvent):
                results.append(event.agent_result)
            elif isinstance(event, TextDeltaEvent):
                combined_answer += event.content
            elif isinstance(event, OrchestrationCompleteEvent):
                execution_mode = event.execution_mode

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
            execution_mode=execution_mode,
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
            execution_mode=f"{mode}+revision",
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
                results.append(result)
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

    async def _execute_pipeline(
        self,
        query: str,
        mode: str = "sequential",
    ) -> AsyncGenerator:
        """Core orchestration pipeline. Yields events for both run() and run_stream()."""
        if self.use_revision:
            logger.warning("Revision logic is not supported in streaming mode; falling back to direct execution.")

        start = time.time()

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
        else:
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
