"""tests/test_orchestrator_cov.py -- branch coverage for koboi/orchestration/orchestrator.py.

Targets the uncovered paths: dynamic-agent resolution, the legacy (non-stream)
sequential/parallel/revision executors, the DAG + conditional graph schedulers,
the LLM synthesis combine paths, and the dynamic (planner) mode including the
re-plan-on-failure loop. No real LLM calls -- specialists are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.events import (
    OrchestrationCompleteEvent,
    TextDeltaEvent,
)
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator, QualityEvaluator
from koboi.orchestration.planner import PlanResult, PlanStep
from koboi.types import AgentBlueprint, AgentResponse, AgentResult, RoutingDecision


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_orchestration_streaming.py patterns)
# ---------------------------------------------------------------------------


def _decision(agents=None, domain_label=None) -> RoutingDecision:
    return RoutingDecision(
        query="q",
        agents=agents or ["a1"],
        confidence=0.9,
        method="keyword",
        reasoning="r",
        domain_label=domain_label,
    )


class _Router:
    def __init__(self, decision: RoutingDecision):
        self._d = decision

    async def route(self, query: str) -> RoutingDecision:
        return self._d


class _Agent:
    """Stand-in specialist. ``run()`` returns an object with ``.content``."""

    def __init__(self, answer: str = "ans", fail: bool = False) -> None:
        self._answer = answer
        self._fail = fail
        self.memory = MagicMock()
        self.memory.get_messages.return_value = []

    async def run(self, query: str):
        if self._fail:
            raise RuntimeError("agent failed")
        result = MagicMock()
        result.content = self._answer
        return result


def _stream_client(content: str = "synthesis") -> MagicMock:
    """Client whose complete() + complete_stream() both succeed."""
    client = MagicMock()
    client.complete = AsyncMock(return_value=AgentResponse(content=content))

    async def _gen(*a, **k):
        yield TextDeltaEvent(content=content)

    client.complete_stream = _gen
    return client


def _make_orch(**kwargs) -> Orchestrator:
    """Orchestrator with a no-op router + stream client by default."""
    router = kwargs.pop("router", _Router(_decision()))
    client = kwargs.pop("client", _stream_client())
    return Orchestrator(client=client, router=router, **kwargs)


def _dyn_builder(blueprint: AgentBlueprint):
    """A DynamicAgentBuilder stand-in whose build_blueprint() returns ``blueprint``."""
    builder = MagicMock()
    builder.build_blueprint = AsyncMock(return_value=blueprint)
    return builder


# ---------------------------------------------------------------------------
# _resolve_dynamic_agents
# ---------------------------------------------------------------------------


class TestResolveDynamicAgents:
    async def test_resolves_dynamic_token_via_existing_builder(self):
        orch = _make_orch(enable_dynamic=True)
        blueprint = AgentBlueprint(
            name="dyn_resolved",
            domain_label="support",
            system_prompt="x",
            chunks=[],
            chunker_config={},
        )
        orch._dynamic_builder = _dyn_builder(blueprint)
        decision = _decision(agents=["dynamic", "hr"], domain_label="support")

        resolved = await orch._resolve_dynamic_agents("query", decision)

        assert resolved == ["dyn_resolved", "hr"]
        assert "dyn_resolved" in orch._dynamic_blueprints
        assert orch._dynamic_blueprints["dyn_resolved"].domain_label == "support"

    async def test_constructs_builder_when_absent(self, monkeypatch):
        orch = _make_orch(enable_dynamic=True)
        assert orch._dynamic_builder is None
        blueprint = AgentBlueprint(
            name="dyn_new",
            domain_label="x",
            system_prompt="x",
            chunks=[],
            chunker_config={},
        )
        fake_builder = MagicMock()
        fake_builder.build_blueprint = AsyncMock(return_value=blueprint)
        monkeypatch.setattr(
            "koboi.orchestration.factory.DynamicAgentBuilder",
            lambda **kw: fake_builder,
        )

        resolved = await orch._resolve_dynamic_agents("query", _decision(agents=["dynamic"]))

        assert resolved == ["dyn_new"]
        assert orch._dynamic_builder is fake_builder


# ---------------------------------------------------------------------------
# run() decision-is-None fallback (line 222)
# ---------------------------------------------------------------------------


class TestRunDecisionFallback:
    async def test_run_with_empty_pipeline_falls_back_to_empty_decision(self):
        # When the pipeline yields no RoutingDecisionEvent, run() builds a fallback
        # RoutingDecision(agents=[]). RoutingDecision.__post_init__ rejects empty
        # agents -> the safety fallback itself raises ValueError (a known source
        # quirk). We assert that behaviour so the fallback line is exercised.
        orch = _make_orch()

        async def _empty_pipeline(query, mode="sequential"):
            if False:  # pragma: no cover - makes this an async generator
                yield

        orch._execute_pipeline = _empty_pipeline

        with pytest.raises(ValueError, match="agents cannot be empty"):
            await orch.run("q", mode="sequential")


# ---------------------------------------------------------------------------
# _run_with_revision_legacy (logger + dynamic resolve + summary)
# ---------------------------------------------------------------------------


class TestRunWithRevisionLegacy:
    async def test_revision_legacy_with_logger_and_dynamic(self, monkeypatch):
        logger = MagicMock()
        client = _stream_client()
        # evaluator quality response -> good (no revision loop)
        client.complete = AsyncMock(
            return_value=AgentResponse(content='{"score": 0.9, "feedback": "good", "needs_revision": false}')
        )
        evaluator = QualityEvaluator(client, threshold=0.6)
        blueprint = AgentBlueprint(
            name="dyn_resolved",
            domain_label="support",
            system_prompt="x",
            chunks=[],
            chunker_config={},
        )
        orch = Orchestrator(
            client=client,
            router=_Router(_decision(agents=["dynamic"], domain_label="support")),
            logger=logger,
            evaluator=evaluator,
            use_revision=True,
            enable_dynamic=True,
            agents_map={"dyn_resolved": _Agent("answer")},
        )
        orch._dynamic_builder = _dyn_builder(blueprint)

        result = await orch.run("q", mode="sequential")

        assert result.execution_mode == "sequential+revision"
        logger.log_routing.assert_called_once()
        logger.log_orchestration_summary.assert_called_once()
        assert result.routing.agents == ["dynamic"]


# ---------------------------------------------------------------------------
# Legacy _execute_sequential / _execute_parallel (dead-code executors)
# ---------------------------------------------------------------------------


class TestExecuteSequential:
    async def test_runs_each_agent_and_logs(self):
        logger = MagicMock()
        orch = _make_orch(
            logger=logger,
            agents_map={"a": _Agent("alpha"), "b": _Agent("beta")},
        )
        # _run_single builds a real AgentLogger (mkdirs) from self.logger; bypass it.
        orch._make_agent_logger = lambda name: None
        results = await orch._execute_sequential("q", ["a", "b"])
        assert [r.agent_name for r in results] == ["a", "b"]
        assert [r.answer for r in results] == ["alpha", "beta"]
        assert logger.log_agent_dispatch.call_count == 2
        assert logger.log_agent_result.call_count == 2


class TestExecuteParallel:
    async def test_success_preserves_order_and_logs(self):
        logger = MagicMock()
        orch = _make_orch(logger=logger)
        orch._run_single = AsyncMock(
            side_effect=[
                AgentResult(agent_name="b", answer="B", elapsed_seconds=0, tokens_used=0),
                AgentResult(agent_name="a", answer="A", elapsed_seconds=0, tokens_used=0),
            ]
        )
        results = await orch._execute_parallel("q", ["a", "b"])
        # sorted back into declared order a, b
        assert [r.agent_name for r in results] == ["a", "b"]
        logger.log_agent_result.assert_called()

    async def test_exception_becomes_error_result(self):
        orch = _make_orch()
        orch._run_single = AsyncMock(side_effect=RuntimeError("boom"))
        results = await orch._execute_parallel("q", ["a"])
        assert len(results) == 1
        assert "Error:" in results[0].answer
        assert results[0].agent_name == "unknown"


# ---------------------------------------------------------------------------
# _execute_with_revision
# ---------------------------------------------------------------------------


class TestExecuteWithRevision:
    async def test_no_evaluator_breaks_after_first_attempt(self):
        logger = MagicMock()
        orch = _make_orch(logger=logger, agents_map={"a": _Agent("ans")})
        orch._make_agent_logger = lambda name: None  # bypass real AgentLogger mkdirs
        # evaluator is None -> immediate break
        results = await orch._execute_with_revision("q", ["a"])
        assert len(results) == 1
        assert results[0].revision_count == 0

    async def test_revision_loops_on_low_quality_until_cap(self):
        logger = MagicMock()
        client = MagicMock()
        evaluator = QualityEvaluator(client, threshold=0.8)
        # Always needs revision, below threshold
        evaluator.evaluate = AsyncMock(return_value=(0.3, "too vague", True))
        orch = _make_orch(client=client, logger=logger, evaluator=evaluator, max_revisions=1)
        orch._agents_map = {"a": _Agent("draft")}
        orch._make_agent_logger = lambda name: None  # bypass real AgentLogger mkdirs
        results = await orch._execute_with_revision("q", ["a"], mode="sequential")
        # max_revisions + 1 attempts = 2
        assert len(results) == 1
        assert results[0].revision_count == 1
        assert results[0].quality_score == 0.3
        assert evaluator.evaluate.await_count == 2
        logger.log_agent_dispatch.assert_called()
        logger.log_agent_result.assert_called_once()


# ---------------------------------------------------------------------------
# _run_single (dynamic-blueprint + token-error paths)
# ---------------------------------------------------------------------------


class TestRunSingle:
    async def test_dynamic_blueprint_path(self, monkeypatch):
        orch = _make_orch()
        blueprint = AgentBlueprint(
            name="dynx",
            domain_label="support",
            system_prompt="x",
            chunks=[],
            chunker_config={},
        )
        orch._dynamic_blueprints = {"dynx": blueprint}
        monkeypatch.setattr(
            "koboi.orchestration.factory.AgentFactory.create_dynamic_agent",
            lambda blueprint, client, logger=None: _Agent("dyn-answer"),
        )
        result = await orch._run_single("dynx", "q")
        assert result.is_dynamic is True
        assert result.domain_label == "support"
        assert result.answer == "dyn-answer"

    async def test_token_estimate_exception_yields_zero(self):
        orch = _make_orch()
        agent = _Agent("ans")
        agent.memory.get_messages.side_effect = ValueError("corrupt")
        orch._agents_map = {"a": agent}
        result = await orch._run_single("a", "q")
        assert result.tokens_used == 0
        assert result.answer == "ans"


# ---------------------------------------------------------------------------
# DAG wave scheduler (pipeline dag branch + _run_dag_waves_with_flow + interrupt)
# ---------------------------------------------------------------------------


class TestDagMode:
    def _dag_orch(self, deps, agents, interrupt_nodes=None, full_graph=False) -> Orchestrator:
        scheduler = DagScheduler(deps=deps, interrupt_nodes=interrupt_nodes or set())
        agents_map = {name: _Agent(f"out-{name}") for name in agents}
        return _make_orch(
            router=_Router(_decision(agents=list(agents))),
            agents_map=agents_map,
            dag_scheduler=scheduler,
            full_graph=full_graph,
        )

    async def test_dag_runs_in_dependency_order(self):
        orch = self._dag_orch(deps={"b": ["a"]}, agents=["a", "b"])
        result = await orch.run("q", mode="dag")
        assert result.execution_mode == "dag"
        names = [r.agent_name for r in result.agent_results]
        assert names.index("a") < names.index("b")

    async def test_dag_interrupt_node_marker_emitted(self):
        orch = self._dag_orch(deps={"b": ["a"]}, agents=["a", "b"], interrupt_nodes={"a"})
        result = await orch.run("q", mode="dag")
        assert "[NODE_INTERRUPT]" in result.final_answer
        assert "a" in result.final_answer

    async def test_dag_records_node_completion(self, tmp_path):
        db = str(tmp_path / "dag.db")
        scheduler = DagScheduler(deps={"b": ["a"]}, db_path=db)
        orch = _make_orch(
            router=_Router(_decision(agents=["a", "b"])),
            agents_map={"a": _Agent("oa"), "b": _Agent("ob")},
            dag_scheduler=scheduler,
        )
        await orch.run("q", mode="dag")
        completed = DagScheduler.list_completed_nodes(db, scheduler.graph_run_id)  # type: ignore[arg-type]
        assert set(completed.keys()) == {"a", "b"}

    async def test_dag_no_scheduler_warns_and_runs_sequentially(self, caplog):
        orch = _make_orch(
            router=_Router(_decision(agents=["a"])),
            agents_map={"a": _Agent("ans")},
            dag_scheduler=None,
        )
        with caplog.at_level("WARNING", logger="koboi.orchestration.orchestrator"):
            result = await orch.run("q", mode="dag")
        assert result.execution_mode == "dag"
        assert any("no DagScheduler configured" in r.message for r in caplog.records)

    async def test_full_graph_uses_entire_agents_map(self):
        # router returns only ["a"], but full_graph runs the whole map (a, b)
        scheduler = DagScheduler(deps={"b": ["a"]})
        orch = _make_orch(
            router=_Router(_decision(agents=["a"])),
            agents_map={"a": _Agent("oa"), "b": _Agent("ob")},
            dag_scheduler=scheduler,
            full_graph=True,
        )
        result = await orch.run("q", mode="dag")
        assert {r.agent_name for r in result.agent_results} == {"a", "b"}


# ---------------------------------------------------------------------------
# _eval_conditional (pure static method)
# ---------------------------------------------------------------------------


class TestEvalConditional:
    def test_contains_match_is_case_insensitive(self):
        assert Orchestrator._eval_conditional({"contains": "GO"}, "let's go now") is True

    def test_contains_miss(self):
        assert Orchestrator._eval_conditional({"contains": "missing"}, "hello") is False

    def test_regex_match(self):
        assert Orchestrator._eval_conditional({"regex": r"\d{3}"}, "code 123") is True

    def test_regex_miss(self):
        assert Orchestrator._eval_conditional({"regex": r"\d{3}"}, "no digits") is False

    def test_field_greater_than(self):
        out = '{"price": 10}'
        assert Orchestrator._eval_conditional({"field": "price", "op": ">", "value": 5}, out) is True
        assert Orchestrator._eval_conditional({"field": "price", "op": ">", "value": 10}, out) is False

    def test_field_gte_lte(self):
        out = '{"price": 10}'
        assert Orchestrator._eval_conditional({"field": "price", "op": ">=", "value": 10}, out) is True
        assert Orchestrator._eval_conditional({"field": "price", "op": "<=", "value": 10}, out) is True
        assert Orchestrator._eval_conditional({"field": "price", "op": "<", "value": 10}, out) is False

    def test_field_equal_aliases(self):
        out = '{"ok": 1}'
        assert Orchestrator._eval_conditional({"field": "ok", "op": "==", "value": 1}, out) is True
        assert Orchestrator._eval_conditional({"field": "ok", "op": "=", "value": 2}, out) is False

    def test_field_not_equal(self):
        out = '{"ok": 1}'
        assert Orchestrator._eval_conditional({"field": "ok", "op": "!=", "value": 2}, out) is True

    def test_field_missing_returns_false_on_value_error(self):
        # not JSON -> ValueError -> False
        assert Orchestrator._eval_conditional({"field": "x", "op": ">", "value": 1}, "not json") is False

    def test_field_absent_val_is_none(self):
        out = '{"a": 1}'
        # field "b" absent -> val None -> op ">" returns False
        assert Orchestrator._eval_conditional({"field": "b", "op": ">", "value": 0}, out) is False

    def test_no_recognized_predicate_returns_false(self):
        assert Orchestrator._eval_conditional({"unknown": "x"}, "anything") is False

    def test_empty_output(self):
        assert Orchestrator._eval_conditional({"contains": "x"}, "") is False


# ---------------------------------------------------------------------------
# _run_conditional_graph (pipeline dag+conditionals branch)
# ---------------------------------------------------------------------------


class TestConditionalGraph:
    async def test_branch_fires_and_target_runs(self):
        scheduler = DagScheduler(
            deps={"b": ["a"]},
            conditionals={"a": [{"to": "b", "when": {"contains": "go"}}]},
        )
        orch = _make_orch(
            router=_Router(_decision(agents=["a", "b"])),
            agents_map={"a": _Agent("let's go"), "b": _Agent("branch-out")},
            dag_scheduler=scheduler,
        )
        result = await orch.run("q", mode="dag")
        names = {r.agent_name for r in result.agent_results}
        assert names == {"a", "b"}  # b enabled because a said "go"

    async def test_branch_does_not_fire_target_skipped(self, caplog):
        scheduler = DagScheduler(
            deps={"b": ["a"]},
            conditionals={"a": [{"to": "b", "when": {"contains": "go"}}]},
        )
        orch = _make_orch(
            router=_Router(_decision(agents=["a", "b"])),
            agents_map={"a": _Agent("nothing relevant"), "b": _Agent("branch-out")},
            dag_scheduler=scheduler,
        )
        with caplog.at_level("WARNING", logger="koboi.orchestration.orchestrator"):
            result = await orch.run("q", mode="dag")
        names = {r.agent_name for r in result.agent_results}
        assert names == {"a"}  # b never enabled
        assert any("could not be reached" in r.message for r in caplog.records)

    async def test_conditional_interrupt_node(self):
        scheduler = DagScheduler(
            deps={"b": ["a"]},
            conditionals={"a": [{"to": "b", "when": {"contains": "go"}}]},
            interrupt_nodes={"a"},
        )
        orch = _make_orch(
            router=_Router(_decision(agents=["a", "b"])),
            agents_map={"a": _Agent("let's go"), "b": _Agent("bx")},
            dag_scheduler=scheduler,
        )
        result = await orch.run("q", mode="dag")
        assert "[NODE_INTERRUPT]" in result.final_answer


# ---------------------------------------------------------------------------
# _combine_results (non-streaming legacy, used by revision path)
# ---------------------------------------------------------------------------


class TestCombineResults:
    async def test_empty_returns_no_agent_message(self):
        orch = _make_orch()
        out = await orch._combine_results([], "q")
        assert "No agent available" in out

    async def test_multi_synthesis_success(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content="synthesized"))
        orch = _make_orch(client=client)
        results = [
            AgentResult(agent_name="a", answer="A", elapsed_seconds=0, tokens_used=0),
            AgentResult(agent_name="b", answer="B", elapsed_seconds=0, tokens_used=0),
        ]
        out = await orch._combine_results(results, "q")
        assert out == "synthesized"

    async def test_multi_synthesis_failure_falls_back_to_concatenation(self):
        client = MagicMock()
        client.complete = AsyncMock(side_effect=RuntimeError("boom"))
        orch = _make_orch(client=client)
        results = [
            AgentResult(agent_name="a", answer="A", elapsed_seconds=0, tokens_used=0),
            AgentResult(agent_name="b", answer="B", elapsed_seconds=0, tokens_used=0),
        ]
        out = await orch._combine_results(results, "q")
        assert "=== Answer from A Agent ===" in out
        assert "A" in out and "B" in out


# ---------------------------------------------------------------------------
# _combine_results_stream
# ---------------------------------------------------------------------------


class TestCombineResultsStream:
    async def _collect(self, gen):
        out = []
        async for e in gen:
            out.append(e)
        return out

    async def test_empty_yields_nothing(self):
        orch = _make_orch()
        events = await self._collect(orch._combine_results_stream([], "q"))
        assert events == []

    async def test_single_result_yields_answer(self):
        orch = _make_orch()
        results = [AgentResult(agent_name="a", answer="only", elapsed_seconds=0, tokens_used=0)]
        events = await self._collect(orch._combine_results_stream(results, "q"))
        assert len(events) == 1
        assert events[0].content == "only"

    async def test_multi_streams_synthesis(self):
        orch = _make_orch(client=_stream_client(content="final-synth"))
        results = [
            AgentResult(agent_name="a", answer="A", elapsed_seconds=0, tokens_used=0),
            AgentResult(agent_name="b", answer="B", elapsed_seconds=0, tokens_used=0),
        ]
        events = await self._collect(orch._combine_results_stream(results, "q"))
        assert any(isinstance(e, TextDeltaEvent) and e.content == "final-synth" for e in events)

    async def test_multi_stream_failure_falls_back_to_concat(self):
        client = MagicMock()

        async def _bad_stream(*a, **k):
            raise RuntimeError("stream down")
            yield  # make it an async generator

        client.complete_stream = _bad_stream
        orch = _make_orch(client=client)
        results = [
            AgentResult(agent_name="a", answer="A", elapsed_seconds=0, tokens_used=0),
            AgentResult(agent_name="b", answer="B", elapsed_seconds=0, tokens_used=0),
        ]
        events = await self._collect(orch._combine_results_stream(results, "q"))
        assert events and "=== Answer from A Agent ===" in events[-1].content


# ---------------------------------------------------------------------------
# _run_dynamic (planner mode) + pipeline dynamic dispatch
# ---------------------------------------------------------------------------


class TestRunDynamicMode:
    def _workflow_plan(self) -> PlanResult:
        return PlanResult(
            needs_workflow=True,
            reason="multi-step",
            steps=[
                PlanStep(id="s1", instruction="do first"),
                PlanStep(id="s2", instruction="do second", depends_on=["s1"]),
            ],
        )

    async def test_direct_path_answers_without_workflow(self, monkeypatch):
        monkeypatch.setattr(
            "koboi.orchestration.planner.plan_or_skip",
            AsyncMock(return_value=PlanResult(needs_workflow=False, reason="simple")),
        )
        orch = _make_orch()
        monkeypatch.setattr(
            "koboi.orchestration.factory.AgentFactory.create_configured_agent",
            lambda agent_def, client, **kw: _Agent("direct-answer"),
        )
        result = await orch.run("hello", mode="dynamic")
        assert result.execution_mode == "dynamic"
        assert len(result.agent_results) == 1
        assert result.agent_results[0].agent_name == "assistant"

    async def test_workflow_path_runs_planned_steps(self, monkeypatch):
        monkeypatch.setattr(
            "koboi.orchestration.planner.plan_or_skip",
            AsyncMock(return_value=self._workflow_plan()),
        )
        orch = _make_orch()
        monkeypatch.setattr(
            "koboi.orchestration.factory.AgentFactory.create_configured_agent",
            lambda agent_def, client, **kw: _Agent(f"out-{agent_def.name}"),
        )
        result = await orch.run("research and summarize", mode="dynamic")
        assert result.execution_mode == "dynamic"
        names = {r.agent_name for r in result.agent_results}
        assert names == {"s1", "s2"}

    async def test_replan_loop_on_node_failure(self, monkeypatch):
        # First plan needs a workflow; patched _run_single returns failed results,
        # so the re-plan loop runs once more (max_replans=1) before exiting.
        mock_plan = AsyncMock(side_effect=[self._workflow_plan(), self._workflow_plan()])
        monkeypatch.setattr("koboi.orchestration.planner.plan_or_skip", mock_plan)
        orch = _make_orch(max_replans=1)

        async def _failing_single(name, query):
            return AgentResult(agent_name=name, answer="Error: boom", elapsed_seconds=0, tokens_used=0, failed=True)

        orch._run_single = _failing_single
        monkeypatch.setattr(
            "koboi.orchestration.factory.AgentFactory.create_configured_agent",
            lambda agent_def, client, **kw: _Agent("x"),
        )
        result = await orch.run("multi-step that fails", mode="dynamic")
        assert result.execution_mode == "dynamic"
        # planner called twice (initial plan + one re-plan within the loop)
        assert mock_plan.await_count == 2

    async def test_replan_exits_when_second_plan_is_simple(self, monkeypatch):
        # First plan = workflow (failed nodes); replan yields a simple plan -> break.
        monkeypatch.setattr(
            "koboi.orchestration.planner.plan_or_skip",
            AsyncMock(
                side_effect=[
                    self._workflow_plan(),
                    PlanResult(needs_workflow=False, reason="simplified"),
                ]
            ),
        )
        orch = _make_orch(max_replans=2)

        async def _failing_single(name, query):
            return AgentResult(agent_name=name, answer="Error: x", elapsed_seconds=0, tokens_used=0, failed=True)

        orch._run_single = _failing_single
        monkeypatch.setattr(
            "koboi.orchestration.factory.AgentFactory.create_configured_agent",
            lambda agent_def, client, **kw: _Agent("x"),
        )
        result = await orch.run("fails then simplifies", mode="dynamic")
        assert result.execution_mode == "dynamic"


# ---------------------------------------------------------------------------
# _execute_pipeline: enable_dynamic resolve + parallel task exception
# ---------------------------------------------------------------------------


class TestPipelineBranches:
    async def test_enable_dynamic_resolves_in_normal_pipeline(self):
        blueprint = AgentBlueprint(
            name="dyn_resolved",
            domain_label="support",
            system_prompt="x",
            chunks=[],
            chunker_config={},
        )
        orch = _make_orch(
            router=_Router(_decision(agents=["dynamic"], domain_label="support")),
            enable_dynamic=True,
            agents_map={"dyn_resolved": _Agent("resolved-answer")},
        )
        orch._dynamic_builder = _dyn_builder(blueprint)
        result = await orch.run("q", mode="sequential")
        assert any(r.agent_name == "dyn_resolved" for r in result.agent_results)

    async def test_parallel_task_exception_becomes_failed_result(self):
        orch = _make_orch(
            router=_Router(_decision(agents=["a", "b"])),
            agents_map={"a": _Agent("ok"), "b": _Agent("ok")},
        )

        async def _flaky_single(name, query):
            if name == "a":
                raise RuntimeError("task exploded")
            return AgentResult(agent_name=name, answer="ok", elapsed_seconds=0, tokens_used=0)

        orch._run_single = _flaky_single
        result = await orch.run("q", mode="parallel")
        failed = [r for r in result.agent_results if r.failed]
        assert len(failed) == 1
        assert "Error:" in failed[0].answer

    async def test_use_revision_in_streaming_logs_warning(self, caplog):
        orch = _make_orch(
            agents_map={"a1": _Agent("ans")},
            use_revision=True,
        )
        with caplog.at_level("WARNING", logger="koboi.orchestration.orchestrator"):
            events = []
            async for e in orch.run_stream("q", mode="sequential"):
                events.append(e)
        assert any("not supported in streaming mode" in r.message for r in caplog.records)
        assert any(isinstance(e, OrchestrationCompleteEvent) for e in events)


# ---------------------------------------------------------------------------
# run_stream dynamic mode yields the full event trio
# ---------------------------------------------------------------------------


class TestStreamDynamic:
    async def test_stream_dynamic_workflow_path(self, monkeypatch):
        plan = PlanResult(
            needs_workflow=True,
            reason="multi",
            steps=[PlanStep(id="s1", instruction="a"), PlanStep(id="s2", instruction="b", depends_on=["s1"])],
        )
        monkeypatch.setattr("koboi.orchestration.planner.plan_or_skip", AsyncMock(return_value=plan))
        orch = _make_orch()
        monkeypatch.setattr(
            "koboi.orchestration.factory.AgentFactory.create_configured_agent",
            lambda agent_def, client, **kw: _Agent(f"out-{agent_def.name}"),
        )
        events = []
        async for e in orch.run_stream("do the thing", mode="dynamic"):
            events.append(e)
        types_seen = {type(e).__name__ for e in events}
        assert "RoutingDecisionEvent" in types_seen
        assert "AgentDispatchEvent" in types_seen
        assert "AgentResultEvent" in types_seen
        assert "OrchestrationCompleteEvent" in types_seen
