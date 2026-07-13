"""Tests for W2 deep-research: research primitives (koboi.orchestration.research),
plan_research (planner), and the Orchestrator._run_deep_research loop."""

from __future__ import annotations

import json

from koboi.events import OrchestrationCompleteEvent, TextDeltaEvent
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.planner import plan_research
from koboi.orchestration.research import (
    CoverageEvaluator,
    ResearchBudget,
    ResearchContext,
    SourceStore,
)
from koboi.orchestration.router import KeywordRouter
from koboi.types import AgentResponse, ToolCall


# ---------------------------------------------------------------------------
# FakeClient: routes complete() by prompt content; complete_stream yields a report.
# ---------------------------------------------------------------------------


class _FakeClient:
    """Duck-typed LLM client dispatching on prompt content (planner / coverage / synthesis / node).

    ``emit_search_call``: when True, a node's first turn emits a ``web_search`` tool_call (so the
    injected CountingProvider meters a real call); the next turn (with the tool result) returns
    ``node_answer``. Used to exercise real per-call budget metering + provider wiring (A0/A4).
    """

    def __init__(
        self,
        node_answer: str = "Found: the topic is X and Y.",
        coverage_score: float = 0.4,
        follow_ups: list[str] | None = None,
        plan_needs_workflow: bool = True,
        synthesis: str = "## Report\nThe topic is X [1] and Y [1].",
        emit_search_call: bool = False,
    ) -> None:
        self.node_answer = node_answer
        self.coverage_score = coverage_score
        self.follow_ups = follow_ups if follow_ups is not None else ["deeper query"]
        self.plan_needs_workflow = plan_needs_workflow
        self.synthesis = synthesis
        self.emit_search_call = emit_search_call
        # AgentCore's SESSION_START hook emit reads client.model; the duck-typed fake needs it.
        self.model = "fake-model"
        self.provider = "fake"

    async def complete(self, messages, tools=None, response_format=None):
        text = " ".join(m.get("content", "") for m in messages)
        if "research planner" in text:
            if not self.plan_needs_workflow:
                return AgentResponse(
                    content=json.dumps({"needs_workflow": False, "reason": "simple", "steps": []}),
                    tool_calls=[],
                )
            return AgentResponse(
                content=json.dumps(
                    {
                        "needs_workflow": True,
                        "reason": "research",
                        "steps": [
                            {
                                "id": "research_topic",
                                "instruction": "Investigate the topic",
                                "depends_on": [],
                                "search_queries": ["topic overview"],
                            }
                        ],
                    }
                ),
                tool_calls=[],
            )
        if "evaluating how thoroughly" in text:
            return AgentResponse(
                content=json.dumps(
                    {
                        "overall_score": self.coverage_score,
                        "coverage": {"Investigate the topic": self.coverage_score},
                        "follow_up_queries": self.follow_ups,
                    }
                ),
                tool_calls=[],
            )
        if "synthesizing a cited research report" in text:
            return AgentResponse(content=self.synthesis, tool_calls=[])
        # Node turn: optionally emit one web_search call before the final answer.
        if self.emit_search_call and not any(m.get("role") == "tool" for m in messages):
            return AgentResponse(
                content="",
                tool_calls=[ToolCall(id="tc_search", name="web_search", arguments=json.dumps({"query": "python"}))],
            )
        return AgentResponse(content=self.node_answer, tool_calls=[])

    async def complete_stream(self, messages, tools=None):
        yield TextDeltaEvent(content=self.synthesis)


# ---------------------------------------------------------------------------
# ResearchBudget
# ---------------------------------------------------------------------------


class TestResearchBudget:
    def test_remaining_until_cap(self):
        b = ResearchBudget(max_searches=2, max_fetches=5, max_depth=3)
        assert b.remaining() is True
        b.record_searches(2)
        assert b.remaining() is False  # searches cap hit

    def test_fetch_cap(self):
        b = ResearchBudget(max_searches=10, max_fetches=1)
        b.record_fetches(1)
        assert b.remaining() is False

    def test_no_token_cap_when_zero(self):
        b = ResearchBudget(max_tokens=0)
        b.record_tokens(999999)
        assert b.remaining() is True


# ---------------------------------------------------------------------------
# SourceStore
# ---------------------------------------------------------------------------


class TestSourceStore:
    def test_add_returns_running_citation_id(self):
        s = SourceStore()
        assert s.add_findings("a", "text a") == 1
        assert s.add_findings("b", "text b") == 2

    def test_dedup_by_node_updates_in_place(self):
        s = SourceStore()
        s.add_findings("a", "first")
        cid = s.add_findings("a", "second")  # same node -> same id, text replaced
        assert cid == 1
        assert s.resolve(1) == "second"
        assert s.citation_ids() == {1}

    def test_empty_text_returns_zero(self):
        assert SourceStore().add_findings("a", "   ") == 0

    def test_format_for_synthesis_numbered(self):
        s = SourceStore()
        s.add_findings("a", "alpha")
        s.add_findings("b", "beta")
        out = s.format_for_synthesis()
        assert "[1]" in out and "alpha" in out
        assert "[2]" in out and "beta" in out

    def test_sources_list(self):
        s = SourceStore()
        s.add_findings("a", "alpha")
        assert s.sources_list() == [{"citation_id": 1, "node_id": "a"}]


# ---------------------------------------------------------------------------
# ResearchContext round-trip
# ---------------------------------------------------------------------------


class TestResearchContext:
    def test_json_round_trip(self):
        ctx = ResearchContext(sub_questions=["q1", "q2"], depth=2)
        ctx.add_findings("node_a", "finding a")
        ctx.coverage_map = {"q1": 0.8}
        ctx.budget.record_searches(3)
        ctx.graph_run_id = "run-xyz"

        restored = ResearchContext.from_json(ctx.to_json())
        assert restored.sub_questions == ["q1", "q2"]
        assert restored.depth == 2
        assert restored.graph_run_id == "run-xyz"
        assert restored.source_store.resolve(1) == "finding a"
        assert restored.coverage_map == {"q1": 0.8}
        assert restored.budget.used_searches == 3

    def test_final_report_round_trip(self):
        # W7: final_report survives serialization (so GET /v1/sessions/{id} can surface it).
        ctx = ResearchContext(query="Tell me about X")
        ctx.final_report = "## Report\nThe topic is X [1].\n\n## Sources\n- [1] node_a"
        restored = ResearchContext.from_json(ctx.to_json())
        assert restored.query == "Tell me about X"
        assert restored.final_report == ctx.final_report
        # Default is empty string (backward-compatible with older journaled JSON).
        assert ResearchContext.from_json(ResearchContext().to_json()).final_report == ""


# ---------------------------------------------------------------------------
# CoverageEvaluator
# ---------------------------------------------------------------------------


class TestCoverageEvaluator:
    async def test_parses_score_and_followups(self):
        ctx = ResearchContext(sub_questions=["q1"])
        ctx.add_findings("n", "a finding")
        score, follow_ups, covmap = await CoverageEvaluator(_FakeClient(coverage_score=0.5)).evaluate(ctx)
        assert score == 0.5
        assert follow_ups == ["deeper query"]
        assert covmap == {"Investigate the topic": 0.5}

    async def test_fail_safe_on_empty_findings(self):
        ctx = ResearchContext(sub_questions=["q1"])  # no findings
        score, follow_ups, _ = await CoverageEvaluator(_FakeClient()).evaluate(ctx)
        assert score == 1.0 and follow_ups == []

    async def test_fail_safe_on_client_error(self):
        class _Boom:
            async def complete(self, *a, **k):
                raise RuntimeError("boom")

        ctx = ResearchContext(sub_questions=["q1"])
        ctx.add_findings("n", "finding")
        score, follow_ups, _ = await CoverageEvaluator(_Boom()).evaluate(ctx)
        assert score == 1.0 and follow_ups == []  # stop iterating


# ---------------------------------------------------------------------------
# plan_research
# ---------------------------------------------------------------------------


class TestPlanResearch:
    async def test_returns_steps_with_search_queries(self):
        plan = await plan_research(_FakeClient(), "research X")
        assert plan.needs_workflow is True
        assert len(plan.steps) == 1
        assert plan.steps[0].search_queries == ["topic overview"]

    async def test_fail_safe_when_simple(self):
        plan = await plan_research(_FakeClient(plan_needs_workflow=False), "hi")
        assert plan.needs_workflow is False
        assert plan.steps == []


# ---------------------------------------------------------------------------
# Orchestrator._run_deep_research (integration)
# ---------------------------------------------------------------------------


def _orch(client, research, tmp_path):
    return Orchestrator(
        client=client,
        router=KeywordRouter(),
        research=research,
        dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
    )


class TestRunDeepResearch:
    async def test_iterates_to_max_depth_then_synthesizes(self, tmp_path):
        # coverage 0.4 < threshold 0.9 -> iterate until max_depth=2.
        orch = _orch(_FakeClient(coverage_score=0.4), {"max_depth": 2, "coverage_threshold": 0.9}, tmp_path)
        events = [e async for e in orch._run_deep_research("Tell me about X")]

        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete and complete[0].execution_mode == "deep_research"
        assert complete[0].metadata["depth"] == 2  # iterated to max_depth
        assert complete[0].metadata["research_sources"]  # citations carried
        # Cited synthesis: inline marker + Sources footer.
        assert "[1]" in complete[0].final_answer
        assert "## Sources" in complete[0].final_answer

    async def test_stops_early_when_covered(self, tmp_path):
        # coverage 0.95 >= threshold 0.7 -> stop after 1 round.
        orch = _orch(_FakeClient(coverage_score=0.95), {"max_depth": 5, "coverage_threshold": 0.7}, tmp_path)
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete and complete[0].metadata["depth"] == 1  # one round, covered

    async def test_budget_hard_stop(self, tmp_path):
        # W4: real per-call metering -- the node actually calls web_search (emit_search_call),
        # the CountingProvider charges it, and after 1 search (max_searches=1) the budget is
        # exhausted -> the loop stops after round 1.
        orch = _orch(
            _FakeClient(coverage_score=0.1, emit_search_call=True),
            {"max_depth": 5, "coverage_threshold": 0.9, "max_searches": 1, "max_fetches": 50},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete and complete[0].metadata["depth"] == 1  # budget stopped it at round 1

    async def test_journals_research_context(self, tmp_path):
        db_path = str(tmp_path / "r.db")
        orch = _orch(_FakeClient(coverage_score=0.95), {"max_depth": 1, "coverage_threshold": 0.7}, tmp_path)
        _ = [e async for e in orch._run_deep_research("Tell me about X")]
        # The run journaled its context (one row).
        rows = DagScheduler.load_research_context(db_path, "__never__")  # ensure read path works
        assert rows is None  # unknown run_id -> None (sanity)
        # The real run_id is in metadata; verify a row exists for it by reading all rows.
        import sqlite3

        from koboi.memory_sqlite import ensure_research_context_table

        conn = sqlite3.connect(db_path)
        try:
            ensure_research_context_table(conn)
            count = conn.execute("SELECT COUNT(*) FROM research_context").fetchone()[0]
        finally:
            conn.close()
        assert count >= 1  # at least one journal row written

    async def test_persists_final_report_session_scoped(self, tmp_path):
        # W7: after synthesis the final report is journaled, tagged with session_id,
        # and recoverable via load_research_context_for_session.
        db_path = str(tmp_path / "r.db")
        orch = Orchestrator(
            client=_FakeClient(coverage_score=0.95, synthesis="## Report\nX [1]."),
            router=KeywordRouter(),
            research={"max_depth": 1, "coverage_threshold": 0.7},
            dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=db_path),
            session_id="sess-123",
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete and complete[0].final_answer  # synthesis produced a report

        ctx_json = DagScheduler.load_research_context_for_session(db_path, "sess-123")
        assert ctx_json is not None, "session-scoped research context not found"
        ctx = ResearchContext.from_json(ctx_json)
        assert ctx.query == "Tell me about X"
        assert ctx.final_report  # the synthesized report was persisted
        assert "[1]" in ctx.final_report
        # A different session has no row.
        assert DagScheduler.load_research_context_for_session(db_path, "other-session") is None

    async def test_no_session_id_still_journals(self, tmp_path):
        # Non-server callers (no session_id) still journal -- the column stays NULL,
        # and load_research_context_for_session returns None (no session tag to match).
        db_path = str(tmp_path / "r.db")
        orch = _orch(_FakeClient(coverage_score=0.95), {"max_depth": 1, "coverage_threshold": 0.7}, tmp_path)
        _ = [e async for e in orch._run_deep_research("Tell me about X")]
        assert DagScheduler.load_research_context_for_session(db_path, "any") is None  # untagged

    async def test_persists_findings_to_corpus_file(self, tmp_path):
        out_path = str(tmp_path / "findings.jsonl")
        orch = _orch(
            _FakeClient(coverage_score=0.95),
            {"max_depth": 1, "coverage_threshold": 0.7, "persist_findings": out_path},
            tmp_path,
        )
        _ = [e async for e in orch._run_deep_research("Tell me about X")]
        # The run wrote its findings as jsonl (one row per source, with text).
        import json
        import os

        assert os.path.exists(out_path)
        rows = [json.loads(line) for line in open(out_path, encoding="utf-8") if line.strip()]
        assert rows  # at least one finding row
        assert {"citation_id", "node_id", "text"} <= set(rows[0])


class TestWave4Correctness:
    """A1/A3/A5/A7/A8 — the W4 correctness fixes."""

    async def test_a1_verify_citations_drops_unresolvable(self):
        # A1: [1] resolves -> kept; [99] does not -> stripped.
        from koboi.orchestration.orchestrator import _verify_citations
        from koboi.orchestration.research import ResearchContext

        ctx = ResearchContext()
        ctx.add_findings("node_a", "real finding")
        cleaned, referenced = _verify_citations("see [1] and [99] for details", ctx)
        assert "[1]" in cleaned
        assert "[99]" not in cleaned
        assert referenced == [1]

    async def test_a3_error_answer_not_a_finding(self, tmp_path):
        # A3: a node whose answer is "Error: ..." must NOT become a cited finding.
        orch = _orch(
            _FakeClient(node_answer="Error: connection failed", coverage_score=0.95),
            {"max_depth": 1, "coverage_threshold": 0.7},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete
        assert complete[0].metadata["research_sources"] == []  # no findings from the error node

    async def test_a5_agent_results_dedup_across_rounds(self, tmp_path):
        # A5: two rounds reusing the same step id -> one agent_results entry (not duplicated).
        orch = _orch(
            _FakeClient(coverage_score=0.1),  # < 0.9 -> iterate
            {"max_depth": 2, "coverage_threshold": 0.9},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete
        names = [r.agent_name for r in complete[0].agent_results]
        assert names.count("research_topic") == 1  # deduped, not duplicated across rounds

    async def test_a7_simple_fallback_stamps_deep_research(self, tmp_path):
        # A7: a simple request (plan_needs_workflow=False) -> execution_mode="deep_research"
        # (not "dynamic" from the old _run_dynamic fallback).
        orch = _orch(
            _FakeClient(plan_needs_workflow=False),
            {"max_depth": 3, "coverage_threshold": 0.7},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("hi")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete and complete[0].execution_mode == "deep_research"

    async def test_a8_max_depth_zero_clamped(self, tmp_path):
        # A8: max_depth=0 is clamped to 1 -> exactly one round (no crash).
        orch = _orch(
            _FakeClient(coverage_score=0.1),
            {"max_depth": 0, "coverage_threshold": 0.9},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete and complete[0].metadata["depth"] == 1

    async def test_a0_node_reaches_configured_provider(self, tmp_path):
        # A0/A4: a node that calls web_search (emit_search_call) must go through the
        # CountingProvider (which enforces the budget). max_searches=1 -> one search ->
        # budget exhausted -> loop stops after round 1 (proves the node reached the wired
        # provider, not the default mock which wouldn't be counted).
        orch = _orch(
            _FakeClient(coverage_score=0.1, emit_search_call=True),
            {"max_depth": 5, "coverage_threshold": 0.9, "max_searches": 1},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete and complete[0].metadata["depth"] == 1  # budget stopped it


class TestWave5Tracing:
    """B4: orchestrator-level LLM calls emit hook events (SESSION_START/END + PRE/POST_LLM_CALL)
    so Langfuse + other hooks trace them (they bypass AgentCore._emit)."""

    async def test_research_run_emits_session_hooks(self, tmp_path):
        from koboi.hooks.chain import HookEvent

        class _SpyChain:
            def __init__(self) -> None:
                self.events: list = []

            async def emit(self, ctx):
                self.events.append(ctx.event)
                return ctx

            def find_hook(self, _pred):
                return None

        spy = _SpyChain()
        orch = Orchestrator(
            client=_FakeClient(plan_needs_workflow=False),  # simple -> fallback path (fast)
            router=KeywordRouter(),
            research={"max_depth": 1, "coverage_threshold": 0.7},
            dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
            hook_chain=spy,
        )
        _ = [e async for e in orch._run_deep_research("hi")]
        # SESSION_START (open trace) + the plan PRE/POST + SESSION_END (close trace).
        assert HookEvent.SESSION_START in spy.events
        assert HookEvent.SESSION_END in spy.events
        assert HookEvent.PRE_LLM_CALL in spy.events  # the plan_research call is traced
        assert HookEvent.POST_LLM_CALL in spy.events

    async def test_no_hook_chain_is_safe(self, tmp_path):
        # No hook_chain -> _emit_research_hook is a no-op; the run still completes.
        orch = _orch(_FakeClient(plan_needs_workflow=False), {"max_depth": 1, "coverage_threshold": 0.7}, tmp_path)
        events = [e async for e in orch._run_deep_research("hi")]
        assert any(isinstance(e, OrchestrationCompleteEvent) for e in events)


class TestWave6MetadataPropagation:
    """C1a: _run_orchestrator propagates OrchestratorResult.metadata into RunResult.metadata."""

    async def test_research_metadata_reaches_run_result(self, tmp_path):
        from koboi.facade import _run_orchestrator

        orch = _orch(
            _FakeClient(coverage_score=0.95),  # covered after 1 round
            {"max_depth": 1, "coverage_threshold": 0.7},
            tmp_path,
        )
        orch.default_mode = "deep_research"  # so _run_orchestrator dispatches to deep_research
        result = await _run_orchestrator(orch, "Tell me about X")
        # The orchestrator's metadata (research_sources/coverage/depth) propagated into RunResult.
        assert "research_sources" in result.metadata
        assert "coverage" in result.metadata
        assert "depth" in result.metadata
        assert result.metadata["execution_mode"] == "deep_research"


class TestHigh3Resume:
    """HIGH-3: W5.1 resume-and-finish -- journal a ctx, then resume from it."""

    async def test_resume_synthesizes_from_journaled_ctx(self, tmp_path):
        from koboi.orchestration.dag_scheduler import DagScheduler as _DS

        db_path = str(tmp_path / "resume.db")
        # Step 1: run deep_research once -> journals a ctx with findings.
        orch = _orch(
            _FakeClient(coverage_score=0.95, synthesis="## Report\nFound [1] something."),
            {"max_depth": 1, "coverage_threshold": 0.7},
            tmp_path,
        )
        # Override the db_path to our test file.
        orch._dag_scheduler = _DS(agents_map={}, deps={}, db_path=db_path)
        # Override the db_path to our test file.
        orch._dag_scheduler = DagScheduler(agents_map={}, deps={}, db_path=db_path)
        _ = [e async for e in orch._run_deep_research("Research Python")]

        # Step 2: load the latest journaled ctx.
        ctx_json = _DS.load_latest_research_context(db_path)
        assert ctx_json is not None, "expected a journaled research context"

        # Step 3: set _resume_ctx_json + run again -> resume branch fires.
        orch2 = _orch(
            _FakeClient(plan_needs_workflow=True, coverage_score=0.95, synthesis="## Report\nResumed [1] finding."),
            {"max_depth": 1, "coverage_threshold": 0.7},
            tmp_path,
        )
        orch2._dag_scheduler = DagScheduler(agents_map={}, deps={}, db_path=db_path)
        orch2._resume_ctx_json = ctx_json
        events = [e async for e in orch2._run_deep_research("Research Python")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete, "expected OrchestrationCompleteEvent from resume"
        assert complete[0].metadata.get("resumed") is True
        assert "[1]" in complete[0].final_answer


class TestHigh5SynthesisFallback:
    """HIGH-5: _synthesize_research falls back to raw findings on LLM failure."""

    async def test_synthesis_fallback_on_llm_error(self, tmp_path):
        class _SynthesisBoomClient(_FakeClient):
            async def complete(self, messages, tools=None, response_format=None):
                text = " ".join(m.get("content", "") for m in messages)
                if "synthesizing a cited research report" in text:
                    raise RuntimeError("LLM down")
                return await super().complete(messages, tools, response_format)

        orch = _orch(
            _SynthesisBoomClient(coverage_score=0.95),
            {"max_depth": 1, "coverage_threshold": 0.7},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Research Python")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete
        # Fallback: the report is the raw format_for_synthesis() output (numbered findings),
        # NOT a crash. The findings contain the node_answer text.
        assert "Found: the topic is X and Y" in complete[0].final_answer


class TestHigh6LoadLatestOrdering:
    """HIGH-6: load_latest_research_context returns the most recent row by updated_at."""

    def test_returns_latest_by_updated_at(self, tmp_path):
        import time as _time

        from koboi.orchestration.dag_scheduler import DagScheduler
        from koboi.orchestration.research import ResearchContext

        db_path = str(tmp_path / "ordering.db")
        ctx_a = ResearchContext(query="query_A")
        ctx_a.add_findings("nodeA", "finding A")
        ctx_b = ResearchContext(query="query_B")
        ctx_b.add_findings("nodeB", "finding B")

        DagScheduler.persist_research_context(db_path, "run_A", ctx_a.to_json())
        _time.sleep(0.02)  # ensure updated_at is strictly later
        DagScheduler.persist_research_context(db_path, "run_B", ctx_b.to_json())

        latest = DagScheduler.load_latest_research_context(db_path)
        assert latest is not None
        restored = ResearchContext.from_json(latest)
        assert restored.query == "query_B"  # the later run, not the earlier

    def test_returns_none_when_empty(self, tmp_path):
        from koboi.orchestration.dag_scheduler import DagScheduler

        assert DagScheduler.load_latest_research_context(str(tmp_path / "empty.db")) is None


class TestMedium7CoverageMalformed:
    """M7: CoverageEvaluator defensive branches for malformed LLM responses."""

    async def test_non_dict_response_defaults_to_stop(self):
        # LLM returns a JSON array (not an object) -> score 1.0 (stop iterating).
        from koboi.orchestration.research import CoverageEvaluator, ResearchContext

        class _ArrayClient:
            model = "fake"

            async def complete(self, messages, tools=None, response_format=None):
                return AgentResponse(content='["not", "an", "object"]', tool_calls=[])

        ctx = ResearchContext()
        ctx.add_findings("n", "finding")
        score, follow_ups, covmap = await CoverageEvaluator(_ArrayClient()).evaluate(ctx)
        assert score == 1.0
        assert follow_ups == []

    async def test_string_score_value_error_defaults_to_stop(self):
        from koboi.orchestration.research import CoverageEvaluator, ResearchContext

        class _StringScoreClient:
            model = "fake"

            async def complete(self, messages, tools=None, response_format=None):
                return AgentResponse(
                    content='{"overall_score": "high", "coverage": {}, "follow_up_queries": []}',
                    tool_calls=[],
                )

        ctx = ResearchContext()
        ctx.add_findings("n", "finding")
        score, _, _ = await CoverageEvaluator(_StringScoreClient()).evaluate(ctx)
        assert score == 1.0  # float("high") -> ValueError -> default 1.0

    async def test_coverage_as_list_not_dict(self):
        from koboi.orchestration.research import CoverageEvaluator, ResearchContext

        class _ListCovClient:
            model = "fake"

            async def complete(self, messages, tools=None, response_format=None):
                return AgentResponse(
                    content='{"overall_score": 0.5, "coverage": [1, 2, 3], "follow_up_queries": []}',
                    tool_calls=[],
                )

        ctx = ResearchContext(sub_questions=["q1"])  # must have sub_questions + findings
        ctx.add_findings("n", "finding")
        score, _, covmap = await CoverageEvaluator(_ListCovClient()).evaluate(ctx)
        assert score == 0.5
        assert covmap == {}  # list not dict -> empty covmap

    async def test_score_out_of_range_clamped(self):
        from koboi.orchestration.research import CoverageEvaluator, ResearchContext

        class _HighScoreClient:
            model = "fake"

            async def complete(self, messages, tools=None, response_format=None):
                return AgentResponse(
                    content='{"overall_score": 1.5, "coverage": {}, "follow_up_queries": []}',
                    tool_calls=[],
                )

        ctx = ResearchContext()
        ctx.add_findings("n", "finding")
        score, _, _ = await CoverageEvaluator(_HighScoreClient()).evaluate(ctx)
        assert score == 1.0  # clamped from 1.5


class TestMedium8EmptyFollowupsStops:
    """M8: coverage < threshold but follow_ups=[] -> loop generates generic follow-ups + continues.

    After Fix 3: the loop NO LONGER breaks when follow_ups is empty but coverage is below
    threshold. It generates generic drill queries from the sub-questions and continues.
    """

    async def test_continues_when_no_follow_ups_but_low_coverage(self, tmp_path):
        # coverage 0.3 < threshold 0.9 BUT follow_ups=[] -> Fix 3 generates generic follow-ups
        # from sub_questions -> the loop continues to depth 2 (not depth 1).
        orch = _orch(
            _FakeClient(coverage_score=0.3, follow_ups=[]),
            {"max_depth": 5, "coverage_threshold": 0.9},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete
        assert complete[0].metadata["depth"] >= 2  # iterated at least once more (Fix 3 safety net)

    async def test_stops_when_no_follow_ups_but_good_coverage(self, tmp_path):
        # coverage 0.95 >= threshold 0.7 AND follow_ups=[] -> correct stop (no Fix 3 needed).
        orch = _orch(
            _FakeClient(coverage_score=0.95, follow_ups=[]),
            {"max_depth": 5, "coverage_threshold": 0.7},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
        assert complete
        assert complete[0].metadata["depth"] == 1  # coverage >= threshold -> correct stop


class TestMedium11WebConfProviderWiring:
    """M11: web_conf with a spy provider proves the configured provider reaches nodes."""

    async def test_configured_provider_reaches_node(self, tmp_path):
        from koboi.web.base import BaseSearchProvider
        from koboi.web.registry import search_provider_registry
        from koboi.web.types import SearchResult

        # Class-level call tracking (build_search_provider creates a NEW instance from the class,
        # so instance-level tracking won't work -- must track at class scope).
        class _SpySearch(BaseSearchProvider):
            calls: list[str] = []  # class-level -- shared across instances

            def __init__(self) -> None:
                pass  # no instance state needed

            async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
                _SpySearch.calls.append(query)
                return [SearchResult(title="spy", url="https://spy.example", snippet="s")]

        _SpySearch.calls = []  # reset before the run
        search_provider_registry.register("__spy_test__", _SpySearch, description="spy")
        try:
            orch = Orchestrator(
                client=_FakeClient(coverage_score=0.95, emit_search_call=True),
                router=KeywordRouter(),
                research={"max_depth": 1, "coverage_threshold": 0.7, "max_searches": 5},
                dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "spy.db")),
                web_conf={"search": {"provider": "__spy_test__"}},
            )
            _ = [e async for e in orch._run_deep_research("Research Python")]
            # The spy was called through the CountingProvider chain (A0 wiring proven end-to-end).
            assert len(_SpySearch.calls) > 0, "spy provider should have been called by a research node"
        finally:
            if "__spy_test__" in search_provider_registry._entries:
                del search_provider_registry._entries["__spy_test__"]
