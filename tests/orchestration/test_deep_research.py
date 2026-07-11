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
from koboi.types import AgentResponse


# ---------------------------------------------------------------------------
# FakeClient: routes complete() by prompt content; complete_stream yields a report.
# ---------------------------------------------------------------------------


class _FakeClient:
    """Duck-typed LLM client dispatching on prompt content (planner / coverage / node)."""

    def __init__(
        self,
        node_answer: str = "Found: the topic is X and Y.",
        coverage_score: float = 0.4,
        follow_ups: list[str] | None = None,
        plan_needs_workflow: bool = True,
    ) -> None:
        self.node_answer = node_answer
        self.coverage_score = coverage_score
        self.follow_ups = follow_ups if follow_ups is not None else ["deeper query"]
        self.plan_needs_workflow = plan_needs_workflow

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
        return AgentResponse(content=self.node_answer, tool_calls=[])

    async def complete_stream(self, messages, tools=None):
        yield TextDeltaEvent(content="## Report\nThe topic is X [1] and Y [1].")


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
        # max_searches=1 -> after the first round's 1 search, budget exhausted -> stop.
        orch = _orch(
            _FakeClient(coverage_score=0.1),
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
