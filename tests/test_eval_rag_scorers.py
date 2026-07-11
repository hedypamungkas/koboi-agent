"""Tests for the mock-safe RAG retrieval/citation/CI scorers and t primitives (Tier 0).

Covers: retrieval_metric math, RetrievalMetricScorer, citation_grounding, bootstrap_ci,
and the t.rankingMetric / t.citationResolves / t.abstains primitives.
"""

from __future__ import annotations

import pytest

from koboi.eval.scorers.ci import BootstrapCIScorer, bootstrap_ci
from koboi.eval.scorers.citation_grounding import CitationGroundingScorer, citation_precision
from koboi.eval.scorers.retrieval_metric import (
    RetrievalMetricScorer,
    compute_ranking_metric,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from koboi.eval.t.assertions import Severity
from koboi.eval.t.context import TestContext
from koboi.types import EvalCase, RunResult


# --------------------------------------------------------------------------
# pure IR metric functions
# --------------------------------------------------------------------------

_RETR = [
    "Annual leave: permanent employees 12 days per year",
    "Paid time off overview",
    "Benefits include health and 401k",
    "AcmeERP price 15000 per year",
    "Remote work 2 days per week",
]


class TestRetrievalMetrics:
    def test_recall_single_gold_in_top_k(self):
        assert recall_at_k(_RETR, "12 days", k=3) == 1.0

    def test_recall_single_gold_below_top_k(self):
        # 'AcmeERP' first appears at rank 4 -> not in top-3
        assert recall_at_k(_RETR, "AcmeERP", k=3) == 0.0
        assert recall_at_k(_RETR, "AcmeERP", k=4) == 1.0

    def test_recall_multi_gold_coverage(self):
        # two gold needles, one in top-3 (12 days) one not (AcmeERP) -> 0.5
        assert recall_at_k(_RETR, ["12 days", "AcmeERP"], k=3) == 0.5

    def test_recall_empty_gold_is_one(self):
        assert recall_at_k(_RETR, "", k=3) == 1.0

    def test_precision_at_k(self):
        # top-3 has one relevant chunk (12 days) -> 1/3
        assert precision_at_k(_RETR, "12 days", k=3) == pytest.approx(1 / 3)

    def test_hit_rate(self):
        assert hit_rate(_RETR, "12 days", k=3) == 1.0
        assert hit_rate(_RETR, "nonexistent", k=3) == 0.0

    def test_mrr_rank1(self):
        assert mrr(_RETR, "12 days", k=5) == 1.0

    def test_mrr_rank4(self):
        # 'AcmeERP' at rank 4 -> 1/4
        assert mrr(_RETR, "AcmeERP", k=5) == pytest.approx(0.25)

    def test_mrr_absent(self):
        assert mrr(_RETR, "nonexistent", k=5) == 0.0

    def test_ndcg_decreases_with_depth(self):
        perfect = ndcg_at_k(["gold a", "gold b", "irrelevant"], "gold", k=3)
        buried = ndcg_at_k(["irrelevant", "irrelevant2", "gold a"], "gold", k=3)
        assert perfect == 1.0
        assert 0.0 < buried < perfect

    def test_compute_ranking_metric_dispatch(self):
        assert compute_ranking_metric("recall", _RETR, "12 days", 5) == 1.0
        with pytest.raises(ValueError):
            compute_ranking_metric("bogus", _RETR, "x", 5)

    def test_compute_clamps_to_unit_interval(self):
        # defensive clamp (no realistic input exceeds 1.0, but the contract is [0,1])
        assert 0.0 <= compute_ranking_metric("ndcg", _RETR, "12 days", 5) <= 1.0


# --------------------------------------------------------------------------
# RetrievalMetricScorer
# --------------------------------------------------------------------------


def _case(gold_needles=None, expected_keywords=None) -> EvalCase:
    return EvalCase(
        name="t",
        user_message="q?",
        metadata={"gold_needles": gold_needles} if gold_needles else {},
        expected_keywords=expected_keywords or [],
    )


def _rag(contents) -> list[dict]:
    return [{"content": c, "score": 0.5, "source": "doc"} for c in contents]


class TestRetrievalMetricScorer:
    async def test_score_recall_from_context(self):
        scorer = RetrievalMetricScorer(metric="recall", k=3)
        score = await scorer.score(_case(gold_needles=["12 days"]), "", {"rag_results": _rag(_RETR)})
        assert score.name == "retrieval_recall"
        assert score.value == 1.0

    async def test_score_no_rag_results(self):
        scorer = RetrievalMetricScorer(metric="recall", k=3)
        score = await scorer.score(_case(gold_needles=["12 days"]), "", {})
        assert score.value == 0.0
        assert "no rag_results" in score.reason

    async def test_score_no_gold(self):
        scorer = RetrievalMetricScorer(metric="recall", k=3)
        score = await scorer.score(_case(), "", {"rag_results": _rag(_RETR)})
        assert score.value == 0.0

    async def test_score_falls_back_to_expected_keywords(self):
        scorer = RetrievalMetricScorer(metric="hit", k=5)
        score = await scorer.score(_case(expected_keywords=["12 days"]), "", {"rag_results": _rag(_RETR)})
        assert score.value == 1.0


# --------------------------------------------------------------------------
# citation grounding
# --------------------------------------------------------------------------


class TestCitationGrounding:
    def test_positional_resolves(self):
        precision, resolved, total = citation_precision("answer [1]", _rag(["a", "b"]))
        assert (precision, resolved, total) == (1.0, 1, 1)

    def test_dangling_positional(self):
        precision, resolved, total = citation_precision("see [9]", _rag(["a", "b"]))
        assert precision == 0.0
        assert (resolved, total) == (0, 1)

    def test_named_source_resolves(self):
        rag = [{"source": "company_policy.md"}]
        precision, _, total = citation_precision("per [Source: company_policy.md]", rag)
        assert precision == 1.0
        assert total == 1

    def test_named_source_unresolved(self):
        rag = [{"source": "company_policy.md"}]
        precision, _, _ = citation_precision("per [Source: hallucinated.md]", rag)
        assert precision == 0.0

    def test_no_citations_vacuous_pass(self):
        precision, resolved, total = citation_precision("plain answer", _rag(["a"]))
        assert (precision, resolved, total) == (1.0, 0, 0)

    async def test_scorer(self):
        scorer = CitationGroundingScorer()
        rag = [{"content": "a", "source": "p"}, {"content": "b", "source": "h"}]
        score = await scorer.score(_case(), "answer [1] and [Source: p]", {"rag_results": rag})
        assert score.name == "citation_grounding"
        assert score.value == 1.0


# --------------------------------------------------------------------------
# bootstrap CI
# --------------------------------------------------------------------------


class TestBootstrapCI:
    def test_empty(self):
        ci = bootstrap_ci([])
        assert (ci.mean, ci.lower, ci.upper, ci.n) == (0.0, 0.0, 0.0, 0)

    def test_single_sample(self):
        ci = bootstrap_ci([0.7])
        assert ci.lower == ci.upper == 0.7
        assert ci.half_width == 0.0

    def test_deterministic_with_seed(self):
        data = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        assert bootstrap_ci(data) == bootstrap_ci(data)

    def test_bounds_bracket_mean(self):
        ci = bootstrap_ci([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        assert ci.lower <= ci.mean <= ci.upper
        assert ci.half_width == pytest.approx((ci.upper - ci.lower) / 2)

    async def test_scorer_reads_metadata_samples(self):
        scorer = BootstrapCIScorer()
        case = EvalCase(name="t", user_message="q?", metadata={"samples": [1.0, 1.0, 1.0, 0.0]})
        score = await scorer.score(case, "", {})
        assert score.name == "bootstrap_ci"
        assert 0.0 < score.value < 1.0  # CI lower bound with one zero

    async def test_scorer_no_samples(self):
        scorer = BootstrapCIScorer()
        score = await scorer.score(_case(), "", {})
        assert score.value == 0.0


# --------------------------------------------------------------------------
# registry wiring
# --------------------------------------------------------------------------


class TestRegistry:
    def test_retrieval_metric_factories_resolve(self):
        from koboi.eval.registry import ScorerRegistry, register_default_scorers

        register_default_scorers()
        for name, metric in [
            ("retrieval_recall", "recall"),
            ("retrieval_precision", "precision"),
            ("retrieval_hit", "hit"),
            ("retrieval_mrr", "mrr"),
            ("retrieval_ndcg", "ndcg"),
        ]:
            scorer = ScorerRegistry.create(name)
            assert scorer.metric == metric
        assert ScorerRegistry.create("citation_grounding").__class__.__name__ == "CitationGroundingScorer"
        assert ScorerRegistry.create("bootstrap_ci").__class__.__name__ == "BootstrapCIScorer"


# --------------------------------------------------------------------------
# t primitives
# --------------------------------------------------------------------------


def _ctx_with_rag(rag_results: list[dict], reply: str = "reply") -> TestContext:
    """Build a TestContext with one turn whose metadata carries rag_results."""
    from koboi.loop import AgentCore
    from koboi.memory import ConversationMemory
    from tests.conftest import MockClient, make_tool_registry

    from koboi.facade import KoboiAgent

    core = AgentCore(client=MockClient([]), memory=ConversationMemory(), tools=make_tool_registry())
    ctx = TestContext(KoboiAgent(core=core))
    ctx._turns.append(RunResult(content=reply, metadata={"rag_results": rag_results}, success=True))  # noqa: SLF001
    return ctx


def _outcome(ctx: TestContext, idx: int):
    return ctx.collect()[idx].outcome()


class TestRankingMetricPrimitive:
    async def test_recall_pass(self):
        ctx = _ctx_with_rag(_rag(_RETR))
        ctx.rankingMetric("12 days", k=3, metric="recall", min_score=1.0)
        out = _outcome(ctx, 0)
        assert out.passed and out.value == 1.0

    async def test_mrr_reports_rank_value(self):
        ctx = _ctx_with_rag(_rag(_RETR))
        ctx.rankingMetric("AcmeERP", k=5, metric="mrr", min_score=0.5)
        out = _outcome(ctx, 0)
        assert out.value == pytest.approx(0.25)  # rank 4
        assert not out.passed  # 0.25 < 0.5

    async def test_no_rag_results(self):
        ctx = _ctx_with_rag([])
        ctx.rankingMetric("12 days", k=3, metric="recall", min_score=1.0)
        out = _outcome(ctx, 0)
        assert not out.passed and out.value == 0.0

    async def test_soft_does_not_gate(self):
        ctx = _ctx_with_rag(_rag(_RETR))
        ctx.rankingMetric("AcmeERP", k=5, metric="mrr", min_score=0.5, severity=Severity.SOFT)
        out = _outcome(ctx, 0)
        assert out.value == pytest.approx(0.25)


class TestCitationResolvesPrimitive:
    async def test_positional_resolves(self):
        ctx = _ctx_with_rag(_rag(["a", "b"]), reply="answer [1]")
        ctx.citationResolves(1)
        assert _outcome(ctx, 0).passed

    async def test_positional_dangling(self):
        ctx = _ctx_with_rag(_rag(["a", "b"]), reply="see [9]")
        ctx.citationResolves(1)  # [1] not even cited
        assert not _outcome(ctx, 0).passed

    async def test_all_resolve(self):
        ctx = _ctx_with_rag(_rag(["a", "b"]), reply="[1] and [2]")
        ctx.citationResolves()
        assert _outcome(ctx, 0).passed

    async def test_all_with_dangling(self):
        ctx = _ctx_with_rag(_rag(["a", "b"]), reply="[1] and [9]")
        ctx.citationResolves()
        out = _outcome(ctx, 0)
        assert not out.passed and out.value < 1.0


class TestAbstainsPrimitive:
    async def test_empty_rag_abstains(self):
        ctx = _ctx_with_rag([], reply="anything")
        ctx.abstains()
        assert _outcome(ctx, 0).passed

    async def test_refusal_marker_abstains(self):
        ctx = _ctx_with_rag(_rag(["noise"]), reply="I don't have that information.")
        ctx.abstains()
        assert _outcome(ctx, 0).passed

    async def test_neither_empty_nor_refusal_fails(self):
        ctx = _ctx_with_rag(_rag(["12 days"]), reply="The answer is 12 days.")
        ctx.abstains()
        assert not _outcome(ctx, 0).passed


# --------------------------------------------------------------------------
# live_ready / require_live (Tier 2 self-skip)
# --------------------------------------------------------------------------


def _ctx_scripted(rag_results=None, reply="reply") -> TestContext:
    """Build a TestContext whose agent uses a ScriptedClient (mimics --mock)."""
    from koboi.facade import KoboiAgent
    from koboi.loop import AgentCore
    from koboi.memory import ConversationMemory
    from koboi.eval.t.mock import ScriptedClient, scripted_response
    from tests.conftest import make_tool_registry

    core = AgentCore(
        client=ScriptedClient([scripted_response(reply)]),
        memory=ConversationMemory(),
        tools=make_tool_registry(),
    )
    ctx = TestContext(KoboiAgent(core=core))
    ctx._turns.append(RunResult(content=reply, metadata={"rag_results": rag_results or []}, success=True))  # noqa: SLF001
    return ctx


class TestLiveReady:
    def test_live_ready_false_under_scripted_client(self):
        # ScriptedClient => mock mode => never live-ready (deterministic branch).
        assert _ctx_scripted().live_ready() is False

    def test_live_ready_accepts_no_extra(self):
        # Retrieval-only live evals (semantic/hybrid) pass extra=None (no judge dep).
        # Still False here because the scripted client signals mock mode.
        assert _ctx_scripted().live_ready(extra=None) is False

    async def test_require_live_skips_and_records(self):
        ctx = _ctx_scripted()
        assert ctx.require_live() is False
        names = [a.name for a in ctx.collect()]
        assert "live_skip" in names
        skip = [a for a in ctx.collect() if a.name == "live_skip"][0]
        out = skip.outcome()
        assert out.passed is True and out.value == 1.0  # SOFT pass, not a gate failure


# --------------------------------------------------------------------------
# loop._run_metadata rag_results stamp (Tier 2/3 additive keys)
# --------------------------------------------------------------------------


class TestRunMetadataStamp:
    def test_rag_results_stamp_carries_retrieval_method_and_doc_id(self):
        """The additive stamp keys (retrieval_method, doc_id) let semantic/hybrid evals
        detect a silent degrade and let golden qrels match by stable id."""
        from koboi.loop import AgentCore
        from koboi.memory import ConversationMemory
        from koboi.rag.types import Chunk, RetrievalResult
        from tests.conftest import MockClient, make_tool_registry

        core = AgentCore(client=MockClient([]), memory=ConversationMemory(), tools=make_tool_registry())

        class _FakeAug:
            last_results = [
                RetrievalResult(
                    chunk=Chunk(id="c1", doc_id="company_policy.md", content="Permanent: 12 days"),
                    score=0.9,
                    retrieval_method="keyword",
                )
            ]

        core.augmentation = _FakeAug()  # type: ignore[assignment]
        meta = core._run_metadata(resumed=False, last_step=0)  # noqa: SLF001
        rr = meta["rag_results"][0]
        assert rr["content"] == "Permanent: 12 days"
        assert rr["score"] == 0.9
        assert rr["retrieval_method"] == "keyword"
        assert rr["doc_id"] == "company_policy.md"
