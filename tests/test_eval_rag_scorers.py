"""Tests for the mock-safe RAG retrieval/citation/CI scorers and t primitives (Tier 0).

Covers: retrieval_metric math, RetrievalMetricScorer, citation_grounding, bootstrap_ci,
and the t.rankingMetric / t.citationResolves / t.abstains primitives.
"""

from __future__ import annotations

import pytest

from koboi.eval.scorers.ci import BootstrapCIScorer, bootstrap_ci
from koboi.eval.scorers.citation_grounding import CitationGroundingScorer, citation_precision
from koboi.eval.scorers.ragas_scorer import _composite_weighted, _extract_ragas_score, _judge_openai_creds
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

    def test_single_sample_is_uninformative_full_width(self):
        # N=1 carries ~no spread info: the honest conservative CI is full-width, so a
        # CI-lower-bound gate FAILS at N=1 (you cannot pass on one sample).
        ci = bootstrap_ci([0.7])
        assert ci.n == 1
        assert ci.mean == 0.7
        assert ci.lower == 0.0 and ci.upper == 1.0
        assert ci.half_width == 0.5

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


# --------------------------------------------------------------------------
# RAGAS composite honesty fixes: _composite_weighted + _extract_ragas_score
# --------------------------------------------------------------------------


class _FakeResult:
    """Minimal stand-in for a ragas EvaluationResult (dict-like over _scores_dict)."""

    def __init__(self, scores_dict: dict):
        self._scores_dict = scores_dict

    def __getitem__(self, key):
        return self._scores_dict[key]  # KeyError if absent -> triggers prefix fallback


class TestCompositeWeighted:
    """The composite must keep real zeros (drag the mean down) and only exclude
    metrics that did not run (None) -- the bug dropped zeros, manufacturing 1.0."""

    def _w(self):
        return {"faithfulness": 0.3, "answer_relevancy": 0.3, "context_precision": 0.2, "context_recall": 0.2}

    def test_real_zero_counts(self):
        # 1,1,1,0 -> (0.3+0.3+0.2+0)/1.0 = 0.8 (was 1.0 under the old drop-zeros bug).
        s = {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 0.0}
        assert _composite_weighted(self._w(), s) == pytest.approx(0.8)

    def test_none_metric_excluded(self):
        # 1,1,1,None -> excluded, not penalized: (0.3+0.3+0.2)/0.8 = 1.0.
        s = {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": None}
        assert _composite_weighted(self._w(), s) == pytest.approx(1.0)

    def test_all_none_is_zero(self):
        s = dict.fromkeys(self._w(), None)
        assert _composite_weighted(self._w(), s) == 0.0

    def test_zero_not_silently_dropped_against_high_legs(self):
        # The regression the audit found: a single 0 among 1.0s must NOT round-trip to 1.0.
        s = {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 0.0}
        assert _composite_weighted(self._w(), s) < 1.0


class TestExtractRagasScore:
    """None vs 0.0 distinction (did-not-run vs scored-zero)."""

    def test_present_value(self):
        assert _extract_ragas_score(_FakeResult({"faithfulness": [1.0]}), "faithfulness") == 1.0

    def test_present_real_zero_is_zero_not_none(self):
        # A genuine 0 must stay 0.0 (so the composite counts it), not become None.
        assert _extract_ragas_score(_FakeResult({"faithfulness": [0.0]}), "faithfulness") == 0.0

    def test_absent_key_is_none(self):
        assert _extract_ragas_score(_FakeResult({}), "faithfulness") is None

    def test_nan_is_none(self):
        assert _extract_ragas_score(_FakeResult({"faithfulness": [float("nan")]}), "faithfulness") is None

    def test_empty_list_is_none(self):
        assert _extract_ragas_score(_FakeResult({"faithfulness": []}), "faithfulness") is None

    def test_prefix_fallback_for_mode_keyed_metric(self):
        # FactualCorrectness keys under factual_correctness(mode=f1).
        r = _FakeResult({"factual_correctness(mode=f1)": [1.0]})
        assert _extract_ragas_score(r, "factual_correctness") == 1.0


class TestJudgeDecouple:
    """Path B1: the RAGAS judge is decoupled from the answer model (self-preference guard)."""

    def test_judge_creds_precedence(self, monkeypatch):
        # RAGAS_JUDGE_* wins over OPENAI_* (separate model / key / base_url).
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "gen-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gen")
        monkeypatch.setenv("RAGAS_JUDGE_MODEL", "gpt-5.4")
        monkeypatch.setenv("RAGAS_JUDGE_API_KEY", "judge-key")
        monkeypatch.setenv("RAGAS_JUDGE_BASE_URL", "https://judge")
        model, key, base = _judge_openai_creds()
        assert (model, key, base) == ("gpt-5.4", "judge-key", "https://judge")

    def test_judge_creds_fallback_to_generator(self, monkeypatch):
        # When RAGAS_JUDGE_* unset, falls back to OPENAI_* (same gateway, same model).
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "gen-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gen")
        monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
        monkeypatch.delenv("RAGAS_JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("RAGAS_JUDGE_BASE_URL", raising=False)
        monkeypatch.delenv("RAGAS_REQUIRE_SEPARATE_JUDGE", raising=False)
        model, key, base = _judge_openai_creds()
        assert (model, key, base) == ("gpt-5.4-mini", "gen-key", "https://gen")

    def test_guard_warns_when_judge_equals_generator(self, monkeypatch, caplog):
        # Same model -> a WARNING is logged (not a raise).
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
        monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
        monkeypatch.delenv("RAGAS_REQUIRE_SEPARATE_JUDGE", raising=False)
        import logging

        with caplog.at_level(logging.WARNING, logger="koboi.eval.scorers.ragas_scorer"):
            _judge_openai_creds()
        assert any("self-preference bias" in r.message for r in caplog.records)

    def test_guard_raises_in_strict_mode(self, monkeypatch):
        # RAGAS_REQUIRE_SEPARATE_JUDGE=1 + same model -> RuntimeError (release gate).
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
        monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
        monkeypatch.setenv("RAGAS_REQUIRE_SEPARATE_JUDGE", "1")
        with pytest.raises(RuntimeError, match="self-preference"):
            _judge_openai_creds()

    def test_no_warning_when_decoupled(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
        monkeypatch.setenv("RAGAS_JUDGE_MODEL", "gpt-5.4")
        monkeypatch.delenv("RAGAS_REQUIRE_SEPARATE_JUDGE", raising=False)
        import logging

        with caplog.at_level(logging.WARNING, logger="koboi.eval.scorers.ragas_scorer"):
            _judge_openai_creds()
        assert not any("self-preference bias" in r.message for r in caplog.records)
