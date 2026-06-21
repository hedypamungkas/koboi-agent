"""Tests for koboi/eval/scorers/retrieval_scorer.py -- deterministic retrieval metrics."""

from __future__ import annotations

import pytest

from koboi.eval.scorers.retrieval_scorer import RetrievalScorer, compute_retrieval_metrics
from koboi.rag.retriever import KeywordRetriever
from koboi.rag.types import Chunk, RetrievalResult
from koboi.types import EvalCase


def _case(**kwargs):
    defaults = dict(name="t", user_message="q", expected_tools=[], expected_keywords=[], max_iterations=10)
    defaults.update(kwargs)
    return EvalCase(**defaults)


def _result(doc_id: str, content: str, score: float = 0.5) -> RetrievalResult:
    """Build a RetrievalResult with a chunk from ``doc_id``."""
    return RetrievalResult(
        chunk=Chunk(id=f"{doc_id}_c0", doc_id=doc_id, content=content),
        score=score,
        retrieval_method="test",
    )


class _StubRetriever:
    """Deterministic retriever returning a preset result list (capped by top_k)."""

    def __init__(self, results: list[RetrievalResult]):
        self._results = results

    async def retrieve(self, query: str, top_k: int = 3):
        return self._results[:top_k]


def _factual(key_facts: list[str], source_doc: str = "product_catalog") -> EvalCase:
    return _case(metadata={"needs_retrieval": True, "source_doc": source_doc, "key_facts": key_facts})


class TestRetrievalScorerConstruction:
    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            RetrievalScorer("bogus")

    @pytest.mark.asyncio
    async def test_no_retriever_fail_open(self):
        score = await RetrievalScorer("recall").score(_factual(["x"]), "out", {})
        assert score.value == 0.0
        assert "no retriever" in score.reason

    def test_registered(self):
        import koboi.eval  # noqa: F401 -- triggers default scorer registration
        from koboi.eval.registry import ScorerRegistry, register_default_scorers

        register_default_scorers()  # idempotent; defend against ScorerRegistry.clear() in other tests
        available = ScorerRegistry.list_available()
        for name in ("retrieval_recall", "retrieval_precision", "retrieval_hit", "retrieval_gate_noise"):
            assert name in available


class TestRecall:
    @pytest.mark.asyncio
    async def test_all_facts_found(self):
        retriever = _StubRetriever(
            [_result("product_catalog", "AcmeERP costs $15,000 with perpetual license and 10 users.")]
        )
        scorer = RetrievalScorer("recall", retriever=retriever, top_k=5)
        score = await scorer.score(_factual(["$15,000", "perpetual license", "10 users"]), "out", {})
        assert score.name == "retrieval_recall@5"
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_facts(self):
        retriever = _StubRetriever([_result("product_catalog", "AcmeERP has a perpetual license.")])
        scorer = RetrievalScorer("recall", retriever=retriever)
        score = await scorer.score(_factual(["$15,000", "perpetual license"]), "out", {})
        assert score.value == 0.5  # 1 of 2 facts found

    @pytest.mark.asyncio
    async def test_no_facts_found(self):
        retriever = _StubRetriever([_result("product_catalog", "Unrelated content about hardware.")])
        scorer = RetrievalScorer("recall", retriever=retriever)
        score = await scorer.score(_factual(["$15,000", "perpetual license"]), "out", {})
        assert score.value == 0.0


class TestPrecision:
    @pytest.mark.asyncio
    async def test_all_relevant(self):
        retriever = _StubRetriever(
            [
                _result("product_catalog", "costs $15,000 perpetual license"),
                _result("product_catalog", "minimum 10 users perpetual license"),
            ]
        )
        scorer = RetrievalScorer("precision", retriever=retriever, top_k=5)
        score = await scorer.score(_factual(["$15,000", "perpetual license"]), "out", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_mixed_relevant(self):
        retriever = _StubRetriever(
            [
                _result("product_catalog", "costs $15,000 perpetual license"),  # relevant
                _result("employee_handbook", "onboarding week one orientation"),  # wrong doc
            ]
        )
        scorer = RetrievalScorer("precision", retriever=retriever, top_k=5)
        score = await scorer.score(_factual(["$15,000", "perpetual license"]), "out", {})
        assert score.value == 0.5  # 1 of 2 retrieved is relevant

    @pytest.mark.asyncio
    async def test_wrong_doc_no_precision(self):
        # Same fact text but from the wrong document -> not relevant for precision.
        retriever = _StubRetriever([_result("employee_handbook", "costs $15,000 perpetual license")])
        scorer = RetrievalScorer("precision", retriever=retriever, top_k=5)
        score = await scorer.score(_factual(["$15,000"], source_doc="product_catalog"), "out", {})
        assert score.value == 0.0


class TestHit:
    @pytest.mark.asyncio
    async def test_hit_present(self):
        retriever = _StubRetriever([_result("product_catalog", "costs $15,000 perpetual license")])
        scorer = RetrievalScorer("hit", retriever=retriever, top_k=5)
        score = await scorer.score(_factual(["$15,000"]), "out", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_hit_absent(self):
        retriever = _StubRetriever([_result("employee_handbook", "onboarding process details")])
        scorer = RetrievalScorer("hit", retriever=retriever, top_k=5)
        score = await scorer.score(_factual(["$15,000"]), "out", {})
        assert score.value == 0.0


class TestGateNoise:
    @pytest.mark.asyncio
    async def test_gating_returns_max_score(self):
        retriever = _StubRetriever([_result("product_catalog", "x", score=0.42), _result("x", "y", score=0.7)])
        scorer = RetrievalScorer("gate_noise", retriever=retriever, top_k=5)
        case = _case(metadata={"needs_retrieval": False})
        score = await scorer.score(case, "out", {})
        assert score.name == "retrieval_gate_noise@5"
        assert score.value == 0.7

    @pytest.mark.asyncio
    async def test_gating_no_results(self):
        retriever = _StubRetriever([])
        scorer = RetrievalScorer("gate_noise", retriever=retriever, top_k=5)
        case = _case(metadata={"needs_retrieval": False})
        score = await scorer.score(case, "out", {})
        assert score.value == 0.0  # retriever correctly fired on nothing


class TestFailOpenAndNA:
    @pytest.mark.asyncio
    async def test_factual_without_gold_labels(self):
        retriever = _StubRetriever([_result("product_catalog", "anything")])
        scorer = RetrievalScorer("recall", retriever=retriever)
        score = await scorer.score(_case(metadata={"needs_retrieval": True}), "out", {})
        assert score.value == 1.0
        assert "no gold" in score.reason

    @pytest.mark.asyncio
    async def test_metric_not_applicable_to_case_type(self):
        retriever = _StubRetriever([_result("d", "c")])
        # gate_noise on a factual case -> n/a
        s1 = await RetrievalScorer("gate_noise", retriever=retriever).score(_factual(["x"]), "out", {})
        assert s1.value == 1.0
        assert "n/a" in s1.reason
        # recall on a gating negative -> n/a
        s2 = await RetrievalScorer("recall", retriever=retriever).score(
            _case(metadata={"needs_retrieval": False}), "out", {}
        )
        assert s2.value == 1.0
        assert "n/a" in s2.reason

    @pytest.mark.asyncio
    async def test_retriever_from_context(self):
        retriever = _StubRetriever([_result("product_catalog", "costs $15,000 perpetual license")])
        # No retriever in ctor; supplied via context instead.
        scorer = RetrievalScorer("recall")
        score = await scorer.score(_factual(["$15,000"]), "out", {"retriever": retriever})
        assert score.value == 1.0


class TestComputeMetricsPure:
    def test_empty_results_factual(self):
        case = _factual(["$15,000"])
        metrics = compute_retrieval_metrics([], case)
        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["hit"] == 0.0
        assert metrics["retrieved"] == 0

    def test_empty_results_gating(self):
        case = _case(metadata={"needs_retrieval": False})
        metrics = compute_retrieval_metrics([], case)
        assert metrics["gate_noise"] == 0.0


class TestIntegrationWithKeywordRetriever:
    @pytest.mark.asyncio
    async def test_real_retriever_end_to_end(self):
        """Integration: a real KeywordRetriever over synthetic chunks returns a valid score."""
        chunks = [
            Chunk(id="d1_c0", doc_id="d1", content="Python is a programming language with great libraries."),
            Chunk(id="d2_c0", doc_id="d2", content="The weather today is sunny and warm."),
        ]
        retriever = KeywordRetriever(chunks=chunks)
        scorer = RetrievalScorer("recall", retriever=retriever, top_k=2)
        case = _case(
            user_message="python programming",
            metadata={"needs_retrieval": True, "source_doc": "d1", "key_facts": ["python"]},
        )
        score = await scorer.score(case, "out", {})
        assert 0.0 <= score.value <= 1.0
        assert score.name == "retrieval_recall@2"
        # The query "python programming" should surface the python chunk -> recall 1.0.
        assert score.value == 1.0
