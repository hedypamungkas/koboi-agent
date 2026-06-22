"""Tests for koboi.rag.augmentation module."""

from __future__ import annotations

from unittest.mock import MagicMock

from koboi.rag.types import Chunk, RetrievalResult
from koboi.rag.retriever import KeywordRetriever
from koboi.rag.augmentation import InMemoryAugmentation, OnTheFlyAugmentation


def _make_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(id="c0", doc_id="d1", content="Python is a versatile programming language."),
        Chunk(id="c1", doc_id="d1", content="JavaScript is used for web development."),
        Chunk(id="c2", doc_id="d2", content="Python has excellent data science libraries."),
    ]


def _make_retriever():
    return KeywordRetriever(chunks=_make_chunks())


class TestInMemoryAugmentation:
    async def test_augment_for_memory_returns_augmented_text(self):
        retriever = _make_retriever()
        aug = InMemoryAugmentation(retriever=retriever, top_k=2)
        result = await aug.augment_for_memory("What is Python?")

        assert "What is Python?" in result
        assert "Python" in result
        assert "Document context:" in result
        assert "---" in result

    async def test_augment_for_memory_includes_relevant_chunks(self):
        retriever = _make_retriever()
        aug = InMemoryAugmentation(retriever=retriever, top_k=2)
        result = await aug.augment_for_memory("Python data science")

        assert "versatile programming language" in result or "data science libraries" in result

    async def test_augment_for_memory_returns_original_when_no_results(self):
        chunks = [Chunk(id="c0", doc_id="d1", content="XYZZZ nonmatching gibberish")]
        retriever = KeywordRetriever(chunks=chunks)
        aug = InMemoryAugmentation(retriever=retriever, top_k=3)
        result = await aug.augment_for_memory("What is Python?")

        assert result == "What is Python?"

    async def test_augment_for_memory_with_single_result(self):
        retriever = _make_retriever()
        aug = InMemoryAugmentation(retriever=retriever, top_k=1)
        result = await aug.augment_for_memory("JavaScript web")

        assert "JavaScript" in result
        assert "Document context:" in result

    async def test_augment_for_memory_respects_top_k(self):
        retriever = _make_retriever()
        aug = InMemoryAugmentation(retriever=retriever, top_k=1)
        result = await aug.augment_for_memory("Python")

        assert "Question: Python" in result


class TestOnTheFlyAugmentation:
    async def test_augment_for_llm_modifies_last_user_message(self):
        retriever = _make_retriever()
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=2)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me about Python"},
        ]
        result = await aug.augment_for_llm(messages)

        assert len(result) == 2
        assert result[0] == messages[0]
        assert "Document context:" in result[1]["content"]
        assert "Tell me about Python" in result[1]["content"]
        assert "Python" in result[1]["content"]

    async def test_augment_for_llm_does_not_mutate_original(self):
        retriever = _make_retriever()
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=2)
        messages = [
            {"role": "user", "content": "Python programming"},
        ]
        original_content = messages[0]["content"]
        result = await aug.augment_for_llm(messages)

        assert messages[0]["content"] == original_content
        assert result[0]["content"] != original_content

    async def test_augment_for_llm_with_no_user_message(self):
        retriever = _make_retriever()
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=2)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Hello there"},
        ]
        result = await aug.augment_for_llm(messages)

        assert result == messages

    async def test_augment_for_llm_uses_last_user_message(self):
        retriever = _make_retriever()
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=2)
        messages = [
            {"role": "user", "content": "First message about XYZ nonmatching"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Tell me about Python"},
        ]
        result = await aug.augment_for_llm(messages)

        assert "Document context:" in result[2]["content"]
        assert "Tell me about Python" in result[2]["content"]
        assert result[0]["content"] == "First message about XYZ nonmatching"


class TestAugmentationEmptyResults:
    async def test_empty_retrieval_returns_original_for_memory(self):
        chunks = [Chunk(id="c0", doc_id="d1", content="XYZZZZ completely unrelated")]
        retriever = KeywordRetriever(chunks=chunks)
        aug = InMemoryAugmentation(retriever=retriever, top_k=3)
        result = await aug.augment_for_memory("Python programming")
        assert result == "Python programming"

    async def test_empty_retrieval_returns_original_messages_for_llm(self):
        chunks = [Chunk(id="c0", doc_id="d1", content="XYZZZZ unrelated content")]
        retriever = KeywordRetriever(chunks=chunks)
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=3)
        messages = [
            {"role": "user", "content": "Python programming"},
        ]
        result = await aug.augment_for_llm(messages)
        assert result[0]["content"] == "Python programming"


class TestOnTheFlyCaching:
    async def test_second_call_with_same_query_uses_cache(self):
        retriever = _make_retriever()
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=2)

        original_retrieve = aug._retrieve_and_format
        call_count = [0]

        async def tracking_retrieve(query):
            call_count[0] += 1
            return await original_retrieve(query)

        aug._retrieve_and_format = tracking_retrieve

        messages = [
            {"role": "user", "content": "Python programming"},
        ]

        result1 = await aug.augment_for_llm(messages)
        assert call_count[0] == 1

        result2 = await aug.augment_for_llm(messages)
        assert call_count[0] == 1

        assert result1[0]["content"] == result2[0]["content"]

    async def test_cache_populated_after_first_call(self):
        retriever = _make_retriever()
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=2)
        messages = [
            {"role": "user", "content": "Python programming"},
        ]

        assert len(aug._cache) == 0

        await aug.augment_for_llm(messages)

        assert "Python programming" in aug._cache

    async def test_different_queries_not_cached_together(self):
        retriever = _make_retriever()
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=2)

        messages_a = [{"role": "user", "content": "Python programming"}]
        messages_b = [{"role": "user", "content": "JavaScript web"}]

        result_a = await aug.augment_for_llm(messages_a)
        result_b = await aug.augment_for_llm(messages_b)

        assert len(aug._cache) == 2
        assert "Python programming" in aug._cache
        assert "JavaScript web" in aug._cache

        assert result_a[0]["content"] != result_b[0]["content"]


def _rr(doc_id: str, content: str, score: float, method: str = "keyword") -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(id=f"{doc_id}_{content}", doc_id=doc_id, content=content),
        score=score,
        retrieval_method=method,
    )


class _ScoredStubRetriever:
    """Returns a preset result list, capped by top_k."""

    def __init__(self, results: list[RetrievalResult]):
        self._results = results

    async def retrieve(self, query: str, top_k: int = 3):
        return self._results[:top_k]


class TestMinScoreFiltering:
    async def test_min_score_drops_low_score_chunks(self):
        results = [_rr("d", "highscore", 0.9), _rr("d", "lowscore", 0.1)]
        aug = InMemoryAugmentation(retriever=_ScoredStubRetriever(results), top_k=5, min_score=0.5, normalize=False)
        out = await aug.augment_for_memory("q")
        assert "highscore" in out
        assert "lowscore" not in out

    async def test_default_min_score_keeps_all(self):
        results = [_rr("d", "a", 0.9), _rr("d", "b", 0.1)]
        aug = InMemoryAugmentation(retriever=_ScoredStubRetriever(results), top_k=5, normalize=False)
        out = await aug.augment_for_memory("q")
        assert "a" in out and "b" in out

    async def test_all_filtered_returns_original_message(self):
        results = [_rr("d", "low", 0.1)]
        aug = InMemoryAugmentation(retriever=_ScoredStubRetriever(results), top_k=5, min_score=0.5, normalize=False)
        out = await aug.augment_for_memory("question")
        assert out == "question"  # nothing survived the filter

    async def test_fetch_factor_widens_candidate_pool(self):
        results = [_rr("d", str(i), 0.5) for i in range(12)]
        retriever = _ScoredStubRetriever(results)
        aug = InMemoryAugmentation(retriever=retriever, top_k=3, fetch_factor=4, normalize=False)
        seen = {}

        async def spy_retrieve(query, top_k=3):
            seen["top_k"] = top_k
            return retriever._results[:top_k]

        retriever.retrieve = spy_retrieve
        out = await aug.augment_for_memory("q")
        assert seen["top_k"] == 12  # top_k(3) * fetch_factor(4)
        # after truncation, only top_k injected
        assert out.count("[Source:") <= 3

    async def test_logs_mean_injected_score(self):
        results = [_rr("d", "a", 0.8), _rr("d", "b", 0.6)]
        logger = MagicMock()
        aug = InMemoryAugmentation(retriever=_ScoredStubRetriever(results), top_k=5, logger=logger, normalize=False)
        await aug.augment_for_memory("q")
        logger.log_rag_filter.assert_called_once()
        kw = logger.log_rag_filter.call_args.kwargs
        assert kw["injected"] == 2
        assert abs(kw["mean_score"] - 0.7) < 1e-9
        assert kw["min_score"] == 0.0

    async def test_normalize_handles_hybrid_rrf_scale(self):
        # Hybrid RRF scores (~0.01-0.03) normalized to [0,1] before filtering.
        results = [
            _rr("d", "top", 0.03, method="hybrid"),
            _rr("d", "mid", 0.02, method="hybrid"),
            _rr("d", "low", 0.01, method="hybrid"),
        ]
        aug = InMemoryAugmentation(retriever=_ScoredStubRetriever(results), top_k=3, min_score=0.5, normalize=True)
        out = await aug.augment_for_memory("q")
        assert "top" in out  # normalized to 1.0, kept
        assert "mid" in out  # normalized to 0.5, kept (>= 0.5)
        assert "low" not in out  # normalized to 0.0, dropped
