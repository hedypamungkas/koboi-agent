"""Tests for koboi.rag.augmentation module."""

from __future__ import annotations


from koboi.rag.types import Chunk
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

    async def test_augment_for_memory_injects_marker_when_no_results(self):
        chunks = [Chunk(id="c0", doc_id="d1", content="XYZZZ nonmatching gibberish")]
        retriever = KeywordRetriever(chunks=chunks)
        aug = InMemoryAugmentation(retriever=retriever, top_k=3)
        result = await aug.augment_for_memory("What is Python?")

        # A2: empty retrieval injects the abstention marker (wrapped in the
        # standard Document-context block) instead of silently passing the bare
        # question through to the LLM.
        assert "[RETRIEVAL_EMPTY]" in result
        assert "What is Python?" in result
        assert "Document context:" in result
        assert result != "What is Python?"

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
    async def test_empty_retrieval_injects_marker_for_memory(self):
        chunks = [Chunk(id="c0", doc_id="d1", content="XYZZZZ completely unrelated")]
        retriever = KeywordRetriever(chunks=chunks)
        aug = InMemoryAugmentation(retriever=retriever, top_k=3)
        result = await aug.augment_for_memory("Python programming")
        assert "[RETRIEVAL_EMPTY]" in result
        assert "Python programming" in result

    async def test_empty_retrieval_injects_marker_for_llm(self):
        chunks = [Chunk(id="c0", doc_id="d1", content="XYZZZZ unrelated content")]
        retriever = KeywordRetriever(chunks=chunks)
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=3)
        messages = [
            {"role": "user", "content": "Python programming"},
        ]
        result = await aug.augment_for_llm(messages)
        assert "[RETRIEVAL_EMPTY]" in result[0]["content"]
        assert "Python programming" in result[0]["content"]
        assert result[0]["content"] != "Python programming"


class TestAbstentionMarker:
    """A2: the abstention marker fires on every empty-retrieval path (no hits,
    threshold-sweep collapse, dedup collapse) and is absent whenever results are
    present."""

    async def test_retrieve_and_format_returns_marker_on_no_hits(self):
        chunks = [Chunk(id="c0", doc_id="d1", content="XYZZZ nonmatching gibberish")]
        aug = InMemoryAugmentation(retriever=KeywordRetriever(chunks=chunks), top_k=3)
        context, results = await aug._retrieve_and_format("What is Python?")  # noqa: SLF001
        assert results == []
        assert "[RETRIEVAL_EMPTY]" in context

    async def test_retrieve_and_format_returns_marker_on_threshold_sweep(self):
        # relevance_threshold collapses low-score results to empty -> marker fires.
        chunks = [Chunk(id="c1", doc_id="d", content="Annual leave is 12 days.")]
        aug = OnTheFlyAugmentation(
            retriever=KeywordRetriever(chunks=chunks), top_k=10, relevance_threshold=0.99
        )
        context, results = await aug._retrieve_and_format("annual leave days")  # noqa: SLF001
        assert results == []
        assert "[RETRIEVAL_EMPTY]" in context

    async def test_marker_not_injected_when_results_present(self):
        # Control: non-empty retrieval -> NO marker, real context.
        aug = InMemoryAugmentation(retriever=_make_retriever(), top_k=2)
        context, results = await aug._retrieve_and_format("What is Python?")  # noqa: SLF001
        assert len(results) > 0
        assert "[RETRIEVAL_EMPTY]" not in context

    async def test_in_memory_marker_is_wrapped_in_document_context_block(self):
        # Documents the in_memory stored shape (marker rides through _build_augmented_message).
        chunks = [Chunk(id="c0", doc_id="d1", content="XYZZZ nonmatching gibberish")]
        aug = InMemoryAugmentation(retriever=KeywordRetriever(chunks=chunks), top_k=3)
        result = await aug.augment_for_memory("What is Python?")
        assert result.startswith("Document context:\n---\n")
        assert "[RETRIEVAL_EMPTY]" in result
        assert "Question: What is Python?" in result

    async def test_on_the_fly_marker_cached_as_empty(self):
        # The empty-query cache entry holds the (truthy) marker so repeated empty
        # queries reuse it instead of re-retrieving.
        chunks = [Chunk(id="c0", doc_id="d1", content="XYZZZ nonmatching gibberish")]
        aug = OnTheFlyAugmentation(retriever=KeywordRetriever(chunks=chunks), top_k=3)
        messages = [{"role": "user", "content": "Python programming"}]
        await aug.augment_for_llm(messages)
        assert "Python programming" in aug._cache  # noqa: SLF001
        assert "[RETRIEVAL_EMPTY]" in aug._cache["Python programming"]  # noqa: SLF001


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
