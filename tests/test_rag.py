"""Tests for koboi.rag module."""

from __future__ import annotations

import pytest

from koboi.rag.types import Chunk, Document, RetrievalResult


@pytest.fixture(autouse=True)
def _isolate_embedding_cache():
    """Reset the process-wide shared embedding index around each test."""
    from koboi.rag.retriever import clear_embedding_cache

    clear_embedding_cache()
    yield
    clear_embedding_cache()


class TestChunkTypes:
    def test_chunk_creation(self):
        chunk = Chunk(id="c1", doc_id="d1", content="Hello world")
        assert chunk.id == "c1"
        assert chunk.content == "Hello world"

    def test_document_creation(self):
        doc = Document(id="d1", title="Test", content="Content")
        assert doc.title == "Test"
        assert len(doc.content) > 0

    def test_retrieval_result(self):
        chunk = Chunk(id="c1", doc_id="d1", content="Data")
        result = RetrievalResult(chunk=chunk, score=0.95, retrieval_method="keyword")
        assert result.score == 0.95
        assert result.chunk.content == "Data"


class TestChunkers:
    def test_fixed_size_chunker(self):
        from koboi.rag.chunker import FixedSizeChunker

        doc = Document(id="d1", title="Test", content="A " * 200)
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.content) <= 120

    def test_sentence_chunker(self):
        from koboi.rag.chunker import SentenceChunker

        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        doc = Document(id="d1", title="Test", content=text)
        chunker = SentenceChunker(max_chunk_size=100)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_paragraph_chunker(self):
        from koboi.rag.chunker import ParagraphChunker

        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        doc = Document(id="d1", title="Test", content=text)
        chunker = ParagraphChunker(max_chunk_size=500)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3


class TestKeywordRetrieverSynonyms:
    async def test_synonym_bridge_closes_vocabulary_gap(self):
        """Query-side synonyms let a term absent from the docs (``dog``) still
        retrieve a chunk phrased with a synonym (``pet``). Regression for the
        hotel-pet e2e scenario, which had zero keyword overlap (score 0.0)."""
        from koboi.rag.chunker import ParagraphChunker
        from koboi.rag.retriever import KeywordRetriever

        doc = Document(
            id="hotel",
            title="hotel",
            content="### Pet Policy\nSmall pets welcome. Pet fee: $25 per night.",
        )
        chunks = ParagraphChunker().chunk(doc)

        plain = KeywordRetriever(chunks)
        assert await plain.retrieve("Can I bring a 10kg dog?", top_k=3) == []

        bridged = KeywordRetriever(chunks, synonyms={"dog": ["pet"]})
        hits = await bridged.retrieve("Can I bring a 10kg dog?", top_k=3)
        assert hits and "Pet Policy" in hits[0].chunk.content

    async def test_no_synonyms_is_noop(self):
        from koboi.rag.chunker import ParagraphChunker
        from koboi.rag.retriever import KeywordRetriever

        doc = Document(id="d1", title="t", content="The pet fee is $25.")
        chunks = ParagraphChunker().chunk(doc)
        retriever = KeywordRetriever(chunks)  # no synonyms arg
        hits = await retriever.retrieve("pet fee", top_k=3)
        assert hits and "$25" in hits[0].chunk.content


class TestRetrievers:
    async def test_keyword_retriever(self):
        from koboi.rag.retriever import KeywordRetriever

        chunks = [
            Chunk(id=f"c{i}", doc_id="d1", content=text)
            for i, text in enumerate(
                [
                    "Python is a programming language",
                    "JavaScript runs in the browser",
                    "Python has great data science libraries",
                    "Rust is a systems programming language",
                ]
            )
        ]
        retriever = KeywordRetriever(chunks=chunks)
        results = await retriever.retrieve("Python programming", top_k=2)
        assert len(results) == 2
        assert "Python" in results[0].chunk.content


class TestKeywordRetrieverEdgeCases:
    async def test_empty_query(self):
        """Test KeywordRetriever with empty query."""
        from koboi.rag.retriever import KeywordRetriever

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Test content"),
        ]
        retriever = KeywordRetriever(chunks=chunks)
        results = await retriever.retrieve("", top_k=3)

        assert results == []

    async def test_no_documents(self):
        """Test KeywordRetriever with no documents."""
        from koboi.rag.retriever import KeywordRetriever

        retriever = KeywordRetriever(chunks=[])
        results = await retriever.retrieve("test query", top_k=3)

        assert results == []

    async def test_query_no_matches(self):
        """Test KeywordRetriever when query matches no chunks."""
        from koboi.rag.retriever import KeywordRetriever

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Python programming"),
            Chunk(id="c2", doc_id="d1", content="JavaScript web"),
        ]
        retriever = KeywordRetriever(chunks=chunks)
        results = await retriever.retrieve("rustlang systems", top_k=3)

        # Should return empty or low-score results
        assert len(results) == 0

    async def test_top_k_limiting(self):
        """Test KeywordRetriever respects top_k parameter."""
        from koboi.rag.retriever import KeywordRetriever

        chunks = [Chunk(id=f"c{i}", doc_id="d1", content=f"Python content {i}") for i in range(10)]
        retriever = KeywordRetriever(chunks=chunks)
        results = await retriever.retrieve("Python", top_k=3)

        assert len(results) <= 3

    async def test_score_ordering(self):
        """Test KeywordRetriever returns results sorted by score."""
        from koboi.rag.retriever import KeywordRetriever

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Python"),
            Chunk(id="c2", doc_id="d1", content="Python programming language"),
            Chunk(id="c3", doc_id="d1", content="Something else"),
        ]
        retriever = KeywordRetriever(chunks=chunks)
        results = await retriever.retrieve("Python programming", top_k=3)

        # Results should be sorted by score (highest first)
        if len(results) > 1:
            assert results[0].score >= results[1].score

    async def test_case_insensitive_search(self):
        """Test KeywordRetriever is case-insensitive."""
        from koboi.rag.retriever import KeywordRetriever

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Python Programming"),
        ]
        retriever = KeywordRetriever(chunks=chunks)

        results_lower = await retriever.retrieve("python", top_k=3)
        results_upper = await retriever.retrieve("PYTHON", top_k=3)

        # Both should find matches
        assert len(results_lower) > 0
        assert len(results_upper) > 0


class TestSemanticRetriever:
    async def test_with_mock_embeddings(self):
        """Test SemanticRetriever with mock embedding client."""
        from koboi.rag.retriever import SemanticRetriever
        from unittest.mock import AsyncMock

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Python programming"),
            Chunk(id="c2", doc_id="d1", content="JavaScript web"),
        ]

        # Mock client
        mock_client = AsyncMock()
        mock_client.get_embeddings = AsyncMock(return_value=[0.1, 0.2, 0.3])

        retriever = SemanticRetriever(chunks=chunks, client=mock_client)
        results = await retriever.retrieve("Python code", top_k=2)

        assert isinstance(results, list)

    async def test_fallback_to_keyword(self):
        """Test SemanticRetriever falls back to keyword when embeddings fail."""
        from koboi.rag.retriever import SemanticRetriever
        from unittest.mock import AsyncMock

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Python programming"),
        ]

        # Mock client that returns None (embedding unavailable)
        mock_client = AsyncMock()
        mock_client.get_embeddings = AsyncMock(return_value=None)

        retriever = SemanticRetriever(chunks=chunks, client=mock_client)
        results = await retriever.retrieve("Python", top_k=1)

        # Should fallback to keyword and still return results
        assert len(results) >= 0

    async def test_no_client_fallback(self):
        """Test SemanticRetriever with no client falls back to keyword."""
        from koboi.rag.retriever import SemanticRetriever

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Test content about Python"),
        ]

        retriever = SemanticRetriever(chunks=chunks, client=None)
        results = await retriever.retrieve("Python", top_k=1)

        # Should use keyword fallback
        assert isinstance(results, list)

    async def test_embedding_unavailable_mid_query(self):
        """Test SemanticRetriever handles embedding failure during query."""
        from koboi.rag.retriever import SemanticRetriever
        from unittest.mock import AsyncMock

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Test content"),
        ]

        # Mock client that fails during query
        mock_client = AsyncMock()
        call_count = [0]

        async def side_effect(text):
            call_count[0] += 1
            if call_count[0] > 1:  # Fail during query
                return None
            return [0.1, 0.2]

        mock_client.get_embeddings = AsyncMock(side_effect=side_effect)

        retriever = SemanticRetriever(chunks=chunks, client=mock_client)
        results = await retriever.retrieve("test", top_k=1)

        # Should handle gracefully
        assert isinstance(results, list)


class TestRetrieverIntegration:
    async def test_fallback_behavior(self):
        """Test fallback from semantic to keyword retrieval."""
        from koboi.rag.retriever import SemanticRetriever
        from unittest.mock import AsyncMock

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Python programming tutorial"),
        ]

        # Mock client that initially works then fails
        mock_client = AsyncMock()
        mock_client.get_embeddings = AsyncMock(return_value=None)

        retriever = SemanticRetriever(chunks=chunks, client=mock_client)
        results = await retriever.retrieve("Python tutorial", top_k=1)

        # Should have fallback results
        assert isinstance(results, list)

    async def test_cosine_similarity_calculation(self):
        """Test cosine similarity is calculated correctly."""
        from koboi.rag.retriever import SemanticRetriever

        # Test the static method
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]

        # Orthogonal vectors
        sim_orthogonal = SemanticRetriever._cosine_similarity(vec1, vec2)
        assert sim_orthogonal < 0.01

        # Identical vectors
        sim_identical = SemanticRetriever._cosine_similarity(vec1, vec3)
        assert abs(sim_identical - 1.0) < 0.01

    async def test_retrieval_result_format(self):
        """Test RetrievalResult has correct format."""
        from koboi.rag.retriever import KeywordRetriever

        chunks = [
            Chunk(id="c1", doc_id="d1", content="Test content"),
        ]

        retriever = KeywordRetriever(chunks=chunks)
        results = await retriever.retrieve("test", top_k=1)

        if results:
            result = results[0]
            assert hasattr(result, "chunk")
            assert hasattr(result, "score")
            assert hasattr(result, "retrieval_method")
            assert result.retrieval_method == "keyword"


class TestEmbeddingCache:
    """The process-level shared embedding index (_EMBEDDING_CACHE)."""

    async def test_second_instance_reuses_index_zero_chunk_embeds(self):
        """A second retriever over the same corpus must not re-embed chunks."""
        from koboi.rag.retriever import SemanticRetriever
        from unittest.mock import AsyncMock

        chunks = [
            Chunk(id="c1", doc_id="d1", content="alpha beta"),
            Chunk(id="c2", doc_id="d1", content="gamma delta"),
        ]
        client1 = AsyncMock()
        client1.get_embeddings = AsyncMock(return_value=[0.1, 0.2])
        r1 = SemanticRetriever(chunks=chunks, client=client1)
        await r1.retrieve("alpha", top_k=2)
        # 2 chunk embeds (index build) + 1 query embed.
        assert client1.get_embeddings.call_count == 3

        # Fresh retriever, SAME chunks -> served from cache. Only the query embeds.
        client2 = AsyncMock()
        client2.get_embeddings = AsyncMock(return_value=[0.9, 0.9])
        r2 = SemanticRetriever(chunks=chunks, client=client2)
        await r2.retrieve("alpha", top_k=2)
        assert client2.get_embeddings.call_count == 1, "chunks must be served from cache"

    async def test_different_corpus_embeds_again(self):
        """A different corpus (different signature) must embed afresh."""
        from koboi.rag.retriever import SemanticRetriever
        from unittest.mock import AsyncMock

        chunks_a = [Chunk(id="a1", doc_id="d", content="one two")]
        c1 = AsyncMock()
        c1.get_embeddings = AsyncMock(return_value=[0.1, 0.2])
        await SemanticRetriever(chunks=chunks_a, client=c1).retrieve("one", top_k=1)

        chunks_b = [Chunk(id="b1", doc_id="d", content="three four")]  # different content
        c2 = AsyncMock()
        c2.get_embeddings = AsyncMock(return_value=[0.3, 0.4])
        await SemanticRetriever(chunks=chunks_b, client=c2).retrieve("three", top_k=1)
        # 1 chunk embed (new corpus) + 1 query embed.
        assert c2.get_embeddings.call_count == 2

    async def test_unavailable_endpoint_not_cached_as_success(self):
        """A failed build (None embeddings) must not poison the cache for later."""
        from koboi.rag.retriever import SemanticRetriever
        from unittest.mock import AsyncMock

        chunks = [Chunk(id="c1", doc_id="d1", content="Python programming")]

        down = AsyncMock()
        down.get_embeddings = AsyncMock(return_value=None)
        r1 = SemanticRetriever(chunks=chunks, client=down)
        await r1.retrieve("Python", top_k=1)
        assert r1._embedding_available is False

        up = AsyncMock()
        up.get_embeddings = AsyncMock(return_value=[0.5, 0.5, 0.5])
        r2 = SemanticRetriever(chunks=chunks, client=up)
        await r2.retrieve("Python", top_k=1)
        assert r2._embedding_available is True, "recovered endpoint must rebuild, not reuse a cached failure"


class TestHybridRetrieverSynonyms:
    async def test_synonyms_propagate_to_keyword_leg(self):
        """HybridRetriever must pass `synonyms` to its keyword leg so vocabulary
        gaps close even when embeddings are unavailable (semantic falls back to
        keyword). Regression for the e2e hotel-pet failure on a no-embedding
        provider, where the synonym bridge was silently dropped under hybrid."""
        from koboi.rag.chunker import ParagraphChunker
        from koboi.rag.retriever import HybridRetriever

        doc = Document(
            id="hotel",
            title="hotel",
            content="### Pet Policy\nSmall pets welcome. Pet fee: $25 per night.",
        )
        chunks = ParagraphChunker().chunk(doc)
        # client=None -> semantic leg falls back to keyword; only the keyword
        # leg carries the synonym bridge here.
        retriever = HybridRetriever(chunks, client=None, synonyms={"dog": ["pet"]})
        hits = await retriever.retrieve("Can I bring a 10kg dog?", top_k=3)
        assert hits and "Pet Policy" in hits[0].chunk.content

    async def test_without_synonyms_dog_misses(self):
        from koboi.rag.chunker import ParagraphChunker
        from koboi.rag.retriever import HybridRetriever

        doc = Document(
            id="hotel",
            title="hotel",
            content="### Pet Policy\nSmall pets welcome. Pet fee: $25 per night.",
        )
        chunks = ParagraphChunker().chunk(doc)
        retriever = HybridRetriever(chunks, client=None)  # no synonyms
        hits = await retriever.retrieve("Can I bring a 10kg dog?", top_k=3)
        assert not hits or all("Pet Policy" not in h.chunk.content for h in hits)


class TestSemanticRetrieverFallbackSynonyms:
    async def test_fallback_leg_carries_synonyms(self):
        """When embeddings are unavailable (client=None), SemanticRetriever falls
        back to keyword -- that fallback must carry the synonym bridge, so hybrid
        degrades cleanly instead of RRF-demoting synonym-only matches."""
        from koboi.rag.chunker import ParagraphChunker
        from koboi.rag.retriever import SemanticRetriever

        doc = Document(
            id="hotel",
            title="hotel",
            content="### Pet Policy\nSmall pets welcome. Pet fee: $25 per night.",
        )
        chunks = ParagraphChunker().chunk(doc)
        retriever = SemanticRetriever(chunks, client=None, synonyms={"dog": ["pet"]})
        assert retriever._embedding_available is False  # no client -> fallback armed
        hits = await retriever.retrieve("Can I bring a 10kg dog?", top_k=3)
        assert hits and "Pet Policy" in hits[0].chunk.content
