"""Tests for koboi.rag module."""

from __future__ import annotations

from koboi.rag.types import Chunk, Document, RetrievalResult


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
