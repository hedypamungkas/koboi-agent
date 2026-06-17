"""RAG pipeline benchmarks."""
import pytest

from koboi.rag.chunker import FixedSizeChunker, SentenceChunker, ParagraphChunker
from koboi.rag.retriever import KeywordRetriever
from koboi.rag.augmentation import InMemoryAugmentation
from koboi.rag.types import Document


def test_fixed_chunking(benchmark, fixed_size_chunker, sample_document):
    """Benchmark FixedSizeChunker on 50KB text."""
    result = benchmark(fixed_size_chunker.chunk, sample_document)
    assert len(result) > 0


def test_fixed_chunking_small_overlap(benchmark, sample_document):
    """Benchmark FixedSizeChunker with small overlap."""
    chunker = FixedSizeChunker(chunk_size=500, overlap=25)
    result = benchmark(chunker.chunk, sample_document)
    assert len(result) > 0


def test_fixed_chunking_large_chunks(benchmark, sample_document):
    """Benchmark FixedSizeChunker with larger chunk size."""
    chunker = FixedSizeChunker(chunk_size=1000, overlap=100)
    result = benchmark(chunker.chunk, sample_document)
    assert len(result) > 0


def test_sentence_chunking(benchmark, sentence_chunker, sample_document):
    """Benchmark SentenceChunker on 50KB text."""
    result = benchmark(sentence_chunker.chunk, sample_document)
    assert len(result) > 0


def test_sentence_chunking_small_max(benchmark, sample_document):
    """Benchmark SentenceChunker with smaller max size."""
    chunker = SentenceChunker(max_chunk_size=400)
    result = benchmark(chunker.chunk, sample_document)
    assert len(result) > 0


def test_paragraph_chunking(benchmark, sample_document):
    """Benchmark ParagraphChunker."""
    chunker = ParagraphChunker(max_chunk_size=1000)
    result = benchmark(chunker.chunk, sample_document)
    assert len(result) > 0


def test_keyword_retrieval(benchmark, keyword_retriever):
    """Benchmark KeywordRetriever with 100 chunks."""
    import asyncio

    def run_retrieve():
        return asyncio.run(keyword_retriever.retrieve("search test query", top_k=5))

    result = benchmark(run_retrieve)
    assert len(result) >= 0


def test_keyword_retrieval_single_chunk(benchmark, sample_chunks):
    """Benchmark KeywordRetriever with single chunk."""
    import asyncio
    retriever = KeywordRetriever(chunks=sample_chunks[:1])

    def run_retrieve():
        return asyncio.run(retriever.retrieve("search", top_k=1))

    result = benchmark(run_retrieve)
    assert len(result) >= 0


def test_keyword_retrieval_top_10(benchmark, keyword_retriever):
    """Benchmark KeywordRetriever with top_k=10."""
    import asyncio

    def run_retrieve():
        return asyncio.run(keyword_retriever.retrieve("search test query", top_k=10))

    result = benchmark(run_retrieve)
    assert len(result) >= 0


def test_keyword_indexing(benchmark, sample_chunks):
    """Benchmark building TF-IDF index."""
    def build_index():
        return KeywordRetriever(chunks=sample_chunks)

    result = benchmark(build_index)
    assert len(result._chunks) == len(sample_chunks)


def test_augmentation_in_memory(benchmark, keyword_retriever):
    """Benchmark InMemoryAugmentation augment_for_memory."""
    import asyncio

    augmentation = InMemoryAugmentation(retriever=keyword_retriever, top_k=3)

    def run_augment():
        return asyncio.run(augmentation.augment_for_memory("What is search?"))

    result = benchmark(run_augment)
    assert isinstance(result, str)


def test_augmentation_in_memory_no_results(benchmark):
    """Benchmark InMemoryAugmentation with empty retriever."""
    import asyncio

    from koboi.rag.retriever import KeywordRetriever
    from koboi.rag.types import Chunk

    empty_retriever = KeywordRetriever(chunks=[
        Chunk(id="c1", doc_id="d1", content="Unrelated content", metadata={})
    ])
    augmentation = InMemoryAugmentation(retriever=empty_retriever, top_k=3)

    def run_augment():
        return asyncio.run(augmentation.augment_for_memory("xyznonexistent"))

    result = benchmark(run_augment)
    assert isinstance(result, str)


def test_chunk_creation(benchmark):
    """Benchmark creating Chunk objects."""
    from koboi.rag.types import Chunk

    def make_chunks():
        return [
            Chunk(
                id=f"chunk_{i}",
                doc_id=f"doc_{i % 10}",
                content=f"Content {i}: " + "test " * 20,
                metadata={"index": i},
            )
            for i in range(100)
        ]

    result = benchmark(make_chunks)
    assert len(result) == 100


def test_document_creation(benchmark, sample_text_50kb):
    """Benchmark creating Document objects."""
    def make_doc():
        return Document(
            id="doc_1",
            title="Sample Document",
            content=sample_text_50kb,
        )

    result = benchmark(make_doc)
    assert result.content == sample_text_50kb


def test_retrieval_result_creation(benchmark, sample_chunks):
    """Benchmark creating RetrievalResult objects."""
    from koboi.rag.types import RetrievalResult

    def make_results():
        return [
            RetrievalResult(
                chunk=chunk,
                score=0.5 + (i * 0.01),
                retrieval_method="keyword",
            )
            for i, chunk in enumerate(sample_chunks[:10])
        ]

    result = benchmark(make_results)
    assert len(result) == 10
