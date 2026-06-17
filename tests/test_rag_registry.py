"""Tests for koboi/rag/registry.py -- RAG component registry."""
from __future__ import annotations

import copy

import pytest
from unittest.mock import MagicMock

from koboi.rag.registry import (
    ComponentRegistry,
    ComponentEntry,
    chunker_registry,
    retriever_registry,
    augmentation_registry,
    build_rag,
    register_chunker,
    register_retriever,
    register_augmentation,
    load_custom_components,
    _extract_parameters,
    _resolve_kwargs,
)
from koboi.rag.types import Chunk, Document
from koboi.rag.retriever import HybridRetriever, KeywordRetriever
from koboi.rag.chunker import SemanticChunker
from koboi.rag.augmentation import RerankerRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Save and restore registry state for test isolation."""
    saved_chunkers = copy.deepcopy(chunker_registry._entries)
    saved_retrievers = copy.deepcopy(retriever_registry._entries)
    saved_augmentations = copy.deepcopy(augmentation_registry._entries)
    yield
    chunker_registry._entries = saved_chunkers
    retriever_registry._entries = saved_retrievers
    augmentation_registry._entries = saved_augmentations


@pytest.fixture
def sample_chunks():
    return [
        Chunk(id=f"c{i}", doc_id="d1", content=f"Content chunk {i}")
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# ComponentRegistry
# ---------------------------------------------------------------------------


class TestComponentRegistry:
    def test_register_and_get(self):
        reg = ComponentRegistry("test")
        reg.register("foo", str)
        entry = reg.get("foo")
        assert entry is not None
        assert entry.cls is str

    def test_get_unknown_returns_none(self):
        reg = ComponentRegistry("test")
        assert reg.get("nonexistent") is None

    def test_list_available(self):
        reg = ComponentRegistry("test")
        reg.register("b", int)
        reg.register("a", str)
        assert reg.list_available() == ["a", "b"]

    def test_clear(self):
        reg = ComponentRegistry("test")
        reg.register("foo", str)
        reg.clear()
        assert reg.get("foo") is None
        assert reg.list_available() == []

    def test_register_validates_config_aliases(self):
        reg = ComponentRegistry("test")
        with pytest.raises(ValueError, match="config_aliases"):
            reg.register(
                "bad", str,
                config_aliases={"yaml_key": "nonexistent_param"},
            )


class TestComponentEntry:
    def test_creation(self):
        entry = ComponentEntry(
            cls=str, parameters={}, description="test", inject=["client"],
        )
        assert entry.cls is str
        assert entry.description == "test"
        assert entry.inject == ["client"]

    def test_defaults(self):
        entry = ComponentEntry(cls=str, parameters={})
        assert entry.config_aliases == {}
        assert entry.inject == []


# ---------------------------------------------------------------------------
# _extract_parameters
# ---------------------------------------------------------------------------


class TestExtractParameters:
    def test_basic_class(self):
        class Foo:
            def __init__(self, x: int, y: str = "hello"):
                pass

        params = _extract_parameters(Foo)
        assert "x" in params
        assert "y" in params
        assert params["y"]["default"] == "hello"

    def test_skips_self(self):
        class Foo:
            def __init__(self, x: int):
                pass

        params = _extract_parameters(Foo)
        assert "self" not in params

    def test_skips_varargs(self):
        class Foo:
            def __init__(self, x: int, *args, **kwargs):
                pass

        params = _extract_parameters(Foo)
        assert "x" in params
        assert len(params) == 1


# ---------------------------------------------------------------------------
# _resolve_kwargs
# ---------------------------------------------------------------------------


class TestResolveKwargs:
    def test_basic_resolution(self):
        entry = ComponentEntry(
            cls=str, parameters={"chunk_size": {"default": 500}},
        )
        kwargs = _resolve_kwargs(entry, {"chunk_size": 200})
        assert kwargs == {"chunk_size": 200}

    def test_uses_default_when_missing(self):
        entry = ComponentEntry(
            cls=str, parameters={"chunk_size": {"default": 500}},
        )
        kwargs = _resolve_kwargs(entry, {})
        assert kwargs == {"chunk_size": 500}

    def test_config_aliases(self):
        entry = ComponentEntry(
            cls=str,
            parameters={"chunk_size": {"default": 500}},
            config_aliases={"size": "chunk_size"},
        )
        kwargs = _resolve_kwargs(entry, {"size": 300})
        assert kwargs == {"chunk_size": 300}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


class TestDecorators:
    def test_register_chunker_decorator(self):
        @register_chunker("test_chunker", description="test")
        class TestChunker:
            def __init__(self, size: int = 100):
                pass

        entry = chunker_registry.get("test_chunker")
        assert entry is not None
        assert entry.cls is TestChunker
        assert entry.description == "test"

    def test_register_retriever_decorator(self):
        @register_retriever("test_retriever", inject=["client"])
        class TestRetriever:
            def __init__(self, chunks, client=None):
                pass

        entry = retriever_registry.get("test_retriever")
        assert entry is not None
        assert entry.inject == ["client"]

    def test_register_augmentation_decorator(self):
        @register_augmentation("test_aug")
        class TestAugmentation:
            def __init__(self, retriever, top_k=3):
                pass

        entry = augmentation_registry.get("test_aug")
        assert entry is not None


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------


class TestBuiltinRegistrations:
    def test_chunkers_registered(self):
        from koboi.rag.chunker import _register_builtins
        _register_builtins()
        assert "fixed" in chunker_registry.list_available()
        assert "sentence" in chunker_registry.list_available()
        assert "paragraph" in chunker_registry.list_available()

    def test_retrievers_registered(self):
        from koboi.rag.retriever import _register_builtins
        _register_builtins()
        assert "keyword" in retriever_registry.list_available()
        assert "semantic" in retriever_registry.list_available()

    def test_augmentations_registered(self):
        from koboi.rag.augmentation import _register_builtins
        _register_builtins()
        assert "in_memory" in augmentation_registry.list_available()
        assert "on_the_fly" in augmentation_registry.list_available()

    def test_semantic_retriever_injects_client(self):
        from koboi.rag.retriever import _register_builtins
        _register_builtins()
        entry = retriever_registry.get("semantic")
        assert "client" in entry.inject

    def test_hybrid_retriever_registered(self):
        from koboi.rag.retriever import _register_builtins
        _register_builtins()
        assert "hybrid" in retriever_registry.list_available()
        entry = retriever_registry.get("hybrid")
        assert "client" in entry.inject

    def test_fixed_chunker_config_aliases(self):
        from koboi.rag.chunker import _register_builtins
        _register_builtins()
        entry = chunker_registry.get("fixed")
        assert "chunk_size" in entry.config_aliases
        assert "overlap" in entry.config_aliases


# ---------------------------------------------------------------------------
# build_rag
# ---------------------------------------------------------------------------


class TestBuildRag:
    def test_disabled_returns_none(self):
        assert build_rag({"enabled": False}) is None
        assert build_rag({}) is None
        assert build_rag(None) is None

    def test_no_documents_returns_none(self, tmp_path):
        result = build_rag({
            "enabled": True,
            "documents": [{"path": str(tmp_path / "nonexistent.md")}],
        })
        assert result is None

    def test_build_with_keyword_retriever(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("This is a test document about Python programming.")

        result = build_rag({
            "enabled": True,
            "chunker": "paragraph",
            "retriever": "keyword",
            "augmentation": "in_memory",
            "top_k": 2,
            "documents": [{"path": str(doc)}],
        })
        assert result is not None
        assert result.top_k == 2

    def test_build_with_on_the_fly(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("Test content for on-the-fly augmentation.")

        result = build_rag({
            "enabled": True,
            "retriever": "keyword",
            "augmentation": "on_the_fly",
            "documents": [{"path": str(doc)}],
        })
        assert result is not None
        from koboi.rag.augmentation import OnTheFlyAugmentation
        assert isinstance(result, OnTheFlyAugmentation)

    def test_unknown_chunker_falls_back(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("Test content.")

        result = build_rag({
            "enabled": True,
            "chunker": "nonexistent",
            "retriever": "keyword",
            "documents": [{"path": str(doc)}],
        })
        assert result is not None

    def test_unknown_retriever_falls_back(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("Test content.")

        result = build_rag({
            "enabled": True,
            "retriever": "nonexistent",
            "documents": [{"path": str(doc)}],
        })
        assert result is not None

    def test_unknown_augmentation_falls_back(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("Test content.")

        result = build_rag({
            "enabled": True,
            "retriever": "keyword",
            "augmentation": "nonexistent",
            "documents": [{"path": str(doc)}],
        })
        assert result is not None

    def test_passes_logger(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("Test content.")

        logger = MagicMock()
        result = build_rag(
            {
                "enabled": True,
                "retriever": "keyword",
                "documents": [{"path": str(doc)}],
            },
            logger=logger,
        )
        assert result is not None
        assert result.logger is logger

    def test_custom_chunker_via_registry(self, tmp_path):
        from koboi.rag.chunker import BaseChunker

        @register_chunker("custom_test")
        class CustomChunker(BaseChunker):
            def __init__(self, custom_param: int = 42):
                self.custom_param = custom_param

            def chunk(self, document):
                return [Chunk(id="c0", doc_id=document.id, content=document.content)]

        doc = tmp_path / "test.md"
        doc.write_text("Test content.")

        result = build_rag({
            "enabled": True,
            "chunker": "custom_test",
            "custom_param": 99,
            "retriever": "keyword",
            "documents": [{"path": str(doc)}],
        })
        assert result is not None

    def test_build_with_hybrid_retriever(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("Python is a programming language. JavaScript runs in browsers.")

        result = build_rag({
            "enabled": True,
            "retriever": "hybrid",
            "augmentation": "in_memory",
            "documents": [{"path": str(doc)}],
        })
        assert result is not None
        from koboi.rag.augmentation import InMemoryAugmentation
        assert isinstance(result, InMemoryAugmentation)
        assert isinstance(result.retriever, HybridRetriever)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    async def test_returns_results(self, sample_chunks):
        retriever = HybridRetriever(chunks=sample_chunks)
        results = await retriever.retrieve("Content chunk", top_k=3)
        assert len(results) > 0
        assert all(r.retrieval_method == "hybrid" for r in results)

    async def test_respects_top_k(self, sample_chunks):
        retriever = HybridRetriever(chunks=sample_chunks)
        results = await retriever.retrieve("Content", top_k=2)
        assert len(results) <= 2

    async def test_rrf_merges_scores(self):
        """Both keyword and semantic results should contribute to final score."""
        chunks = [
            Chunk(id="c0", doc_id="d1", content="Python programming language"),
            Chunk(id="c1", doc_id="d1", content="JavaScript web browser"),
            Chunk(id="c2", doc_id="d1", content="Python data science"),
        ]
        retriever = HybridRetriever(chunks=chunks, rrf_k=60)
        results = await retriever.retrieve("Python", top_k=3)
        assert len(results) > 0
        # Python chunks should rank higher
        assert "Python" in results[0].chunk.content

    async def test_empty_chunks(self):
        retriever = HybridRetriever(chunks=[])
        results = await retriever.retrieve("test", top_k=3)
        assert results == []


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class TestSemanticChunker:
    def test_falls_back_to_sentence_chunker(self):
        """Without embeddings, SemanticChunker falls back to SentenceChunker."""
        chunker = SemanticChunker(similarity_threshold=0.5, max_chunk_size=200)
        doc = Document(
            id="d1",
            title="Test",
            content="First sentence. Second sentence. Third sentence. Fourth sentence.",
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_single_sentence(self):
        chunker = SemanticChunker()
        doc = Document(id="d1", title="Test", content="Single sentence.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Single sentence."

    def test_empty_document(self):
        chunker = SemanticChunker()
        doc = Document(id="d1", title="Test", content="")
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_cosine_similarity(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]

        assert SemanticChunker._cosine_similarity(vec1, vec2) < 0.01
        assert abs(SemanticChunker._cosine_similarity(vec1, vec3) - 1.0) < 0.01

    def test_registered_in_registry(self):
        from koboi.rag.chunker import _register_builtins
        _register_builtins()
        from koboi.rag.registry import chunker_registry
        assert "semantic" in chunker_registry.list_available()


# ---------------------------------------------------------------------------
# RerankerRetriever
# ---------------------------------------------------------------------------


class TestRerankerRetriever:
    async def test_reranks_results(self):
        """Reranker should re-score and reorder results."""
        chunks = [
            Chunk(id="c0", doc_id="d1", content="Python is a programming language used for software development"),
            Chunk(id="c1", doc_id="d1", content="JavaScript runs in web browsers for frontend development"),
            Chunk(id="c2", doc_id="d1", content="Python data science libraries are great for programming"),
            Chunk(id="c3", doc_id="d1", content="Rust is a systems programming language for development"),
        ]
        base = KeywordRetriever(chunks)
        reranker = RerankerRetriever(base_retriever=base)
        results = await reranker.retrieve("Python programming", top_k=2)

        assert len(results) == 2
        assert all("reranked" in r.retrieval_method for r in results)
        # Python chunks should rank higher
        assert "Python" in results[0].chunk.content

    async def test_respects_top_k(self):
        chunks = [
            Chunk(id=f"c{i}", doc_id="d1", content=f"Content {i}")
            for i in range(10)
        ]
        base = KeywordRetriever(chunks)
        reranker = RerankerRetriever(base_retriever=base)
        results = await reranker.retrieve("Content", top_k=3)
        assert len(results) <= 3

    async def test_preserves_original_when_few_results(self):
        """If base retriever returns <= top_k results, pass through unchanged."""
        chunks = [
            Chunk(id="c0", doc_id="d1", content="Python programming"),
        ]
        base = KeywordRetriever(chunks)
        reranker = RerankerRetriever(base_retriever=base)
        results = await reranker.retrieve("Python", top_k=5)
        assert len(results) <= 1

    async def test_empty_results(self):
        base = KeywordRetriever(chunks=[])
        reranker = RerankerRetriever(base_retriever=base)
        results = await reranker.retrieve("test", top_k=3)
        assert results == []


# ---------------------------------------------------------------------------
# load_custom_components
# ---------------------------------------------------------------------------


class TestLoadCustomComponents:
    def test_import_failure_warns(self):
        # Should not raise, just warn
        load_custom_components(["nonexistent.module.rag"])

    def test_empty_list_noop(self):
        load_custom_components([])
