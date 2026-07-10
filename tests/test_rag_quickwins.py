"""tests/test_rag_quickwins.py -- Regression guards for the 7 RAG quick-win capabilities.

Each test asserts a capability that was previously absent (gap #3/#5/#7/#8/#11/#12).
All additive/opt-in: defaults are unchanged, so these guard both the new behaviour
and the absence of regressions in the default path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from koboi.rag.augmentation import InMemoryAugmentation, OnTheFlyAugmentation, RerankerRetriever
from koboi.rag.registry import _load_documents, build_rag, retriever_registry
from koboi.rag.retriever import (
    BM25Retriever,
    KeywordRetriever,
    SemanticRetriever,
    _EmbeddingIndexCache,
    _EMBEDDING_CACHE,
    clear_embedding_cache,
    set_embedding_cache_path,
)
from koboi.rag.types import Chunk


class _CountClient:
    """Counts get_embeddings calls; returns a deterministic vector."""

    model = "mock-count"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, tools=None):
        raise RuntimeError

    async def complete_stream(self, messages, tools=None):
        raise RuntimeError

    async def get_embeddings(self, text):
        self.calls += 1
        return [float(len(text)), 1.0]


# --------------------------------------------------------------------------- #
# #8 BM25 retriever (additive; keyword stays default)
# --------------------------------------------------------------------------- #
async def test_bm25_retriever_registered_and_ranks():
    assert "bm25" in retriever_registry.list_available()
    chunks = [
        Chunk(id="short", doc_id="d", content="cat", metadata={}),
        Chunk(id="long", doc_id="d", content="cat cat cat cat cat", metadata={}),
        Chunk(id="none", doc_id="d", content="dog dog dog", metadata={}),
    ]
    results = await BM25Retriever(chunks).retrieve("cat", top_k=3)
    assert results and results[0].retrieval_method == "bm25"
    # saturation: the long doc does NOT dominate linearly -- BM25 differentiates
    # both matching docs and ranks the non-matching one off the list.
    ids = [r.chunk.id for r in results]
    assert "none" not in ids
    assert "short" in ids and "long" in ids


# --------------------------------------------------------------------------- #
# #7 Query-embedding cache (corpus cached separately; query cached per retriever)
# --------------------------------------------------------------------------- #
async def test_query_embedding_cached_across_retrieves():
    clear_embedding_cache()
    chunks = [Chunk(id=f"c{i}", doc_id="d", content=f"doc {i}", metadata={}) for i in range(3)]
    client = _CountClient()
    retriever = SemanticRetriever(chunks, client=client)
    await retriever.retrieve("same query", top_k=2)
    after_first = client.calls
    await retriever.retrieve("same query", top_k=2)  # identical query
    assert client.calls == after_first  # query NOT re-embedded


# --------------------------------------------------------------------------- #
# #5 On-disk embedding cache (persists across "process restart" = fresh instance)
# --------------------------------------------------------------------------- #
async def test_on_disk_embedding_cache_persists(tmp_path):
    cache_file = tmp_path / "emb.json"
    calls = {"n": 0}

    async def embed(text):
        calls["n"] += 1
        return [float(len(text))]

    chunks = [Chunk(id=f"c{i}", doc_id="d", content=f"doc {i}", metadata={}) for i in range(3)]

    c1 = _EmbeddingIndexCache(cache_path=str(cache_file))
    await c1.get_or_build(chunks, embed)  # "first process": embeds + saves
    embedded_once = calls["n"]
    assert cache_file.exists()
    json.loads(cache_file.read_text())  # valid JSON

    c2 = _EmbeddingIndexCache(cache_path=str(cache_file))  # "restart": fresh instance
    await c2.get_or_build(chunks, embed)
    assert calls["n"] == embedded_once  # zero re-embeds -- loaded from disk


def test_build_rag_wires_embedding_cache_path(tmp_path):
    cache_file = tmp_path / "emb.json"
    doc = tmp_path / "kb.md"
    doc.write_text("alpha beta")
    try:
        build_rag(
            {
                "enabled": True,
                "retriever": "keyword",
                "embedding_cache_path": str(cache_file),
                "documents": [{"path": str(doc)}],
            }
        )
        assert _EMBEDDING_CACHE._cache_path == str(cache_file)
    finally:
        set_embedding_cache_path(None)
        clear_embedding_cache()


# --------------------------------------------------------------------------- #
# #12 Citation numbering + #11b content-hash dedup
# --------------------------------------------------------------------------- #
async def test_citations_numbered_and_duplicates_deduped():
    chunks = [
        Chunk(id="a", doc_id="d", content="refund 30 days", metadata={"source": "docA"}),
        Chunk(id="b", doc_id="d", content="refund 30 days", metadata={"source": "docB"}),  # dup of a
        Chunk(id="c", doc_id="d", content="invoice net 14", metadata={"source": "docC"}),
    ]
    out = await InMemoryAugmentation(retriever=KeywordRetriever(chunks), top_k=5).augment_for_memory("refund invoice")
    assert "[1]" in out and "[2]" in out  # numbered citations
    assert "[Source: docA]" in out
    assert out.count("refund 30 days") == 1  # duplicate content deduped


async def test_on_the_fly_augment_also_numbers_and_dedups():
    chunks = [
        Chunk(id="a", doc_id="d", content="same text", metadata={"source": "s1"}),
        Chunk(id="b", doc_id="d", content="same text", metadata={"source": "s2"}),
    ]
    msgs = await OnTheFlyAugmentation(retriever=KeywordRetriever(chunks), top_k=5).augment_for_llm(
        [{"role": "user", "content": "same text"}]
    )
    content = msgs[-1]["content"]
    assert "[1]" in content and "[2]" not in content  # duplicate content -> single chunk


# --------------------------------------------------------------------------- #
# #11a Reranker opt-in (default off; on -> wraps the retriever)
# --------------------------------------------------------------------------- #
def test_rerank_flag_wraps_retriever(tmp_path):
    doc = tmp_path / "kb.md"
    doc.write_text("alpha beta gamma")
    default = build_rag({"enabled": True, "retriever": "keyword", "documents": [{"path": str(doc)}]})
    wrapped = build_rag({"enabled": True, "retriever": "keyword", "rerank": True, "documents": [{"path": str(doc)}]})
    assert not isinstance(default.retriever, RerankerRetriever)
    assert isinstance(wrapped.retriever, RerankerRetriever)


# --------------------------------------------------------------------------- #
# #3 Globbing + directory recursion (and no more IsADirectoryError)
# --------------------------------------------------------------------------- #
def test_glob_and_directory_load_documents(tmp_path):
    (tmp_path / "a.md").write_text("file a")
    (tmp_path / "b.md").write_text("file b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("file c")
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02\x03PNG\x00binary\xff\xfe")  # NUL bytes -> binary -> skipped

    _, globbed = _load_documents({"enabled": True, "documents": [{"path": str(tmp_path / "*.md")}]})
    assert len(globbed) == 2  # a.md, b.md (non-recursive)

    _, recursed = _load_documents({"enabled": True, "documents": [{"path": str(tmp_path)}]})
    assert len(recursed) == 3  # a.md, b.md, sub/c.md (binary skipped)
    assert {c.metadata["source"] for c in recursed} == {"a", "b", "c"}
