"""tests/test_rag_composition.py -- Integration tests proving RAG features compose.

These test REAL-WORLD configs that stack multiple features together (the gap
identified in the QA audit: features were unit-tested in isolation, never combined).
All fully mocked (no network / no LLM) -> deterministic, CI-safe.
"""

from __future__ import annotations

import koboi.rag.sources as sources
from koboi.rag.augmentation import OnTheFlyAugmentation, RerankerRetriever
from koboi.rag.registry import _load_documents, build_rag
from koboi.rag.retriever import BM25Retriever, KeywordRetriever
from koboi.rag.types import Chunk
from koboi.types import AgentResponse as AgentResponseT


# ---- shared mock ----
class _Chat:
    model = "mock-chat"

    async def complete(self, messages, tools=None, **kwargs):
        return AgentResponseT(content="refund policy", tool_calls=[])

    async def complete_stream(self, messages, tools=None, **kwargs):
        raise RuntimeError

    async def get_embeddings(self, text):
        return None


def _chunks():
    return [
        Chunk(id="a", doc_id="d", content="ALPHA refund", metadata={"source": "policy", "year": 2024}),
        Chunk(id="b", doc_id="d", content="BETA refund", metadata={"source": "handbook", "year": 2024}),
        Chunk(id="c", doc_id="d", content="GAMMA invoice", metadata={"source": "policy", "year": 2021}),
    ]


# --------------------------------------------------------------------------- #
# 1. Ingestion pipeline: http(mocked) + parse + cache + size-cap
# --------------------------------------------------------------------------- #
def test_ingestion_pipeline_http_parse_cache_sizecap(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_fetch(url, *, headers=None, timeout=None, max_bytes=None):
        calls["n"] += 1
        return b"<p>Refund window is 30 days</p>"  # HTML -> stripped to text

    monkeypatch.setattr(sources, "fetch_http", fake_fetch)
    cache = str(tmp_path / "dc")

    conf = {
        "enabled": True,
        "document_cache_path": cache,
        "max_document_size_mb": 10,
        "documents": [{"source": "http", "url": "https://example.com/policy.html"}],
    }
    _, chunks = _load_documents(conf)
    assert len(chunks) >= 1
    assert "Refund window is 30 days" in chunks[0].content
    assert chunks[0].metadata.get("source_format") == "html"

    # 2nd build: cache hit -> no new fetch
    _load_documents(conf)
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# 2. rewrite + filter + retrieve (query rewritten, then chunks filtered)
# --------------------------------------------------------------------------- #
async def test_rewrite_then_filter_compose():
    chat = _Chat()
    retrieved_queries: list[str] = []
    kw = KeywordRetriever(_chunks())
    orig = kw.retrieve

    async def spy(query, top_k=3, metadata_filter=None):
        retrieved_queries.append(query)
        return await orig(query, top_k, metadata_filter=metadata_filter)

    kw.retrieve = spy
    aug = OnTheFlyAugmentation(
        retriever=kw,
        top_k=5,
        query_rewrite=True,
        rewrite_client=chat,
        metadata_filter={"year": {"$gte": 2024}},
    )
    out = await aug.augment_for_llm([{"role": "user", "content": "hey what about refunds?"}])
    content = out[-1]["content"]

    # The retriever saw the REWRITTEN query ("refund policy"), not the raw.
    assert retrieved_queries == ["refund policy"]
    # The filter excluded chunk c (year 2021); only ALPHA + BETA survive.
    assert "ALPHA" in content and "BETA" in content
    assert "GAMMA" not in content
    # Rewrite metadata stamped.
    assert aug.last_rewrite is not None and aug.last_rewrite["method"] == "llm"


# --------------------------------------------------------------------------- #
# 3. filter + rerank (filtered set is reranked)
# --------------------------------------------------------------------------- #
async def test_filter_then_rerank_compose():
    base = KeywordRetriever(_chunks())
    rr = RerankerRetriever(base)
    results = await rr.retrieve("refund", top_k=1, metadata_filter={"year": {"$gte": 2024}})
    ids = {r.chunk.id for r in results}
    assert "c" not in ids  # 2021 filtered out
    assert len(results) == 1  # top_k=1 respected after rerank
    # Reranker marks the method.
    assert all("reranked" in r.retrieval_method for r in results)


# --------------------------------------------------------------------------- #
# 4. Full build_rag: bm25 + query_rewrite + filter + rerank in one config
# --------------------------------------------------------------------------- #
def test_full_build_rag_composes_all_features(tmp_path):
    doc = tmp_path / "kb.md"
    doc.write_text("refund policy 30 days")
    chat = _Chat()
    aug = build_rag(
        {
            "enabled": True,
            "retriever": "bm25",
            "top_k": 3,
            "rerank": True,
            "query_rewrite": True,
            "rewrite": {"timeout": 5.0},
            "filter": {"source": {"$in": ["policy", "handbook"]}},
            "documents": [{"path": str(doc)}],
        },
        chat_client=chat,
    )
    assert aug is not None
    # Reranker wraps BM25.
    assert isinstance(aug.retriever, RerankerRetriever)
    assert isinstance(aug.retriever._base, BM25Retriever)
    # Rewrite + filter wired.
    assert aug._query_rewrite is True
    assert aug.metadata_filter == {"source": {"$in": ["policy", "handbook"]}}
