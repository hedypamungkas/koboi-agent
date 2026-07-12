"""tests/test_rag_rerank.py -- Cross-encoder rerank backends + wrapper + wiring.

Fully mocked (no network / no LLM): deterministic, CI-safe. Covers response parsing,
fail-soft semantics, the over-fetch+rescore wrapper, the build_rerank_client factory,
the build_rag dict-vs-bool branch, the local-BGE import gate, batch clamping, and
lifecycle close.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.llm.base import LLMInvalidRequestError, LLMServerError
from koboi.rag.augmentation import RerankerRetriever
from koboi.rag.registry import build_rag
from koboi.rag.rerank import (
    CrossEncoderReranker,
    CohereRerankBackend,
    JinaRerankBackend,
    LocalBGERerankBackend,
    RerankBackend,
    build_rerank_client,
)
from koboi.rag.retriever import BaseRetriever, BM25Retriever
from koboi.rag.types import Chunk, RetrievalResult
from koboi.types import AgentResponse as AgentResponseT


# ---- shared fakes -------------------------------------------------------- #


def _http_backend(cls, response, *, side_effect=None, api_key="k", **kw):
    """Instantiate an HTTP backend (Jina/Cohere) with its transport mocked."""
    backend = cls(api_key=api_key, **kw)
    backend._transport = MagicMock()
    if side_effect is not None:
        backend._transport.post = AsyncMock(side_effect=side_effect)
    else:
        backend._transport.post = AsyncMock(return_value=response)
    return backend


class _MockBackend(RerankBackend):
    """Deterministic backend for wrapper tests: returns a preset ranking (or None)."""

    provider = "mock"

    def __init__(self, ranked=None):
        self._ranked = ranked
        self.closed = False

    async def rerank(self, query, documents, top_n):
        return None if self._ranked is None else list(self._ranked)[:top_n]

    async def close(self):
        self.closed = True


class _FakeBaseRetriever(BaseRetriever):
    """Returns a fixed result list; records the top_k it was called with (over-fetch)."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_top_k = None

    async def retrieve(self, query, top_k=3, metadata_filter=None):
        self.last_top_k = top_k
        return [RetrievalResult(chunk=c, score=0.5, retrieval_method="bm25") for c in self._chunks[:top_k]]


def _chunks(n):
    return [Chunk(id=str(i), doc_id=f"d{i}", content=f"doc-{i}") for i in range(n)]


class _Chat:
    model = "mock-chat"

    async def complete(self, messages, tools=None, **kwargs):
        return AgentResponseT(content="ctx", tool_calls=[])

    async def complete_stream(self, messages, tools=None, **kwargs):
        raise RuntimeError

    async def get_embeddings(self, text):
        return None


# --------------------------------------------------------------------------- #
# Backend response parsing + fail-soft
# --------------------------------------------------------------------------- #


class TestJinaBackend:
    async def test_parses_response(self):
        backend = _http_backend(
            JinaRerankBackend,
            {"results": [{"index": 0, "relevance_score": 0.98, "document": {"text": "x"}}]},
        )
        ranked = await backend.rerank("q", ["a", "b"], 2)
        assert ranked == [(0, 0.98)]
        # Body sent to the transport is well-formed (path + body are positional).
        backend._transport.post.assert_awaited_once()
        path, body = backend._transport.post.call_args.args
        assert path == "/rerank"
        assert body == {"model": backend._model, "query": "q", "documents": ["a", "b"], "top_n": 2}

    async def test_clamps_out_of_range_score(self):
        backend = _http_backend(JinaRerankBackend, {"results": [{"index": 1, "relevance_score": 1.4}]})
        ranked = await backend.rerank("q", ["a", "b"], 2)
        assert ranked == [(1, 1.0)]

    async def test_fail_soft_returns_none(self):
        backend = _http_backend(JinaRerankBackend, None, side_effect=LLMServerError("boom"))
        assert await backend.rerank("q", ["a", "b"], 2) is None


class TestCohereBackend:
    async def test_parses_response_no_document_echo(self):
        backend = _http_backend(
            CohereRerankBackend, {"results": [{"index": 1, "relevance_score": 0.42}]}
        )
        ranked = await backend.rerank("q", ["a", "b"], 1)
        assert ranked == [(1, 0.42)]

    async def test_fail_soft_returns_none(self):
        backend = _http_backend(CohereRerankBackend, None, side_effect=LLMServerError("5xx"))
        assert await backend.rerank("q", ["a", "b"], 2) is None


# --------------------------------------------------------------------------- #
# build_rerank_client factory
# --------------------------------------------------------------------------- #


class TestBuildRerankClient:
    def test_jina_default(self):
        backend = build_rerank_client({"api_key": "k"})
        assert isinstance(backend, JinaRerankBackend)
        assert backend.provider == "jina"

    def test_cohere_explicit(self):
        assert isinstance(build_rerank_client({"provider": "cohere", "api_key": "k"}), CohereRerankBackend)

    def test_none_without_apikey(self):
        assert build_rerank_client({"provider": "jina"}) is None
        assert build_rerank_client(None) is None
        assert build_rerank_client({}) is None

    def test_local_no_key_required(self):
        # local doesn't need an api_key (the gate is the import, tested separately).
        # We can't construct it here (extra not installed), so just assert the keyless
        # branch is taken by checking it doesn't return None for the key reason.
        with pytest.raises(LLMInvalidRequestError, match="rerank-local"):
            build_rerank_client({"provider": "local"})

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMInvalidRequestError, match="Unknown rerank provider"):
            build_rerank_client({"provider": "voyageai", "api_key": "k"})

    def test_empty_string_model_falls_back_to_default(self):
        # `${RERANK_MODEL:}` env interpolation resolves to "" -- must NOT pass an empty
        # model to the provider (would 400 -> fail-soft -> silent bare BM25). Empty string
        # falls back to the provider default (jina-reranker-v3), same as a missing key.
        backend = build_rerank_client({"provider": "jina", "api_key": "k", "model": "", "base_url": ""})
        assert isinstance(backend, JinaRerankBackend)
        assert backend._model == "jina-reranker-v3"


# --------------------------------------------------------------------------- #
# CrossEncoderReranker wrapper
# --------------------------------------------------------------------------- #


class TestCrossEncoderReranker:
    async def test_overfetches_and_rescores_in_new_order(self):
        base = _FakeBaseRetriever(_chunks(6))
        # Backend promotes chunk index 5 to the top, then 0, then 1.
        wrapper = CrossEncoderReranker(base, _MockBackend(ranked=[(5, 0.9), (0, 0.8), (1, 0.7)]))
        out = await wrapper.retrieve("q", top_k=3)
        assert [r.chunk.id for r in out] == ["5", "0", "1"]
        assert all(r.retrieval_method == "rerank:mock(bm25)" for r in out)
        assert [r.score for r in out] == [0.9, 0.8, 0.7]
        # Over-fetch: base called with fetch_k = max(3*3, 3+5) = 9, but only 6 exist.
        assert base.last_top_k == 9

    async def test_no_rerank_when_results_le_top_k(self):
        base = _FakeBaseRetriever(_chunks(2))
        backend = _MockBackend(ranked=[(0, 1.0)])
        wrapper = CrossEncoderReranker(base, backend)
        out = await wrapper.retrieve("q", top_k=3)
        # Only 2 results <= top_k=3 -> returned as-is, backend NOT consulted.
        assert len(out) == 2
        assert all(r.retrieval_method == "bm25" for r in out)

    async def test_falls_back_on_backend_failure_default(self):
        base = _FakeBaseRetriever(_chunks(6))
        wrapper = CrossEncoderReranker(base, _MockBackend(ranked=None))  # backend "fails"
        out = await wrapper.retrieve("q", top_k=3)
        # Base order preserved, truncated to top_k; method stays the base method (silent).
        assert [r.chunk.id for r in out] == ["0", "1", "2"]
        assert all(r.retrieval_method == "bm25" for r in out)

    async def test_fallback_false_stamps_failed_method(self):
        base = _FakeBaseRetriever(_chunks(6))
        wrapper = CrossEncoderReranker(base, _MockBackend(ranked=None), fallback=False)
        out = await wrapper.retrieve("q", top_k=3)
        assert all("rerank:failed(mock" in r.retrieval_method for r in out)

    async def test_score_threshold_drops_low_scores(self):
        base = _FakeBaseRetriever(_chunks(6))
        wrapper = CrossEncoderReranker(
            base, _MockBackend(ranked=[(5, 0.9), (0, 0.1), (1, 0.8)]), score_threshold=0.5
        )
        out = await wrapper.retrieve("q", top_k=3)
        # 0.1 dropped; only 5 and 1 survive.
        assert [r.chunk.id for r in out] == ["5", "1"]

    async def test_fetch_k_clamped_to_provider_cap(self):
        base = _FakeBaseRetriever(_chunks(150))
        cohere = _http_backend(CohereRerankBackend, {"results": [{"index": 0, "relevance_score": 0.9}]})
        wrapper = CrossEncoderReranker(base, cohere)  # cohere cap = 100
        await wrapper.retrieve("q", top_k=50)  # naive fetch_k = 155
        assert base.last_top_k == 100  # clamped to the cohere batch cap

    async def test_close_closes_backend(self):
        base = _FakeBaseRetriever(_chunks(6))
        backend = _MockBackend(ranked=[(0, 1.0)])
        wrapper = CrossEncoderReranker(base, backend)
        await wrapper.close()
        assert backend.closed is True

    async def test_close_closes_http_transport(self):
        base = _FakeBaseRetriever(_chunks(6))
        backend = JinaRerankBackend(api_key="k")
        backend._transport = MagicMock()
        backend._transport.close = AsyncMock()
        wrapper = CrossEncoderReranker(base, backend)
        await wrapper.close()
        backend._transport.close.assert_awaited_once()


# --------------------------------------------------------------------------- #
# LocalBGE import gate
# --------------------------------------------------------------------------- #


class TestLocalBGEGate:
    def test_missing_extra_raises_clear_error(self, monkeypatch):
        # Force the import to fail even if sentence-transformers happens to be installed.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(LLMInvalidRequestError, match=r"\[rerank-local\]"):
            LocalBGERerankBackend()


# --------------------------------------------------------------------------- #
# build_rag wiring: dict -> cross-encoder, bool -> heuristic (regression)
# --------------------------------------------------------------------------- #


class TestBuildRagWiring:
    def test_dict_rerank_wires_cross_encoder(self, tmp_path):
        doc = tmp_path / "kb.md"
        doc.write_text("refund policy 30 days")
        aug = build_rag(
            {
                "enabled": True,
                "retriever": "bm25",
                "top_k": 3,
                "rerank": {"provider": "jina", "api_key": "x"},
                "documents": [{"path": str(doc)}],
            },
            chat_client=_Chat(),
        )
        assert aug is not None
        assert isinstance(aug.retriever, CrossEncoderReranker)
        assert isinstance(aug.retriever._base, BM25Retriever)
        assert aug.retriever._provider == "jina"

    def test_bool_rerank_still_heuristic(self, tmp_path):
        doc = tmp_path / "kb.md"
        doc.write_text("refund policy 30 days")
        aug = build_rag(
            {
                "enabled": True,
                "retriever": "bm25",
                "top_k": 3,
                "rerank": True,  # legacy bool -> heuristic, unchanged
                "documents": [{"path": str(doc)}],
            },
            chat_client=_Chat(),
        )
        assert aug is not None
        assert isinstance(aug.retriever, RerankerRetriever)
        assert isinstance(aug.retriever._base, BM25Retriever)

    def test_dict_rerank_without_key_falls_through_to_heuristic(self, tmp_path):
        # No api_key -> build_rerank_client returns None -> no cross-encoder wrap (and no
        # heuristic either, since the dict branch doesn't fall back to the bool path).
        doc = tmp_path / "kb.md"
        doc.write_text("refund policy 30 days")
        aug = build_rag(
            {
                "enabled": True,
                "retriever": "bm25",
                "top_k": 3,
                "rerank": {"provider": "jina"},  # no api_key
                "documents": [{"path": str(doc)}],
            },
            chat_client=_Chat(),
        )
        assert aug is not None
        # Bare BM25 (neither wrapper applied).
        assert isinstance(aug.retriever, BM25Retriever)
