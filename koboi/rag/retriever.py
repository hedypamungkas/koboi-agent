from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

from koboi.rag.types import Chunk, RetrievalResult

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from koboi.llm.base import LLMClient


def resolve_retriever(config: dict, chunks: list[Chunk], client: LLMClient | None = None) -> BaseRetriever:
    """Build a retriever from a RAG config dict.

    Reads 'retriever' (default 'keyword').
    """
    from koboi.rag.registry import retriever_registry

    retriever_name = config.get("retriever", "keyword")
    entry = retriever_registry.get(retriever_name)
    if entry is None:
        _logger.warning("Unknown retriever '%s', falling back to keyword", retriever_name)
        entry = retriever_registry.get("keyword")
        if entry is None:
            raise ValueError("No retrievers registered")

    kwargs: dict = {"chunks": chunks}
    if "client" in entry.inject:
        kwargs["client"] = client
    return entry.cls(**kwargs)


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        ...


class KeywordRetriever(BaseRetriever):
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._tfidf_index: dict[str, dict[str, float]] = {}
        self._idf: dict[str, float] = {}
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _build_index(self) -> None:
        doc_freq: dict[str, int] = {}
        total_docs = len(self._chunks)

        term_counts_per_chunk: list[Counter] = []
        for chunk in self._chunks:
            terms = self._tokenize(chunk.content)
            counts = Counter(terms)
            term_counts_per_chunk.append(counts)
            for term in counts:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        self._idf = {
            term: math.log((total_docs + 1) / (freq + 1)) + 1
            for term, freq in doc_freq.items()
        }

        for i, counts in enumerate(term_counts_per_chunk):
            chunk_id = self._chunks[i].id
            total_terms = sum(counts.values()) or 1
            self._tfidf_index[chunk_id] = {
                term: (count / total_terms) * self._idf.get(term, 1.0)
                for term, count in counts.items()
            }

    def _score(self, query_terms: list[str], chunk_id: str) -> float:
        chunk_vec = self._tfidf_index.get(chunk_id, {})
        query_counts = Counter(query_terms)
        total = sum(query_counts.values()) or 1
        query_vec = {
            term: (count / total) * self._idf.get(term, 1.0)
            for term, count in query_counts.items()
        }

        dot = sum(query_vec.get(t, 0) * chunk_vec.get(t, 0) for t in query_vec)
        norm_q = sum(v * v for v in query_vec.values()) ** 0.5
        norm_c = sum(v * v for v in chunk_vec.values()) ** 0.5

        if norm_q == 0 or norm_c == 0:
            return 0.0
        return dot / (norm_q * norm_c)

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored = [
            (chunk, self._score(query_terms, chunk.id))
            for chunk in self._chunks
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(chunk=chunk, score=score, retrieval_method="keyword")
            for chunk, score in scored[:top_k]
            if score > 0
        ]


class SemanticRetriever(BaseRetriever):
    def __init__(
        self,
        chunks: list[Chunk],
        client: LLMClient | None = None,
    ):
        self._chunks = chunks
        self._client = client
        self._embedding_available = True
        self._fallback: KeywordRetriever | None = None
        self._chunk_embeddings: dict[str, list[float]] = {}
        self._index_built = False
        self._building = False
        self._build_index()

    async def _get_embedding(self, text: str) -> list[float] | None:
        if not self._client:
            return None
        result = await self._client.get_embeddings(text)
        if result is None:
            self._embedding_available = False
        return result

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-10)

    def _build_index(self) -> None:
        if not self._client:
            self._embedding_available = False
            self._fallback = KeywordRetriever(self._chunks)
            _logger.warning(
                "SemanticRetriever: no embedding client provided -- "
                "falling back to keyword retrieval for %d chunks",
                len(self._chunks),
            )
            return

    async def _ensure_index_built(self) -> None:
        if self._index_built:
            return
        if self._building:
            return
        self._building = True

        for chunk in self._chunks:
            emb = await self._get_embedding(chunk.content)
            if emb is None:
                self._fallback = KeywordRetriever(self._chunks)
                self._embedding_available = False
                self._index_built = True
                self._building = False
                _logger.info(
                    "SemanticRetriever: falling back to keyword retrieval "
                    "for %d chunks (embedding endpoint unavailable).",
                    len(self._chunks),
                )
                return
            self._chunk_embeddings[chunk.id] = emb
            chunk.embedding = emb

        self._index_built = True
        self._building = False

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        await self._ensure_index_built()

        if not self._embedding_available:
            if self._fallback is None:
                self._fallback = KeywordRetriever(self._chunks)
            results = await self._fallback.retrieve(query, top_k)
            for r in results:
                r.retrieval_method = "semantic (fallback to keyword)"
            return results

        query_emb = await self._get_embedding(query)
        if query_emb is None:
            if self._fallback is None:
                self._fallback = KeywordRetriever(self._chunks)
            return await self._fallback.retrieve(query, top_k)

        scored = [
            (chunk, self._cosine_similarity(query_emb, emb))
            for chunk, emb in zip(
                self._chunks,
                [self._chunk_embeddings.get(c.id, []) for c in self._chunks],
            )
            if emb
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(chunk=chunk, score=score, retrieval_method="semantic")
            for chunk, score in scored[:top_k]
        ]


class HybridRetriever(BaseRetriever):
    """Combines keyword (TF-IDF) and semantic (embedding) retrieval with RRF.

    Reciprocal Rank Fusion merges two ranked lists by score =
    1/(k + rank_keyword) + 1/(k + rank_semantic), where k=60 is the
    standard smoothing constant.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        client: LLMClient | None = None,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ):
        self._chunks = chunks
        self._rrf_k = rrf_k
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
        self._keyword = KeywordRetriever(chunks)
        self._semantic = SemanticRetriever(chunks, client=client)

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        # Get results from both retrievers (fetch more than top_k for fusion)
        fetch_k = max(top_k * 3, 10)
        kw_results = await self._keyword.retrieve(query, top_k=fetch_k)
        sem_results = await self._semantic.retrieve(query, top_k=fetch_k)

        # Build RRF scores: score = weight / (k + rank)
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, r in enumerate(kw_results, start=1):
            cid = r.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self._keyword_weight / (self._rrf_k + rank)
            chunk_map[cid] = r.chunk

        for rank, r in enumerate(sem_results, start=1):
            cid = r.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self._semantic_weight / (self._rrf_k + rank)
            chunk_map[cid] = r.chunk

        # Sort by combined RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=score,
                retrieval_method="hybrid",
            )
            for cid, score in sorted_items[:top_k]
        ]


# ---------------------------------------------------------------------------
# Register built-in retrievers with the RAG registry
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register built-in retrievers. Called lazily on first use."""
    from koboi.rag.registry import register_retriever as _reg

    _reg("keyword", description="TF-IDF cosine similarity retrieval")(KeywordRetriever)

    _reg(
        "semantic",
        description="Embedding-based retrieval with keyword fallback",
        inject=["client"],
    )(SemanticRetriever)

    _reg(
        "hybrid",
        description="Combines keyword and semantic retrieval with Reciprocal Rank Fusion",
        inject=["client"],
    )(HybridRetriever)
