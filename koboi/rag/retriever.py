from __future__ import annotations

import asyncio
import hashlib
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
    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]: ...


class KeywordRetriever(BaseRetriever):
    def __init__(self, chunks: list[Chunk], synonyms: dict[str, list[str]] | None = None):
        self._chunks = chunks
        # Optional lexical-bridge map (e.g. {"dog": ["pet"]}) applied to the
        # QUERY only, so vocabulary that differs from the document (synonyms /
        # paraphrase) still matches. Opt-in via ``rag.synonyms`` in config; no
        # effect when unset. Cheap (no re-indexing). Complements -- but does not
        # replace -- semantic retrieval, which is the general fix but needs an
        # embedding endpoint.
        self._synonyms: dict[str, list[str]] = {k.lower(): v for k, v in (synonyms or {}).items()}
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

        self._idf = {term: math.log((total_docs + 1) / (freq + 1)) + 1 for term, freq in doc_freq.items()}

        for i, counts in enumerate(term_counts_per_chunk):
            chunk_id = self._chunks[i].id
            total_terms = sum(counts.values()) or 1
            self._tfidf_index[chunk_id] = {
                term: (count / total_terms) * self._idf.get(term, 1.0) for term, count in counts.items()
            }

    def _score(self, query_terms: list[str], chunk_id: str) -> float:
        chunk_vec = self._tfidf_index.get(chunk_id, {})
        query_counts = Counter(query_terms)
        total = sum(query_counts.values()) or 1
        query_vec = {term: (count / total) * self._idf.get(term, 1.0) for term, count in query_counts.items()}

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

        # Expand the query with configured synonyms (query-side only; the index
        # is untouched). Bridges vocabulary gaps like "dog" vs document "pet".
        if self._synonyms:
            expanded = [a for term in query_terms for a in self._synonyms.get(term, [])]
            if expanded:
                query_terms = query_terms + expanded

        scored = [(chunk, self._score(query_terms, chunk.id)) for chunk in self._chunks]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(chunk=chunk, score=score, retrieval_method="keyword")
            for chunk, score in scored[:top_k]
            if score > 0
        ]


class _EmbeddingIndexCache:
    """Process-level shared embedding index, keyed by a corpus signature.

    Retrievers built from the same corpus (same chunk ids + contents) reuse one
    embedding pass instead of each re-embedding every chunk. This makes semantic
    / hybrid retrieval affordable for multi-session deployments (e.g. the e2e
    suite, which builds a fresh agent per session): the corpus is embedded once
    per process, not per session (~76 chunks once vs ~70 s x N sessions).

    Only successful builds are cached; an unavailable embedding endpoint yields
    a miss that is not stored, so a later retry once it recovers still works.
    Concurrency: one ``asyncio.Lock`` per signature dedupes concurrent
    first-builds (mirrors ``koboi/server/pool.py``'s per-key lock pattern).

    Note: the signature is content-only and assumes a single embedding model per
    process. A model change requires a process restart, which clears this
    module-level cache.
    """

    def __init__(self) -> None:
        self._index: dict[str, dict[str, list[float]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _signature(chunks: list[Chunk]) -> str:
        h = hashlib.sha256()
        for c in chunks:
            h.update(c.id.encode())
            h.update(b"\0")
            h.update(c.content.encode())
            h.update(b"\0")
        return h.hexdigest()

    async def get_or_build(self, chunks, embed_fn) -> tuple[dict[str, list[float]] | None, bool]:
        """Return ``(id -> embedding, ok)``. ``ok=False`` means unavailable."""
        sig = self._signature(chunks)
        cached = self._index.get(sig)
        if cached is not None:
            return cached, True
        # dict.setdefault is atomic in a single-threaded asyncio loop, so the
        # lock is created safely before any await.
        lock = self._locks.setdefault(sig, asyncio.Lock())
        async with lock:
            cached = self._index.get(sig)  # double-check after acquiring lock
            if cached is not None:
                return cached, True
            emb_map: dict[str, list[float]] = {}
            for c in chunks:
                emb = await embed_fn(c.content)
                if emb is None:
                    return None, False  # endpoint unavailable; do not cache
                emb_map[c.id] = emb
            self._index[sig] = emb_map
            return emb_map, True

    def clear(self) -> None:
        self._index.clear()
        self._locks.clear()


#: Process-wide shared embedding index (see ``_EmbeddingIndexCache``).
_EMBEDDING_CACHE = _EmbeddingIndexCache()


def clear_embedding_cache() -> None:
    """Reset the shared embedding index (test isolation / forced rebuild)."""
    _EMBEDDING_CACHE.clear()


class SemanticRetriever(BaseRetriever):
    def __init__(
        self,
        chunks: list[Chunk],
        client: LLMClient | None = None,
        synonyms: dict[str, list[str]] | None = None,
    ):
        self._chunks = chunks
        self._client = client
        # Query-side synonym bridge, propagated to every keyword fallback so that
        # when embeddings are unavailable (and semantic degrades to keyword) the
        # bridge still closes vocabulary gaps -- keeping hybrid correct under
        # fallback instead of RRF-demoting synonym-only matches.
        self._synonyms = synonyms
        self._embedding_available = True
        self._fallback: KeywordRetriever | None = None
        self._chunk_embeddings: dict[str, list[float]] = {}
        self._index_built = False
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
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-10)

    def _build_index(self) -> None:
        if not self._client:
            self._embedding_available = False
            self._fallback = KeywordRetriever(self._chunks, synonyms=self._synonyms)
            _logger.warning(
                "SemanticRetriever: no embedding client provided -- falling back to keyword retrieval for %d chunks",
                len(self._chunks),
            )
            return

    async def _ensure_index_built(self) -> None:
        if self._index_built:
            return
        if not self._client:
            # No embedding client: ``_build_index`` already armed the keyword fallback.
            self._index_built = True
            return
        # Shared, process-level index: the corpus is embedded once per process,
        # not per retriever/session. Concurrent first-builds are deduped by the
        # cache's per-signature lock.
        emb_map, ok = await _EMBEDDING_CACHE.get_or_build(self._chunks, self._client.get_embeddings)
        if not ok:
            self._fallback = KeywordRetriever(self._chunks, synonyms=self._synonyms)
            self._embedding_available = False
            self._index_built = True
            _logger.warning(
                "SemanticRetriever: falling back to keyword retrieval for %d chunks "
                "(the provider returned no embeddings). Set a top-level `embedding:` "
                "section to enable true semantic retrieval.",
                len(self._chunks),
            )
            return
        self._chunk_embeddings = emb_map
        for c in self._chunks:
            c.embedding = emb_map.get(c.id)
        self._index_built = True

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        await self._ensure_index_built()

        if not self._embedding_available:
            if self._fallback is None:
                self._fallback = KeywordRetriever(self._chunks, synonyms=self._synonyms)
            results = await self._fallback.retrieve(query, top_k)
            for r in results:
                r.retrieval_method = "semantic (fallback to keyword)"
            return results

        query_emb = await self._get_embedding(query)
        if query_emb is None:
            if self._fallback is None:
                self._fallback = KeywordRetriever(self._chunks, synonyms=self._synonyms)
            return await self._fallback.retrieve(query, top_k)

        scored = [
            (chunk, self._cosine_similarity(query_emb, emb))
            for chunk, emb in zip(
                self._chunks,
                [self._chunk_embeddings.get(c.id, []) for c in self._chunks],
                strict=False,
            )
            if emb
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(chunk=chunk, score=score, retrieval_method="semantic") for chunk, score in scored[:top_k]
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
        synonyms: dict[str, list[str]] | None = None,
    ):
        self._chunks = chunks
        self._rrf_k = rrf_k
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
        # Propagate the query-side synonym bridge to the keyword leg so that
        # vocabulary gaps (e.g. user "dog" vs document "pet") are closed even
        # when embeddings are unavailable and the semantic leg falls back.
        self._keyword = KeywordRetriever(chunks, synonyms=synonyms)
        self._semantic = SemanticRetriever(chunks, client=client, synonyms=synonyms)

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
