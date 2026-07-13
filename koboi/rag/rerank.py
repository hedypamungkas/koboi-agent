"""koboi/rag/rerank.py -- Pluggable cross-encoder rerank stage for the RAG pipeline.

A true cross-encoder reranker (unlike the heuristic keyword-overlap ``RerankerRetriever`` in
``augmentation.py``). Three backends share one ``RerankBackend`` ABC:

* ``jina``   -- Jina Reranker API (default; per-token billing, large doc capacity).
* ``cohere`` -- Cohere v2 Rerank API (per-call; default model is multilingual -- override ``model:``
  with ``rerank-english-v3.0`` for an English-specialized edge).
* ``local``  -- BGE cross-encoder via sentence-transformers (no egress; the ``[rerank-local]``
  extra, mirrors the ``[tokenizer]``/tiktoken gate).

This is a PRODUCTION pipeline stage -- when enabled it runs on every retrieval. It mirrors the
LLM stack's transport/auth/adapter split (reuses ``HttpTransport`` + ``BearerAuth`` + the
``LLMError`` hierarchy) but co-locates the whole small concern in one module. Fail-soft like
``OpenAIAdapter.get_embeddings``: on any provider hiccup the wrapper returns the base retriever's
results unchanged so retrieval never breaks.

Enabled via ``rag.rerank`` as a DICT (the legacy ``rerank: true`` bool still selects the
heuristic). ``build_rag`` wires it (``registry.py``); the wrapper stamps a distinctive
``retrieval_method`` (e.g. ``rerank:jina(bm25)``) that flows to ``RunResult.metadata['rag_results']``
so evals can detect the rerank provider.
"""

from __future__ import annotations

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from typing import Any

from koboi.llm.auth import BearerAuth
from koboi.llm.base import LLMInvalidRequestError
from koboi.llm.http_transport import HttpTransport
from koboi.rag.retriever import BaseRetriever
from koboi.rag.types import RetrievalResult

_logger = logging.getLogger(__name__)

# Per-provider document-per-call caps. ``CrossEncoderReranker`` clamps its over-fetch to the
# active provider's cap so we never exceed a batch limit (v1: no multi-call batching).
_PROVIDER_MAX_BATCH: dict[str, int] = {"jina": 2048, "cohere": 100, "local": 10_000}


def _clamp01(value: float) -> float:
    """Clamp a score to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


class RerankBackend(ABC):
    """Scores ``(query, document)`` pairs via a cross-encoder.

    Implementations MUST return ``None`` on any failure (network, auth, parse) so the wrapper
    can fall back to the base retriever's results -- retrieval never breaks on a rerank hiccup.
    """

    #: Short provider label, surfaced in ``retrieval_method`` (e.g. ``rerank:jina(...)``).
    provider: str = "cross"

    @abstractmethod
    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
        """Return ``[(original_index, score in [0,1]), ...]`` sorted desc (len <= top_n),
        or ``None`` on failure."""
        ...

    async def close(self) -> None:  # noqa: B027 - intentional optional override (HTTP backends close; local is a no-op)
        """Release HTTP transports / models. Default no-op; HTTP backends override."""
        ...


class JinaRerankBackend(RerankBackend):
    """Jina Reranker API: ``POST /rerank`` -> ``{results:[{index, relevance_score, document}]}``."""

    provider = "jina"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "jina-reranker-v3",
        base_url: str = "https://api.jina.ai/v1",
        timeout: float = 30.0,
    ):
        self._model = model
        self._transport = HttpTransport(base_url, BearerAuth(api_key), timeout=timeout)

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
        try:
            data = await self._transport.post(
                "/rerank",
                {"model": self._model, "query": query, "documents": documents, "top_n": top_n},
            )
            return _parse_rerank_results(data)
        except Exception as e:  # noqa: BLE001 - fail-soft, mirror get_embeddings
            _logger.warning("Jina rerank failed: %s", e)
            return None

    async def close(self) -> None:
        await self._transport.close()


class CohereRerankBackend(RerankBackend):
    """Cohere v2 Rerank: ``POST /rerank`` -> ``{results:[{index, relevance_score}]}`` (no doc echo)."""

    provider = "cohere"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-multilingual-v3.0",
        base_url: str = "https://api.cohere.com/v2",
        timeout: float = 30.0,
    ):
        self._model = model
        self._transport = HttpTransport(base_url, BearerAuth(api_key), timeout=timeout)

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
        try:
            data = await self._transport.post(
                "/rerank",
                {"model": self._model, "query": query, "documents": documents, "top_n": top_n},
            )
            return _parse_rerank_results(data)
        except Exception as e:  # noqa: BLE001 - fail-soft, mirror get_embeddings
            _logger.warning("Cohere rerank failed: %s", e)
            return None

    async def close(self) -> None:
        await self._transport.close()


class LocalBGERerankBackend(RerankBackend):
    """Local BGE cross-encoder via sentence-transformers (logits -> sigmoid).

    Heavy: pulls torch. Import-gated behind the ``[rerank-local]`` extra (mirrors the
    ``[tokenizer]``/tiktoken pattern) so the default install stays lean.
    """

    provider = "local"

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        try:
            from sentence_transformers import CrossEncoder  # import-gated (the [rerank-local] extra)
        except ImportError as e:
            raise LLMInvalidRequestError(
                "rag.rerank provider 'local' requires the [rerank-local] extra: pip install 'koboi-agent[rerank-local]'"
            ) from e
        self._model_name = model
        self._model = CrossEncoder(model)  # sync; rerank() runs predict() off-loop

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
        try:
            pairs = [[query, d] for d in documents]
            # predict() is CPU-bound -> run off the event loop.
            scores = await asyncio.to_thread(self._model.predict, pairs)
            ranked = sorted(enumerate(list(scores)), key=lambda x: x[1], reverse=True)[:top_n]
            return [(i, _sigmoid(float(s))) for i, s in ranked]
        except Exception as e:  # noqa: BLE001 - fail-soft, mirror get_embeddings
            _logger.warning("Local BGE rerank failed: %s", e)
            return None


def build_rerank_client(rerank_config: dict | None) -> RerankBackend | None:
    """Build a rerank backend from the ``rag.rerank`` dict.

    Mirrors ``build_embedding_client``'s None-on-unconfigured contract: for HTTP providers
    (jina/cohere) returns ``None`` when no ``api_key`` is set so the caller uses the base retriever
    **unwrapped (no reranking)** with a warning -- it does NOT fall back to the heuristic
    ``RerankerRetriever`` (that path is only reached via the legacy ``rerank: true`` bool config).
    Unknown providers raise ``LLMInvalidRequestError`` at build time (fail-fast, like
    ``create_client``). ``provider`` defaults to ``jina``.
    """
    cfg = rerank_config or {}
    if not isinstance(cfg, dict):
        return None
    provider = str(cfg.get("provider") or "jina").lower()
    timeout = float(cfg.get("timeout", 30.0))

    if provider == "local":
        return LocalBGERerankBackend(model=cfg.get("model") or "BAAI/bge-reranker-v2-m3")

    api_key = cfg.get("api_key") or ""
    if not api_key:
        _logger.warning(
            "rerank provider %r has no api_key; cross-encoder rerank disabled (falling back).",
            provider,
        )
        return None

    if provider == "jina":
        return JinaRerankBackend(
            api_key=api_key,
            model=cfg.get("model") or "jina-reranker-v3",
            base_url=cfg.get("base_url") or "https://api.jina.ai/v1",
            timeout=timeout,
        )
    if provider == "cohere":
        return CohereRerankBackend(
            api_key=api_key,
            model=cfg.get("model") or "rerank-multilingual-v3.0",
            base_url=cfg.get("base_url") or "https://api.cohere.com/v2",
            timeout=timeout,
        )

    raise LLMInvalidRequestError(f"Unknown rerank provider: {provider!r}. Available: jina, cohere, local.")


class CrossEncoderReranker(BaseRetriever):
    """Wraps a base retriever, over-fetches, and re-scores via a cross-encoder backend.

    Same wrapper shape as the heuristic ``RerankerRetriever``: delegates the over-fetch to
    ``self._base.retrieve(...)`` (which carries ``_chunks``) so metadata filtering stays
    base-retriever-owned; does NOT set ``self._chunks``. Fail-soft: if the backend returns
    ``None``/empty, returns the base results (original order, truncated to ``top_k``) so
    retrieval never breaks. The degradation is **always observable** -- results are stamped
    ``retrieval_method="rerank:failed(<provider>,<base>)"`` (surfaced in
    ``RunResult.metadata['rag_results']``) plus a warning log, so a persistent outage
    (bad key / quota / bad model) never silently reads as healthy ``bm25``.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        backend: RerankBackend,
        fetch_multiplier: int = 3,
        score_threshold: float | None = None,
    ):
        self._base = base_retriever
        self._backend = backend
        self._fetch_multiplier = fetch_multiplier
        self._score_threshold = score_threshold
        self._provider = backend.provider

    async def retrieve(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        cap = _PROVIDER_MAX_BATCH.get(self._provider, 2048)
        fetch_k = min(max(top_k * self._fetch_multiplier, top_k + 5), cap)
        results = await self._base.retrieve(query, top_k=fetch_k, metadata_filter=metadata_filter)

        if not results or len(results) <= top_k:
            return results

        documents = [r.chunk.content for r in results]
        ranked = await self._backend.rerank(query, documents, top_n=top_k)
        if not ranked:  # backend failed/empty -> fail-soft to base order (ALWAYS observable)
            base_method = results[0].retrieval_method
            _logger.warning(
                "Rerank backend %s returned no usable results; returning base results (fail-soft).",
                self._provider,
            )
            method = f"rerank:failed({self._provider},{base_method})"
            return [RetrievalResult(chunk=r.chunk, score=r.score, retrieval_method=method) for r in results[:top_k]]

        out: list[RetrievalResult] = []
        for idx, score in ranked:
            if idx < 0 or idx >= len(results):
                continue  # defensive: provider returned a stale/out-of-range index
            if self._score_threshold is not None and score < self._score_threshold:
                continue
            base = results[idx]
            out.append(
                RetrievalResult(
                    chunk=base.chunk,
                    score=score,
                    retrieval_method=f"rerank:{self._provider}({base.retrieval_method})",
                )
            )
        # If everything was filtered out by score_threshold, fall back to base order.
        return out[:top_k] if out else results[:top_k]

    async def close(self) -> None:
        """Close the backend's HTTP transport(s). Called by ``KoboiAgent.close()``."""
        await self._backend.close()


def _int(value: Any) -> int:
    """Coerce a JSON index to int (providers may return it as a numpy/float-ish value)."""
    return int(value)


def _parse_rerank_results(data: dict) -> list[tuple[int, float]]:
    """Parse a Jina/Cohere rerank response into ``[(index, score in [0,1])]``.

    Tolerates a single malformed row (missing/non-numeric ``index`` or ``relevance_score``)
    by skipping it rather than discarding the whole batch -- a provider returning 99 good
    scores + 1 bad row keeps the 99. Both backends share this parser.
    """
    out: list[tuple[int, float]] = []
    for r in data.get("results", []):
        try:
            out.append((_int(r["index"]), _clamp01(r.get("relevance_score", 0.0))))
        except (KeyError, TypeError, ValueError):
            continue  # skip one malformed row; keep the rest
    return out


def _sigmoid(x: float) -> float:
    """Logits -> [0,1] probability, clamped."""
    if x >= 0:
        z = math.exp(-x)
        return _clamp01(1.0 / (1.0 + z))
    z = math.exp(x)
    return _clamp01(z / (1.0 + z))
