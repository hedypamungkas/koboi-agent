"""koboi/rag/rewrite.py -- opt-in query rewriting / HyDE for better retrieval recall.

The raw user message is often verbose/conversational and under-retrieves. This module
produces a retrieval-optimized query before ``retrieve()`` runs:

- **rule-based** normalization (always on when rewriting is enabled): deterministic,
  zero-cost, zero-hallucination (strip stopwords/filler, collapse whitespace).
- **LLM rewrite** (``rag.query_rewrite``): an LLM call produces a concise search query.
- **HyDE** (``rag.hyde``): the LLM writes a short hypothetical answer; the retriever
  embeds *that* instead of the query (semantic/hybrid only).

The output is **ephemeral** -- used ONLY as the retrieval query, never stored in memory
or shown to the user. Any failure falls back to the rule-normalized query (or the raw
query if fallback is disabled). Requires a **chat-capable** client (NOT the embedding
client passed to ``build_rag``).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from koboi.llm.base import LLMClient

_logger = logging.getLogger(__name__)

# Conservative filler removal -- preserves all content terms. Matches the spirit of the
# query-side synonym expansion already in KeywordRetriever, generalized to every retriever.
_STOPWORDS = frozenset(
    "a an the of to in on at for and or is are was were be been being do does did "
    "i you he she it we they me him her us them my your his its our their this that "
    "these those with from by as can could would should will shall please hey um like".split()
)

QUERY_REWRITE_PROMPT = (
    "Rewrite the following user question into a concise, search-optimized query for a "
    "document retrieval system. Preserve the core intent and key entities (names, numbers, "
    "terms). Remove greetings, filler, and conversational language. Do NOT answer the "
    "question. Output ONLY the rewritten query.\n\nQuestion: {query}\n\nRewritten query:"
)

HYDE_PROMPT = (
    "Write a short (2-3 sentence) hypothetical answer to the following question, as if you "
    "had the relevant document in front of you. It will be used ONLY to find similar "
    "documents via embedding similarity -- do not hedge, just write a plausible, factual "
    "answer. Output ONLY the answer.\n\nQuestion: {query}\n\nAnswer:"
)

_MAX_QUERY_CHARS = 1000  # cap to bound the rewrite prompt (prompt-injection surface)


def rule_based_rewrite(query: str) -> str:
    """Deterministic normalization: lowercase, drop filler stopwords, collapse whitespace.

    Conservative: only removes obvious filler, keeps every content term. Always safe.
    """
    tokens = re.findall(r"\w+", query.lower())
    kept = [t for t in tokens if t not in _STOPWORDS]
    text = " ".join(kept) if kept else query.strip()
    return re.sub(r"\s+", " ", text).strip()


class QueryRewriter:
    """Opt-in query rewriting (rule-based + LLM) and HyDE, with caching + fallback."""

    def __init__(
        self,
        client: LLMClient | None = None,
        config: dict | None = None,
    ) -> None:
        self._client = client
        self._config = config or {}
        self._cache: dict[str, str] = {}
        self._max_cache = int(self._config.get("query_cache_size", 256))
        self._fallback = bool(self._config.get("fallback_to_raw", True))

    async def _llm_complete(self, prompt: str) -> str | None:
        if not self._client:
            return None
        try:
            resp = await self._client.complete([{"role": "user", "content": prompt}], tools=None)
        except Exception as exc:  # network / provider -> fallback path handles it
            _logger.warning("query rewrite LLM call failed: %s", exc)
            return None
        return (resp.content or "").strip() if resp and resp.content else None

    async def rewrite(self, query: str, *, mode: str) -> tuple[str, dict]:
        """Return ``(effective_query, meta)``.

        ``mode`` in ``{"rule", "llm", "hyde"}``. ``meta`` = ``{original, rewritten, method}``
        (surfaced to ``RunResult.metadata['rag_rewrite']`` for eval/observability).
        """
        normalized = rule_based_rewrite(query)
        if mode == "rule" or not self._client:
            return normalized or query, {"original": query, "rewritten": normalized, "method": "rule"}

        cached = self._cache.get(query)
        if cached is not None:
            return cached, {"original": query, "rewritten": cached, "method": "cache"}

        template = HYDE_PROMPT if mode == "hyde" else QUERY_REWRITE_PROMPT
        rewritten = await self._llm_complete(template.format(query=query[:_MAX_QUERY_CHARS]))
        if not rewritten:
            # LLM unavailable/failed -> fall back to the rule-normalized (or raw) query.
            effective = (normalized or query) if self._fallback else query
            return effective, {"original": query, "rewritten": effective, "method": "rule-fallback"}

        if len(self._cache) >= self._max_cache:
            self._cache.pop(next(iter(self._cache)), None)  # FIFO evict
        self._cache[query] = rewritten
        return rewritten, {"original": query, "rewritten": rewritten, "method": mode}
