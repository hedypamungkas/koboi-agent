from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from koboi.rag.types import RetrievalResult
from koboi.tokens import estimate_single

if TYPE_CHECKING:
    from koboi.logger import AgentLogger
    from koboi.rag.retriever import BaseRetriever


class AugmentationStrategy(ABC):
    def __init__(
        self,
        retriever: BaseRetriever,
        top_k: int = 3,
        logger: AgentLogger | None = None,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.logger = logger

    async def _retrieve_and_format(self, query: str) -> tuple[str, list[RetrievalResult]]:
        results = await self.retriever.retrieve(query, top_k=self.top_k)

        if not results:
            return "", results

        context_parts = []
        for r in results:
            source = r.chunk.metadata.get("source", r.chunk.doc_id)
            context_parts.append(f"[Source: {source}]\n{r.chunk.content}")
        context = "\n---\n".join(context_parts)

        if self.logger:
            method = results[0].retrieval_method if results else "none"
            self.logger.log_rag_retrieval(query, results, method)

        return context, results

    @staticmethod
    def _build_augmented_message(context: str, user_message: str) -> str:
        return f"Document context:\n---\n{context}\n---\n\nQuestion: {user_message}"

    async def augment_for_memory(self, user_message: str) -> str:
        return user_message

    async def augment_for_llm(self, messages: list[dict]) -> list[dict]:
        return messages


class InMemoryAugmentation(AugmentationStrategy):
    async def augment_for_memory(self, user_message: str) -> str:
        context, results = await self._retrieve_and_format(user_message)

        if not context:
            return user_message

        augmented = self._build_augmented_message(context, user_message)

        if self.logger:
            original_tokens = estimate_single({"role": "user", "content": user_message})
            augmented_tokens = estimate_single({"role": "user", "content": augmented})
            self.logger.log_rag_augmentation(
                strategy="IN_MEMORY",
                original=user_message,
                augmented=augmented,
                delta=augmented_tokens - original_tokens,
            )

        return augmented


class OnTheFlyAugmentation(AugmentationStrategy):
    def __init__(
        self,
        retriever: BaseRetriever,
        top_k: int = 3,
        logger: AgentLogger | None = None,
    ):
        super().__init__(retriever=retriever, top_k=top_k, logger=logger)
        self._cache: dict[str, str] = {}

    async def augment_for_llm(self, messages: list[dict]) -> list[dict]:
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return messages

        user_content = messages[last_user_idx].get("content", "")

        if user_content in self._cache:
            context = self._cache[user_content]
        else:
            context, _ = await self._retrieve_and_format(user_content)
            self._cache[user_content] = context

        if not context:
            return messages

        augmented = self._build_augmented_message(context, user_content)

        if self.logger:
            original_tokens = estimate_single({"role": "user", "content": user_content})
            augmented_tokens = estimate_single({"role": "user", "content": augmented})
            self.logger.log_rag_augmentation(
                strategy="ON_THE_FLY",
                original=user_content,
                augmented=augmented,
                delta=augmented_tokens - original_tokens,
            )

        result = [m.copy() for m in messages]
        result[last_user_idx] = {
            "role": "user",
            "content": augmented,
        }
        return result


# ---------------------------------------------------------------------------
# Reranker augmentation -- re-scores retrieved results for higher precision
# ---------------------------------------------------------------------------


class RerankerRetriever:
    """Wraps a retriever and re-scores results using keyword overlap scoring.

    This is a lightweight cross-encoder style reranker that doesn't require
    an external model. It scores chunks by:
    1. Query term coverage (what fraction of query terms appear in the chunk)
    2. Term frequency of query terms in the chunk
    3. Chunk length penalty (prefer concise, focused chunks)
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        fetch_multiplier: int = 3,
        length_penalty: float = 0.1,
    ):
        self._base = base_retriever
        self._fetch_multiplier = fetch_multiplier
        self._length_penalty = length_penalty

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        # Fetch more results from base retriever for reranking
        fetch_k = max(top_k * self._fetch_multiplier, top_k + 5)
        results = await self._base.retrieve(query, top_k=fetch_k)

        if not results or len(results) <= top_k:
            return results

        # Re-score using keyword overlap
        query_terms = set(query.lower().split())
        if not query_terms:
            return results[:top_k]

        rescored: list[tuple[RetrievalResult, float]] = []
        for r in results:
            content_lower = r.chunk.content.lower()
            content_words = content_lower.split()
            content_terms = set(content_words)

            # Term coverage: fraction of query terms found in chunk
            coverage = len(query_terms & content_terms) / len(query_terms)

            # Term frequency: how often query terms appear (word-level)
            tf = sum(content_words.count(t) for t in query_terms)
            tf_score = tf / (len(content_words) + 1)

            # Length penalty: prefer shorter, focused chunks
            length_score = 1.0 / (1.0 + self._length_penalty * len(r.chunk.content) / 100)

            # Combined score
            rerank_score = coverage * 0.5 + tf_score * 0.3 + length_score * 0.2

            rescored.append((r, rerank_score))

        rescored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                chunk=r.chunk,
                score=score,
                retrieval_method=f"reranked({r.retrieval_method})",
            )
            for r, score in rescored[:top_k]
        ]


# ---------------------------------------------------------------------------
# Register built-in augmentation strategies with the RAG registry
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register built-in augmentation strategies. Called lazily on first use."""
    from koboi.rag.registry import register_augmentation as _reg

    _reg("in_memory", description="Augments user message with retrieved context before storing")(InMemoryAugmentation)

    _reg("on_the_fly", description="Augments last user message in-place before each LLM call")(OnTheFlyAugmentation)
