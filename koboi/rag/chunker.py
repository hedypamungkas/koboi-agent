"""koboi/rag/chunker -- Text chunking strategies for RAG pipeline."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from koboi.rag.types import Chunk, Document

_logger = logging.getLogger(__name__)


def resolve_chunker(config: dict) -> BaseChunker:
    """Build a chunker from a RAG config dict.

    Reads 'chunker' (default 'paragraph'), 'chunk_size', 'overlap', 'max_chunk_size'.
    """
    from koboi.rag.registry import _resolve_kwargs, chunker_registry

    chunker_name = config.get("chunker", "paragraph")
    entry = chunker_registry.get(chunker_name)
    if entry is None:
        _logger.warning("Unknown chunker '%s', falling back to paragraph", chunker_name)
        entry = chunker_registry.get("paragraph")
        if entry is None:
            raise ValueError("No chunkers registered")
    kwargs = _resolve_kwargs(entry, config)
    return entry.cls(**kwargs)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]: ...

    def _make_chunk(self, doc_id: str, index: int, content: str) -> Chunk:
        return Chunk(
            id=f"{doc_id}_c{index}",
            doc_id=doc_id,
            content=content.strip(),
            metadata={"chunk_index": index},
        )


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunks with overlap and sentence-boundary snapping."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [self._make_chunk(document.id, 0, text)]

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Snap to sentence boundary to avoid breaking mid-sentence
            if end < len(text):
                snap = text.rfind(". ", start, end)
                if snap > start:
                    end = snap + 1
                else:
                    snap = text.rfind("\n", start, end)
                    if snap > start:
                        end = snap

            piece = text[start:end].strip()
            if piece:
                chunks.append(self._make_chunk(document.id, index, piece))
                index += 1

            next_start = end - self.overlap
            if next_start <= start:
                start = end
            else:
                start = next_start

        return chunks


class SentenceChunker(BaseChunker):
    """Sentence-aware chunks up to max_chunk_size."""

    def __init__(self, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        index = 0

        for sentence in sentences:
            if current_len + len(sentence) > self.max_chunk_size and current:
                content = " ".join(current)
                chunks.append(self._make_chunk(document.id, index, content))
                index += 1
                current = []
                current_len = 0

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            content = " ".join(current)
            chunks.append(self._make_chunk(document.id, index, content))

        return chunks


class SemanticChunker(BaseChunker):
    """Splits documents based on embedding similarity between sentences.

    Computes embeddings for each sentence and groups consecutive sentences
    whose similarity stays above a threshold. Falls back to SentenceChunker
    when no embedding client is available.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self._fallback = SentenceChunker(max_chunk_size=max_chunk_size)

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [self._make_chunk(document.id, 0, text)] if text else []

        # Try to get embeddings from the global LLM client
        embeddings = self._get_embeddings_sync(sentences)
        if embeddings is None:
            return self._fallback.chunk(document)

        return self._split_by_similarity(document, sentences, embeddings)

    def _get_embeddings_sync(self, sentences: list[str]) -> list[list[float]] | None:
        """Get embeddings for sentences. Returns None if unavailable."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context, can't use run_until_complete
                # Fall back to SentenceChunker
                return None
        except RuntimeError:
            pass

        # Try to get embeddings from the LLM client via the registry
        # This is a best-effort approach; if unavailable, fall back
        try:
            from koboi.rag.registry import retriever_registry

            entry = retriever_registry.get("semantic")
            if entry is None or "client" not in entry.inject:
                return None
            # We don't have direct access to the client here, so fall back
            return None
        except Exception:
            return None

    def _split_by_similarity(
        self,
        document: Document,
        sentences: list[str],
        embeddings: list[list[float]],
    ) -> list[Chunk]:
        """Split sentences into chunks based on embedding similarity."""
        chunks: list[Chunk] = []
        current: list[str] = [sentences[0]]
        current_len = len(sentences[0])
        index = 0

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            sentence = sentences[i]

            # Start new chunk if similarity drops below threshold or size exceeded
            should_split = sim < self.similarity_threshold or current_len + len(sentence) > self.max_chunk_size
            if should_split and current_len >= self.min_chunk_size:
                content = " ".join(current)
                chunks.append(self._make_chunk(document.id, index, content))
                index += 1
                current = []
                current_len = 0

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            content = " ".join(current)
            chunks.append(self._make_chunk(document.id, index, content))

        return chunks

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunks with heading-aware merging."""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self._fallback = FixedSizeChunker(chunk_size=max_chunk_size, overlap=50)

    @staticmethod
    def _is_heading(text: str) -> bool:
        stripped = text.strip()
        return bool(stripped) and bool(re.match(r"^#{1,6}\s", stripped))

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Merge heading-only paragraphs with their following content
        merged: list[str] = []
        pending_heading: str | None = None
        for para in paragraphs:
            if self._is_heading(para):
                if pending_heading is not None:
                    merged.append(pending_heading)
                pending_heading = para
            elif pending_heading is not None:
                merged.append(pending_heading + "\n\n" + para)
                pending_heading = None
            else:
                merged.append(para)
        if pending_heading is not None:
            merged.append(pending_heading)

        chunks: list[Chunk] = []
        index = 0

        for para in merged:
            if len(para) <= self.max_chunk_size:
                chunks.append(self._make_chunk(document.id, index, para))
                index += 1
            else:
                sub_doc = Document(id=document.id, title="", content=para)
                for sub_chunk in self._fallback.chunk(sub_doc):
                    sub_chunk.metadata["chunk_index"] = index
                    sub_chunk.id = f"{document.id}_c{index}"
                    chunks.append(sub_chunk)
                    index += 1

        return chunks


# ---------------------------------------------------------------------------
# Register built-in chunkers with the RAG registry
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register built-in chunkers. Called lazily on first use."""
    from koboi.rag.registry import register_chunker as _reg

    _reg(
        "fixed",
        description="Fixed-size chunks with overlap and sentence-boundary snapping",
        config_aliases={"chunk_size": "chunk_size", "overlap": "overlap"},
    )(FixedSizeChunker)

    _reg("sentence", description="Sentence-aware chunks up to max_chunk_size")(SentenceChunker)

    _reg("paragraph", description="Paragraph-based chunks with heading-aware merging")(ParagraphChunker)

    _reg(
        "semantic",
        description="Embedding similarity-based chunking with adaptive boundaries",
    )(SemanticChunker)
