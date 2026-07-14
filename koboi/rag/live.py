"""koboi/rag/live.py -- live, mutable corpus + retriever (W3).

Lets an agent grow its knowledge mid-conversation: ``ingest_url`` appends chunks to a shared
``LiveCorpus``; the agent's ``LiveRetriever`` reads from the same corpus. ``add_chunks`` is a
cheap ``list.extend`` -- the ``KeywordRetriever`` delegate is rebuilt lazily on ``retrieve()``
only when the corpus is dirty (TF-IDF IDF is corpus-global, so an incremental add would need a
full re-index anyway; deferring it to the next retrieval keeps adds cheap).
"""

from __future__ import annotations

from koboi.rag.retriever import BaseRetriever, KeywordRetriever
from koboi.rag.types import Chunk, RetrievalResult


class LiveCorpus:
    """Mutable shared chunk store (a thin wrapper over ``list[Chunk]`` + a dirty flag)."""

    def __init__(self, seed: list[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = list(seed or [])
        # Seed chunks need an initial delegate build on first retrieve().
        self.dirty: bool = bool(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if chunks:
            self._chunks.extend(chunks)
            self.dirty = True

    def mark_clean(self) -> None:
        self.dirty = False

    @classmethod
    def from_corpus_file(cls, path: str) -> LiveCorpus | None:
        """Load a LiveCorpus from a research-findings jsonl (``SourceStore.to_corpus_file`` output).

        Each row is ``{citation_id, node_id, text}``. Returns None if the file is missing/empty so
        the caller can fall back to static chunks. Tolerant of malformed rows (skipped). This is the
        W5 convergence seam: a deep_research run's persisted findings seed a later session's corpus.
        """
        import json
        from pathlib import Path

        p = Path(path)
        if not p.is_file():
            return None
        chunks: list[Chunk] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            node_id = str(row.get("node_id", "research"))
            meta: dict = {"source": node_id}
            if "citation_id" in row:
                meta["citation_id"] = row["citation_id"]
            chunks.append(
                Chunk(
                    id=f"src_{row.get('citation_id', len(chunks))}",
                    doc_id=node_id,
                    content=text,
                    metadata=meta,
                )
            )
        return cls(chunks) if chunks else None


class LiveRetriever(BaseRetriever):
    """``KeywordRetriever``-backed retriever over a shared mutable ``LiveCorpus``.

    ``LiveCorpus.add_chunks`` is cheap (list.extend); the ``KeywordRetriever`` delegate is
    rebuilt lazily on ``retrieve()`` only when the corpus is dirty, then reused while clean.
    """

    def __init__(self, corpus: LiveCorpus) -> None:
        self._corpus = corpus
        # The same list object the corpus mutates in place (add_chunks uses list.extend), so
        # this reference always reflects the current chunks (satisfies BaseRetriever._chunks).
        self._chunks = corpus.chunks
        self._delegate: KeywordRetriever | None = None

    def _ensure_delegate(self) -> KeywordRetriever:
        if self._delegate is None or self._corpus.dirty:
            self._delegate = KeywordRetriever(self._corpus.chunks)
            self._corpus.mark_clean()
        return self._delegate

    async def retrieve(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        return await self._ensure_delegate().retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
