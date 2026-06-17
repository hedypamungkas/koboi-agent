"""koboi/rag/types -- Data types for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class Document:
    id: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    retrieval_method: str  # "keyword" | "semantic"
