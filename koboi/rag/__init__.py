"""koboi/rag -- RAG (Retrieval-Augmented Generation) pipeline."""

from __future__ import annotations

from koboi.rag.types import Chunk, Document, RetrievalResult
from koboi.rag.chunker import (
    BaseChunker,
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    SemanticChunker,
    resolve_chunker,
)
from koboi.rag.retriever import (
    BaseRetriever,
    KeywordRetriever,
    SemanticRetriever,
    HybridRetriever,
    resolve_retriever,
)
from koboi.rag.augmentation import (
    AugmentationStrategy,
    InMemoryAugmentation,
    OnTheFlyAugmentation,
    RerankerRetriever,
)
from koboi.rag.registry import (
    ComponentRegistry,
    ComponentEntry,
    chunker_registry,
    retriever_registry,
    augmentation_registry,
    build_rag,
    register_chunker,
    register_retriever,
    register_augmentation,
    load_custom_components,
)

# Register built-in components (lazy, idempotent)
from koboi.rag.chunker import _register_builtins as _reg_chunkers
from koboi.rag.retriever import _register_builtins as _reg_retrievers
from koboi.rag.augmentation import _register_builtins as _reg_augmentations
from koboi.rag.parsers import _register_builtins as _reg_parsers

_reg_chunkers()
_reg_retrievers()
_reg_augmentations()
_reg_parsers()

__all__ = [
    # Types
    "Chunk",
    "Document",
    "RetrievalResult",
    # Chunkers
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "SemanticChunker",
    "resolve_chunker",
    # Retrievers
    "BaseRetriever",
    "KeywordRetriever",
    "SemanticRetriever",
    "HybridRetriever",
    "resolve_retriever",
    # Augmentations
    "AugmentationStrategy",
    "InMemoryAugmentation",
    "OnTheFlyAugmentation",
    "RerankerRetriever",
    # Registry
    "ComponentRegistry",
    "ComponentEntry",
    "chunker_registry",
    "retriever_registry",
    "augmentation_registry",
    "parser_registry",
    "build_rag",
    "register_chunker",
    "register_retriever",
    "register_augmentation",
    "register_parser",
    "load_custom_components",
]
