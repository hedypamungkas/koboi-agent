"""koboi/rag/registry.py -- Decorator-based RAG component registry.

Provides @register_chunker, @register_retriever, @register_augmentation
decorators and a build_rag() orchestrator that composes them from config.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from koboi.llm.base import LLMClient
    from koboi.logger import AgentLogger
    from koboi.rag.augmentation import AugmentationStrategy
    from koboi.rag.chunker import BaseChunker
    from koboi.rag.retriever import BaseRetriever
    from koboi.rag.types import Chunk

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic component registry
# ---------------------------------------------------------------------------


class ComponentEntry:
    """Metadata for a registered RAG component."""

    __slots__ = ("cls", "parameters", "description", "config_aliases", "inject")

    def __init__(
        self,
        cls: type,
        parameters: dict[str, dict[str, Any]],
        description: str = "",
        config_aliases: dict[str, str] | None = None,
        inject: list[str] | None = None,
    ):
        self.cls = cls
        self.parameters = parameters
        self.description = description
        self.config_aliases = config_aliases or {}
        self.inject = inject or []


class ComponentRegistry:
    """Generic registry for RAG components (chunkers, retrievers, augmentations)."""

    def __init__(self, component_type: str) -> None:
        self._component_type = component_type
        self._entries: dict[str, ComponentEntry] = {}

    def register(
        self,
        name: str,
        cls: type,
        *,
        description: str = "",
        config_aliases: dict[str, str] | None = None,
        inject: list[str] | None = None,
    ) -> None:
        params = _extract_parameters(cls)
        if config_aliases:
            valid_params = set(params.keys())
            for yaml_key, param_name in config_aliases.items():
                if param_name not in valid_params:
                    raise ValueError(
                        f"config_aliases maps '{yaml_key}' to '{param_name}', "
                        f"but {cls.__name__}.__init__ has no such parameter. "
                        f"Available: {valid_params}"
                    )
        self._entries[name] = ComponentEntry(
            cls=cls,
            parameters=params,
            description=description,
            config_aliases=config_aliases,
            inject=inject,
        )

    def get(self, name: str) -> ComponentEntry | None:
        return self._entries.get(name)

    def list_available(self) -> list[str]:
        return sorted(self._entries.keys())

    def clear(self) -> None:
        self._entries.clear()


def _extract_parameters(cls: type) -> dict[str, dict[str, Any]]:
    """Extract constructor parameters via introspection.

    Returns dict mapping param_name -> {"default": ..., "annotation": ...}.
    Skips 'self' and *args/**kwargs.
    """
    sig = inspect.signature(cls.__init__)  # type: ignore[misc]  # __init__ params drive config->kwargs resolution
    params: dict[str, dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        entry: dict[str, Any] = {}
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        if param.annotation is not inspect.Parameter.empty:
            entry["annotation"] = param.annotation
        params[name] = entry
    return params


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

chunker_registry = ComponentRegistry("chunker")
retriever_registry = ComponentRegistry("retriever")
augmentation_registry = ComponentRegistry("augmentation")


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def register_chunker(
    name: str,
    description: str = "",
    *,
    config_aliases: dict[str, str] | None = None,
):
    """Decorator to register a chunker class.

    Usage::

        @register_chunker("paragraph", description="Splits on double newlines")
        class MyChunker(BaseChunker):
            ...
    """

    def decorator(cls: type) -> type:
        chunker_registry.register(
            name,
            cls,
            description=description,
            config_aliases=config_aliases,
        )
        return cls

    return decorator


def register_retriever(
    name: str,
    description: str = "",
    *,
    inject: list[str] | None = None,
):
    """Decorator to register a retriever class.

    Usage::

        @register_retriever("semantic", inject=["client"])
        class SemanticRetriever(BaseRetriever):
            ...
    """

    def decorator(cls: type) -> type:
        retriever_registry.register(
            name,
            cls,
            description=description,
            inject=inject,
        )
        return cls

    return decorator


def register_augmentation(
    name: str,
    description: str = "",
):
    """Decorator to register an augmentation strategy class.

    Usage::

        @register_augmentation("on_the_fly")
        class OnTheFlyAugmentation(AugmentationStrategy):
            ...
    """

    def decorator(cls: type) -> type:
        augmentation_registry.register(name, cls, description=description)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------


def _build_chunker(rag_conf: dict[str, Any]) -> BaseChunker:
    """Build a chunker from RAG config using the chunker registry."""
    chunker_name = rag_conf.get("chunker", "paragraph")
    entry = chunker_registry.get(chunker_name)
    if entry is None:
        _logger.warning(
            "Unknown chunker '%s', falling back to 'paragraph'. Available: %s",
            chunker_name,
            chunker_registry.list_available(),
        )
        entry = chunker_registry.get("paragraph")
        if entry is None:
            raise ValueError("No chunkers registered")

    kwargs = _resolve_kwargs(entry, rag_conf)
    return entry.cls(**kwargs)


def _build_retriever(
    chunks: list[Chunk],
    rag_conf: dict[str, Any],
    client: LLMClient | None = None,
) -> BaseRetriever:
    """Build a retriever from config, injecting dependencies as needed."""
    retriever_name = rag_conf.get("retriever", "keyword")
    entry = retriever_registry.get(retriever_name)
    if entry is None:
        _logger.warning(
            "Unknown retriever '%s', falling back to 'keyword'. Available: %s",
            retriever_name,
            retriever_registry.list_available(),
        )
        entry = retriever_registry.get("keyword")
        if entry is None:
            raise ValueError("No retrievers registered")

    kwargs = _resolve_kwargs(entry, rag_conf)
    kwargs["chunks"] = chunks
    if "client" in entry.inject:
        kwargs["client"] = client

    return entry.cls(**kwargs)


def _resolve_kwargs(
    entry: ComponentEntry,
    rag_conf: dict[str, Any],
) -> dict[str, Any]:
    """Resolve constructor kwargs from config using entry metadata."""
    config_aliases = entry.config_aliases
    kwargs: dict[str, Any] = {}
    for param_name in entry.parameters:
        yaml_key = param_name
        for yk, pn in config_aliases.items():
            if pn == param_name:
                yaml_key = yk
                break
        if yaml_key in rag_conf:
            kwargs[param_name] = rag_conf[yaml_key]
        elif "default" in entry.parameters[param_name]:
            kwargs[param_name] = entry.parameters[param_name]["default"]
    return kwargs


def _load_documents(
    rag_conf: dict[str, Any],
) -> tuple[BaseChunker, list[Chunk]]:
    """Load and chunk documents from RAG config.

    Returns (chunker, chunks) so callers can reuse the chunker if needed.
    """
    from koboi.rag.types import Document

    chunker = _build_chunker(rag_conf)
    doc_paths = rag_conf.get("documents", [])
    all_chunks: list[Chunk] = []

    for doc_conf in doc_paths:
        path = doc_conf.get("path", "")
        if path:
            from pathlib import Path as PathlibPath

            p = PathlibPath(path)
            if p.exists():
                content = p.read_text()
                doc = Document(id=p.stem, title=p.stem, content=content)
                chunks = chunker.chunk(doc)
                for chunk in chunks:
                    chunk.metadata["source"] = doc.title
                all_chunks.extend(chunks)

    return chunker, all_chunks


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def build_rag(
    rag_conf: dict[str, Any],
    *,
    client: LLMClient | None = None,
    logger: AgentLogger | None = None,
) -> AugmentationStrategy | None:
    """Build a complete RAG augmentation pipeline from config.

    Composes chunker, retriever, and augmentation strategy from their
    respective registries based on the config dict.

    Args:
        rag_conf: RAG config dict (from YAML ``rag:`` section).
        client: LLM client for embedding-based retrievers.
        logger: Optional logger for RAG events.

    Returns:
        Configured AugmentationStrategy, or None if RAG is disabled / no docs.
    """
    if not rag_conf or not rag_conf.get("enabled"):
        return None

    _, all_chunks = _load_documents(rag_conf)
    if not all_chunks:
        _logger.warning("RAG enabled but no documents loaded")
        return None

    retriever = _build_retriever(all_chunks, rag_conf, client=client)

    aug_name = rag_conf.get("augmentation", "in_memory")
    entry = augmentation_registry.get(aug_name)
    if entry is None:
        _logger.warning(
            "Unknown augmentation '%s', falling back to 'in_memory'. Available: %s",
            aug_name,
            augmentation_registry.list_available(),
        )
        entry = augmentation_registry.get("in_memory")
        if entry is None:
            raise ValueError("No augmentation strategies registered")

    kwargs: dict[str, Any] = {
        "retriever": retriever,
        "top_k": rag_conf.get("top_k", 3),
    }
    if logger is not None:
        kwargs["logger"] = logger

    for param_name in entry.parameters:
        if param_name not in kwargs and param_name in rag_conf:
            kwargs[param_name] = rag_conf[param_name]

    return entry.cls(**kwargs)


# ---------------------------------------------------------------------------
# Custom module loading (YAML-driven extensibility)
# ---------------------------------------------------------------------------


def load_custom_components(custom_modules: list[str]) -> None:
    """Import modules to trigger @register_* decorators.

    YAML config example::

        rag:
          custom_modules:
            - my_package.rag.chunkers
            - my_package.rag.retrievers
    """
    for module_path in custom_modules:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            _logger.warning(
                "Failed to import custom RAG module '%s': %s",
                module_path,
                e,
            )
