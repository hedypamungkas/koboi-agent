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
parser_registry = ComponentRegistry("parser")


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


def register_parser(
    name: str,
    description: str = "",
):
    """Decorator to register a document-format parser.

    Usage::

        @register_parser("csv", description="CSV via pandas")
        class CsvParser(BaseParser):
            ...
    """

    def decorator(cls: type) -> type:
        parser_registry.register(name, cls, description=description)
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
    """Load, parse, and chunk documents from RAG config.

    Returns ``(chunker, chunks)``. Each ``documents[]`` entry selects a source:

    - ``{path: "..."}`` or a bare string -- local file / glob (``*.md``) / directory
      (recursed). Backward compatible with pre-existing configs.
    - ``{source: http, url: "..."}`` -- fetch over HTTP(S) via httpx (presigned URLs
      work for R2/S3 public-ish objects). Zero new dependency.
    - ``{source: s3, bucket: "...", key: "prefix/", endpoint_url: "${R2_ENDPOINT}",
      region: "auto"}`` -- S3-compatible (Cloudflare R2) via boto3 (``[rag-cloud]``).

    Fetched/loaded bytes are parsed by format (text/html/pdf/docx) via the parser
    registry; unreadable/binary files are skipped. ``document_cache_path`` caches
    remote fetches across the per-session rebuilds in ``koboi/server/pool.py``.
    """
    import glob as _glob
    from pathlib import Path as PathlibPath

    from koboi.rag.parsers import dispatch_parser
    from koboi.rag.sources import DocumentCache, fetch_http_entry, fetch_s3_entry
    from koboi.rag.types import Document

    chunker = _build_chunker(rag_conf)
    doc_cache_path = rag_conf.get("document_cache_path")
    doc_cache = DocumentCache(doc_cache_path) if doc_cache_path else None

    def _resolve_files(path: str) -> list[PathlibPath]:
        # #3: expand glob patterns and directories into a concrete file list.
        if any(ch in path for ch in "*?["):
            return sorted(PathlibPath(p) for p in _glob.glob(path, recursive=True) if PathlibPath(p).is_file())
        p = PathlibPath(path)
        if p.is_dir():
            return sorted(f for f in p.rglob("*") if f.is_file())
        return [p] if p.is_file() else []

    def _resolve_entry(entry: Any):
        # Yield (name, bytes) for one documents[] entry from any source.
        if isinstance(entry, str):
            for fp in _resolve_files(entry):
                try:
                    yield fp.name, fp.read_bytes()
                except OSError:
                    continue
            return
        if not isinstance(entry, dict):
            return
        source = (entry.get("source") or "file").lower()
        if source in ("file", "local"):
            path = entry.get("path", "")
            if path:
                for fp in _resolve_files(path):
                    try:
                        yield fp.name, fp.read_bytes()
                    except OSError:
                        continue
            return
        if source == "http" or "url" in entry:
            yield from fetch_http_entry(entry, doc_cache)
            return
        if source == "s3":
            yield from fetch_s3_entry(entry, doc_cache)
            return
        _logger.warning("Unknown document source %r; skipping", source)

    max_mb = int(rag_conf.get("max_document_size_mb", 10))
    max_bytes = max_mb * 1024 * 1024
    all_chunks: list[Chunk] = []
    for entry in rag_conf.get("documents", []):
        fmt_hint = entry.get("format") if isinstance(entry, dict) else None
        for name, data in _resolve_entry(entry):
            if len(data) > max_bytes:
                _logger.warning("Skipping %s: %d bytes exceeds max_document_size_mb=%d", name, len(data), max_mb)
                continue
            text, meta = dispatch_parser(name, data, format_hint=fmt_hint)
            if not text or not text.strip():
                # binary / unreadable / empty -> skip (never ingest mojibake)
                continue
            stem = PathlibPath(name).stem or name
            doc = Document(id=stem, title=stem, content=text)
            for chunk in chunker.chunk(doc):
                chunk.metadata["source"] = stem
                if meta.get("source_format"):
                    chunk.metadata["source_format"] = meta["source_format"]
                all_chunks.append(chunk)

    return chunker, all_chunks


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def build_rag(
    rag_conf: dict[str, Any],
    *,
    client: LLMClient | None = None,
    logger: AgentLogger | None = None,
    chat_client: LLMClient | None = None,
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

    # #5: opt-in on-disk embedding cache -> avoid re-embedding the corpus on restart.
    cache_path = rag_conf.get("embedding_cache_path")
    if cache_path:
        from koboi.rag.retriever import set_embedding_cache_path

        set_embedding_cache_path(cache_path)

    _, all_chunks = _load_documents(rag_conf)
    if not all_chunks:
        _logger.warning("RAG enabled but no documents loaded")
        return None

    retriever = _build_retriever(all_chunks, rag_conf, client=client)

    # #11a: opt-in rerank stage. A DICT selects a true cross-encoder backend
    # (jina/cohere/local -- see koboi/rag/rerank.py); the legacy ``True`` bool keeps
    # the lightweight heuristic RerankerRetriever. Both wrap the chosen retriever.
    rerank_conf = rag_conf.get("rerank")
    if isinstance(rerank_conf, dict):
        from koboi.rag.rerank import CrossEncoderReranker, build_rerank_client

        backend = build_rerank_client(rerank_conf)
        if backend is not None:
            retriever = CrossEncoderReranker(
                retriever,
                backend,
                fetch_multiplier=rerank_conf.get("fetch_multiplier", 3),
                score_threshold=rerank_conf.get("score_threshold"),
            )
        # else: build_rerank_client already warned; base retriever used unwrapped.
    elif rerank_conf:
        from koboi.rag.augmentation import RerankerRetriever

        retriever = RerankerRetriever(retriever)

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
    # #9: opt-in query rewriting / HyDE -- thread the chat client (distinct from the
    # embedding `client`) and the `rewrite:` config into the augmentation.
    if rag_conf.get("query_rewrite") or rag_conf.get("hyde"):
        kwargs["query_rewrite"] = bool(rag_conf.get("query_rewrite"))
        kwargs["hyde"] = bool(rag_conf.get("hyde"))
        kwargs["rewrite_client"] = chat_client
        kwargs["rewrite_config"] = rag_conf.get("rewrite") or {}

    # #10: opt-in metadata filter for relevance scoping (NOT an ACL boundary).
    if rag_conf.get("filter"):
        kwargs["metadata_filter"] = rag_conf.get("filter")

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
