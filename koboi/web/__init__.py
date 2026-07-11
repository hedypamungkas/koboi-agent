"""koboi/web -- web I/O abstraction: search + fetch provider registries.

Two decorator-based registries (mirroring ``koboi/rag``): ``@register_search_provider``
and ``@register_fetch_provider``. Built-in search providers: mock (offline default),
ddg (fallback), brave, firecrawl. Fetch providers land in Wave 1. ``web_search`` /
``web_fetch`` tools (``koboi/tools/builtin/web.py``) delegate to providers injected via
the tool registry's dep store.
"""

from __future__ import annotations

from koboi.web.base import BaseFetchProvider, BaseSearchProvider
from koboi.web.types import FetchResult, SearchResult

# Register built-in providers (idempotent; decorators fire on import of each module).
from koboi.web.providers import brave as _brave  # noqa: F401
from koboi.web.providers import ddg as _ddg  # noqa: F401
from koboi.web.providers import firecrawl as _firecrawl  # noqa: F401
from koboi.web.providers import mock as _mock  # noqa: F401

from koboi.web.registry import (
    ProviderEntry,
    ProviderRegistry,
    build_fetch_provider,
    build_search_provider,
    fetch_provider_registry,
    load_custom_components,
    register_fetch_provider,
    register_search_provider,
    search_provider_registry,
)

__all__ = [
    # Types
    "SearchResult",
    "FetchResult",
    # ABCs
    "BaseSearchProvider",
    "BaseFetchProvider",
    # Registry
    "ProviderRegistry",
    "ProviderEntry",
    "search_provider_registry",
    "fetch_provider_registry",
    "register_search_provider",
    "register_fetch_provider",
    "build_search_provider",
    "build_fetch_provider",
    "load_custom_components",
]
