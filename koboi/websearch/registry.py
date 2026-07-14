"""koboi/websearch/registry.py -- decorator-based web provider registry.

Provides ``@register_search_provider`` / ``@register_fetch_provider`` decorators and
``build_search_provider`` / ``build_fetch_provider`` resolvers that compose a provider
from the ``websearch:`` YAML section. Mirrors ``koboi/rag/registry.py`` (the ComponentRegistry
pattern), with two web-specific differences:

1. **Nested per-provider config.** A provider's kwargs are read from
   ``websearch.<stage>.<provider_name>.<key>`` (e.g. ``websearch.search.brave.api_key``), with
   shared top-level knobs (e.g. ``websearch.search.max_results``) as fallback. RAG reads a
   flat config because only one chunker/retriever is ever selected per build; web keeps
   each provider's credentials isolated in its own sub-dict.
2. **Secret redaction.** Web providers carry credentials (``api_key``/``token``), so any
   config that reaches a log line is passed through ``_redact``. RAG has no such filtering.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

from koboi.websearch.base import BaseFetchProvider, BaseSearchProvider

_logger = logging.getLogger(__name__)

# Keys whose values must never reach logs / cache-key material (provider credentials).
_SECRET_KEYS = frozenset({"api_key", "token", "secret", "password", "x_subscription_token"})


# ---------------------------------------------------------------------------
# Generic provider registry
# ---------------------------------------------------------------------------


class ProviderEntry:
    """Metadata for a registered web provider."""

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


class ProviderRegistry:
    """Generic registry for web providers (search, fetch)."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, ProviderEntry] = {}

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
        self._entries[name] = ProviderEntry(
            cls=cls,
            parameters=params,
            description=description,
            config_aliases=config_aliases,
            inject=inject,
        )

    def get(self, name: str) -> ProviderEntry | None:
        return self._entries.get(name)

    def list_available(self) -> list[str]:
        return sorted(self._entries.keys())

    def clear(self) -> None:
        self._entries.clear()


def _extract_parameters(cls: type) -> dict[str, dict[str, Any]]:
    """Return ``param_name -> {"default": ..., "annotation": ...}`` for ``cls.__init__``."""
    sig = inspect.signature(cls.__init__)  # type: ignore[misc]  # __init__ params drive config->kwargs resolution
    params: dict[str, dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
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

search_provider_registry = ProviderRegistry("search")
fetch_provider_registry = ProviderRegistry("fetch")


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def register_search_provider(
    name: str,
    description: str = "",
    *,
    config_aliases: dict[str, str] | None = None,
    inject: list[str] | None = None,
) -> Callable[[type], type]:
    """Register a search provider class.

    Usage::

        @register_search_provider("brave", description="Brave Search API")
        class BraveSearchProvider(BaseSearchProvider):
            ...
    """

    def decorator(cls: type) -> type:
        search_provider_registry.register(
            name, cls, description=description, config_aliases=config_aliases, inject=inject
        )
        return cls

    return decorator


def register_fetch_provider(
    name: str,
    description: str = "",
    *,
    config_aliases: dict[str, str] | None = None,
    inject: list[str] | None = None,
) -> Callable[[type], type]:
    """Register a fetch provider class.

    Usage::

        @register_fetch_provider("firecrawl", description="Firecrawl scrape -> markdown")
        class FirecrawlFetchProvider(BaseFetchProvider):
            ...
    """

    def decorator(cls: type) -> type:
        fetch_provider_registry.register(
            name, cls, description=description, config_aliases=config_aliases, inject=inject
        )
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _redact(conf: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``conf`` with secret values masked (for safe logging)."""
    redacted: dict[str, Any] = {}
    for k, v in conf.items():
        redacted[k] = "***" if k in _SECRET_KEYS and v else v
    return redacted


def _merged_provider_conf(parent_conf: dict[str, Any], provider_name: str) -> dict[str, Any]:
    """Build the kwargs source for a provider: shared top-level knobs + per-provider sub-dict.

    Shared knobs are the non-dict keys on ``parent_conf`` (excluding the ``provider``
    selector). The per-provider sub-dict (``parent_conf[provider_name]``) overrides them.
    """
    merged: dict[str, Any] = {}
    for k, v in parent_conf.items():
        if k == "provider" or isinstance(v, dict):
            continue
        merged[k] = v
    merged.update(parent_conf.get(provider_name, {}) or {})
    return merged


def _resolve_kwargs(entry: ProviderEntry, conf: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``__init__`` kwargs from ``conf`` using entry metadata.

    Iterates the provider's introspected params (not the config keys), so only declared
    params are ever forwarded -- unknown YAML keys are silently ignored. ``config_aliases``
    remaps a YAML key to a param name.
    """
    config_aliases = entry.config_aliases
    kwargs: dict[str, Any] = {}
    for param_name in entry.parameters:
        yaml_key = param_name
        for yk, pn in config_aliases.items():
            if pn == param_name:
                yaml_key = yk
                break
        if yaml_key in conf:
            kwargs[param_name] = conf[yaml_key]
        elif "default" in entry.parameters[param_name]:
            kwargs[param_name] = entry.parameters[param_name]["default"]
    return kwargs


def _build_provider(
    registry: ProviderRegistry,
    kind: str,
    parent_conf: dict[str, Any],
    fallback_name: str,
) -> BaseSearchProvider | BaseFetchProvider:
    """Resolve ``parent_conf['provider']`` -> instance, with ``fallback_name`` on miss.

    ``parent_conf`` is the stage section (``websearch.search`` or ``websearch.fetch``). The provider's
    kwargs come from ``_merged_provider_conf``.
    """
    provider_name = parent_conf.get("provider", fallback_name)
    entry = registry.get(provider_name)
    if entry is None:
        _logger.warning(
            "Unknown %s provider '%s', falling back to '%s'. Available: %s. Config: %s",
            kind,
            provider_name,
            fallback_name,
            registry.list_available(),
            _redact(parent_conf),
        )
        entry = registry.get(fallback_name)
        if entry is None:
            raise ValueError(f"No {kind} providers registered")

    conf = _merged_provider_conf(parent_conf, provider_name)
    kwargs = _resolve_kwargs(entry, conf)
    return entry.cls(**kwargs)  # type: ignore[no-any-return]


def build_search_provider(websearch_conf: dict[str, Any] | None) -> BaseSearchProvider:
    """Build a search provider from the ``websearch:`` config (``websearch.search.*``).

    Defaults to ``mock`` when unset (offline-safe). Unknown provider -> ``mock``.
    """
    return _build_provider(  # type: ignore[return-value]
        search_provider_registry,
        "search",
        (websearch_conf or {}).get("search", {}) or {},
        fallback_name="mock",
    )


def build_fetch_provider(websearch_conf: dict[str, Any] | None) -> BaseFetchProvider:
    """Build a fetch provider from the ``websearch:`` config (``websearch.fetch.*``).

    Defaults to ``httpx`` (readability extractor) when unset. Fetch providers are
    registered in Wave 1; until then only the default resolves.
    """
    return _build_provider(  # type: ignore[return-value]
        fetch_provider_registry,
        "fetch",
        (websearch_conf or {}).get("fetch", {}) or {},
        fallback_name="httpx",
    )


# ---------------------------------------------------------------------------
# Custom module loading (YAML-driven extensibility)
# ---------------------------------------------------------------------------


def load_custom_components(custom_modules: list[str]) -> None:
    """Import modules so ``@register_*`` decorators fire on import.

    YAML config example::

        websearch:
          custom_modules:
            - mycorp.web_providers.bing
    """
    for module_path in custom_modules:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            _logger.warning("Failed to import custom web module '%s': %s", module_path, e)
