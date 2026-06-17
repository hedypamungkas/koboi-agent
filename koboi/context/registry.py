"""koboi/context/registry.py -- Decorator-based context strategy registry.

Provides @register_context_strategy decorator and a build_context() factory
that resolves strategies from config, following the same pattern as rag/registry.py.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from koboi.rag.registry import ComponentRegistry

if TYPE_CHECKING:
    from koboi.context.manager import ContextManager
    from koboi.logger import AgentLogger
    from koboi.client import Client

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

context_registry = ComponentRegistry("context_strategy")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def register_context_strategy(name: str, description: str = ""):
    """Decorator to register a context strategy class.

    Usage::

        @register_context_strategy("truncation", description="Keep last N messages")
        class TruncationManager(ContextManager):
            ...
    """

    def decorator(cls: type) -> type:
        context_registry.register(name, cls, description=description)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Custom module loading
# ---------------------------------------------------------------------------


def load_custom_context_modules(custom_modules: list[str]) -> None:
    """Import modules to trigger @register_context_strategy decorators.

    YAML config example::

        context:
          custom_modules:
            - my_package.context.strategies
    """
    for module_path in custom_modules:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            _logger.warning(
                "Failed to import custom context module '%s': %s",
                module_path,
                e,
            )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_context(
    strategy: str,
    *,
    logger: AgentLogger | None = None,
    client: Client | None = None,
    **kwargs: Any,
) -> ContextManager | None:
    """Build a context manager from the registry by strategy name.

    Args:
        strategy: Strategy name (e.g. "truncation", "sliding_window").
        logger: Optional logger instance.
        client: Optional LLM client (needed by sliding_window).
        **kwargs: Extra constructor args (keep_last, summarization_truncation, etc.).

    Returns:
        Configured ContextManager, or None if strategy is "noop" or unknown.
    """
    if strategy == "noop":
        return None

    entry = context_registry.get(strategy)
    if entry is None:
        _logger.warning(
            "Unknown context strategy '%s'. Available: %s",
            strategy,
            context_registry.list_available(),
        )
        return None

    ctor_kwargs: dict[str, Any] = {}
    if logger is not None and "logger" in entry.parameters:
        ctor_kwargs["logger"] = logger
    if client is not None and "client" in entry.parameters:
        ctor_kwargs["client"] = client
    ctor_kwargs.update(kwargs)

    return entry.cls(**ctor_kwargs)
