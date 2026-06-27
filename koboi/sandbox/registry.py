"""koboi/sandbox/registry -- Sandbox backend registry + build_sandbox() orchestrator.

Reuses :class:`koboi.rag.registry.ComponentRegistry` (same shape as
``koboi/context/registry.py``) so backends are discovered by name and their
constructor kwargs resolved from config via signature introspection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from koboi.rag.registry import ComponentRegistry, _resolve_kwargs

if TYPE_CHECKING:
    from koboi.logger import AgentLogger
    from koboi.sandbox.base import BaseSandbox

_logger = logging.getLogger(__name__)

# Reuse the generic RAG component registry -- one shape, many subsystems.
sandbox_registry = ComponentRegistry("sandbox_backend")


def register_sandbox(name: str, description: str = ""):
    """Decorator to register a sandbox backend class.

    Usage::

        @register_sandbox("restricted", description="Cwd/env/rlimit containment")
        class RestrictedProcessBackend(BaseSandbox):
            ...
    """

    def decorator(cls: type) -> type:
        sandbox_registry.register(name, cls, description=description)
        return cls

    return decorator


def build_sandbox(sandbox_conf: dict | None, *, logger: AgentLogger | None = None) -> BaseSandbox:
    """Build a sandbox backend from the ``sandbox:`` config dict.

    Defaults to ``passthrough`` when the section is absent or the backend is
    unknown -- a misconfigured sandbox must never brick the agent, so this never
    raises on bad config; it logs a warning and falls back instead.
    """
    conf = dict(sandbox_conf or {})
    backend = conf.pop("backend", "passthrough")

    entry = sandbox_registry.get(backend)
    if entry is None:
        _logger.warning(
            "Unknown sandbox backend '%s', falling back to 'passthrough'. Available: %s",
            backend,
            sandbox_registry.list_available(),
        )
        entry = sandbox_registry.get("passthrough")
        if entry is None:
            raise RuntimeError("No sandbox backends registered")
        conf = {}  # passthrough takes no kwargs; drop unknown keys

    kwargs = _resolve_kwargs(entry, conf)
    return entry.cls(**kwargs)


def register_builtin_sandboxes() -> None:
    """Register the shipped backends. Idempotent; safe at import time."""
    from koboi.sandbox.passthrough import PassthroughBackend
    from koboi.sandbox.restricted import RestrictedProcessBackend

    if sandbox_registry.get("passthrough") is None:
        sandbox_registry.register("passthrough", PassthroughBackend, description="Default, behavior-preserving")
    if sandbox_registry.get("restricted") is None:
        sandbox_registry.register(
            "restricted",
            RestrictedProcessBackend,
            description="Cwd/env/PATH/network/rlimit containment",
        )
