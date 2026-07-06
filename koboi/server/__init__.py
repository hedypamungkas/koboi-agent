"""koboi/server -- FastAPI serving layer (interactive SSE + extensibility).

The public surface (``create_app``, ``serve_app``, ``AgentPool``, ``PoolFull``,
``sse_stream``, ``DONE_FRAME``) is exported LAZILY via ``__getattr__`` (PEP 562)
so that importing a pure submodule -- e.g. ``from koboi.server.keys_cli import
create_key`` -- does NOT transitively import ``koboi.server.app`` (and thus
``fastapi``). The ``[api]`` extra is required only when one of those names is
actually accessed; the pure submodules (``sse``, ``schema``, ``pool``,
``health``, ``middleware``, ``idempotency``, ``keys_cli``, ``protocols``) remain
importable without it.
"""

from __future__ import annotations

__all__ = ["create_app", "serve_app", "AgentPool", "PoolFull", "sse_stream", "DONE_FRAME"]

# Map each public attribute to the (submodule, attribute) that defines it.
# Resolved on first access via __getattr__ and cached in globals() -- so the
# parent package import never eagerly pulls fastapi via koboi.server.app.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "create_app": ("koboi.server.app", "create_app"),
    "serve_app": ("koboi.server.app", "serve_app"),
    "AgentPool": ("koboi.server.pool", "AgentPool"),
    "PoolFull": ("koboi.server.pool", "PoolFull"),
    "sse_stream": ("koboi.server.sse", "sse_stream"),
    "DONE_FRAME": ("koboi.server.sse", "DONE_FRAME"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'koboi.server' has no attribute {name!r}")
    module_name, attr = target
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
