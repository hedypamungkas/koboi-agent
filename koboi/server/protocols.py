"""koboi/server/protocols -- Protocol types for SaaS-ready state swap (M5, doc §16.20).

These Protocols define the public surface that a future Redis-backed or
multi-tenant implementation must satisfy. The current in-memory classes
(``AgentPool``, ``JobRegistry``, ``JobStore``) already conform structurally —
these are annotations, not runtime checks.

Usage in M5: import these for type hints in ``create_app`` and route handlers.
In a future SaaS milestone, a ``RedisSessionStore`` / ``RedisJobStore`` would
implement these Protocols and swap in at the ``create_app`` wiring point.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SessionStore(Protocol):
    """Per-session agent lifecycle (currently ``AgentPool``)."""

    async def get_or_create(self, session_id: str) -> Any: ...

    def get(self, session_id: str) -> Any | None: ...

    async def evict(self, session_id: str) -> bool: ...

    async def close_all(self) -> None: ...

    def __len__(self) -> int: ...


@runtime_checkable
class LockProvider(Protocol):
    """Per-session serialization (currently ``AgentPool.session_lock``)."""

    def session_lock(self, session_id: str) -> Any: ...


@runtime_checkable
class EventBuffer(Protocol):
    """Capped event buffer for SSE replay.

    ``JobRegistry`` (the current in-memory impl) now implements this surface
    directly (``append_event``/``get_events``); route handlers read via
    ``get_events`` so a future Redis EventBuffer swaps in transparently.
    """

    def append_event(self, key: str, event: Any) -> None: ...

    def get_events(self, key: str) -> list[Any]: ...


@runtime_checkable
class IdempotencyStore(Protocol):
    """TTL dedup for the /chat/stream Idempotency-Key (currently ``IdempotencyRegistry``)."""

    def check_and_record(self, key: str) -> bool: ...

    def __len__(self) -> int: ...
