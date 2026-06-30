"""koboi/server/idempotency -- in-memory TTL registry for ``/chat/stream`` Idempotency-Key.

409-reject semantics (no replay): a dedup key seen within its TTL window is rejected.
The dedup key is ``(owner, session_id, idempotency_key)`` -- see ``app.py::chat_stream``.
Single-threaded asyncio + a synchronous ``check_and_record`` make the check atomic
w.r.t. the event loop, so no lock is needed.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class IdempotencyRegistry:
    """Track recently-seen dedup keys with a TTL (lazy purge on access)."""

    def __init__(
        self, ttl_seconds: float = 600.0, max_entries: int = 10000, clock: Callable[[], float] | None = None
    ) -> None:
        self._seen: dict[str, float] = {}  # dedup_key -> timestamp
        self._ttl = ttl_seconds
        self._max_entries = max_entries  # H6: bound memory against key-storms
        self._clock = clock or time.monotonic

    def _purge(self, now: float) -> None:
        expired = [k for k, ts in self._seen.items() if ts < now - self._ttl]
        for k in expired:
            self._seen.pop(k, None)

    def check_and_record(self, dedup_key: str) -> bool:
        """``True`` if NEW (and now recorded); ``False`` if already seen within the TTL."""
        now = self._clock()
        self._purge(now)
        if dedup_key in self._seen:
            return False
        # H6: hard cap -- evict the oldest entry when full so a key-storm can't
        # grow _seen without bound (lazy TTL purge runs first).
        if len(self._seen) >= self._max_entries:
            oldest = min(self._seen, key=self._seen.get)  # type: ignore[arg-type]
            self._seen.pop(oldest, None)
        self._seen[dedup_key] = now
        return True

    def __len__(self) -> int:
        return len(self._seen)
