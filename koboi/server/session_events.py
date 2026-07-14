"""koboi/server/session_events.py -- per-session capped event buffer for SSE replay (B2).

A supervisor/human operator can ``GET /v1/sessions/{id}/stream`` to REPLAY a
session's buffered event history + LIVE-TAIL the current/next turn -- closing the
B1 post-handover blindness gap (the per-invocation ``/chat/stream`` queue is GC'd
when the stream ends, so a supervisor connecting after a handover otherwise sees
nothing).

Mirrors ``JobRegistry``'s buffer mechanics (``koboi/server/jobs.py``) -- a plain
capped ``list`` per key, ``append_event``/``get_events`` -- minus the job-specific
``JobRecord`` fields (status/task/terminal/register-by-job). Conforms to the
``EventBuffer`` Protocol (``koboi/server/protocols.py``): the same surface a future
Redis-backed buffer would swap in. In-memory, per-process (lost on restart);
retained across turns + after a ``/chat/stream`` ends until ``DELETE /sessions/{id}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _SessionBuffer:
    events: list = field(default_factory=list)


class SessionEventRegistry:
    """Per-session capped event buffer for SSE replay (B2).

    Conforms to the ``EventBuffer`` Protocol (``append_event``/``get_events`` keyed
    by ``session_id``). The producer is the ``/chat/stream`` ``_run_agent`` loop
    (single-threaded per turn under ``pool.session_lock``); replay readers poll
    ``get_events`` -- unsynchronized, mirroring ``JobRegistry``.
    """

    def __init__(self, max_events: int = 1000) -> None:
        self._buffers: dict[str, _SessionBuffer] = {}
        self._max_events = max_events

    def append_event(self, session_id: str, event: Any) -> None:
        buf = self._buffers.setdefault(session_id, _SessionBuffer())
        buf.events.append(event)
        if len(buf.events) > self._max_events:
            # Slice-trim, keep newest (mirror jobs.py JobRegistry.append_event).
            buf.events = buf.events[-self._max_events :]

    def get_events(self, session_id: str) -> list:
        """Return a COPY of the capped event list (empty if session unseen)."""
        buf = self._buffers.get(session_id)
        return list(buf.events) if buf else []

    def forget(self, session_id: str) -> None:
        """Drop the buffer (on DELETE /sessions/{id}). No-op if unseen."""
        self._buffers.pop(session_id, None)
