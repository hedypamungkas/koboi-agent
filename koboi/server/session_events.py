"""koboi/server/session_events.py -- per-session capped event buffer for SSE replay (B2).

A supervisor/human operator can ``GET /v1/sessions/{id}/stream`` to REPLAY a
session's buffered event history + LIVE-TAIL the current/next turn -- closing the
B1 post-handover blindness gap (the per-invocation ``/chat/stream`` queue is GC'd
when the stream ends, so a supervisor connecting after a handover otherwise sees
nothing).

CR-1 fix: events carry a monotonic **sequence number** so the replay reader tracks
``last_seq`` (stable across buffer trims) instead of a list index (which stalled
permanently once the sliding-window buffer reached capacity). ``get_events_since``
returns events appended after a given seq + the latest seq.

In-memory, per-process (lost on restart); retained across turns + after a
``/chat/stream`` ends until ``DELETE /sessions/{id}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _SessionBuffer:
    # (seq, event) tuples; oldest trimmed when cap exceeded.
    events: list[tuple[int, Any]] = field(default_factory=list)
    next_seq: int = 1


class SessionEventRegistry:
    """Per-session capped event buffer for SSE replay (B2).

    The producer is the ``/chat/stream`` ``_run_agent`` loop (single-threaded per
    turn under ``pool.session_lock``); replay readers poll ``get_events_since``.
    """

    def __init__(self, max_events: int = 1000) -> None:
        self._buffers: dict[str, _SessionBuffer] = {}
        self._max_events = max_events

    def append_event(self, session_id: str, event: Any) -> None:
        buf = self._buffers.setdefault(session_id, _SessionBuffer())
        seq = buf.next_seq
        buf.events.append((seq, event))
        buf.next_seq += 1
        if len(buf.events) > self._max_events:
            buf.events = buf.events[-self._max_events :]

    def get_events(self, session_id: str) -> list:
        """Return a COPY of all buffered events (without seq). Legacy compat."""
        buf = self._buffers.get(session_id)
        return [ev for _, ev in buf.events] if buf else []

    def get_events_since(self, session_id: str, after_seq: int = 0) -> tuple[list, int]:
        """Return ``(events appended after after_seq, latest_seq)``.

        Uses the monotonic sequence (not a list index) so the cursor is stable
        across buffer trims — the reader never stalls (CR-1 fix).
        """
        buf = self._buffers.get(session_id)
        if not buf or not buf.events:
            return [], after_seq
        events = [ev for s, ev in buf.events if s > after_seq]
        latest_seq = buf.events[-1][0]
        return events, latest_seq

    def forget(self, session_id: str) -> None:
        """Drop the buffer (on DELETE /sessions/{id}). No-op if unseen."""
        self._buffers.pop(session_id, None)
