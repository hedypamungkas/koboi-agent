"""koboi/tracing_context -- W3C Trace Context propagation for cross-instance A2A (P4).

A ``contextvars.ContextVar`` carries the current W3C trace context across the async
fan-out, so ``invoke_peer`` (deep in the tool pipeline / orchestrator, with no HTTP
request in scope) can stamp an outbound ``traceparent`` header without any signature
threading. ``asyncio.gather`` / ``create_task`` copy the context for free, so a
parallel fan-out to N peers shares the caller's trace-id automatically.

W3C format: ``00-<trace-id 32 hex>-<parent-id 16 hex>-<flags 2 hex>``. Each A2A hop
carries a *child* traceparent (same trace-id, fresh parent-id) so the whole fan-out
is one trace tree, correlatable across instances via the shared trace-id (stamped in
the step journal + linked into Langfuse trace metadata).
"""

from __future__ import annotations

import re
import secrets
from contextvars import ContextVar
from dataclasses import dataclass

_TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")
_DEFAULT_FLAGS = "01"  # sampled


@dataclass(frozen=True)
class TraceContext:
    """A parsed W3C trace context (the three fields after the ``00`` version)."""

    trace_id: str  # 32 hex
    parent_id: str  # 16 hex
    flags: str  # 2 hex

    def __post_init__(self) -> None:
        # Self-protecting: the factories (parse/mint/child) always produce valid values,
        # but the type shouldn't rely on convention -- reject invalid construction.
        tp = f"00-{self.trace_id}-{self.parent_id}-{self.flags}"
        if not _TRACEPARENT_RE.match(tp) or self.trace_id == "0" * 32 or self.parent_id == "0" * 16:
            raise ValueError(f"invalid TraceContext: {tp!r}")

    def as_traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.parent_id}-{self.flags}"


_ctx: ContextVar[TraceContext | None] = ContextVar("koboi_trace_ctx", default=None)


def parse_traceparent(value: str | None) -> TraceContext | None:
    """Parse + validate a W3C ``traceparent`` header; None if absent/malformed/invalid.

    Rejects format mismatches and the W3C-invalid all-zero trace-id / parent-id.
    """
    if not value:
        return None
    value = value.strip().lower()
    if not _TRACEPARENT_RE.match(value):
        return None
    _, trace_id, parent_id, flags = value.split("-")
    if trace_id == "0" * 32 or parent_id == "0" * 16:
        return None
    return TraceContext(trace_id=trace_id, parent_id=parent_id, flags=flags)


def mint_root() -> TraceContext:
    """Mint a fresh root trace context (new trace-id + parent-id)."""
    return TraceContext(trace_id=secrets.token_hex(16), parent_id=secrets.token_hex(8), flags=_DEFAULT_FLAGS)


def child(tc: TraceContext) -> TraceContext:
    """A child context for an outbound hop: same trace-id, fresh parent-id (proper W3C)."""
    return TraceContext(trace_id=tc.trace_id, parent_id=secrets.token_hex(8), flags=tc.flags)


def current() -> TraceContext | None:
    """The current trace context (None if no trace is active)."""
    return _ctx.get()


def current_trace_id() -> str | None:
    """Just the trace-id of the current context (None if none). The cross-instance key."""
    tc = _ctx.get()
    return tc.trace_id if tc is not None else None


def current_traceparent() -> str | None:
    """The current context as a ``traceparent`` header value (None if no trace)."""
    tc = _ctx.get()
    return tc.as_traceparent() if tc is not None else None


def set_context(tc: TraceContext | None) -> None:
    """Set the current trace context (no token/reset -- callers are task-scoped)."""
    _ctx.set(tc)


def begin_request(incoming_header: str | None) -> TraceContext:
    """Start tracing for an inbound request: honor a valid ``traceparent`` else mint a root.

    Sets the ContextVar for the current async context (the route handler / job run and
    everything it awaits -- the agent loop, tools, ``invoke_peer``). Returns the context.
    """
    tc = parse_traceparent(incoming_header) or mint_root()
    _ctx.set(tc)
    return tc
