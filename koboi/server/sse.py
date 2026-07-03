"""koboi/server/sse -- pure SSE wire encoder for StreamEvent.

No FastAPI/Pydantic imports so it unit-tests without the ``api`` extra installed.
``sse_stream`` wraps any ``AsyncGenerator[StreamEvent]`` and always terminates
with ``data: [DONE]\\n\\n`` -- even when the agent raises (an ``ErrorEvent``
frame is emitted first, so clients never see a truncated stream).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from koboi.events import ErrorEvent, event_to_dict

if TYPE_CHECKING:
    from koboi.events import StreamEvent

#: Terminator frame (OpenAI/Anthropic SSE convention). New in M1.
DONE_FRAME = b"data: [DONE]\n\n"

#: SSE comment keepalive frame — ignored by all clients per spec (RFC 8895 §9.1),
#: but resets Cloudflare's ~100s idle-connection timeout during HITL waits and
#: sparse autonomous-job phases.
KEEPALIVE_FRAME = b": keepalive\n\n"

#: Default keepalive interval (seconds). Chosen well under Cloudflare's ~100s limit.
DEFAULT_KEEPALIVE_INTERVAL = 15.0


def _frame(obj: dict) -> bytes:
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n".encode()


async def sse_stream(
    event_gen: AsyncGenerator[StreamEvent, None],
    keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL,
) -> AsyncGenerator[bytes, None]:
    """Wrap a StreamEvent generator into SSE byte frames, ending with [DONE].

    Emits a keepalive comment frame every ``keepalive_interval`` seconds when
    the generator is silent (HITL approval waits, sparse job phases).

    Uses a persistent asyncio.Task for the next-item fetch so that a timeout
    does NOT cancel the underlying __anext__() call — the generator stays live
    and resumes after each keepalive.
    """
    _sentinel = object()

    async def _anext() -> object:
        try:
            return await event_gen.__anext__()
        except StopAsyncIteration:
            return _sentinel

    pending: asyncio.Task | None = None
    try:
        pending = asyncio.ensure_future(_anext())
        while True:
            done, _ = await asyncio.wait({pending}, timeout=keepalive_interval)
            if not done:
                yield KEEPALIVE_FRAME
                continue
            result = pending.result()  # re-raises any exception from the generator
            if result is _sentinel:
                break
            yield _frame(event_to_dict(result))  # type: ignore[arg-type]
            pending = asyncio.ensure_future(_anext())
    except Exception as exc:  # never drop the stream without an error frame + [DONE]
        yield _frame(event_to_dict(ErrorEvent(error=exc)))
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, Exception):
                pass
        yield DONE_FRAME
