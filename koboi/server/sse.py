"""koboi/server/sse -- pure SSE wire encoder for StreamEvent.

No FastAPI/Pydantic imports so it unit-tests without the ``api`` extra installed.
``sse_stream`` wraps any ``AsyncGenerator[StreamEvent]`` and always terminates
with ``data: [DONE]\\n\\n`` -- even when the agent raises (an ``ErrorEvent``
frame is emitted first, so clients never see a truncated stream).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from koboi.events import ErrorEvent, event_to_dict

if TYPE_CHECKING:
    from koboi.events import StreamEvent

#: Terminator frame (OpenAI/Anthropic SSE convention). New in M1.
DONE_FRAME = b"data: [DONE]\n\n"


def _frame(obj: dict) -> bytes:
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n".encode()


async def sse_stream(event_gen: AsyncGenerator[StreamEvent, None]) -> AsyncGenerator[bytes, None]:
    """Wrap a StreamEvent generator into SSE byte frames, ending with [DONE]."""
    try:
        async for event in event_gen:
            yield _frame(event_to_dict(event))
    except Exception as exc:  # never drop the stream without an error frame + [DONE]
        yield _frame(event_to_dict(ErrorEvent(error=exc)))
    finally:
        yield DONE_FRAME
