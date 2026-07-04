"""koboi/server/middleware -- request-id middleware (doc §16.26).

Mints (or honors an incoming) ``X-Request-Id`` per request, stashes it on
``request.state.request_id``, and echoes it on the response so logs, traces,
and Langfuse (M5 enrichment) can correlate end-to-end.
"""

from __future__ import annotations

from uuid import uuid4

REQUEST_ID_HEADER = "X-Request-Id"


def derive_request_id(incoming: str | None) -> str:
    """Honor a non-empty incoming id; otherwise mint a uuid4 hex."""
    if incoming and incoming.strip():
        return incoming.strip()
    return uuid4().hex


async def request_id_middleware(request, call_next):  # type: ignore[no-untyped-def]
    rid = derive_request_id(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = rid
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = rid
    return response
