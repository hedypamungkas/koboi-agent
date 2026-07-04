"""koboi/server -- FastAPI serving layer (interactive SSE + extensibility).

Importing this package requires the ``api`` extra (``fastapi``/``uvicorn``).
The pure submodules (``sse``, ``schema``, ``pool``, ``health``, ``middleware``)
do NOT import FastAPI and unit-test without it.
"""

from koboi.server.app import create_app, serve_app
from koboi.server.pool import AgentPool, PoolFull
from koboi.server.sse import DONE_FRAME, sse_stream

__all__ = ["create_app", "serve_app", "AgentPool", "PoolFull", "sse_stream", "DONE_FRAME"]
