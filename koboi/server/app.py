"""koboi/server/app -- create_app composition root + ``koboi serve`` entrypoint.

Composes: AgentPool (lifecycle brain) + HealthRegistry + request-id middleware +
routes (sessions CRUD, /chat/stream SSE, /healthz, /readyz). Routes are inline
in M1; M2 (/approve) and M3 (auth) extend additively.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Sequence

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from koboi.config import Config
from koboi.server.health import HealthRegistry, make_db_check, make_pool_alive_check
from koboi.server.middleware import request_id_middleware
from koboi.server.pool import AgentPool, PoolFull, is_safe_session_id
from koboi.server.schema import (
    ChatStreamRequest,
    CreateSessionResponse,
    ErrorDetail,
    ErrorResponse,
    ReadyzCheck,
    ReadyzResponse,
    SessionDeletedResponse,
    SessionResponse,
)
from koboi.server.sse import sse_stream

ExtraRouteRegistrar = Callable[[FastAPI, AgentPool], None]


def create_app(
    config: Config,
    *,
    client_factory: Callable[[], Any] | None = None,
    extra_tools: Sequence = (),
    extra_hooks: Sequence = (),
    approval_handler: Any | None = None,
    extra_middleware: Sequence = (),
    extra_routes: Sequence[ExtraRouteRegistrar] = (),
    workspace_root: str = "./workspace",
    cap: int = 100,
    enable_cors: bool = True,
) -> FastAPI:
    """Build the FastAPI app (composition root -- single place wiring happens).

    ``client_factory`` is the test seam: when set, each session's agent is built
    then has its LLM client swapped to ``client_factory()`` (e.g. a MockClient),
    so integration tests run with no network. Production leaves it ``None``.
    """
    pool = AgentPool(
        config,
        client_factory=client_factory,
        workspace_root=workspace_root,
        cap=cap,
        extra_tools=tuple(extra_tools),
        extra_hooks=tuple(extra_hooks),
        approval_handler=approval_handler,
    )

    health = HealthRegistry()
    health.register("pool", make_pool_alive_check(pool))
    health.register("db", make_db_check(config))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await pool.close_all()

    app = FastAPI(title=f"koboi-{config.agent_name}", version="0.1.0", lifespan=lifespan)
    app.state.pool = pool
    app.state.health = health

    if enable_cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.middleware("http")(request_id_middleware)
    for mw in extra_middleware:
        app.middleware("http")(mw)

    _register_routes(app, pool, health)
    for registrar in extra_routes:
        registrar(app, pool)
    return app


def _register_routes(app: FastAPI, pool: AgentPool, health: HealthRegistry) -> None:
    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", response_model=ReadyzResponse)
    async def readyz(response: Response) -> ReadyzResponse:
        results = await health.run_all()
        ok = all(r.ok for r in results)
        response.status_code = 200 if ok else 503
        return ReadyzResponse(
            status="ok" if ok else "down",
            checks=[ReadyzCheck(**r.to_dict()) for r in results],
        )

    @app.post("/v1/sessions", status_code=201)
    async def create_session(request: Request, response: Response) -> Response:
        sid = pool.new_session_id()
        try:
            await pool.get_or_create(sid)  # materialize so GET/DELETE work immediately
        except PoolFull as exc:
            return _error_response(429, "pool_full", str(exc), request)
        response.headers["X-Session-Id"] = sid
        return CreateSessionResponse(session_id=sid)

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        if pool.get(session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        messages = await pool.get_messages(session_id)
        return SessionResponse(session_id=session_id, messages=messages)

    @app.delete("/v1/sessions/{session_id}", response_model=SessionDeletedResponse)
    async def delete_session(session_id: str, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        evicted = await pool.evict(session_id)
        if not evicted:
            raise HTTPException(status_code=404, detail="session not found")
        return SessionDeletedResponse(session_id=session_id, evicted=True)

    @app.post("/v1/chat/stream")
    async def chat_stream(body: ChatStreamRequest, request: Request):
        try:
            message = body.user_message()
        except ValueError as exc:
            return _error_response(400, "bad_request", str(exc), request)

        header_sid = request.headers.get("X-Session-Id")
        if header_sid is not None and not is_safe_session_id(header_sid):
            return _error_response(400, "bad_request", "invalid X-Session-Id", request)
        session_id = header_sid or pool.new_session_id()
        try:
            await pool.get_or_create(session_id)
        except PoolFull as exc:
            return _error_response(429, "pool_full", str(exc), request)

        async def event_gen():
            async for ev in pool.run_stream(session_id, message):
                yield ev

        return StreamingResponse(
            sse_stream(event_gen()),
            media_type="text/event-stream",
            headers={
                "X-Session-Id": session_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx/proxy buffering
            },
        )


def _error_response(status: int, code: str, message: str, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            error=ErrorDetail(code=code, message=message, request_id=getattr(request.state, "request_id", None))
        ).model_dump(),
    )


def serve_app(config_path: str | Path, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """``koboi serve`` entrypoint: load config, build app, run uvicorn."""
    import logging

    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        # M1 ships without auth (M3 adds it). Warn loudly on non-loopback binds.
        logging.getLogger(__name__).warning(
            "Binding to %s with NO authentication (auth lands in M3). "
            "Do not expose this server to untrusted networks yet.",
            host,
        )
    cfg = Config.from_yaml(config_path)
    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port)  # pragma: no cover (blocking server)
