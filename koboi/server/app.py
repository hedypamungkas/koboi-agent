"""koboi/server/app -- create_app composition root + ``koboi serve`` entrypoint.

Composes: AgentPool + ApprovalRegistry (HITL) + KeyStore (auth) +
OwnershipStore (tenant) + HealthRegistry + request-id middleware + routes.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Sequence

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from koboi.config import Config
from koboi.events import ErrorEvent
from koboi.guardrails.approval import AsyncCallbackApprovalHandler
from koboi.guardrails.approval_types import ApprovalResponse
from koboi.server.approvals import ApprovalCoordinator, ApprovalRegistry
from koboi.server.auth import KeyStore, make_auth_middleware
from koboi.server.health import HealthRegistry, make_db_check, make_pool_alive_check
from koboi.server.middleware import request_id_middleware
from koboi.server.ownership import OwnershipStore
from koboi.server.pool import AgentPool, PoolFull, is_safe_session_id
from koboi.server.schema import (
    ApproveRequest,
    ApproveResponse,
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

#: Default approval timeout (seconds). Overridable via config in a future rev.
APPROVAL_TIMEOUT = 120.0


def _build_key_store(config: Config, api_keys: list[str] | None = None) -> KeyStore:
    """Load API keys from file + env + config (or the ``api_keys`` test seam)."""
    ks = KeyStore()
    ks.load_from_file(config.get("server", "api_keys_file", default=None))
    if api_keys:
        ks.load_from_env(",".join(api_keys))
    else:
        env_val = os.environ.get("KOBOI_API_KEYS", "")
        if env_val:
            ks.load_from_env(env_val)
        cfg_keys = config.get("server", "api_keys", default=[])
        if isinstance(cfg_keys, list):
            for k in cfg_keys:
                ks.load_from_env(str(k))
    return ks


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
    api_keys: list[str] | None = None,
) -> FastAPI:
    """Build the FastAPI app (composition root -- single place wiring happens).

    ``api_keys`` (test seam): when provided, enables auth with those plaintext
    keys. When ``None`` (default), keys are loaded from file + env + config.
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
    approvals = ApprovalRegistry()

    # M3: API-key auth (keys file + env back-compat; dev-allow when empty).
    key_store = _build_key_store(config, api_keys)

    # M3: session ownership (SQLite sidecar; same db_path as memory when sqlite).
    memory_backend = config.get("memory", "backend", default="sqlite")
    memory_db_path = config.get("memory", "db_path", default="koboi_memory.db")
    ownership_db = memory_db_path if memory_backend == "sqlite" else ":memory:"
    ownership = OwnershipStore(db_path=ownership_db)

    health = HealthRegistry()
    health.register("pool", make_pool_alive_check(pool))
    health.register("db", make_db_check(config))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await pool.close_all()
            ownership.close()

    app = FastAPI(title=f"koboi-{config.agent_name}", version="0.3.0", lifespan=lifespan)
    app.state.pool = pool
    app.state.approvals = approvals
    app.state.ownership = ownership
    app.state.health = health

    if enable_cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Middleware: registration order is the REVERSE of execution order.
    # request_id is registered LAST → executes FIRST (outermost), wrapping auth
    # so 401/403 responses carry X-Request-Id.
    app.middleware("http")(make_auth_middleware(key_store))
    for mw in extra_middleware:
        app.middleware("http")(mw)
    app.middleware("http")(request_id_middleware)

    _register_routes(app, pool, health, approvals, ownership)
    for registrar in extra_routes:
        registrar(app, pool)
    return app


def _check_owner(ownership: OwnershipStore, session_id: str, request: Request) -> JSONResponse | None:
    """Returns an error response if the caller is not the session owner.

    Sessions with NO owner set (header-provided, never POSTed) are allowed
    (back-compat — the session_id is the secret). Only explicitly-owned sessions
    (created via POST /v1/sessions) are restricted to their owner.
    """
    owner = getattr(request.state, "api_key_id", "dev")
    actual = ownership.get_owner(session_id)
    if actual is not None and actual != owner:
        return _error_response(403, "forbidden", "not the session owner", request)
    return None


def _register_routes(
    app: FastAPI,
    pool: AgentPool,
    health: HealthRegistry,
    approvals: ApprovalRegistry,
    ownership: OwnershipStore,
) -> None:
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
            await pool.get_or_create(sid)
        except PoolFull as exc:
            return _error_response(429, "pool_full", str(exc), request)
        ownership.set_owner(sid, getattr(request.state, "api_key_id", "dev"))
        response.headers["X-Session-Id"] = sid
        return CreateSessionResponse(session_id=sid)

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        if pool.get(session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        messages = await pool.get_messages(session_id)
        return SessionResponse(session_id=session_id, messages=messages)

    @app.delete("/v1/sessions/{session_id}", response_model=SessionDeletedResponse)
    async def delete_session(session_id: str, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        evicted = await pool.evict(session_id)
        if not evicted:
            raise HTTPException(status_code=404, detail="session not found")
        ownership.delete(session_id)
        return SessionDeletedResponse(session_id=session_id, evicted=True)

    # ---- M2: /chat/stream with queue-bridged HITL + /approve ----

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

        # M3: ownership — check for existing sessions (header-provided), set for
        # NEW sessions AFTER get_or_create succeeds (avoids orphan rows on PoolFull
        # and overwriting an existing owner after eviction+re-create).
        owner = getattr(request.state, "api_key_id", "dev")
        is_new_session = header_sid is None and pool.get(session_id) is None
        if header_sid is not None:
            err = _check_owner(ownership, session_id, request)
            if err:
                return err

        try:
            agent = await pool.get_or_create(session_id)
        except PoolFull as exc:
            return _error_response(429, "pool_full", str(exc), request)

        if is_new_session:
            ownership.set_owner(session_id, owner)

        queue: asyncio.Queue = asyncio.Queue()
        coordinator = ApprovalCoordinator(queue, timeout=APPROVAL_TIMEOUT)
        approvals.register(session_id, coordinator)
        handler = AsyncCallbackApprovalHandler(
            callback=coordinator.request,
            trust_db=agent.trust_db,
            audit_trail=agent._core.audit_trail,
            timeout=APPROVAL_TIMEOUT,
        )

        async def _run_agent():
            try:
                async with pool.session_lock(session_id):
                    if hasattr(agent._core, "_tool_pipeline"):
                        del agent._core._tool_pipeline
                    agent._core.approval_handler = handler
                    async for ev in agent.run_stream(message):
                        await queue.put(ev)
            except Exception as exc:
                await queue.put(ErrorEvent(error=exc))
            finally:
                await queue.put(None)

        async def event_gen():
            task = asyncio.create_task(_run_agent())
            try:
                while True:
                    ev = await queue.get()
                    if ev is None:
                        break
                    yield ev
                await task
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                approvals.unregister(session_id)

        return StreamingResponse(
            sse_stream(event_gen()),
            media_type="text/event-stream",
            headers={
                "X-Session-Id": session_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/sessions/{session_id}/approve", response_model=ApproveResponse)
    async def approve(session_id: str, body: ApproveRequest, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        coordinator = approvals.get(session_id)
        if coordinator is None:
            raise HTTPException(status_code=404, detail="no active session or pending approval")
        approved = body.decision == "approve"
        always_allow = body.scope == "always"
        resolved = coordinator.resolve(
            body.approval_id,
            ApprovalResponse(approved=approved, always_allow=always_allow),
        )
        if not resolved:
            raise HTTPException(status_code=404, detail="approval not found or already resolved")
        return ApproveResponse(approval_id=body.approval_id, resolved=True)


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
        logging.getLogger(__name__).warning(
            "Binding to %s with NO authentication (auth lands in M3). "
            "Do not expose this server to untrusted networks yet.",
            host,
        )
    cfg = Config.from_yaml(config_path)
    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port)  # pragma: no cover (blocking server)
