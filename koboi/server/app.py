"""koboi/server/app -- create_app composition root + ``koboi serve`` entrypoint.

Composes: AgentPool + ApprovalRegistry (HITL) + KeyStore (auth) +
OwnershipStore (tenant) + HealthRegistry + request-id middleware + routes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from koboi.config import Config
from koboi.events import ErrorEvent, HandoverEvent
from koboi.exceptions import AgentHandoverError
from koboi.guardrails.approval import AsyncCallbackApprovalHandler
from koboi.guardrails.approval_types import ApprovalResponse
from koboi.modes import AgentMode, ModeManager
from koboi.server.approvals import ApprovalCoordinator, ApprovalRegistry
from koboi.server.auth import KeyStore, make_auth_middleware
from koboi.server.health import HealthRegistry, make_db_check, make_pool_alive_check
from koboi.server.idempotency import IdempotencyRegistry
from koboi.server.jobs import (
    DuplicateIdempotencyKey,
    JobRegistry,
    JobStore,
    drain_webhook_tasks,
    new_job_id,
    resume_on_startup,
    run_job,
    _emit_handover_webhook,
)
from koboi.server.middleware import request_id_middleware
from koboi.server.ownership import OwnershipStore
from koboi.server.pool import AgentPool, PoolFull, is_safe_session_id
from koboi.server.session_events import SessionEventRegistry
from koboi.server.handoff_digest import HandoffDigest
from koboi.server.schema import (
    ApproveRequest,
    ApproveResponse,
    ChatStreamRequest,
    CreateSessionResponse,
    ErrorDetail,
    ErrorResponse,
    TransferRequest,
    TransferResponse,
    JobStatusResponse,
    JobSubmitRequest,
    McpServerCreateRequest,
    McpServerListResponse,
    McpServerResponse,
    ReadyzCheck,
    ReadyzResponse,
    SessionDeletedResponse,
    SessionForkResponse,
    SessionListItem,
    SessionListResponse,
    SessionResponse,
)
from koboi.server.sse import sse_stream

_logger = logging.getLogger(__name__)

ExtraRouteRegistrar = Callable[[FastAPI, AgentPool], None]

#: Default approval timeout (seconds). Overridable via config in a future rev.
APPROVAL_TIMEOUT = 120.0


def _cleanup_workdirs(workspace_root: str, ttl_seconds: float) -> int:
    """Remove session workdirs older than TTL. Returns count removed."""
    import time

    root = Path(workspace_root)
    if not root.is_dir():
        return 0
    cutoff = time.time() - ttl_seconds
    count = 0
    for d in root.iterdir():
        if d.is_dir():
            try:
                mtime = d.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    count += 1
            except OSError:
                pass
    return count


async def _workdir_gc_loop(workspace_root: str, ttl_seconds: float, interval: float = 300) -> None:
    """Periodic background sweep of expired session workdirs."""
    while True:
        await asyncio.sleep(interval)
        removed = _cleanup_workdirs(workspace_root, ttl_seconds)
        if removed:
            _logger.info("Workdir GC: removed %d expired director(s)", removed)


async def _job_ttl_gc_loop(
    job_store: JobStore, job_registry: JobRegistry, ttl_seconds: float, interval: float = 300
) -> None:
    """Periodic reaping of terminal jobs older than ``ttl_seconds`` (G5c-a)."""
    while True:
        await asyncio.sleep(interval)
        cutoff = time.time() - ttl_seconds
        reaped = job_store.reap_terminal_older_than(cutoff)
        if reaped:
            job_registry.forget(reaped)
            _logger.info("Job TTL GC: reaped %d terminal job(s)", len(reaped))


def _build_key_store(config: Config, api_keys: list[str] | None = None) -> KeyStore:
    """Load API keys from file + env + config (or the ``api_keys`` test seam)."""
    ks = KeyStore()
    # M8: KOBOI_API_KEYS_FILE env (set by docker-compose) takes precedence over
    # the YAML server.api_keys_file path; both are absent → load_from_file no-ops.
    keys_file = os.environ.get("KOBOI_API_KEYS_FILE") or config.get("server", "api_keys_file", default=None)
    ks.load_from_file(keys_file)
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


def _sidecar_db_path(memory_backend: str, explicit_db_path: str | None) -> str:
    """Control-plane DB path for ownership/jobs (so ``resume_on_startup`` can work).

    ``sqlite`` → the conversation DB file (matches ``SQLiteMemory``'s own default),
    so sidecars share it. non-sqlite (ephemeral conversations) → a durable file only
    when the deployer set ``memory.db_path``; otherwise ``:memory:`` (explicit opt-out
    that preserves test behavior — e.g. ``test_server_app`` builds in-memory apps).
    """
    if memory_backend == "sqlite":
        return explicit_db_path or "koboi_memory.db"
    return explicit_db_path or ":memory:"


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
    # Issue #1: externalized-state injection seam. Each defaults to None -> the
    # current in-process/SQLite impl is constructed (today's behavior, so serve_app
    # and all examples are unaffected). Pass a compatible store to swap state out
    # of process (e.g. a future Redis backend). The injected object must currently
    # satisfy the concrete surface the routes use (AgentPool/JobStore/JobRegistry/
    # IdempotencyRegistry/OwnershipStore/ApprovalRegistry); the Protocols in
    # ``protocols.py`` capture the minimal contract a full backend should meet.
    session_store: Any | None = None,
    job_store: Any | None = None,
    event_buffer: Any | None = None,
    idempotency_store: Any | None = None,
    ownership_store: Any | None = None,
    approval_registry: Any | None = None,
    session_event_buffer: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app (composition root -- single place wiring happens).

    ``api_keys`` (test seam): when provided, enables auth with those plaintext
    keys. When ``None`` (default), keys are loaded from file + env + config.
    """
    pool = (
        session_store
        if session_store is not None
        else AgentPool(
            config,
            client_factory=client_factory,
            workspace_root=workspace_root,
            cap=cap,
            extra_tools=tuple(extra_tools),
            extra_hooks=tuple(extra_hooks),
            approval_handler=approval_handler,
        )
    )
    approvals = approval_registry if approval_registry is not None else ApprovalRegistry()

    # M3: API-key auth (keys file + env back-compat; dev-allow when empty).
    key_store = _build_key_store(config, api_keys)
    # G5a: per_tenant_max is enforced only when real auth is configured (dev mode → owner "dev").
    auth_enabled = key_store.has_keys
    # C1: honor server.auth_required (default true). When true and no keys are
    # configured, the auth middleware fails closed (401) instead of serving open.
    auth_required = config.get("server", "auth_required", default=True)

    # M3: session ownership + M4: job store. Control-plane state persists to a file
    # so ``resume_on_startup`` works; see ``_sidecar_db_path`` for the resolution rules.
    memory_backend = config.get("memory", "backend", default="sqlite")
    shared_db = _sidecar_db_path(memory_backend, config.get("memory", "db_path"))
    ownership = ownership_store if ownership_store is not None else OwnershipStore(db_path=shared_db)
    if job_store is None:
        job_store = JobStore(db_path=shared_db)

    # 16.16: warn only in the genuinely-bad case — ephemeral sidecar can't resume.
    if shared_db == ":memory:":
        _logger.warning(
            "memory.backend='%s' with no memory.db_path — job/ownership sidecar is :memory:; "
            "resume-on-startup will NOT survive restart. Set memory.db_path to persist.",
            memory_backend,
        )

    # 16.24: workdir TTL GC config.
    workdir_ttl = config.get("server", "workdir_ttl_seconds", default=86400.0) or 86400.0

    # M4: job config.
    job_max_concurrent = config.get("jobs", "max_concurrent", default=64)
    job_timeout = config.get("jobs", "timeout_seconds", default=1800)
    job_per_tenant = config.get("jobs", "per_tenant_max", default=5)  # G5a
    job_queue_depth = config.get("jobs", "queue_depth", default=32)  # G5c-b
    job_ttl = config.get("jobs", "ttl_seconds", default=86400.0) or 86400.0  # G5c-a
    job_max_events = config.get("jobs", "event_buffer", "max_events", default=500) or 500
    job_webhooks = config.get("jobs", "webhooks", default=[]) or []
    # B5: chat-path handover webhooks (mid-conversation HandoverEvent notification).
    handover_webhooks = config.get("handover", "webhooks", default=[]) or []
    job_resume = config.get("jobs", "resume_on_startup", default=True)
    job_registry = event_buffer if event_buffer is not None else JobRegistry(max_events=job_max_events)
    # B2: per-session replayable event buffer (GET /v1/sessions/{id}/stream).
    session_max_events = config.get("server", "limits", "session_event_buffer", "max_events", default=1000) or 1000
    session_streams_per_owner = config.get("server", "limits", "session_streams_per_owner", default=4)
    session_stream_timeout = config.get("server", "limits", "session_stream_timeout", default=3600.0) or 3600.0
    session_events = (
        session_event_buffer if session_event_buffer is not None else SessionEventRegistry(max_events=session_max_events)
    )
    # B4: warm handoff digest (opt-in). Reuses the main llm config for the side-LLM.
    handoff_digest = None
    if config.get("handover", "digest", "enabled", default=False):
        handoff_digest = HandoffDigest(
            provider=config.get("llm", "provider", default="openai"),
            model=config.get("llm", "model", default="gpt-4o-mini"),
            api_key=config.get("llm", "api_key", default=""),
            base_url=config.get("llm", "base_url", default=""),
        )
    # G6: /chat/stream Idempotency-Key (409-reject, in-memory TTL).
    chat_idem_ttl = config.get("server", "idempotency", "chat_ttl_seconds", default=600.0) or 600.0
    chat_idem_max = config.get("server", "idempotency", "max_entries", default=10000)  # H6
    chat_idem = (
        idempotency_store
        if idempotency_store is not None
        else IdempotencyRegistry(ttl_seconds=chat_idem_ttl, max_entries=chat_idem_max)
    )
    chat_queue_maxsize = config.get("server", "limits", "chat_queue_maxsize", default=1000)  # H6
    job_streams_per_owner = config.get("server", "limits", "job_streams_per_owner", default=4)  # M3
    # G2: per-request mode/iteration knobs -- operator policy boundary. unset
    # allowed_modes ⇒ safe default (all except yolo); max_iterations_cap clamps
    # the per-request knob (ceiling, not a limit the caller can exceed).
    allowed_modes = _resolve_allowed_modes(config.get("server", "allowed_modes", default=None))
    max_iter_cap = int(config.get("server", "limits", "max_iterations_cap", default=25))

    health = HealthRegistry()
    health.register("pool", make_pool_alive_check(pool))
    health.register("db", make_db_check(ownership, backend=memory_backend))

    drain_seconds = config.get("server", "timeouts", "drain_seconds", default=60.0) or 60.0
    # G3: in-flight interactive stream producers -- cancelled deterministically
    # in _shutdown so each releases its session lock + we don't rely on uvicorn's
    # graceful shutdown (not deterministic at the app layer).
    stream_tasks: set[asyncio.Task] = set()

    async def _shutdown():
        """Drain: cancel in-flight streams, flush Langfuse, then close pool/store."""
        # G3: cancel interactive stream producers deterministically; each releases
        # its session lock via _run_agent's finally.
        await _cancel_tasks(stream_tasks)
        # G3: explicit Langfuse flush -- agent.close() doesn't (the hook flushes
        # on SESSION_END from the loop, which never fires on shutdown). Runs
        # off-loop + concurrently so a slow Langfuse server can't pin the drain.
        await pool.flush_langfuse()
        job_registry.cancel_all()
        # Drain in-flight webhook deliveries BEFORE closing the job store -- a
        # cancelled/completed job's webhook can still be mid-flight (cancel_all only
        # cancels the run task, not any webhook it already scheduled).
        await drain_webhook_tasks()
        await pool.close_all()
        ownership.close()
        job_store.close()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # M4: resume pending jobs on startup (simplified: running→failed).
        if job_resume:
            requeued = await resume_on_startup(job_store, pool, job_registry, job_timeout, job_webhooks)
            if requeued:
                _logger.info("Resumed %d pending job(s) on startup", requeued)
        # 16.24: workdir TTL GC + G5c-a: job TTL GC background sweeps.
        workdir_gc = asyncio.create_task(_workdir_gc_loop(workspace_root, workdir_ttl))
        job_gc = asyncio.create_task(_job_ttl_gc_loop(job_store, job_registry, job_ttl))
        try:
            yield
        finally:
            workdir_gc.cancel()
            job_gc.cancel()
            for _gc in (workdir_gc, job_gc):
                try:
                    await _gc
                except asyncio.CancelledError:
                    pass
            try:
                await asyncio.wait_for(_shutdown(), timeout=drain_seconds)
            except asyncio.TimeoutError:
                _logger.warning("Shutdown drain exceeded %.1fs; forcing exit", drain_seconds)

    # H7: interactive docs are off by default; enable via server.docs_enabled.
    docs_enabled = config.get("server", "docs_enabled", default=False)
    app = FastAPI(
        title=f"koboi-{config.agent_name}",
        version="0.5.0",
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.pool = pool
    app.state.approvals = approvals
    app.state.ownership = ownership
    app.state.job_store = job_store
    app.state.job_registry = job_registry
    app.state.job_max_concurrent = job_max_concurrent
    app.state.job_timeout = job_timeout
    app.state.health = health
    app.state.job_streams_per_owner = job_streams_per_owner  # M3
    app.state.job_streams = {}  # M3: owner -> active job-stream count
    app.state.session_events = session_events  # B2: per-session replay buffer
    app.state.session_streams_per_owner = session_streams_per_owner  # B2 slowloris guard
    app.state.session_streams = {}  # B2: owner -> active session-stream count
    app.state.session_stream_timeout = session_stream_timeout  # B2: stream deadline
    app.state.mcp_registries = {}  # G6: session_id -> SessionMcpRegistry

    # Middleware: registration order is the REVERSE of execution order.
    # auth is registered FIRST → executes LAST (innermost, closest to routes).
    # request_id is registered SECOND → executes before auth so 401/403
    # responses carry X-Request-Id.
    # CORSMiddleware is registered LAST → executes FIRST (outermost) so it
    # intercepts OPTIONS preflights before auth runs. Pure ASGI middleware
    # (CORSMiddleware) must be outermost; BaseHTTPMiddleware.call_next does not
    # chain into it reliably in Starlette 1.x when it sits innermost.
    app.middleware("http")(make_auth_middleware(key_store, auth_required=auth_required))
    for mw in extra_middleware:
        app.middleware("http")(mw)
    app.middleware("http")(request_id_middleware)

    # C4: CORS is config-driven, never a wildcard default. CORSMiddleware is
    # added ONLY when `server.cors` is explicitly configured; the default (no
    # `cors:` block) adds nothing → no cross-origin reads. An operator who wants
    # open CORS sets `cors: {allow_origins: ["*"]}` explicitly.
    # Registered last (outermost) so it intercepts preflights before auth.
    cors_cfg = config.get("server", "cors", default={}) or {}
    if enable_cors and cors_cfg:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_cfg.get("allow_origins", []),
            allow_methods=cors_cfg.get("allow_methods", ["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
            allow_headers=cors_cfg.get(
                "allow_headers",
                ["Authorization", "Content-Type", "X-Session-Id", "Idempotency-Key", "X-Request-Id"],
            ),
            allow_credentials=cors_cfg.get("allow_credentials", False),
            expose_headers=cors_cfg.get("expose_headers", []),
            max_age=cors_cfg.get("max_age", 600),
        )

    _register_routes(
        app,
        pool,
        health,
        approvals,
        ownership,
        job_store,
        job_registry,
        job_max_concurrent,
        job_timeout,
        chat_idem,
        job_per_tenant,
        job_queue_depth,
        auth_enabled,
        chat_queue_maxsize,
        allowed_modes,
        max_iter_cap,
        stream_tasks,
        memory_backend,
        shared_db,
        job_webhooks,
        session_events,
        session_stream_timeout,
        session_streams_per_owner,
        handoff_digest,
        handover_webhooks,
    )
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


def _enrich_trace(agent: Any, **metadata: str) -> None:
    """Tag the Langfuse trace with serving context (16.21). No-op if no hook."""
    if agent._core and agent._core.hooks:
        lf_hook = agent._core.hooks.find_hook(lambda h: type(h).__name__ == "LangfuseTracingHook")
        if lf_hook:
            lf_hook.set_serving_metadata(**metadata)


def _check_job_access(
    job_store: JobStore, job_id: str, owner: str, request: Request
) -> tuple[dict | None, JSONResponse | None]:
    """Returns (job_dict, error_response). Raises HTTPException(404) if not found."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["owner"] != owner:
        return None, _error_response(403, "forbidden", "not the job owner", request)
    return job, None


def _register_routes(
    app: FastAPI,
    pool: AgentPool,
    health: HealthRegistry,
    approvals: ApprovalRegistry,
    ownership: OwnershipStore,
    job_store: JobStore,
    job_registry: JobRegistry,
    job_max_concurrent: int,
    job_timeout: float,
    chat_idem: IdempotencyRegistry,
    job_per_tenant: int,
    job_queue_depth: int,
    auth_enabled: bool,
    chat_queue_maxsize: int,
    allowed_modes: frozenset[str],
    max_iter_cap: int,
    stream_tasks: set[asyncio.Task],
    memory_backend: str,
    shared_db: str,
    job_webhooks: list[dict] | None = None,
    session_events: Any | None = None,
    session_stream_timeout: float = 3600.0,
    session_streams_per_owner: int = 4,
    handoff_digest: Any | None = None,
    handover_webhooks: list[dict] | None = None,
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
        return CreateSessionResponse(session_id=sid)  # type: ignore[return-value]

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
        return SessionResponse(session_id=session_id, messages=messages)  # type: ignore[return-value]

    # --- G6: per-session MCP server management ---

    def _mcp_registry_for(session_id: str):
        # 29-E: lazily drop registries for sessions the pool no longer holds (LRU eviction
        #       that didn't clear app.state) so the dict can't grow unbounded over time.
        for stale in [s for s in app.state.mcp_registries if s != session_id and pool.get(s) is None]:
            app.state.mcp_registries.pop(stale, None)
        reg = app.state.mcp_registries.get(session_id)
        if reg is None:
            from koboi.server.mcp_registry import SessionMcpRegistry

            reg = SessionMcpRegistry()
            app.state.mcp_registries[session_id] = reg
        agent = pool.get(session_id)
        if agent is not None:
            reg.ensure_populated(list(agent.mcp_clients))
        return reg

    @app.get("/v1/sessions/{session_id}/mcp/servers", response_model=McpServerListResponse)
    async def list_mcp_servers(session_id: str, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        if pool.get(session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        reg = _mcp_registry_for(session_id)
        servers = [McpServerResponse(**e) for e in reg.status()]
        return McpServerListResponse(servers=servers)  # type: ignore[return-value]

    @app.post("/v1/sessions/{session_id}/mcp/servers", response_model=McpServerResponse, status_code=201)
    async def add_mcp_server(session_id: str, body: McpServerCreateRequest, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        agent = pool.get(session_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="session not found")
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        import subprocess

        from koboi.facade import _create_mcp_client
        from koboi.mcp.base import MCPError
        from koboi.types import RiskLevel

        conf = body.model_dump()
        transport = conf.get("transport", "stdio")
        risk_map = {
            "safe": RiskLevel.SAFE,
            "moderate": RiskLevel.MODERATE,
            "destructive": RiskLevel.DESTRUCTIVE,
        }
        risk = risk_map.get(str(conf.get("risk_level", "safe")).lower(), RiskLevel.SAFE)
        async with pool.session_lock(session_id):
            # 29-A: connect() is sync + blocking (subprocess spawn / HTTP handshake);
            #       offload so the event loop + every other session isn't frozen.
            # 29-F: only expected transport/config failures -> 400; anything else -> 500.
            try:
                client = _create_mcp_client(conf, transport, agent._logger, agent.config)
                await asyncio.to_thread(client.connect)
            except (MCPError, ValueError, OSError, subprocess.SubprocessError, TimeoutError, RuntimeError) as e:
                return _error_response(400, "mcp_connect_failed", f"MCP server failed to connect: {e}", request)
            # 29-D: connect() succeeded -- a registration failure (discover_tools/register)
            #       must not orphan the spawned subprocess/httpx client; close it + 502.
            try:
                agent.add_mcp_client(client, group=conf.get("group"), risk_level=risk)
                reg = _mcp_registry_for(session_id)
                sid = reg.register(client)
            except Exception as e:  # noqa: BLE001
                try:
                    client.close()
                except Exception as close_err:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "MCP client close failed after registration error for %r: %s", client.name, close_err
                    )
                return _error_response(
                    502, "mcp_register_failed", f"MCP server connected but tool discovery failed: {e}", request
                )
        entry = next((e for e in reg.status() if e["id"] == sid), {"id": sid})
        return McpServerResponse(**entry)  # type: ignore[return-value]

    @app.delete("/v1/sessions/{session_id}/mcp/servers/{server_id}", response_model=McpServerResponse)
    async def remove_mcp_server(session_id: str, server_id: str, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        agent = pool.get(session_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="session not found")
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        reg = _mcp_registry_for(session_id)
        if reg.get(server_id) is None:
            raise HTTPException(status_code=404, detail="mcp server not found")
        entry = next((e for e in reg.status() if e["id"] == server_id), {"id": server_id})
        tools = agent.core.tools if agent.core is not None else None
        async with pool.session_lock(session_id):
            # Pass the actual _mcp_clients list (not a copy) so remove() can mutate it;
            # the mcp_clients property returns list(self._mcp_clients) (a copy).
            reg.remove(server_id, tools, agent._mcp_clients)
        return McpServerResponse(**entry)  # type: ignore[return-value]

    @app.post(
        "/v1/sessions/{session_id}/mcp/servers/{server_id}/reconnect",
        response_model=McpServerResponse,
    )
    async def reconnect_mcp_server(session_id: str, server_id: str, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        agent = pool.get(session_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="session not found")
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        reg = _mcp_registry_for(session_id)
        if reg.get(server_id) is None:
            raise HTTPException(status_code=404, detail="mcp server not found")
        import subprocess

        from koboi.mcp.base import MCPError

        async with pool.session_lock(session_id):
            try:
                # 29-A: reconnect calls the blocking connect(); offload. 29-F: specific families.
                await asyncio.to_thread(reg.reconnect, server_id)
            except (MCPError, ValueError, OSError, subprocess.SubprocessError, TimeoutError, RuntimeError) as e:
                return _error_response(400, "mcp_reconnect_failed", f"MCP reconnect failed: {e}", request)
        entry = next((e for e in reg.status() if e["id"] == server_id), {"id": server_id})
        return McpServerResponse(**entry)  # type: ignore[return-value]

    @app.delete("/v1/sessions/{session_id}", response_model=SessionDeletedResponse)
    async def delete_session(session_id: str, request: Request) -> Response:
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        # Hold the session lock (if pooled) so a concurrent /chat/stream on the
        # same session finishes before we clear its rows -- otherwise the stream
        # re-inserts orphaned, unowned rows after the delete. No agent creation.
        async with pool.existing_session_lock(session_id):
            evicted = await pool.evict(session_id)
            # Also clear persisted DB rows (messages/steps/session_meta/sessions/
            # tasks) so DELETE isn't pool-only. No-op for non-sqlite backends.
            db_cleared = False
            if memory_backend == "sqlite":
                from koboi.memory_sqlite import SQLiteMemory

                db_cleared = SQLiteMemory.delete_session(shared_db, session_id)
            ownership.delete(session_id)
            # 29-E: drop the session's MCP registry so app.state.mcp_registries can't leak.
            app.state.mcp_registries.pop(session_id, None)
            # B2: drop the session's replay buffer so it can't leak.
            app.state.session_events.forget(session_id)
        if not evicted and not db_cleared:
            raise HTTPException(status_code=404, detail="session not found")
        return SessionDeletedResponse(session_id=session_id, evicted=evicted or db_cleared)  # type: ignore[return-value]

    @app.get("/v1/sessions", response_model=SessionListResponse)
    async def list_sessions_route(request: Request) -> Response:
        """List sessions, owner-scoped when auth is enabled (issue #10a)."""
        if memory_backend != "sqlite":
            return SessionListResponse(sessions=[])  # type: ignore[return-value]
        from koboi.memory_sqlite import SQLiteMemory

        rows = SQLiteMemory.list_sessions(shared_db)
        # Owner-scope: filter to the caller's sessions via the ownership sidecar.
        # Fail CLOSED when auth is on but no caller identity was stamped -- never
        # silently fall back to a real tenant name like "dev" (cross-tenant leak).
        if auth_enabled:
            owner = getattr(request.state, "api_key_id", None)
            if not owner:
                return _error_response(401, "unauthenticated", "no caller identity", request)
            owned = set(ownership.list_owned_sessions(owner))
            rows = [r for r in rows if r.get("session_id") in owned]
        items = [
            SessionListItem(
                session_id=r["session_id"],
                title=r.get("title"),
                owner=r.get("owner"),
                message_count=r.get("message_count") or 0,
                model=r.get("model"),
                agent_name=r.get("agent_name"),
                first_message=r.get("first_message"),
                updated_at=r.get("updated_at"),
            )
            for r in rows
        ]
        return SessionListResponse(sessions=items)  # type: ignore[return-value]

    @app.post("/v1/sessions/{session_id}/fork", response_model=SessionForkResponse, status_code=201)
    async def fork_session_route(session_id: str, request: Request, response: Response) -> Response:
        """Fork a session's persisted messages into a new session (issue #10a)."""
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        if pool.get(session_id) is None and not ownership.get_owner(session_id):
            raise HTTPException(status_code=404, detail="session not found")
        if memory_backend != "sqlite":
            return _error_response(409, "not_persisted", "fork requires memory.backend=sqlite", request)
        from koboi.memory_sqlite import SQLiteMemory

        new_sid = pool.new_session_id()
        SQLiteMemory.fork_session(shared_db, session_id, new_sid)
        owner = getattr(request.state, "api_key_id", "dev")
        ownership.set_owner(new_sid, owner)
        try:
            await pool.get_or_create(new_sid)
        except PoolFull as exc:
            # Roll back the committed fork rows so we don't leave a ghost session
            # (DB + owner rows, no pool entry, no X-Session-Id sent to the client).
            SQLiteMemory.delete_session(shared_db, new_sid)
            ownership.delete(new_sid)
            return _error_response(429, "pool_full", str(exc), request)
        except Exception as exc:
            SQLiteMemory.delete_session(shared_db, new_sid)
            ownership.delete(new_sid)
            return _error_response(500, "fork_failed", str(exc), request)
        response.headers["X-Session-Id"] = new_sid
        return SessionForkResponse(session_id=new_sid, source_session_id=session_id)  # type: ignore[return-value]

    @app.post("/v1/sessions/{session_id}/resume")
    async def resume_session(session_id: str, request: Request) -> Response:
        """Resume an interrupted session (rehydrate journal → continue loop).

        Returns the RunResult as JSON (non-streaming). The client can then call
        ``/chat/stream`` to continue interactively.
        """
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        agent = pool.get(session_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="session not found")
        try:
            async with pool.session_lock(session_id):
                result = await agent.resume()
            return JSONResponse(
                status_code=200,
                content={
                    "session_id": session_id,
                    "content": result.content,
                    "iterations_used": result.iterations_used,
                    "success": result.success,
                    "error": str(result.error) if result.error else None,
                },
            )
        except Exception as exc:
            return _error_response(500, "resume_failed", str(exc), request)

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
        # H1: check an existing header-supplied session, then claim ownership
        # below for ANY session currently without an owner (covers header-supplied
        # NEW sessions, which previously slipped through unowned → IDOR).
        if header_sid is not None:
            err = _check_owner(ownership, session_id, request)
            if err:
                return err

        # G2: validate per-request mode/max_iterations BEFORE consuming a pool
        # slot (mirrors the idempotency pre-check rationale). Interactive path
        # honors server.allowed_modes; yolo is permitted only if the operator
        # explicitly opted in. max_iterations is clamped to the cap (ceiling).
        try:
            effective_mode = _resolve_mode(body.mode, allowed_modes, allow_yolo=True)
        except ValueError as exc:
            return _error_response(400, "invalid_mode", str(exc), request)
        effective_max_iter = min(body.max_iterations, max_iter_cap) if body.max_iterations is not None else None

        try:
            agent = await pool.get_or_create(session_id)
        except PoolFull as exc:
            return _error_response(429, "pool_full", str(exc), request)

        # G6: 409-reject idempotency — same (owner, session, key) within TTL is a duplicate.
        # Checked after pre-checks (so PoolFull/bad-request don't consume a key) and before the
        # agent runs (so duplicates 409 fast, without waiting on the session lock).
        idem_key = request.headers.get("Idempotency-Key")
        if idem_key:
            dedup_key = f"{owner}:{session_id}:{idem_key}"
            if not chat_idem.check_and_record(dedup_key):
                return _error_response(
                    409,
                    "duplicate_request",
                    "Idempotency-Key already used for this session within the window",
                    request,
                )

        # H1: claim ownership for any session without one (header-new or recovered
        # orphan). set_owner is an upsert; an existing owner is never overwritten.
        if ownership.get_owner(session_id) is None:
            ownership.set_owner(session_id, owner)

        queue: asyncio.Queue = asyncio.Queue(maxsize=chat_queue_maxsize)  # H6: backpressure
        coordinator = ApprovalCoordinator(queue, timeout=APPROVAL_TIMEOUT)
        approvals.register(session_id, coordinator)
        # W5 B1: orchestrated configs (execution.mode: dynamic/dag/deep_research) build the
        # agent with core=None -- the orchestrator manages its own per-node agents. HITL approval
        # + per-request mode/cap don't apply during orchestration, so guard every _core access.
        handler = None
        if agent._core is not None:
            handler = AsyncCallbackApprovalHandler(
                callback=coordinator.request,
                trust_db=agent.trust_db,
                audit_trail=agent._core.audit_trail,
                timeout=APPROVAL_TIMEOUT,
            )
        # 16.21: enrich Langfuse trace with serving context.
        _enrich_trace(agent, mode="interactive", request_id=getattr(request.state, "request_id", ""), owner=owner)

        async def _run_agent():
            try:
                async with pool.session_lock(session_id):
                    prior_mode = None
                    prior_max_iter = None
                    if agent._core is not None:
                        if hasattr(agent._core, "_tool_pipeline"):
                            del agent._core._tool_pipeline
                        agent._core.approval_handler = handler
                        # G2: per-request mode + cap; restore in finally (pooled agent is
                        # reused). switch_mode is live (shared ModeManager ref).
                        prior_mode = agent._core.mode_manager.current_mode
                        prior_max_iter = agent._core.max_iterations
                        if effective_mode is not None:
                            agent._core.mode_manager.switch_mode(effective_mode)
                        if effective_max_iter is not None:
                            agent._core.max_iterations = effective_max_iter
                    try:
                        async for ev in agent.run_stream(message):
                            await queue.put(ev)
                            session_events.append_event(session_id, ev)  # B2: buffer for replay
                    finally:
                        if agent._core is not None:
                            agent._core.mode_manager.switch_mode(prior_mode)
                            agent._core.max_iterations = prior_max_iter
            except AgentHandoverError as he:
                # B1: the bot yielded via transfer_to_human -> emit a typed
                # HandoverEvent (NOT ErrorEvent). The exception propagated out of
                # run_stream, so the ``async with pool.session_lock`` above already
                # exited -> lock released (no deadlock). A human operator takes over
                # via POST /v1/sessions/{id}/transfer + a new /chat/stream.
                _summary = he.summary
                if not _summary and handoff_digest is not None:
                    # B4: warm handoff digest (opt-in). Side-LLM summary + redact.
                    # Never raises (the helper is fail-soft; this double-wrap mirrors
                    # ProactiveExtractionHook -- a digest failure MUST NOT lose the
                    # handover by falling through to the ErrorEvent branch below).
                    try:
                        _summary = await handoff_digest.digest(agent._core.memory.get_messages())
                    except Exception:  # nosec - belt-and-suspenders; never lose the handover
                        _summary = ""
                hev = HandoverEvent(
                    handover_id=uuid.uuid4().hex[:12],
                    reason=he.reason,
                    summary=_summary,
                )
                session_events.append_event(session_id, hev)  # B2: buffer for replay
                # B5: notify the host CS platform of the chat-path handover (fire-and-forget).
                _emit_handover_webhook(
                    handover_webhooks, session_id, hev.handover_id, hev.reason, _summary
                )
                await queue.put(hev)
            except Exception as exc:
                err_ev = ErrorEvent(error=exc)
                session_events.append_event(session_id, err_ev)  # B2: buffer for replay
                await queue.put(err_ev)
            finally:
                await queue.put(None)

        async def event_gen():
            task = asyncio.create_task(_run_agent())
            stream_tasks.add(task)
            task.add_done_callback(stream_tasks.discard)
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
        return ApproveResponse(approval_id=body.approval_id, resolved=True)  # type: ignore[return-value]

    @app.post("/v1/sessions/{session_id}/transfer", response_model=TransferResponse)
    async def transfer(session_id: str, body: TransferRequest, request: Request) -> Response:
        """B1: claim ownership of a session to take it over from the bot.

        After the bot yields (``HandoverEvent`` on the stream), the host CS platform
        POSTs ``/transfer`` (with the current owner's key) to reassign ownership to
        the chosen human operator; the operator then POSTs ``/chat/stream`` on the
        same session to drive it. RBAC note: ``_check_owner`` only verifies the
        caller is the CURRENT owner -- the host platform (holding the bot's key) can
        reassign to any operator. Proper RBAC (operators can't reassign each other's
        sessions) is deferred to the enterprise layer.
        """
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        new_owner = body.operator or getattr(request.state, "api_key_id", "dev")
        ownership.set_owner(session_id, new_owner)
        return TransferResponse(  # type: ignore[return-value]
            session_id=session_id, transferred=True, owner=new_owner
        )

    @app.get("/v1/sessions/{session_id}/stream")
    async def stream_session(session_id: str, request: Request):
        """B2: replay a session's buffered event history + live-tail the current/next turn.

        A supervisor/human operator calls this AFTER a handover (or during a turn)
        to see what the bot said (replay) + watch the live turn (tail) before/while
        taking over via POST /transfer + /chat/stream. Long-lived (tails across
        turns until disconnect or the configured deadline); SSE keepalives cover
        silent waits. No 404 on pool-miss -- the buffer is the source of truth for
        replay (an LRU-evicted-but-not-DELETE'd session still has a buffer). RBAC
        gap: _check_owner is owner-equality only (no supervisor role).
        """
        if not is_safe_session_id(session_id):
            return _error_response(400, "bad_request", "invalid session_id", request)
        err = _check_owner(ownership, session_id, request)
        if err:
            return err
        owner = getattr(request.state, "api_key_id", "dev")
        # B2: per-owner stream cap (slowloris guard); skip in dev (mirror jobs).
        if auth_enabled:
            active = app.state.session_streams.get(owner, 0)
            if active >= app.state.session_streams_per_owner:
                return _error_response(429, "too_many_streams", "Max concurrent session streams reached", request)
            app.state.session_streams[owner] = active + 1

        async def event_gen():
            last_index = 0
            deadline = time.monotonic() + session_stream_timeout
            try:
                while True:
                    all_events = session_events.get_events(session_id)
                    for ev in all_events[last_index:]:
                        yield ev
                    last_index = len(all_events)
                    if time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
            finally:
                if auth_enabled:
                    app.state.session_streams[owner] = max(0, app.state.session_streams.get(owner, 0) - 1)

        return StreamingResponse(
            sse_stream(event_gen()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- M4: Jobs (autonomous) ----

    def _start_job(job_id: str) -> None:
        """Admit one job: mark running, spawn run_job, attach drain-on-complete."""
        job = job_store.get(job_id)
        if job is None or job["status"] != "pending":
            return  # cancelled or reaped while queued
        task = asyncio.create_task(
            run_job(
                job_id,
                pool,
                job_registry,
                job_store,
                job["message"],
                job_timeout,
                job.get("mode"),
                job.get("max_iterations"),
                webhooks=job_webhooks,
            )
        )
        job_registry.set_running(job_id, task)
        task.add_done_callback(_on_job_done)

    def _on_job_done(_task: asyncio.Task) -> None:
        """A slot freed — start queued jobs while capacity allows (G5c-b drain)."""
        while job_registry.active_count < job_max_concurrent:
            next_id = job_registry.pop_pending()
            if next_id is None:
                break
            _start_job(next_id)

    @app.post("/v1/jobs", status_code=202)
    async def submit_job(body: JobSubmitRequest, request: Request) -> Response:
        owner = getattr(request.state, "api_key_id", "dev")

        # G2: jobs reject yolo outright (allow_yolo=False) — an autonomous run
        # has no human review, so it must keep the approval gate + rate limiter.
        try:
            job_mode = _resolve_mode(body.mode, allowed_modes, allow_yolo=False)
        except ValueError as exc:
            return _error_response(400, "invalid_mode", str(exc), request)
        job_max_iter = min(body.max_iterations, max_iter_cap) if body.max_iterations is not None else None

        # Idempotency: same key within window → return existing job.
        idem_key = request.headers.get("Idempotency-Key")
        if idem_key:
            existing = job_store.find_by_idempotency_key(idem_key)
            if existing and existing["owner"] == owner:
                return {  # type: ignore[return-value]
                    "job_id": existing["job_id"],
                    "status": existing["status"],
                    "session_id": existing["session_id"],
                }

        # G5c-b: global admission — run now, queue (up to queue_depth), or reject.
        admit = job_registry.peek_admit(job_max_concurrent, job_queue_depth)
        if admit == "reject":
            return _error_response(
                429,
                "queue_full",
                f"max_concurrent ({job_max_concurrent}) + queue_depth ({job_queue_depth}) reached",
                request,
            )

        # G5a: per-tenant running cap — hard 429 (not queued); skipped in dev mode.
        if auth_enabled and job_registry.active_count_for_owner(owner) >= job_per_tenant:
            return _error_response(
                429,
                "too_many_jobs_per_tenant",
                f"per_tenant_max ({job_per_tenant}) reached",
                request,
            )

        # Session: dedicated by default, or reuse existing (with ownership check).
        session_id = body.session_id or pool.new_session_id()
        if body.session_id:
            if not is_safe_session_id(session_id):
                return _error_response(400, "bad_request", "invalid session_id", request)
            err = _check_owner(ownership, session_id, request)
            if err:
                return err
            # H1: claim ownership of a reused session that currently has none.
            if ownership.get_owner(session_id) is None:
                ownership.set_owner(session_id, owner)
        else:
            try:
                await pool.get_or_create(session_id)
            except PoolFull as exc:
                return _error_response(429, "pool_full", str(exc), request)
            # H1: dedicated new session — always acquires an owner.
            ownership.set_owner(session_id, owner)

        job_id = new_job_id()
        try:
            job_store.insert(
                job_id,
                session_id,
                owner,
                body.message,
                idempotency_key=idem_key,
                mode=job_mode.value if job_mode else None,
                max_iterations=job_max_iter,
            )
        except DuplicateIdempotencyKey as exc:
            # M1: a concurrent same-key submit won the race -- return the canonical
            # job (if ours) or 409 so the client retries without the key.
            existing = job_store.get(exc.existing_job_id)
            if existing and existing["owner"] == owner:
                return {  # type: ignore[return-value]  # (same FastAPI route-dict noise as the other handlers)
                    "job_id": existing["job_id"],
                    "status": existing["status"],
                    "session_id": existing["session_id"],
                }
            return _error_response(
                409,
                "duplicate_request",
                "Idempotency-Key already used for this session within the window",
                request,
            )
        job_registry.register(job_id, session_id, owner)
        if admit == "run":
            _start_job(job_id)
        else:  # "queue" — wait for a running slot to free (drained on completion)
            job_registry.enqueue_pending(job_id)
        return {"job_id": job_id, "status": "pending", "session_id": session_id}  # type: ignore[return-value]

    @app.get("/v1/jobs")
    async def list_jobs(request: Request, status: str | None = None) -> list:
        owner = getattr(request.state, "api_key_id", "dev")
        jobs = job_store.list_by_owner(owner, status=status)
        return [{"job_id": j["job_id"], "status": j["status"], "session_id": j["session_id"]} for j in jobs]

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> Response:
        owner = getattr(request.state, "api_key_id", "dev")
        job, err = _check_job_access(job_store, job_id, owner, request)
        if err:
            return err
        result = json.loads(job["result_json"]) if job.get("result_json") else None
        return JobStatusResponse(  # type: ignore[return-value]
            job_id=job_id,
            status=job["status"],
            session_id=job["session_id"],
            result=result,
            error=job.get("error"),
            error_class=job.get("error_class"),
            retriable=bool(job.get("retriable", 0)),
        )

    @app.get("/v1/jobs/{job_id}/stream")
    async def stream_job(job_id: str, request: Request):
        owner = getattr(request.state, "api_key_id", "dev")
        job, err = _check_job_access(job_store, job_id, owner, request)
        if err:
            return err
        record = job_registry.get(job_id)

        # M3: per-owner concurrent-stream cap (slowloris guard). Skipped in dev
        # mode (single "dev" owner). Decremented in the generator's finally so a
        # disconnect/cancel always releases the slot.
        if auth_enabled:
            active = app.state.job_streams.get(owner, 0)
            if active >= app.state.job_streams_per_owner:
                return _error_response(
                    429,
                    "too_many_streams",
                    "Max concurrent job streams reached for this owner",
                    request,
                )
            app.state.job_streams[owner] = active + 1

        async def event_gen():
            last_index = 0
            deadline = time.monotonic() + job_timeout  # M3: bound stream duration
            try:
                while True:
                    if record:
                        # Issue #1: read via the EventBuffer surface (get_events)
                        # so a future Redis EventBuffer swaps in transparently.
                        all_events = job_registry.get_events(job_id)
                        events = all_events[last_index:]
                        last_index = len(all_events)
                        for ev in events:
                            yield ev
                        if record.terminal.is_set():
                            break
                    else:
                        break
                    if time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
            finally:
                if auth_enabled:
                    app.state.job_streams[owner] = max(0, app.state.job_streams.get(owner, 0) - 1)

        return StreamingResponse(
            sse_stream(event_gen()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request) -> Response:
        owner = getattr(request.state, "api_key_id", "dev")
        job, err = _check_job_access(job_store, job_id, owner, request)
        if err:
            return err
        if job["status"] in ("completed", "failed", "timed_out", "cancelled"):
            raise HTTPException(status_code=409, detail=f"job already {job['status']}")
        cancelled = await job_registry.cancel(job_id)
        if not cancelled:
            record = job_registry.get(job_id)
            if record is not None and record.task is not None and record.task.done():
                # Task finished between our status check and cancel.
                raise HTTPException(status_code=409, detail=f"job already {job_store.get(job_id)['status']}")
            # Pending (not yet running) → drop from the queue + mark cancelled.
            job_registry.remove_pending(job_id)
            job_store.update_status(job_id, "cancelled")
            job_registry.set_terminal(job_id, "cancelled")
        return {"job_id": job_id, "status": "cancelled"}  # type: ignore[return-value]


async def _cancel_tasks(tasks: set[asyncio.Task]) -> None:
    """Cancel all then await them concurrently, then clear the set (drain path).

    Extracted from ``_shutdown`` so the cancellation discipline is unit-testable
    without spinning up uvicorn. Cancelling all before awaiting lets each task's
    ``finally`` (e.g. ``_run_agent`` releasing the session lock) run concurrently
    with the others. ``return_exceptions=True`` swallows CancelledError/exceptions
    so one stuck task can't abort the drain.
    """
    snapshot = list(tasks)
    for task in snapshot:
        task.cancel()
    if snapshot:
        await asyncio.gather(*snapshot, return_exceptions=True)
    tasks.clear()


def _error_response(status: int, code: str, message: str, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            error=ErrorDetail(code=code, message=message, request_id=getattr(request.state, "request_id", None))
        ).model_dump(),
    )


# G2 default HTTP mode allowlist: everything except yolo. yolo drops the rate
# limiter, the approval gate, and the CHAT/PLAN mode block (only PolicyHook's
# hardcoded safety remains), so it stays opt-in via server.allowed_modes.
_DEFAULT_ALLOWED_MODES = frozenset({"chat", "plan", "act", "auto"})


def _resolve_allowed_modes(raw: object) -> frozenset[str]:
    """Normalize the operator's ``server.allowed_modes``; raise on invalid entries.

    None/empty -> the safe default (all modes except yolo). Otherwise each entry
    must be a valid ``AgentMode`` value; an invalid entry fails loud at startup so
    a YAML typo can't silently widen or narrow the policy boundary.
    """
    if not raw:
        return _DEFAULT_ALLOWED_MODES
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"server.allowed_modes must be a list, got {type(raw).__name__}")
    resolved: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str):
            raise ValueError(f"server.allowed_modes entry must be a string, got {entry!r}")
        try:
            resolved.add(ModeManager.from_string(entry).value)
        except ValueError as exc:
            raise ValueError(f"server.allowed_modes: {exc}") from exc
    return frozenset(resolved) or _DEFAULT_ALLOWED_MODES


def _resolve_mode(
    mode_str: str | None,
    allowed_modes: frozenset[str],
    *,
    allow_yolo: bool,
) -> AgentMode | None:
    """Validate a per-request mode against the operator allowlist.

    Returns None when the caller omitted ``mode`` (config default applies, so the
    config-only path is unchanged). Raises ValueError (-> 400 invalid_mode) for
    an unknown mode, a mode outside ``allowed_modes``, or yolo when ``allow_yolo``
    is False (e.g. jobs).
    """
    if mode_str is None:
        return None
    # ModeManager.from_string raises ValueError ("Unknown mode ...") on bad input.
    mode = ModeManager.from_string(mode_str)
    if mode.value not in allowed_modes:
        raise ValueError(f"mode '{mode.value}' is not allowed; permitted modes: {sorted(allowed_modes)}")
    if mode is AgentMode.YOLO and not allow_yolo:
        raise ValueError("yolo mode is not allowed for autonomous jobs")
    return mode


def _resolve_bind(config: Config, host: str | None, port: int | None) -> tuple[str, int]:
    """Resolve bind host/port: CLI flag > YAML (server.host/server.port) > defaults.

    ``serve_app`` receives ``None`` when the CLI flag is absent, so a YAML
    ``server.host``/``server.port`` (then the hardcoded defaults) takes effect.
    """
    resolved_host = host or config.get("server", "host", default="127.0.0.1")
    resolved_port = port or config.get("server", "port", default=8000)
    return resolved_host, int(resolved_port)


def serve_app(config_path: str | Path, *, host: str | None = None, port: int | None = None) -> None:
    """``koboi serve`` entrypoint: load config, build app, run uvicorn.

    ``host``/``port`` default to ``None`` (CLI flag absent) so YAML ``server.host``
    / ``server.port`` are honored — see ``_resolve_bind``.
    """
    import logging

    import uvicorn

    cfg = Config.from_yaml(config_path)
    resolved_host, resolved_port = _resolve_bind(cfg, host, port)
    if resolved_host not in ("127.0.0.1", "localhost", "::1"):
        # C1: refuse to start a non-loopback server that would fail open.
        if cfg.get("server", "auth_required", default=True) and not _build_key_store(cfg).has_keys:
            raise SystemExit(
                f"Refusing to bind {resolved_host}:{resolved_port}: auth_required=true with no API "
                "keys configured would leave the server fully open. Run `koboi keys create`, set the "
                "KOBOI_API_KEYS env, or set server.auth_required:false only for local dev."
            )
        logging.getLogger(__name__).warning(
            "Binding to %s (non-loopback). Ensure API keys are configured "
            "(KOBOI_API_KEYS or `koboi keys create`) before exposing to untrusted networks.",
            resolved_host,
        )
    app = create_app(cfg)
    uvicorn.run(app, host=resolved_host, port=resolved_port)  # pragma: no cover (blocking server)
