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
from koboi.server.idempotency import IdempotencyRegistry
from koboi.server.jobs import JobRegistry, JobStore, new_job_id, resume_on_startup, run_job
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
    JobStatusResponse,
    JobSubmitRequest,
    ReadyzCheck,
    ReadyzResponse,
    SessionDeletedResponse,
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
    # G5a: per_tenant_max is enforced only when real auth is configured (dev mode → owner "dev").
    auth_enabled = key_store.has_keys
    # C1: honor server.auth_required (default true). When true and no keys are
    # configured, the auth middleware fails closed (401) instead of serving open.
    auth_required = config.get("server", "auth_required", default=True)

    # M3: session ownership + M4: job store. Control-plane state persists to a file
    # so ``resume_on_startup`` works; see ``_sidecar_db_path`` for the resolution rules.
    memory_backend = config.get("memory", "backend", default="sqlite")
    shared_db = _sidecar_db_path(memory_backend, config.get("memory", "db_path"))
    ownership = OwnershipStore(db_path=shared_db)
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
    job_resume = config.get("jobs", "resume_on_startup", default=True)
    job_registry = JobRegistry(max_events=job_max_events)
    # G6: /chat/stream Idempotency-Key (409-reject, in-memory TTL).
    chat_idem_ttl = config.get("server", "idempotency", "chat_ttl_seconds", default=600.0) or 600.0
    chat_idem = IdempotencyRegistry(ttl_seconds=chat_idem_ttl)

    health = HealthRegistry()
    health.register("pool", make_pool_alive_check(pool))
    health.register("db", make_db_check(ownership, backend=memory_backend))

    drain_seconds = config.get("server", "timeouts", "drain_seconds", default=60.0) or 60.0

    async def _shutdown():
        """Cancel job tasks, close pool/ownership/store (M5: drain with timeout)."""
        job_registry.cancel_all()
        await pool.close_all()
        ownership.close()
        job_store.close()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # M4: resume pending jobs on startup (simplified: running→failed).
        if job_resume:
            requeued = await resume_on_startup(job_store, pool, job_registry, job_timeout)
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

    app = FastAPI(title=f"koboi-{config.agent_name}", version="0.5.0", lifespan=lifespan)
    app.state.pool = pool
    app.state.approvals = approvals
    app.state.ownership = ownership
    app.state.job_store = job_store
    app.state.job_registry = job_registry
    app.state.job_max_concurrent = job_max_concurrent
    app.state.job_timeout = job_timeout
    app.state.health = health

    # C4: CORS is config-driven, never a wildcard default. CORSMiddleware is
    # added ONLY when `server.cors` is explicitly configured; the default (no
    # `cors:` block) adds nothing → no cross-origin reads. An operator who wants
    # open CORS sets `cors: {allow_origins: ["*"]}` explicitly.
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

    # Middleware: registration order is the REVERSE of execution order.
    # request_id is registered LAST → executes FIRST (outermost), wrapping auth
    # so 401/403 responses carry X-Request-Id.
    app.middleware("http")(make_auth_middleware(key_store, auth_required=auth_required))
    for mw in extra_middleware:
        app.middleware("http")(mw)
    app.middleware("http")(request_id_middleware)

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
        is_new_session = header_sid is None and pool.get(session_id) is None
        if header_sid is not None:
            err = _check_owner(ownership, session_id, request)
            if err:
                return err

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
        # 16.21: enrich Langfuse trace with serving context.
        _enrich_trace(agent, mode="interactive", request_id=getattr(request.state, "request_id", ""), owner=owner)

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

    # ---- M4: Jobs (autonomous) ----

    def _start_job(job_id: str) -> None:
        """Admit one job: mark running, spawn run_job, attach drain-on-complete."""
        job = job_store.get(job_id)
        if job is None or job["status"] != "pending":
            return  # cancelled or reaped while queued
        task = asyncio.create_task(run_job(job_id, pool, job_registry, job_store, job["message"], job_timeout))
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

        # Idempotency: same key within window → return existing job.
        idem_key = request.headers.get("Idempotency-Key")
        if idem_key:
            existing = job_store.find_by_idempotency_key(idem_key)
            if existing and existing["owner"] == owner:
                return {
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
        else:
            is_new = pool.get(session_id) is None
            try:
                await pool.get_or_create(session_id)
            except PoolFull as exc:
                return _error_response(429, "pool_full", str(exc), request)
            if is_new:
                ownership.set_owner(session_id, owner)

        job_id = new_job_id()
        job_store.insert(job_id, session_id, owner, body.message, idempotency_key=idem_key)
        job_registry.register(job_id, session_id, owner)
        if admit == "run":
            _start_job(job_id)
        else:  # "queue" — wait for a running slot to free (drained on completion)
            job_registry.enqueue_pending(job_id)
        return {"job_id": job_id, "status": "pending", "session_id": session_id}

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
        return JobStatusResponse(
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

        async def event_gen():
            last_index = 0
            try:
                while True:
                    if record:
                        events = record.events[last_index:]
                        last_index = len(record.events)
                        for ev in events:
                            yield ev
                        if record.terminal.is_set():
                            break
                    else:
                        break
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass

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
        return {"job_id": job_id, "status": "cancelled"}


def _error_response(status: int, code: str, message: str, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            error=ErrorDetail(code=code, message=message, request_id=getattr(request.state, "request_id", None))
        ).model_dump(),
    )


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
