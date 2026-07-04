# koboi/server/ -- REST/SSE serving layer

## What this is
FastAPI app that serves the koboi agent over HTTP/SSE: **interactive SSE chat with
human-in-the-loop (HITL) approvals** and **autonomous background jobs** (no HITL,
durable resume). API-key auth, per-session ownership, idempotency, graceful drain.
Activated by the `[api]` extra (`fastapi`, `uvicorn`); pure submodules
(`sse`, `schema`, `pool`, `health`, `middleware`, `idempotency`, `keys_cli`,
`protocols`) unit-test without it.

Two entrypoints, same composition: **`koboi serve <config>`** (built-in, zero code)
and **`create_app(config, extra_tools=..., extra_hooks=..., approval_handler=...)`**
(customize by code).

## Key files
```
__init__.py        Public surface: create_app, serve_app, AgentPool, PoolFull, sse_stream, DONE_FRAME
app.py             Composition root -- create_app() wires every subsystem + registers routes; serve_app() (koboi serve)
pool.py            AgentPool -- per-session KoboiAgent cache + per-session asyncio.Lock + per-session workdir + LRU eviction + flush_langfuse
jobs.py            JobStore (SQLite 'jobs') + JobRegistry (in-memory task+events+admission) + run_job + resume_on_startup
approvals.py       HITL bridge -- ApprovalCoordinator (PendingApprovalEvent -> SSE queue -> awaits Future) + ApprovalRegistry
auth.py            KeyStore (file+env, SHA-256 hashed) + make_auth_middleware (Bearer, fail-closed)
ownership.py       OwnershipStore -- SQLite 'session_owners' sidecar (session_id -> owner) for tenant isolation
idempotency.py     IdempotencyRegistry -- in-memory TTL for /chat/stream Idempotency-Key (409-reject)
schema.py          Pure-Pydantic-v2 request/response models + ErrorResponse/ErrorDetail envelope
sse.py             Pure SSE wire encoder -- keepalive on silence, always terminates data: [DONE]
health.py          HealthRegistry + /readyz checks (pool alive, DB ping)
middleware.py      X-Request-Id middleware (mint/honor/echo; stashed on request.state)
keys_cli.py        `koboi keys create|list|revoke|rotate` -> ~/.koboi/keys.json (hashed, 0600 atomic writes)
protocols.py       M5 forward-only Protocols (SessionStore/LockProvider/EventBuffer) for a future Redis/SaaS swap
```

## Endpoints
```
GET    /healthz                       Liveness (always open)
GET    /readyz                        Readiness -- pool + DB checks, 503 if any fail (always open)
POST   /v1/sessions                   Create session
GET    /v1/sessions/{id}              Messages (owner-checked)
DELETE /v1/sessions/{id}              Evict from pool
POST   /v1/sessions/{id}/resume       Resume interrupted session (journal rehydrate)
POST   /v1/chat/stream                Interactive SSE chat (lock + HITL + idempotency + per-request mode/cap)
POST   /v1/sessions/{id}/approve      Resolve a pending HITL approval
POST   /v1/jobs                       Submit autonomous job (202; admission + idempotency)
GET    /v1/jobs                       List caller's jobs
GET    /v1/jobs/{id}                  Job status (result/error/retriable)
GET    /v1/jobs/{id}/stream           SSE replay + live tail (per-owner stream cap)
POST   /v1/jobs/{id}/cancel           Cancel pending/running job
```

## The two execution modes
- **Interactive** (`/v1/chat/stream`): SSE + per-session lock + HITL. A tool needing
  approval -> `AsyncCallbackApprovalHandler` -> `PendingApprovalEvent` on the stream ->
  awaits a Future resolved by `POST /v1/sessions/{id}/approve` (120s timeout -> deny).
- **Autonomous job** (`POST /v1/jobs`): 202, no HITL, `AutonomousApprovalHandler`
  (deny-by-default on destructive), restricted sandbox mandatory, `resume_on_startup`
  (running -> failed `InterruptedByRestart`/retriable; pending -> requeued fresh).

## Conventions
- **Error envelope**: `_error_response(status, code, message, request)` -> `JSONResponse(ErrorResponse(...))`
  on every error path. Codes are snake_case (`pool_full`, `invalid_mode`,
  `duplicate_request`, `queue_full`, `too_many_jobs_per_tenant`, `forbidden`).
  `HTTPException` only for true 404s/409s.
- **Config via dotted paths**: `config.get("server", "auth_required", default=True)`,
  `config.get("jobs", "max_concurrent", default=64)`, `config.get("server", "limits",
  "max_iterations_cap", default=25)`. Pydantic `ServerConfig`/`JobsConfig` are cosmetic
  (validation only); dotted-path reads are the runtime truth.
- **Container env hooks** (Dockerfile): `KOBOI_CONFIG`/`KOBOI_HOST`/`KOBOI_PORT` drive the
  `koboi serve` entrypoint; `KOBOI_EXTENSIONS_DIR` is added to `sys.path` at `import koboi`
  (`koboi/_extensions_path.py`) so mounted custom modules (`tools.custom`/`rag.custom_modules`/
  `context.custom_modules`) are importable. For full code customization, derive `FROM` the image
  with a `create_app(extra_tools=…, extra_routes=…)` entrypoint (see `examples/docker/`).
- **SSE protocol**: `data: {json}\n\n` via `event_to_dict()`; keepalive comment every
  15s (resets Cloudflare's ~100s idle); always ends `data: [DONE]\n\n` (after an
  `ErrorEvent` on failure).
- **G2 per-request knobs**: `mode`/`max_iterations` on `/v1/chat/stream` + `/v1/jobs`
  bodies; validated against `server.allowed_modes` (default = chat/plan/act/auto;
  yolo opt-in), clamped to `server.limits.max_iterations_cap`.

## Gotchas
- **AgentCore is NOT concurrent-safe** -- each session gets its own `asyncio.Lock`;
  the lock is acquired inside the stream generator so its lifetime == the stream's.
  Install the per-run approval handler UNDER `pool.session_lock`.
- **Per-session sandbox workdir** (`workdir_for(session_id)` = `{workspace_root}/{id}`);
  `session_id` validated at the route boundary AND in `workdir_for` (defense-in-depth).
  Eagerly `mkdir`-ed; GC'd at `server.workdir_ttl_seconds`.
- **Jobs reject yolo** (`allow_yolo=False` regardless of allowlist) and **require
  `sandbox.backend='restricted'`** (passthrough refused at execution).
- **Pooled agent state must be restored**: snapshot `mode`/`max_iterations`/`_tool_pipeline`/
  `approval_handler` before a run, restore in `finally` (agent is reused; without restore
  a later `mode=None` request inherits this request's mode).
- **`agent.close()` does NOT flush Langfuse** -- the hook flushes on SESSION_END from
  the loop, which never fires on shutdown, so `_shutdown` calls `pool.flush_langfuse()`
  off-loop + concurrently (`asyncio.to_thread` + `gather`); a slow Langfuse server
  can't pin the drain.
- **Drain order** (`_shutdown`): cancel in-flight stream tasks -> flush langfuse ->
  cancel_all jobs -> close_all pool -> close ownership/store, under
  `wait_for(drain_seconds)`.
- **Middleware order is REVERSE of execution**: register auth FIRST (innermost),
  request_id SECOND, CORS LAST (outermost via `add_middleware`). Getting this wrong
  breaks preflight handling and loses `X-Request-Id` on 401/403.
- **`auth_required=true` fails closed**: no keys + auth_required -> 401, never open;
  `serve_app` refuses non-loopback bind with auth_required + no keys.
- **Output guardrail buffering lives in `loop.py`** (not here): when output guardrails
  are configured, TextDeltas buffer until `_process_output` passes (G8). Transparent to
  this package -- `sse_stream` just sees delayed deltas.
- **M5 seams**: `protocols.py` defines SessionStore/LockProvider/EventBuffer for a
  future Redis/SaaS swap; do NOT pre-protocol-ize for one impl.
