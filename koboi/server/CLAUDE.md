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
(customize by code). `create_app` also accepts externalized-state injection kwargs
(`session_store`/`job_store`/`event_buffer`/`idempotency_store`/`ownership_store`/`approval_registry`,
each default None → in-process impl) — the seam for a future Redis/Postgres backend.
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
peers.py           PeerRegistry (outbound peers + hashed inbound tokens; `peers:` config, SSRF-gated) + `invoke_peer`
                   (shared A2A HTTP path to /v1/peer/invoke, used by the call_peer_agent tool + RemoteAgentProxy)
mcp_registry.py    SessionMcpRegistry -- in-process per-session MCP server attach/detach/reconnect (/v1/sessions/{id}/mcp/servers)
idempotency.py     IdempotencyRegistry -- in-memory TTL for /chat/stream Idempotency-Key (409-reject)
schema.py          Pure-Pydantic-v2 request/response models + ErrorResponse/ErrorDetail envelope
sse.py             Pure SSE wire encoder -- keepalive on silence, always terminates data: [DONE]
health.py          HealthRegistry + /readyz checks (pool alive, DB ping)
middleware.py      X-Request-Id middleware (mint/honor/echo; stashed on request.state)
keys_cli.py        `koboi keys create|list|revoke|rotate` -> ~/.koboi/keys.json (hashed, 0600 atomic writes)
protocols.py       M5 forward-only Protocols (SessionStore/LockProvider/EventBuffer) for a future Redis/SaaS swap
session_events.py  SessionEventRegistry -- per-session capped B2 replay buffer (monotonic seq; get_events_since)
workflow_store.py  WorkflowStore -- SQLite owner-scoped workflow bundles + cache sidecar (drives /v1/workflows)
```

## Endpoints
```
GET    /healthz                       Liveness (always open)
GET    /readyz                        Readiness -- pool + DB checks, 503 if any fail (always open)
POST   /v1/sessions                   Create session
GET    /v1/sessions                   List sessions (owner-scoped when auth on; fail-closed 401)
GET    /v1/sessions/{id}              Messages (owner-checked)
DELETE /v1/sessions/{id}              Evict from pool + clear DB rows (under existing_session_lock)
POST   /v1/sessions/{id}/fork         Fork persisted messages into a new session (sqlite only)
POST   /v1/sessions/{id}/resume       Resume interrupted session (journal rehydrate)
GET    /v1/sessions/{id}/mcp/servers  List a session's attached MCP servers (in-process)
POST   /v1/sessions/{id}/mcp/servers  Attach an MCP server to a session (in-process)
DELETE /v1/sessions/{id}/mcp/servers/{sid}  Detach a session MCP server
POST   /v1/sessions/{id}/mcp/servers/{sid}/reconnect  Reconnect a session MCP server
POST   /v1/chat/stream                Interactive SSE chat (lock + HITL + idempotency + per-request mode/cap)
POST   /v1/peer/invoke                A2A inbound receiver (sync JSON; peer-token auth via peers.inbound_tokens; AutonomousApprovalHandler; ephemeral session)
POST   /v1/sessions/{id}/approve      Resolve a pending HITL approval
POST   /v1/sessions/{id}/transfer     Reassign session ownership to a human operator (handover take-over)
GET    /v1/sessions/{id}/stream        Replay buffered events (B2: history + B4 warm-handoff digest) for an operator
POST   /v1/jobs                       Submit autonomous job (202; admission + idempotency; `workflow_ref` + `replay_mode` body fields)
GET    /v1/jobs                       List caller's jobs
GET    /v1/jobs/{id}                  Job status (result/error/retriable)
GET    /v1/jobs/{id}/stream           SSE replay + live tail (per-owner stream cap)
POST   /v1/jobs/{id}/cancel           Cancel pending/running job
POST   /v1/workflows                  Import a workflow bundle (201; owner-scoped SQLite store)
GET    /v1/workflows                  List caller's workflows
GET    /v1/workflows/{name}           Show a workflow bundle
DELETE /v1/workflows/{name}           Delete a workflow
POST   /v1/jobs/{id}/capture          Capture a completed workflow_ref job into a bundle + cache sidecar
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
- **DELETE `/v1/sessions/{id}` holds `pool.existing_session_lock(session_id)`** (NOT
  `session_lock` — it does NOT `get_or_create`/materialize an agent) across the evict +
  DB-row clear + ownership-delete, so a concurrent `/chat/stream` finishes first and can't
  re-insert orphaned unowned rows. `list_sessions`/`delete_session`/`fork_session` all
  self-heal schema via `_ensure_schema_on` (safe on older DBs). Fork rolls back its
  committed DB+owner rows on any `get_or_create` failure.
- **Per-session sandbox workdir** (`workdir_for(session_id)` = `{workspace_root}/{id}`);
  `session_id` validated at the route boundary AND in `workdir_for` (defense-in-depth).
  Eagerly `mkdir`-ed; GC'd at `server.workdir_ttl_seconds`.
- **Jobs reject yolo** (`allow_yolo=False` regardless of allowlist) and **require
  `sandbox.backend='restricted'`** (passthrough refused at execution).
- **Job webhooks** (`jobs.webhooks`): on a terminal status (`completed`/`failed`/
  `timed_out`/`cancelled`), `run_job` fire-and-forgets an HTTP POST (via `_emit_job_webhooks`
  → `httpx.AsyncClient`, retries on 5xx/network error, fail-safe logs only) to each
  matching webhook URL, AFTER `set_terminal` (so the queue isn't blocked). Payload =
  job row (`result` parsed from `result_json`, redacted `error`). `secret` HMAC-SHA256-signs
  the body (`X-Koboi-Signature`). Operator-configured URLs only (never tenant); payload
  carries tenant `result` → use HTTPS + a secret. Config is threaded `create_app` →
  `_register_routes(job_webhooks=...)` → `_start_job` (NOT a create_app closure: routes
  live in `_register_routes`). v1: terminal statuses only.
- **Handover flow** (`handover:` config, PR #40): the `transfer_to_human` tool /
  `HandoverDetectionHook` raise `AgentHandoverError`; `_run_agent` (SSE) converts it to a
  `HandoverEvent` and `run_job` converts it to an `awaiting_human` terminal status. No Future
  is awaited, so `pool.session_lock` releases when the run ends and a human operator takes over
  via `POST /v1/sessions/{id}/transfer` (ownership) + `GET /v1/sessions/{id}/stream` (B2 replay:
  history + B4 digest) + a new `POST /v1/chat/stream`. `handover.webhooks` fire HMAC-signed
  `handover.requested` callbacks mid-conversation (mirror of `jobs.webhooks`); `handover.digest.enabled`
  generates the warm-handoff summary. **PR #57 robustness:** the HandoverEvent is queued to the
  operator (`await queue.put`) *before* the webhook is scheduled, and the webhook's payload/HMAC/POST
  run inside a fire-and-forget task — so a webhook config error (e.g. `timeout: 10s` -> ValueError) can
  never drop the handover event from the operator's stream; the LLM-provided B1 summary is scrubbed via
  `redact_value` before emission (the B4 digest path already redacts); the B2 replay cursor is a monotonic
  sequence (`SessionEventRegistry.get_events_since`), not a list index, so it survives the 1000-event buffer trim.
  See `docs/channel-bridge.md` for the omnichannel surfaces.
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
- **Orchestrated configs (`execution.mode: dynamic|dag|deep_research`) build `core=None`** —
  `KoboiAgent(core=None, orchestrator=...)`. The orchestrator manages its own per-node agents, so
  there's no `AgentCore`/HITL pipeline. Every `_core` access is guarded:
  - `/v1/chat/stream` (`app.py` B1 guard): `if agent._core is not None:` wraps the approval-handler
    build + mode/`max_iterations` snapshot/restore; `handler` stays `None` for core=None.
  - `/v1/jobs` middle path (`jobs.py`): `if agent._core is None:` → config-level
    `sandbox.backend='restricted'` check (passthrough refused) → `agent.run_stream()` / `agent.resume()`.
    Job results come from `OrchestrationCompleteEvent.final_answer` (NOT `CompleteEvent` — distinct class).
  - `GET /v1/sessions/{id}` (`pool._deep_research_messages`): surfaces the query + cited report from
    the session-tagged `research_context` table (returns `[]` only when no research run exists).
  - `pool.py` `_client_factory`/`_approval_handler` seams are guarded too (skipped for core=None).
  See `docs/deep-research-smoke.md` for the production bar.
- **Workflow export/store** (`/v1/workflows`, PR #42): owner-scoped SQLite bundles (`workflow_store.py`).
  `POST /v1/jobs` accepts `workflow_ref` (run a stored bundle as a job) + `replay_mode`
  (`live`/`cache`/`replay`); `POST /v1/jobs/{id}/capture` freezes a completed `workflow_ref` job into a
  bundle + cache sidecar. Plain (non-`workflow_ref`) jobs cannot isolate a run cache — only `workflow_ref`
  jobs freeze a per-job sidecar. The CLI store is filesystem (`KOBOI_WORKFLOWS_DIR`), separate from this
  SQLite store. See `koboi/workflows/CLAUDE.md`.
