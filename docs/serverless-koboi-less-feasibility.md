# Koboi-Less — Serverless Feasibility & Reference Plan

**Status:** Research / feasibility (draft)
**Date:** 2026-07-20
**Scope:** Analyze whether koboi-agent can run serverlessly ("koboi-less": no always-on Docker standby container, scale-to-zero, pay-per-use), and produce a phased reference plan.

---

## 1. TL;DR

Split "serverless" into two distinct goals — their feasibility is opposite:

| Goal | Meaning | Feasibility | Effort |
|---|---|---|---|
| **A. Scale-to-zero container** | Wrap the *existing* Docker image; platform sleeps it when idle, wakes on request. No standby cost. | **✅ Feasible today** | **Low** (days) |
| **B. True function-serverless** | Lambda/Workers per request; fully ephemeral, zero process affinity, state 100% external. | **⚠️ Partial / hybrid only** | **High** (weeks–months) |

**The actual pain (always-on Docker billing) is solved by (A) now.** (B) is a real migration with a hard ceiling: long agent loops, the sandbox, and subprocess tools will *always* need a container somewhere — so the end-state of (B) is a **hybrid (edge function + container executor)**, never pure Lambda.

**Most important finding:** koboi's architecture is **already designed for this** — the externalized-state seams exist; only the externalized backends aren't shipped yet. It's an **engineering gap, not an architectural dead-end.**

**Recommended path:** Wave 1 (separate thin deploy repo, toggle `min-instances=0`, persistent volume for SQLite, **zero engine change**) → Wave 2 (additive Postgres/Redis backends, only when multi-instance parallel is needed) → Wave 3 (engine execution-model changes, only for true function-serverless).

---

## 2. Problem Statement

Today the operator must keep a Docker container **always-on (standby)** so clients can use koboi's features on demand. This is a fixed 24/7 cost regardless of traffic. The question: **can koboi run serverlessly** (scale-to-zero, wake-on-request, no standby cost) — and if so, how, and what needs to change?

Secondary requirement (clarified in discussion): the solution must be **mode-flexible** — able to run standby *or* serverless from the same codebase, adjustable per environment/use-case.

---

## 3. Current-State Analysis (code-grounded)

### 3.1 Why a standby container is needed today

Two things pin koboi to a long-lived process:

**(a) In-process execution of background work.** The job system persists *metadata* to SQLite, but the **actual execution is an `asyncio.Task` inside the server process**:

- `koboi/server/jobs.py:628` — `async def run_job(...)`, spawned via `asyncio.create_task` at `jobs.py:1064` & `:1088`
- Webhooks: `jobs.py:458` `_WEBHOOK_TASKS`, fired at `:573` `asyncio.create_task(_deliver_webhooks(...))`
- Job GC loop: `app.py:449` `job_gc = asyncio.create_task(_job_ttl_gc_loop(...))`

→ If the process dies, the in-flight task dies with it. That is *why* it's kept warm.

**(b) State defaults are SQLite-on-local-disk + in-memory.**

- SQLite-backed (persistent, but **local to the container**): `JobStore` (`jobs.py:72`), `OwnershipStore` (`ownership.py:14`), memory (`memory_sqlite.py`), the `steps` journal, workflow store.
- **Purely in-process**: session pool (`pool.py`), per-session `asyncio.Lock` (`pool.session_lock`), `JobRegistry` event buffer, idempotency store, MCP client connections, response cache, RAG index.

Local SQLite on an ephemeral serverless filesystem is the breaker — no shared volume across instances.

### 3.2 The good news — the seams already exist

This is what makes "koboi-less" realistic rather than a rewrite:

1. **`create_app` already accepts externalized-state injection** (`server/app.py:239–264`):
   ```
   session_store, job_store, event_buffer, idempotency_store,
   ownership_store, approval_registry, session_event_buffer
   ```
   Each defaults to `None` → in-process impl. Pass your own and it's used as-is (`app.py:276–354`).

2. **Protocols define the swap surface** (`server/protocols.py`): `SessionStore`, `EventBuffer`, `IdempotencyStore`; plus `MemoryBackend` (`memory.py:12`). Codebase comments are explicit this was pre-designed:
   - `protocols.py:3` — *"These Protocols define the public surface that a future Redis-backed ... store would [implement]"*
   - `pool.py:15` — *"once a second (Redis) backend exists — do NOT pre-protocol-ize for one impl"*

3. **No Redis/Postgres backend is shipped yet.** Every `redis/postgres` hit is a "future" comment (`protocols.py:9`, `jobs.py:311`, `app.py:255`, `trust.py:64`). The gap is literally: *write the backends the seams were built for.*

4. **Crash/resume already works** — the `steps` journal + `koboi run --resume` rehydrates an interrupted session, and the workflow `replay` mode runs *fully offline* (no API key, raises on cache miss). A cold-start mid-session is survivable, not catastrophic.

---

## 4. Execution-Model Constraints

| Constraint | Source | Serverless implication | Severity |
|---|---|---|---|
| Long agent loops | `max_iterations_cap` default 25/turn; deep-research unbounded; orchestration DAG; `generate_video` 1800s timeout | Blows past Lambda 15-min & Workers CPU limits | **HARD** for functions, OK for containers |
| Per-session single-writer | `AgentCore` not concurrent-safe → one `asyncio.Lock` per session (`pool.session_lock`) | Needs per-session coordination primitive (Durable Object / DB advisory lock) | **SOFT** (replaceable) |
| SSE holds socket for whole loop | `run_stream` buffers TextDeltas until output guardrail passes (G8) | A function can't hold a socket 10 min; B2 replay buffer (`GET /v1/sessions/{id}/stream`) gives partial resumability | **SOFT→HARD** |
| Subprocess/sandbox | `run_shell`, git, filesystem, seccomp; `AutonomousApprovalHandler` requires `sandbox.backend='restricted'` | Needs a real OS; impossible on Workers, fine on containers | **HARD** for functions |
| MCP stdio spawns child processes | `koboi/mcp/` | Process-bound; HTTP/SSE MCP is fine, stdio isn't | **SOFT** (prefer HTTP MCP) |
| Background work in-process | jobs, webhooks, GC loop are `asyncio` tasks | Must move to queue + external worker | **SOFT** (engineering) |
| Cold start | Python + subsystem assembly + optional RAG/embedding load | First request pays seconds→tens of seconds; *why* Docker is kept warm | **SOFT** (mitigatable) |

The two genuinely **HARD** constraints — long loops + arbitrary subprocess — mean pure Lambda/Workers can never be the whole story. A container executor is unavoidable for the heavy paths.

---

## 5. Platform Landscape

| Platform | Long loops | SSE | Real Python | Scale-to-zero | External state | Sandbox/subprocess | Cold start | Fit |
|---|---|---|---|---|---|---|---|---|
| **Modal** | ✅ (hours) | ✅ | ✅ native | ✅ | needs glue | ✅ real containers | low (sub-second) | **Best single fit** |
| **Cloud Run** | ✅ (≤60min req) | ✅ | ✅ | ✅ | needs glue | ✅ container | medium (~10–30s) | **Best default** |
| **Fly Machines** | ✅ | ✅ | ✅ | ✅ (suspend/resume) | needs glue + volumes | ✅ microVM | medium (~5–10s resume) | "warm-per-session" |
| **Cloudflare Containers** | ✅ | ✅ | ✅ | ✅ | R2/KV/D1 | ✅ | medium | Strong |
| **CF Workers + DO + Workflows** | ⚠️ CPU limits | ✅ (WS hibernate) | ⚠️ Py not GA | ✅ | ✅ DO/D1 native | ❌ (need Containers) | very low (edge) | Hybrid edge only |
| **AWS Lambda + Step Fn + DynamoDB** | ⚠️ 15min | ⚠️ (streaming/API GW WS) | ✅ | ✅ | ✅ | ❌ | medium (~1–3s) | Chat-only subset |

### Shortlist

- **Modal** — fastest scale-to-zero win; real Python, sub-second cold start, subprocess/sandbox survive. **Wave 1–2.**
- **Cloud Run / Fly Machines** — production-grade defaults; scale-to-zero; full container. **Wave 1.**
- **Cloudflare (Workers + DO + Workflows + Containers)** — true-serverless-agent end-state (DO = the `asyncio.Lock` replacement); highest ceiling, highest effort; confirm Python-GA status. **Wave 3 hybrid.**

### ⚠️ AWS Lambda calibration

An earlier research draft claimed a "Dec-2025 AWS Lambda Durable Functions" + "Lambda MicroVMs (8h suspend)" capability, sourced from a community blog. This appears to be a **conflation**:
- "Durable Functions" is an **Azure** concept (Azure Durable Functions, GA). AWS does **not** ship a product by that name.
- AWS's durable-orchestration primitive is **Step Functions** (Standard Workflows = state, retries, checkpoint-like behavior).
- Lambda invocations are **ephemeral**; mid-flight checkpoint/resume is **not** a GA Lambda feature. Firecracker suspend/resume is a VM-level capability, not an exposed Lambda product feature.

→ Lambda remains **weak for koboi** (15-min cap, no subprocess, no native single-writer). The *real, GA* AWS pattern is **Step Functions + Lambda + DynamoDB**. Down-rank any "Lambda 75% fit" claim until confirmed from an official AWS source.

### Recurring patterns (how other frameworks solve serverless + stateful agents)

1. **Durable execution + checkpoint/resume** — LangGraph checkpointers (DynamoDB), Cloudflare Workflows (fibers), Inngest, AWS Step Functions. Koboi already has this pillar via the `steps` journal.
2. **Stream-via-storage + polling** — Vercel AI SDK; write chunks to KV/S3, client polls.
3. **Hibernating WebSockets** — Cloudflare DO hibernation API; connection survives scale-to-zero.
4. **Externalized coordinator** — DO single-writer, Azure Durable Entities, or Redis/DB advisory locks (replaces `pool.session_lock`).

---

## 6. Architecture: the 3 Layers (where code lives)

This is the key clarity: **mode (standby vs serverless) is a deploy-time setting, not an engine property.** The engine doesn't know or care whether it's warm or cold.

| Layer | What changes | Where the code lives | Engine change? | Depends on koboi image? |
|---|---|---|---|---|
| **L1 — Deploy wrapper** (Wave 1) | Deploy the *existing* image with `min-instances=0`, mount persistent volume for SQLite, light config | **Separate repo** (sibling, like `koboi-use-cases`) | **No** | Yes — consumes image/pkg |
| **L2 — State backends** (Wave 2) | Implement `PostgresSessionStore` / `RedisEventBuffer` etc. satisfying Protocols | **Additive** — new files, slot into `create_app()` seams. In core behind extra `[postgres]` OR separate OSS pkg | **No** (additive; core logic untouched) | Inverted: backend depends on koboi |
| **L3 — Execution model** (Wave 3) | Move job exec off in-process `asyncio.Task` to queue+worker; replace `pool.session_lock` with distributed/DO lock; resumable SSE cross-instance | **Inside the engine** (`server/jobs.py`, `server/pool.py`, `loop.py`) | **Yes** — direct | — |

### Why Wave 1 needs no engine change

"Standby Docker" today is effectively `min-instances=1, always-on`. Wave 1 just flips it to `min-instances=0, wake-on-request` on the **same image**. The engine runs identically warm or cold, and the `steps` journal + `koboi run --resume` already survive cold restarts. The only thing to preserve is **SQLite not on ephemeral FS** — point `db_path` at a persistent volume (Modal/Fly volume, Cloud Run disk, Litestream-to-S3, or Turso). That's config, not engine change.

### Why L2 backends are additive, not a rewrite

`create_app` already accepts `session_store=`, `job_store=`, etc., and Protocols already define the contract. Writing a Postgres backend = adding new files that fill an existing contract — the core loop (`loop.py`, `loop_pipeline.py`) is **not** restructured. Consistent with the open-core strategy (ship Redis/Postgres state backends in OSS).

### Why L3 must touch the engine

Externalizing job execution to a queue+external worker, replacing the per-session asyncio.Lock with a distributed/DO lock, and making SSE resumable across instances all touch how koboi **runs** — these live inside `server/jobs.py`, `server/pool.py`, `loop.py`. They can't be done purely from outside via injection. Only needed for Wave 3.

---

## 7. Dual-Mode Flexibility (standby ⇄ serverless)

Validated requirement: the solution must support **both** modes, adjustable per environment/use-case.

Because mode lives in the **separate deploy repo (L1)** — not the engine — flexibility is "free." One image, one config, multiple deploy presets:

```
koboi-serverless-deploy/
├── image/            # single Dockerfile → build once
├── configs/
│   └── agent.yaml    # koboi config (db_path → persistent volume)
└── deploy/
    ├── standby.yaml      # min-instances: 1   ← prod, low latency
    ├── serverless.yaml   # min-instances: 0   ← dev/low-traffic, cheap
    └── hybrid.yaml       # standby front + serverless job-worker
```

| Mode | min-instances | Best for | Trade-off |
|---|---|---|---|
| **Standby** | 1 (always-on) | Prod needing stable low latency | Pays idle 24/7 |
| **Serverless** | 0 (scale-to-zero) | Dev, staging, low/intermittent traffic | Cold start on first request |
| **Hybrid** | front=1, worker=0 | Fast chat/SSE + cheap heavy jobs | Slight extra setup |

Also supports **per-environment** (prod=standby, staging=serverless) and **per-route/per-session** (interactive → warm instance; background `generate_video` → scale-to-zero worker).

### The one honest boundary

Dual-mode toggling is **free as long as max 1 active instance at a time** (standby, or serverless scaling 0→1). That covers ~90% of "stop standby Docker."

If serverless must **scale to N>1 concurrent instances**, you hit the **SQLite single-writer wall** — multiple instances writing the same SQLite concurrently = contention/corruption. At that point you need **L2 (external Postgres/Redis backends)** — not just a toggle.

- **Standby ⇄ Serverless (0↔1 instance): flexible, toggle only, engine untouched.** ✅
- **Serverless multi-instance parallel: needs L2 (external backends).** ⚠️ (= Wave 2)

---

## 8. Phased Roadmap

### Wave 1 — "No standby cost" (days, low risk)
**Goal:** kill the always-on bill without touching the engine.
- Deploy the current image to **Cloud Run / Modal / Fly** with `min-instances=0`.
- Move SQLite off ephemeral FS: point `memory.backend` + job/ownership stores at persistent storage (Postgres via existing seams, **or** Modal/Fly volume / Litestream / Turso for SQLite-on-object-storage — the volume route needs no new backend).
- Accept cold-start latency on first hit; use HTTP MCP (not stdio) so connections aren't process-bound.
- **Engine changes:** none. **Where:** separate deploy repo (L1).

### Wave 2 — Externalize state (weeks, medium)
**Goal:** jobs/sessions survive any single instance; enable multi-instance parallel.
Trigger: only when N>1 concurrent instances are required.
- Ship the `Redis`/`Postgres` backends the Protocols were built for (`SessionStore`, `EventBuffer`, `IdempotencyStore`, `MemoryBackend`).
- Move job execution off in-process asyncio to **queue + worker** (Cloud Run Jobs / Modal / CF Queues + Workflows). The `steps` journal already provides resume checkpoints.
- Replace per-session `asyncio.Lock` with a DB advisory lock / Durable Object so any instance can take over a session safely.
- **Engine changes:** additive (L2) for backends; the queue+worker hand-off begins to lean into L3.

### Wave 3 — True hybrid serverless (months, high)
**Goal:** edge function + container executor.
- Cloudflare Workers (or Lambda) for routing + SSE + short stateless chat (cheap, fast cold start).
- Delegate long loops / sandbox / `run_shell` / video generation to **CF Containers / Modal / Cloud Run Jobs**.
- State in Durable Objects + D1/R2; durable multi-step via Workflows / Step Functions.
- **Engine changes:** yes (L3) — execution model in `server/jobs.py`, `server/pool.py`, `loop.py`.

---

## 9. Honest Limits (what will NEVER be pure-Lambda)

- **Arbitrary subprocess execution** (`run_shell`, `git_*`, filesystem) and **seccomp sandbox** require a real OS → always a container/VM tier.
- **Multi-minute / unbounded loops** (deep research, DAG, `generate_video`) exceed function CPU/time limits → always a container/VM tier.

→ "koboi-less" realistically means **"no always-on container; containers spin up on demand"** — which Wave 1 (scale-to-zero) already delivers.

---

## 10. Open Decisions / Next Steps

1. **Wave 1 target platform** — Modal (fastest) vs Cloud Run (production default) vs Fly (warm-per-session). Decide based on latency tolerance + existing cloud.
2. **SQLite persistence strategy** — volume/network-disk (no new backend, fastest) vs Postgres backend (L2, more portable). Recommend volume for Wave 1.
3. **Repo structure** — confirm the `koboi-serverless-deploy` sibling-repo pattern (mirrors `koboi-use-cases`).
4. **Cold-start budget** — acceptable first-request latency → drives min-instances floor and import-path optimization.
5. **Defer L3** until true function-serverless or multi-instance parallel is actually required.

### Concrete next action (proposed)
Scaffold the **Wave 1 L1 repo** — directory structure + Dockerfile (Modal/Cloud Run) + example config pointing `db_path` to a volume + 3 deploy presets (standby / serverless / hybrid) — to make the thinness of this layer concrete and immediately toggleable.

---

## 11. Sources & Grounding

**Code-grounded (koboi side — verified 2026-07-20):**
- Externalized-state seams: `koboi/server/app.py:239–264` (`create_app` injection kwargs)
- Protocols: `koboi/server/protocols.py` (`SessionStore`, `EventBuffer`, `IdempotencyStore`); `koboi/memory.py:12` (`MemoryBackend`)
- Default stores: `JobStore` (`koboi/server/jobs.py:72`), `OwnershipStore` (`koboi/server/ownership.py:14`)
- In-process execution: `run_job` (`jobs.py:628`), `asyncio.create_task` (`jobs.py:1064, 1088`), `_WEBHOOK_TASKS` (`jobs.py:458`), webhook delivery (`jobs.py:573`), job-GC loop (`app.py:449`)
- "Future Redis" comments (no backend shipped): `protocols.py:3,9`, `jobs.py:311`, `pool.py:15`, `app.py:255`, `trust.py:64`
- Crash/resume: `steps` journal (`koboi/journal.py`), `koboi run --resume`, workflow `replay` mode (offline, no API key)

**Platform / framework references:**
- Cloudflare Agents SDK — https://developers.cloudflare.com/agents/
- Cloudflare Workflows (GA) — https://blog.cloudflare.com/workflows-ga-production-ready-durable-execution/
- Cloudflare DO WebSocket hibernation — https://developers.cloudflare.com/durable-objects/best-practices/websockets/
- Cloudflare long-running agent patterns — https://developers.cloudflare.com/agents/concepts/agentic-patterns/long-running-agents/
- AWS Step Functions (durable orchestration, GA) — https://docs.aws.amazon.com/step-functions/
- Firecracker microVM — https://github.com/firecracker-microvm/firecracker
- Azure Durable Functions (the "Durable Functions" name origin) — https://learn.microsoft.com/en-us/azure/azure-functions/durable/durable-functions-overview
- LangGraph + DynamoDB checkpoint — https://aws.amazon.com/blogs/database/build-durable-ai-agents-with-langgraph-and-amazon-dynamodb/
- Google Cloud Run autoscaling — https://docs.cloud.google.com/run/docs/about-instance-autoscaling
- Azure Container Apps scaling (KEDA) — https://learn.microsoft.com/en-us/azure/container-apps/scale-app

*Note: claims about "AWS Lambda Durable Functions (Dec 2025)" and "Lambda MicroVMs (8h suspend)" could not be independently verified (search rate-limited) and appear to be conflation of Azure Durable Functions + AWS Step Functions + Firecracker internals. Treat as unconfirmed.*
