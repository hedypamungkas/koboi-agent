# Koboi Agent Platform -- One-Pager Requirements

## Vision

Self-hosted AI agent platform yang expose API untuk client, mendukung single/multi-agent
dengan lifecycle on-demand, dynamic configuration, dan inter-agent communication.
Dimulai dari single-node Docker Compose, di-design agar bisa scale ke multi-node.

---

## High-Level Architecture

```
                              ┌─────────────────────────────────────────────────┐
                              │              Docker Compose (Node)              │
                              │                                                 │
  ┌──────────┐   HTTP/SSE     │  ┌──────────────┐    NATS     ┌─────────────┐  │
  │          │ ─────────────► │  │              │ ──────────► │             │  │
  │  Client  │                │  │   API        │             │  Worker     │  │
  │  (App/   │ ◄───────────── │  │   Gateway    │ ◄────────── │  Pool       │  │
  │   CLI/   │   SSE stream   │  │  (FastAPI)   │   result    │  Manager    │  │
  │   Web)   │                │  │              │             │             │  │
  └──────────┘                │  └──────┬───────┘             └──────┬──────┘  │
                              │         │                            │         │
                              │         │  NATS                      │ NATS    │
                              │         │                            │         │
                              │  ┌──────▼────────────────────────────▼──────┐  │
                              │  │            NATS Server                   │  │
                              │  │    (message bus + JetStream persistence)  │  │
                              │  └──────┬────────────────────────────┬──────┘  │
                              │         │                            │         │
                              │  ┌──────▼──────┐  ┌──────────┐  ┌──▼───────┐  │
                              │  │  Worker     │  │  Worker   │  │  Worker  │  │
                              │  │  Agent #1   │  │  Agent #2 │  │  Agent #N│  │
                              │  │  (koboi)    │  │  (koboi)  │  │  (koboi) │  │
                              │  └─────────────┘  └──────────┘  └──────────┘  │
                              │                                                 │
                              │  ┌─────────────┐  ┌──────────────────────┐     │
                              │  │ Config      │  │ Observability        │     │
                              │  │ (YAML vol)  │  │ (Langfuse + Redis)   │     │
                              │  └─────────────┘  └──────────────────────┘     │
                              └─────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. API Gateway (FastAPI)

Client-facing HTTP server. Tidak menjalankan agent sendiri — hanya routing.

```
POST   /v1/agents/{agent_id}/run         → RunResult (sync)
POST   /v1/agents/{agent_id}/run/stream  → SSE stream of StreamEvent
GET    /v1/agents                         → list registered agents
GET    /v1/agents/{agent_id}/health       → agent status
POST   /v1/agents/{agent_id}/reset        → clear agent memory

GET    /v1/config                         → list active configs
GET    /v1/config/{agent_id}              → get agent config
PUT    /v1/config/{agent_id}              → update agent config (hot-reload)
POST   /v1/config/validate                → validate YAML without applying

GET    /v1/workers                        → list active workers
POST   /v1/workers/spawn                  → manually spawn worker
POST   /v1/workers/{id}/shutdown          → gracefully shutdown worker
GET    /v1/workers/{id}/health            → worker health check

POST   /v1/agents/{agent_id}/run/async    → returns task_id (async + webhook)

GET    /v1/tasks                          → list tasks across agents
GET    /v1/tasks/{task_id}                → get task status/result

GET    /v1/usage/summary                  → aggregate usage per agent/client/period
GET    /v1/usage/{task_id}                → per-task token usage + estimated cost
```

**Key behaviors:**
- Request masuk → Gateway publish task ke NATS → Worker pick up → Stream hasil via SSE
- Auth via API key header (`X-API-Key`)
- Rate limiting per client
- Request overrides: hanya `temperature`, `max_tokens`, `rag.top_k` yang boleh di-override
- Webhook: client bisa pass `callback_url` di request body — worker POST hasil ke URL saat selesai
- Async mode: `POST /run/async` langsung return `task_id`, client poll `/tasks/{id}` atau terima webhook
- Usage tracking: setiap RunResult simpan `token_usage` + `estimated_cost` ke store, query via `/v1/usage/*`

### 2. Worker Pool Manager

Service yang manage lifecycle worker containers.

```
┌─────────────────────────────────────────────────────────┐
│                 Worker Pool Manager                      │
│                                                          │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │ Warm Pool   │  │ Burst Pool   │  │ Registry     │  │
│   │ (pre-warmed │  │ (on-demand   │  │ (who's alive │  │
│   │  workers)   │  │  spawn)      │  │  + metadata) │  │
│   └─────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│   Config:                                                │
│     warm_pool_size: 2          # always running          │
│     max_burst: 5               # max extra workers       │
│     idle_timeout: 300s         # shutdown if idle         │
│     health_interval: 10s       # heartbeat check          │
│     scale_threshold: 0.8       # spawn when 80% busy     │
└─────────────────────────────────────────────────────────┘
```

**Lifecycle:**
```
  Request masuk
       │
       ▼
  Ada idle worker? ──yes──► Dispatch ke worker
       │
       no
       │
       ▼
  Warm pool < max? ──yes──► Spawn dari warm pool
       │
       no
       │
       ▼
  Burst < max_burst? ──yes──► Spawn burst worker
       │
       no
       │
       ▼
  Queue request (NATS JetStream) → wait for idle worker
```

### 3. Worker (koboi-agent runtime)

Setiap worker adalah satu instance koboi-agent yang subscribe ke NATS.

```
┌────────────────────────────────────────┐
│            Worker Container             │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  NATS Subscriber                  │  │
│  │  - Subscribe: worker.{id}.task    │  │
│  │  - Publish:   task.{id}.result    │  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│  ┌──────────────▼────────────────────┐  │
│  │  KoboiAgent                       │  │
│  │  - Loaded from YAML config        │  │
│  │  - Hot-reload on config change    │  │
│  │  - Streaming via NATS publish     │  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│  ┌──────────────▼────────────────────┐  │
│  │  Heartbeat Publisher              │  │
│  │  - Publish: koboi.health.{id}     │  │
│  │  - Every 10s                      │  │
│  │  - Payload: {status, config_hash, │  │
│  │    uptime, active_tasks}          │  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│  ┌──────────────▼────────────────────┐  │
│  │  Health / Metrics endpoint        │  │
│  │  - GET /health                    │  │
│  │  - GET /metrics                   │  │
│  └───────────────────────────────────┘  │
└────────────────────────────────────────┘
```

### 4. Inter-Agent Communication

**Pattern: NATS + HTTP hybrid**

```
  Agent A                    NATS                    Agent B
     │                        │                        │
     │  request("agent.B.task", payload)               │
     │ ──────────────────────►│                        │
     │                        │  deliver to subscriber │
     │                        │ ──────────────────────►│
     │                        │                        │
     │                        │  publish("task.result", │
     │                        │          partial_chunk) │
     │                        │ ◄──────────────────────│
     │  receive partial       │                        │
     │ ◄──────────────────────│                        │
     │                        │                        │
     │                        │  publish("task.result", │
     │                        │          final_result)  │
     │                        │ ◄──────────────────────│
     │  receive final         │                        │
     │ ◄──────────────────────│                        │
```

**NATS Subject Topology:**
```
koboi.task.{agent_id}          # task dispatch (request-reply)
koboi.result.{task_id}         # result streaming (pub-sub)
koboi.broadcast.{event}        # system events (fan-out)
koboi.control.{command}        # control plane commands
koboi.health.{worker_id}       # worker heartbeats
```

**Kenapa NATS (bukan custom control plane):**

| Aspect              | Custom Control Plane          | NATS                          |
|---------------------|-------------------------------|-------------------------------|
| Latency             | 2 hop (5-20ms each)          | 1 hop (~100us)               |
| SPOF                | Ya, perlu HA replication      | Ya, tapi binary 10MB, mudah HA|
| Request-reply       | Build sendiri (correlation ID)| Built-in `nc.request()`       |
| Backpressure        | Build sendiri                 | Built-in                      |
| Durability          | Build sendiri                 | JetStream (exactly-once)      |
| Streaming           | Proxy semua traffic           | Pub-sub per chunk             |
| Multi-node          | Perlu service discovery       | Zero config (just cluster URL)|
| Complexity          | HIGH (reinventing broker)     | LOW (battle-tested infra)     |

**Trade-off:** Control plane lebih mudah di-observe (semua traffic terpusat), tapi ini
bisa di-handle oleh NATS monitoring + Langfuse tracing tanpa membangun broker sendiri.

### 5. Dynamic Configuration

```
┌──────────────────────────────────────────────────────────┐
│                   Config Layer                            │
│                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ AdminConfig  │    │ RequestOverrides│  │ ConfigWatcher│ │
│  │ (from YAML)  │    │ (per-request) │    │ (file poll)  │ │
│  │              │    │               │    │              │ │
│  │ system_prompt│    │ temperature   │    │ poll every 2s│ │
│  │ model        │    │ max_tokens    │    │ hash check   │ │
│  │ api_key      │    │ rag.top_k     │    │ atomic swap  │ │
│  │ tools        │    │ context.strategy│  │              │ │
│  │ guardrails   │    │               │    │              │ │
│  └──────┬───────┘    └───────┬───────┘    └──────┬──────┘ │
│         │                    │                    │        │
│         │    merge           │     detect         │        │
│         ◄────────────────────┘     change         │        │
│         │                    │                    │        │
│         ▼                    │                    │        │
│  ┌──────────────────┐       │    notify workers   │        │
│  │ Effective Config │       │   ◄─────────────────┘        │
│  │ (per-request     │       │                              │
│  │  snapshot)       │       │                              │
│  └──────────────────┘       │                              │
└──────────────────────────────────────────────────────────┘
```

**Config Field Taxonomy (Security):**

| Field                    | Per-request Override | Admin-only |
|--------------------------|:--------------------:|:----------:|
| `temperature`            | ✅                   |            |
| `max_tokens`             | ✅                   |            |
| `rag.top_k`              | ✅                   |            |
| `context.strategy`       | ✅                   |            |
| `agent.system_prompt`    |                      | ✅         |
| `llm.model`              |                      | ✅         |
| `llm.api_key`            |                      | ✅         |
| `tools` list             |                      | ✅         |
| `guardrails`             |                      | ✅         |

**Risk/Reward per approach:**

| Approach            | Reward                              | Risk                                    |
|---------------------|-------------------------------------|-----------------------------------------|
| Per-request override| Maximum flexibility, A/B testing    | Prompt injection, cost amplification    |
| Hot-reload YAML     | Simple, git-auditable, no restart   | File race conditions, in-flight ambiguity|
| Config API (DB)     | Full CRUD, RBAC, versioning         | Over-engineering for POC, migration debt |

**Recommendation:** Hot-reload YAML + narrow per-request allowlist. Config API nanti
saat ada multiple operators.

---

## Docker Compose Layout

```yaml
# docker-compose.yml
services:
  nats:
    image: nats:2-alpine
    command: ["--jetstream"]
    ports: ["4222:4222", "8222:8222"]

  api-gateway:
    build: .
    command: ["uvicorn", "koboi.api.server:app"]
    ports: ["8000:8000"]
    depends_on: [nats]
    volumes:
      - ./configs:/app/configs:ro
    environment:
      NATS_URL: nats://nats:4222
      KOBOI_CONFIG_DIR: /app/configs

  worker-manager:
    build: .
    command: ["python", "-m", "koboi.platform.manager"]
    depends_on: [nats]
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # spawn workers
      - ./configs:/app/configs:ro
    environment:
      NATS_URL: nats://nats:4222
      WARM_POOL_SIZE: "2"
      MAX_BURST: "5"

  # Workers spawned dynamically by worker-manager
  # Not defined in compose — created via Docker API

  langfuse:
    image: langfuse/langfuse:2
    ports: ["3000:3000"]
    depends_on: [postgres]

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_PASSWORD: ${LANGFUSE_PG_PASSWORD}
```

---

## Module Structure (new code)

```
koboi/
  platform/                    # NEW — platform layer
    __init__.py
    server.py                  # FastAPI app + routes
    models.py                  # API request/response Pydantic models
    auth.py                    # API key auth middleware
    config_api.py              # Config CRUD endpoints
    worker_manager.py          # Worker lifecycle (spawn/shutdown/health)
    nats_bus.py                # NATS client wrapper (publish/subscribe/request)
    worker_runtime.py          # Worker-side: subscribe + run KoboiAgent
    stream_relay.py            # NATS → SSE relay for streaming
    webhook.py                 # Webhook delivery (POST result to callback_url)
    usage_store.py             # Token usage + cost tracking (in-memory → SQLite)
  api/                         # Thin entry point
    __init__.py
    server.py                  # uvicorn entry: `uvicorn koboi.api.server:app`
```

---

## Request Flow (end-to-end)

```
  Client                API Gateway           NATS              Worker
    │                      │                   │                   │
    │ POST /v1/agents/     │                   │                   │
    │ sales/run/stream     │                   │                   │
    │ {message, overrides} │                   │                   │
    │ ────────────────────►│                   │                   │
    │                      │ validate auth     │                   │
    │                      │ validate overrides│                   │
    │                      │ create task_id    │                   │
    │                      │                   │                   │
    │                      │ publish task      │                   │
    │                      │ ─────────────────►│                   │
    │                      │                   │ deliver to worker │
    │                      │                   │ ─────────────────►│
    │                      │                   │                   │
    │                      │                   │   stream chunks   │
    │                      │                   │ ◄─────────────────│
    │  SSE: text_delta     │                   │                   │
    │ ◄────────────────────│◄──────────────────│                   │
    │                      │                   │                   │
    │                      │                   │   stream chunks   │
    │                      │                   │ ◄─────────────────│
    │  SSE: tool_call      │                   │                   │
    │ ◄────────────────────│◄──────────────────│                   │
    │                      │                   │                   │
    │                      │                   │   final result    │
    │                      │                   │ ◄─────────────────│
    │  SSE: complete       │                   │                   │
    │ ◄────────────────────│◄──────────────────│                   │
    │                      │                   │                   │
```

---

## Phased Implementation

```
Phase 1 — API Foundation (MVP)
├── FastAPI server with /run and /run/stream endpoints
├── In-process worker (no Docker spawning yet)
├── NATS integration for task dispatch
├── Per-request config overrides (allowlist only)
├── API key auth
└── Docker Compose: api-gateway + nats

Phase 2 — Worker Pool
├── Worker runtime (standalone KoboiAgent container)
├── Worker Pool Manager (warm pool + burst spawn)
├── Health checks + auto-restart
├── Graceful shutdown via NATS control subject
└── Docker Compose: + worker-manager

Phase 3 — Multi-Agent Communication
├── Inter-agent task delegation via NATS
├── Orchestrator mode (routes to multiple workers)
├── Streaming relay (NATS → SSE)
└── Result aggregation + quality evaluation

Phase 4 — Dynamic Config + Production
├── ConfigWatcher (hot-reload YAML)
├── Config API (CRUD endpoints)
├── Observability dashboard (Langfuse integration)
├── Rate limiting + quota management
└── Multi-node Docker Compose / K8s manifests
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API Framework | FastAPI | Async native, SSE support, Pydantic integration, koboi already uses httpx |
| Message Bus | NATS + JetStream | Single binary 10MB, built-in request-reply, exactly-once, scales to multi-node |
| Worker Lifecycle | Hybrid (warm + burst) | Low latency for common case, elastic for parallel tasks |
| Config Strategy | Hot-reload YAML + per-request allowlist | Simple, auditable, secure (no client access to secrets) |
| Streaming | NATS pub-sub → SSE relay | Works for agent-sized chunks, client gets standard SSE |
| Auth | API key (X-API-Key header) | Simple for POC, upgradeable to JWT/OAuth |
| Observability | Langfuse (existing) + NATS monitoring | Leverage existing Langfuse setup |

---

## Resolved Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Multi-tenancy | **Single-tenant** untuk POC. Isolate via API key + rate limit per client. Namespace pattern (`koboi.{tenant}.*`) bisa ditambah nanti tanpa rewrite. | Cukup untuk POC dan tim kecil. Multi-tenant = overkill sekarang. |
| Cost tracking | **Expose via API**. Setiap RunResult simpan `token_usage` + `estimated_cost`. Tambah `/v1/usage/summary` endpoint untuk aggregate per agent/client/period. Langfuse tetap jalan untuk observability. | Client perlu visibility ke cost. Langfuse untuk deep-dive, API untuk dashboard. |
| Webhook | **Ya, perlu**. Client pass `callback_url` di request body. Worker POST hasil ke URL saat selesai. Async mode: `POST /run/async` langsung return `task_id`. | Cocok untuk async workflow. Client tidak perlu long-poll atau hold SSE connection. |
| Agent discovery | **NATS heartbeat**. Worker publish `{worker_id, status, config_hash, uptime, active_tasks}` ke `koboi.health.{worker_id}` setiap 10s. Worker Manager subscribe `koboi.health.>` dan maintain registry in-memory. Miss 3 heartbeat → mark dead → auto-restart. | Decoupled dari Docker runtime. Tidak perlu service registry terpisah. Docker API bisa ditambah nanti untuk container-level metrics. |
