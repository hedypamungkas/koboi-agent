# koboi-agent — REST/SSE Serving Layer Requirements

**Status:** Draft v0.6 (final checkpoint) — for review before execution
**Date:** 2026-06-28
**Owner:** Hedy Pamungkas
**Stack decision (agreed):** FastAPI + uvicorn

**Decisions locked:**
1. Execution = **in-process asyncio + resume-on-startup** (no external queue).
2. Two modes = **Interactive (SSE + HITL)** and **Autonomous / Job (`POST /jobs`, no HITL)**.
3. Jobs are **autonomous**: policy-gated (Trust DB + PolicyHook); no HITL.
4. Delivery = **SSE replay + live tail + poll** (no webhook in v1).
5. Tenancy = **single-tenant v1, interface-ready for multi-tenant**.
6. Same-session concurrency = **serialize via per-session lock**; **jobs default dedicated session**.
7. **No sync path** — interactive = SSE only.
8. Sandbox = **per-session workdir**.
9. **Built-in (config) + Customize (code)** via `create_app()` factory + `koboi serve` CLI.
10. **Observability (Langfuse)** = loop Hook → auto-traces served runs; serving enriches metadata.
11. **Deployment:** self-hosted first (v1); SaaS = known target, phased later.
12. **API keys** = keys file + `koboi keys` CLI (env `${KOBOI_API_KEYS}` back-compat).
13. **Idempotency** = `Idempotency-Key` header + dedup window (same key → same result).
14. **Workdir** = keep tied to session/job TTL (24h), GC at TTL.
15. **API shape** = koboi-native SSE + jobs (primary); OpenAI-compat adapter additive later (not v1).

**Defaults declared (no question needed):**
- Error contract = structured envelope `{code,message,details}` + `ErrorEvent`; job `failed` carries
  `error_class` + `retriable`.
- Autonomous jobs: input/output guardrails + PolicyHook hardcoded safety **remain active** (no silent
  bypass — anti prompt-injection).
- `/readyz` (DB/pool/langfuse health) + `request-id` middleware (log/trace/Langfuse correlation).
- Drain disconnects in-flight interactive streams after `drain_seconds`.

> Grounding via `file:line` in §22. Fix-first items in §16. Changelog at end.

---

## 1. Konteks & Tujuan

Hygiene feature AI-agent sudah diadopsi (secret hygiene, compaction preservation, tool disable,
sandbox abstraction, step journal/resume, QA gates). Output masih **AI-conversational**, diakses
via CLI/TUI. Tujuan: **menyajikan agent via REST/SSE** + **background otonom (jobs)**, dapat
**self-hosted** out-of-the-box maupun **customize by code**, dgn **observability (Langfuse)**.

**Prinsip:** serving layer = adapter tipis di atas API async yang sudah ada. Codebase sudah
menyediakan `facade.run_stream()` (`loop.py:484`), `event_to_dict()` (`events.py:128`),
`facade.resume()` + `StepJournal` (`loop.py:463`, `journal.py`).

---

## 2. Glossary

| Istilah | Arti |
|---|---|
| SSE | Server-Sent Events (`text/event-stream`) |
| **Interactive mode** | `/chat/stream` (SSE) + HITL. Tidak ada sync. |
| **Autonomous / Job mode** | `POST /jobs`; tanpa HITL; policy-gated; resumable |
| Session | 1 percakapan = 1 `session_id` (uuid4); SQLite |
| AgentPool | `session_id → KoboiAgent` + per-session lock + per-session workdir |
| JobRegistry/JobStore | in-memory `job_id→task` + tabel SQLite `jobs` |
| Trust DB | `TrustDatabase` (`trust.py`); aturan auto-approve persisten |
| Built-in path | `koboi serve --config` (zero code) |
| Customize path | `create_app(config, extra_*=...)` (programmatic) |

> `TaskManager` (`task.py`) = todo-list internal agent, BUKAN job runner.

---

## 3. Scope

### In scope
- `koboi/server/`: FastAPI app, SSE encoder, AgentPool, JobRegistry, per-session lock, per-session
  workdir, **app factory + `koboi serve` CLI + `koboi keys` CLI**.
- Two modes: Interactive (`/chat/stream` SSE + HITL) + Autonomous/Job (`POST /jobs`).
- Session lifecycle (lock + workdir TTL) + Job lifecycle (submit/status/cancel/resume + TTL).
- Auth: **keys file + CLI** (env back-compat) + session/job ownership.
- **Idempotency-Key** dedup on submit endpoints.
- Cancellation/disconnect + graceful drain (jobs + interactive).
- Config: `server:` + `jobs:` + `sandbox.workdir_strategy`.
- **Error contract** (envelope + `ErrorEvent` + job error taxonomy).
- `/healthz` + `/readyz` + request-id middleware.
- Langfuse serving-enrichment (metadata + trace_id + flush).
- Extensibility parity (all customize-by-code active in served context, §6).

### Out of scope (v1)
- Sync `/chat`; WebSocket; webhook delivery (pull-only v1).
- Horizontal scaling / multi-process (single-node; resume-on-startup = crash durability).
- Multi-tenant runtime enforcement (single v1; abstractions ready).
- HITL pada job; external queue/worker; memory → Postgres.
- **OpenAI-compat adapter** (additive later; native = source of truth).
- Artifact retrieval endpoint (deferred; files in per-session workdir).

### Self-hosted (no core change) & Cloud SaaS (phased)
Self-host = jalankan serving layer config-driven, BYO keys, 1 node — covered by v1 (M0–M5).
SaaS = phased extension (agent core/API tak berubah): multi-tenant enforcement + state
protocol-ization + control plane (§12).

---

## 4. Design Principles
1. Adapter, bukan rewrite.
2. Two modes, satu infra (beda instance approval handler + lifecycle).
3. Streaming-only interactive.
4. HITL first-class (interactive), Job autonomous.
5. 1 session = 1 run/waktu (per-session lock).
6. 1 session = 1 workdir (TTL-persisted).
7. Durable by journal.
8. Aman by default (auth wajib; destructive: interactive approve, job deny-by-default; **guardrails
   + PolicyHook tetap aktif di job mode**).
9. Built-in (config) + Customize (code).
10. Interface-ready for multi-tenant.
11. Correctness-first: **idempotent submit**, no double-state.
12. Native fidelity first; interop adapters additive.

---

## 5. Architecture Overview

```
 HTTP client
   Interactive: POST /v1/chat/stream (Bearer, X-Session-Id, Idempotency-Key?) → SSE
   Job:         POST /v1/jobs → 202 ; /jobs/:id[/stream]                     → poll/SSE
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ FastAPI app = create_app(config, **extra)  (koboi/server/app.py) │
│  • API-key auth (keys file; env back-compat) + ownership         │
│  • request-id middleware (log/trace/Langfuse correlation)         │
│  • Idempotency-Key dedup (window)                                 │
│  • request validation (Pydantic v2) • SSE encoder • cancel guard  │
│  • drain on shutdown (jobs + in-flight interactive)               │
└───────────────┬──────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ AgentPool (session_id → KoboiAgent) + per-session Lock + workdir │
│  (workdir TTL-persisted) • approval handler by MODE              │
└───────────────┬───────────────────────────────────┬──────────────┘
                │ interactive (holds lock)          │ job (dedicated session)
                ▼                                    ▼
   Interactive run (SSE live + HITL)    JobRegistry+JobStore+buffer+resume+TTL
                │ both
                ▼
   KoboiAgent(facade) → AgentCore.run_stream → approval gate → Trust DB → PolicyHook
   (+ LangfuseTracingHook; input/output guardrails; custom tools/hooks via §6)
```

Layer baru: `koboi/server/`. Perubahan core = additive wiring §16.

---

## 6. Extensibility — Built-in (config) vs Customize (code)

### Path A — Built-in (zero code)
```bash
koboi serve --config agent.yaml --port 8080        # KOBOI_CONFIG=agent.yaml uvicorn koboi.server:app
```
Customize via YAML: `tools.custom`, `rag.custom_modules`, `context.custom_modules`, plugin entry
points — auto-import (`facade.py:473,506,527`). Zero Python di deploy.

### Path B — Customize by code
```python
from koboi.server import create_app
from koboi.tools.registry import tool
from koboi.hooks.chain import Hook, HookEvent, HookContext
from koboi.rag.registry import register_retriever
from koboi.guardrails.approval import ApprovalHandler

@tool(name="query_crm", ...) def query_crm(...): ...
class AuditHook(Hook): ...
@register_retriever("crm_kb") ...
class CrmApproval(ApprovalHandler): ...

app = create_app(config_path="agent.yaml",
                 extra_tools=[query_crm], extra_hooks=[AuditHook()],
                 approval_handler=CrmApproval())
```
Runtime identik dgn Path A — cuma komposisi beda.

### Extension points (semua sudah ada; aktif di served)
| Subsystem | Mekanisme | Ref |
|---|---|---|
| LLM provider | subclass base ABC + entry point `koboi.providers` | `llm/`, `plugins.py:31` |
| Tool | `@tool()`; `tools.custom` | `tools/registry.py:216`, `facade.py:473` |
| Hook | subclass `Hook` ABC | `hooks/chain.py` |
| Context strategy | `@register_context_strategy` / `context.custom_modules` | `context/registry.py:35`, `facade.py:506` |
| RAG chunker/retriever/augmentation | `@register_*` / `rag.custom_modules` | `rag/registry.py:133,160,187`, `facade.py:527` |
| Guardrail/approval | subclass `ApprovalHandler`; `koboi.guardrails` | `guardrails/approval.py:36` |
| Memory backend | `MemoryBackend` protocol | `memory.py` |
| Eval scorer | `koboi.scorers` | `plugins.py:34` |
| Plugin (semua) | auto-discover entry points | `plugins.py:38` |

**Requirement (GAP):** expose `create_app()` + `koboi serve` CLI (§16.19).

---

## 7. Execution Mode 1 — Interactive (SSE + HITL)

### Aset HITL (sebagian dormant)
| Aset | Status | Bukti |
|---|---|---|
| Pipeline `await` approval handler async | ✅ | `loop_pipeline.py:127` |
| `CallbackApprovalHandler` (sync) | ✅ | `guardrails/approval.py:100`, `facade.py:619` |
| `TrustDatabase` | ⚠️ dibangun, tak dipakai pipeline; global | `facade.py:649,833,922`; `trust.py:44-51,54` |
| `PolicyHook` konfirmasi | ⚠️ di-emit, tak dikonsumsi | `policy_hook.py:74`; cek `abort`(`:174`)/`mode_blocked`(`:197`) |
| Event `pending_approval` | ❌ belum ada | `events.py:97` |

### Alur approval
Gate (`loop_pipeline.py:126`) → `AsyncCallbackApprovalHandler` → cek Trust DB → tak auto → emit
`pending_approval` + `await Future` → `POST /approve {approval_id,decision,scope}` →
`scope=always` → `record_decision`. Timeout 120s → deny.

### Konkurensi
`AgentCore` tak concurrent-safe (`loop.py:123,262,321`) → interactive wajib acquire per-session
lock (serialize).

---

## 8. Execution Mode 2 — Autonomous / Job

**State machine:** `pending → running → completed | failed | timed_out | cancelled` (restart:
running+non-terminal → resume). Tanpa `awaiting_approval`.

**Exec:** submit → `JobStore` → 202 → scheduler (`asyncio.create_task`) → `run_job` (agent dgn
`AutonomousApprovalHandler`) → `run_stream()` drained ke buffer; journal catat step.
**Resume-on-startup:** pending→requeue; running+non-terminal→`resume()`; running+terminal→reconcile.

**Approval job:** Trust DB auto-approve; else **deny** (no pause). **Input/output guardrails +
PolicyHook hardcoded safety tetap aktif** (anti prompt-injection; no silent bypass).

**Delivery:** poll `GET /jobs/:id`; stream `GET /jobs/:id/stream` replay+tail+`[DONE]`. Attach ke
job terminal → return result + `[DONE]`. Buffer in-memory capped; TTL `jobs.ttl_seconds`.

**Cancel/limits:** `POST /jobs/:id/cancel`; `max_concurrent` ≤ pool; `per_tenant_max`; queue →
`queue_depth` → 429. **Dedicated session by default** (no `session_id`).

**Idempotency:** `POST /jobs` menerima `Idempotency-Key` (§11); dedup window → key sama = job_id sama.

---

## 9. API Surface

### Interactive (SSE only)
| Method | Path | Inti |
|---|---|---|
| `POST` | `/v1/sessions` | Buat session |
| `GET`/`DELETE` | `/v1/sessions/:id` | Messages / hapus+evict |
| `POST` | `/v1/chat/stream` | SSE `run_stream()` (lock; Idempotency-Key optional) |
| `POST` | `/v1/sessions/:id/resume` | SSE `resume()` |
| `POST` | `/v1/sessions/:id/approve` | Resolve HITL |
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Readiness (DB/pool/langfuse) |

### Jobs (autonomous)
| Method | Path | Inti | Status |
|---|---|---|---|
| `POST` | `/v1/jobs` | Submit (Idempotency-Key recommended) | **202** |
| `GET` | `/v1/jobs` | List (owner/status) | 200 |
| `GET` | `/v1/jobs/:id` | Status+result (poll) | 200/403/404 |
| `GET` | `/v1/jobs/:id/stream` | SSE replay+tail | 200 |
| `POST` | `/v1/jobs/:id/cancel` | Cancel | 200/409 |

### SSE event protocol
`data: {json}\n\n` via `event_to_dict()` (`events.py:128`). 10 tipe + `pending_approval` (§16.4).
Akhir: `complete`/`error`/`[DONE]`. **Native koboi** (OpenAI-compat adapter = later).

```jsonc
data: {"type":"pending_approval","approval_id":"ap_01H...","tool_name":"run_shell",
       "tool_call_id":"call_...","arguments":"...","risk_level":"destructive","timeout_seconds":120}
data: {"type":"tool_result","tool_name":"run_shell","tool_call_id":"call_...","result":"Error: Denied"}
data: {"type":"complete","content":"...","iterations_used":3,"tools_used":["web_search"],"trace_id":"lf_..."}
data: [DONE]
```

### Error contract (default)
- HTTP error envelope: `{"error":{"code":"...","message":"...","details":{...},"request_id":"..."}}`.
- SSE error: `ErrorEvent` → `{"type":"error","error":"...","code":"...","retriable":bool}` lalu `[DONE]`.
- Job `failed` (JobStore): bawa `error_class` + `retriable` (client bisa retry idempotent).

---

## 10. Session & Job Lifecycle

### AgentPool + per-session lock + per-session workdir (TTL)
- **Lazy create:** `Config.from_yaml(session_id=X)` (`config.py:448`); assign workdir
  `./workspace/<session_id>/`.
- **Per-session lock:** `asyncio.Lock`/session; wajib acquire (serialize; AgentCore tak safe).
- **Workdir TTL:** workdir **persist selama record session/job** (`workdir_ttl_seconds`, default
  24h); GC saat TTL. Resume + future artifact retrieval jalan; tak menyabotase durability.
- **Reuse** instance utk session_id sama. **Evict** (LRU+idle): `agent.close()` (`facade.py:223`)
  → MCP ditutup; workdir **tetap** sampai TTL. **Cap** `pool.max_agents` → 429.

### JobRegistry + JobStore + TTL
JobStore (SQLite `jobs`): `job_id, session_id, owner, status, created/updated, result_json, error,
error_class, retriable, idempotency_key, config_snapshot`. Registry in-memory. TTL GC (record+buffer).

---

## 11. Auth, Idempotency & Tenancy

### API keys (self-host)
- **Keys file + CLI:** `koboi keys create|rotate|revoke|list` → `~/.koboi/keys.json` (atau
  `server.api_keys_file`). Middleware validasi `Authorization: Bearer`.
- **Env back-compat:** `${KOBOI_API_KEYS}` comma-separated tetap dihonor (konsisten pola back-compat
  project, cf. `KOBOI_SANDBOX_DIR`).
- Interface-ready: swap ke DB-backed saat SaaS.

### Idempotency
- Header `Idempotency-Key` di `POST /jobs` (dan opsional `/chat/stream`); dedup window
  (`idempotency.window_seconds`, mis. 86400); key sama → **job_id hasil sama** (Stripe-like). Store
  di JobStore (`idempotency_key` col) + ring cache.

### Tenancy (single v1, interface-ready multi)
- v1: 1 API key = owner; sidecar `session_id→owner`. Job punya `owner`.
- Interface-ready: `TrustStore` (Trust DB global saat ini → future per-tenant), sandbox per-session
  (sudah), rate per-key (middleware). DB: v1 shared; multi → per-tenant.

### Rate limit (dwi-lapis)
Guardrail (per-tool/session, pipeline) + infra (per key, middleware) + per-tenant job conc.

---

## 12. Deployment Models (self-host vs SaaS)

| | **Self-hosted** | **Cloud SaaS** |
|---|---|---|
| Agent core + API contract | ✅ tak berubah | ✅ tak berubah |
| Single-tenant serving | ✅ **ready v1** | ✅ reused |
| Multi-tenant isolation | tak perlu | ⚠️ deferred; interface ready |
| Single-process state | ✅ (1 node) | ❌ gap multi-node → protocol → Redis |
| Control plane (auth/billing/key-pool) | tak perlu | produk layer |
| LLM keys | BYO | pooling + metering |

Self-host = no core change. SaaS = phased (multi-tenant enforcement + state protocol-ization +
control plane). Serving-state protocol-ization (`SessionStore`/`LockProvider`/`EventBuffer`/
`JobStore`) → forward-looking (§16.20).

---

## 13. Cancellation, Disconnect & Drain
- Interactive SSE disconnect → `task.cancel()` + **release session lock**; partial-turn via journal.
- Job cancel → §8. **Drain (SIGTERM):** tak terima baru; tunggu `drain_seconds`; **disconnect
  in-flight interactive stream**; cancel/complete job sesuai policy; flush Langfuse; `agent.close()`
  semua; flush JobStore.

---

## 14. Concurrency & Resources
| Resource | Concern | Requirement |
|---|---|---|
| AgentCore instance | NOT concurrent-safe | per-session lock; 1 run/session/waktu |
| Sandbox workdir | was global | **per-session, TTL-persisted**; `validate_path` (`restricted.py:163`) |
| SQLite WAL | busy_timeout=5s; per-instance conn | konkuren write aman; `db_path` eksplisit |
| JobStore | konkuren update + idempotency | koneksi sendiri; WAL |
| httpx (LLM) | long-lived; concurrent-safe | reuse via pool |
| MCP subprocess/http | leak bila tak close | evict/job-done → close |
| Event buffer (job) | memori tumbuh | capped; drop oldest |
| asyncio Task (job) | tak terbatas | max_concurrent + per-tenant + queue_depth |
| Workdir disk | tumbuh dgn session | TTL GC (24h) |

---

## 15. Roadmap Alignment (skills/RAG/orchestration/subagent/MCP)
| Subsystem | Config-driven? | Yang HTTP layer lakukan |
|---|---|---|
| Skills | `skills.search_paths` | pool per-session → state terjaga |
| RAG | ya (`rag.*`+`custom_modules`) | exposure otomatis; build index blocking I/O (§16.9) |
| Orchestration + subagent | ya (`orchestration.*`) | emit event dikenal SSE; resume tak didukung |
| MCP client | ya (`mcp.servers[]`) | cleanup close() |
| Sandbox | ya (`sandbox.*`) | per-session workdir (TTL); restricted aktif |
| Journal/resume | ya (`journal.*`) | `/resume` + job resume-on-startup |

---

## 16. Pre-requisite Fixes / Adjustments

| # | Item | Lokasi | Mengapa | Ukuran |
|---|---|---|---|---|
| 16.1 | **Wire `TrustDatabase` ke pipeline** | `loop_pipeline.py:122-130` | DB dormant; dipakai kedua mode | S |
| 16.2 | **Konsumsi `policy_needs_confirmation`** | `loop_pipeline.py:174` | sinyal belum dikonsumsi | S |
| 16.3 | **`AsyncCallbackApprovalHandler`** | `guardrails/approval.py`+`facade.py:615` | handler async belum ada | S |
| 16.4 | **Event `PendingApprovalEvent`** | `events.py:97,111` | SSE perlu tahu approval | S |
| 16.5 | **Ownership sidecar** + cek | server layer | tenant isolation | M |
| 16.6 | **Config `server:`+`jobs:`+`sandbox.workdir_strategy`** | `config*.py` | belum ada section | S |
| 16.7 | **Verifikasi partial-turn consistency** | `loop.py:484`+`journal.py` | tak korup | M |
| 16.8 | **Verifikasi timing tool-result persist** + idempotensi doc | `loop.py:470-471`+`memory_sqlite.py:171` | cegah dobel destructive saat resume | M |
| 16.9 | **RAG warm-up / pre-build** | `rag/registry.py`+facade | cold-start lambat | M |
| 16.10 | **`pyproject` extra `[api]`** | `pyproject.toml` | dep baru | S |
| 16.11 | **`AutonomousApprovalHandler`** | `guardrails/approval.py` | instance approval job | S |
| 16.12 | **`JobStore`+`JobRegistry`+`run_job`+resume+buffer+TTL** | `koboi/server/jobs.py` | backbone job | M–L |
| 16.13 | **Per-session `asyncio.Lock`** | `koboi/server/pool.py` | AgentCore tak concurrent-safe | S |
| 16.14 | **Per-session sandbox workdir** | pool + sandbox config | workdir global → leak | S–M |
| 16.15 | **`TrustStore` interface** (future per-tenant) | `trust.py`+facade | interface-ready multi | S |
| 16.16 | **Enforce `memory.backend=sqlite` utk job** | server+config | job resume butuh sqlite | S |
| 16.17 | **Log hygiene** (no secret; per-session ns) | `logger.py`+server | shared/multi-tenant log | S |
| 16.18 | **Drain on shutdown** (jobs + in-flight interactive + langfuse flush) | server layer | graceful exit | S |
| 16.19 | **`create_app()` factory + `koboi serve` CLI + examples** | `koboi/server/app.py` | built-in+customize | M |
| 16.20 | **Serving-state protocol-ization** | `koboi/server/` | self-host in-memory; SaaS swap Redis | M |
| 16.21 | **Langfuse serving-enrichment** (job_id/user/tenant/request_id/mode + `trace_id` return + flush) | `hooks/langfuse_hook.py`+server+`facade.push_langfuse_scores` | tracing sudah jalan; butuh metadata+DX | S |
| 16.22 | **API keys: `koboi keys` CLI + keys file + middleware** (env back-compat) | `koboi/server/auth.py`+`koboi keys` | self-host mint/rotate; no DB | M |
| 16.23 | **Idempotency-Key dedup** (window store; col `idempotency_key`) | server layer + JobStore | no double-job/cost saat retry | S–M |
| 16.24 | **Workdir TTL GC** (keep tied to session/job ttl) | pool + JobStore | artifact + resume correctness; disk bounded | S |
| 16.25 | **Error contract** (envelope + `ErrorEvent` enrichment + job `failed` `error_class`/`retriable`) | server + `events.py` | client resilience; consistent errors | S |
| 16.26 | **`/readyz` + request-id middleware** (DB/pool/langfuse health; log/trace correlation) | server layer | deploy + observability | S |
| 16.27 | **Guardrails-job-active verification** (input/output guardrails + PolicyHook enforce di mode job) | `loop.py`+hooks | anti prompt-injection (job no human review) | S |

> 16.1–16.4 = fondasi pipeline (shared). 16.11–16.12 = job. 16.13–16.18 = serving hardening.
> 16.19 = extensibility. 16.20 = SaaS-ready. 16.21–16.27 = final-checkpoint hardening.
> M0/M4 testable tanpa FastAPI.

---

## 17. Security & Operability
- **Secret leakage:** audit `text_delta` tak memuat secret (`harness/env.py`).
- **Autonomous safety:** job destructive tanpa rule → deny-by-default; PolicyHook hardcoded safety
  always enforced (`pre_ctx.abort`); **input/output guardrails tetap aktif di job mode** (16.27).
- **Log hygiene:** no secret/PII; per-session namespace (16.17).
- **Audit trail:** tool/approval/deny/job-terminal → `AuditTrail` + log.
- **CORS:** configurable, default terbatas.
- **Timeouts:** interactive (request 300s, approval 120s); job (`timeout_seconds` 1800 → timed_out);
  drain (60s).
- **Cost ceiling:** `max_iterations` + hard `server.max_iterations_cap`.
- **Observability (Langfuse):** loop hook → auto-trace; enrichment (16.21). Metrics app-level
  (active sessions/jobs, p95, approval/deny rate, job throughput). `/readyz`+request-id (16.26).

---

## 18. Configuration

```yaml
server:
  enabled: true
  host: "0.0.0.0"
  port: 8080
  api_keys_file: ~/.koboi/keys.json     # CLI `koboi keys` manajemen
  api_keys: ${KOBOI_API_KEYS:}          # env back-compat (comma-separated)
  auth_required: true
  cors: { allow_origins: [] }
  pool: { max_agents: 256, idle_timeout_seconds: 900 }
  timeouts: { request_seconds: 300, approval_seconds: 120, drain_seconds: 60 }
  limits: { max_iterations_cap: 25 }            # G2: clamps per-request max_iterations
  allowed_modes: [chat, plan, act, auto]        # G2: yolo opt-in only; omit = this safe default
  idempotency: { window_seconds: 86400 }
  workdir_ttl_seconds: 86400            # per-session workdir GC

jobs:
  enabled: true
  max_concurrent: 64
  per_tenant_max: 5
  queue_depth: 32
  default_dedicated_session: true
  event_buffer: { max_events: 500 }
  resume_on_startup: true
  timeout_seconds: 1800
  ttl_seconds: 86400

sandbox:
  backend: restricted
  workdir_strategy: per_session
  safe_path: true
  network: false

tracing: { provider: langfuse, public_key: ${LF_PUBLIC_KEY:}, secret_key: ${LF_SECRET_KEY:}, base_url: ${LF_BASE_URL:http://localhost:3300} }
```
Pydantic di `config_models.py`. Webhook slot (`jobs.webhook`) default off.

---

## 19. Testing & Acceptance
**Unit:** SSE encoder; AgentPool (lazy/reuse/evict→close/workdir-keep-until-TTL/cap→429);
per-session lock (run ke-2 nunggu, tak race); per-session workdir (2 session beda; `validate_path`
tolak cross); JobRegistry/JobStore (resume reconcile, TTL GC); event buffer (cap→drop);
`create_app()` (factory dgn/tanpa extra; custom aktif; `koboi serve` config-only); **idempotency**
(key sama → job_id sama dalam window; beda body → 409/ignore); **keys CLI** (create/revoke honored).

**Integration:** Interactive SSE happy; HITL happy/deny+always/timeout; per-session lock serial;
job happy(dedicated)/attach-terminal/autonomous-destructive-deny/cancel/resume-on-startup;
workdir isolation (2 job `write_file report.md` → file beda, tak read silang; **TTL: resume after
idle masih punya file**); disconnect→cancel+lock-release; auth 401/403/429; **readyz** (DB down → 503);
**idempotent retry** (network flake → no double job); **guardrails-job** (prompt-injection via tool
output → blocked); drain (SIGTERM→drain→disconnect interactive→flush langfuse).

**Non-fungsional:** cov ≥80%; 50 job concurrent tanpa fatal `database is locked`; cold-start job
(dgn RAG) < ambang.

**CI gates:** ruff/format/build HARD-fail; mypy/bandit continue-on-error.

---

## 20. Phase Development Plan

> **Urutan:** M0 → M1 → M2 → M3 → M4 → M5. M0 & M4 testable tanpa FastAPI. Self-host siap
> post-M5. SaaS + OpenAI-adapter = post-M5 (additive). Item §16 dirinci per fase di bawah.

### Ringkasan
| Fase | Tujuan | Item §16 | Butuh HTTP? |
|---|---|---|---|
| **M0** Pipeline foundation | Hidupkan fondasi pipeline (dipakai kedua mode) | 16.1–16.4,16.6,16.10 | ❌ |
| **M1** MVP interactive (SSE) + extensibility | Serve interactive SSE end-to-end + create_app | 16.13,14,17,19,26 | ✅ |
| **M2** HITL interactive | Approval end-to-end (interactive) | (M0) 16.3,16.4 | ✅ |
| **M3** Auth & tenancy | API key + ownership + interface-ready multi | 16.5,15,16,22 | ✅ |
| **M4** Jobs (autonomous) | Background jobs + resume-on-startup | 16.11,12,23,24,27 | ✅ |
| **M5** Ops & hardening | Production-grade self-host | 16.7,8,9,18,20,21,25 | ✅ |

---

### M0 — Pipeline Foundation (core wiring, no HTTP)
**Goal:** hidupkan sinyal dormant (Trust DB + policy confirmation) + handler approval async +
event HITL + skeleton config/extra. Dipakai bersama kedua mode. **Testable murni tanpa FastAPI.**

**Yang dibentuk:**
- `koboi/loop_pipeline.py` — wire `TrustDatabase` auto-approve fast-path (16.1); konsumsi
  `policy_needs_confirmation` → route ke flow approval/policy (16.2).
- `koboi/guardrails/approval.py` — `AsyncCallbackApprovalHandler` (await `Future`) (16.3).
- `koboi/events.py` — `PendingApprovalEvent` + tambah ke `StreamEvent` union + `_EVENT_TYPE_MAP`
  + case di `event_to_dict` (16.4).
- `koboi/config_models.py` / `config.py` — Pydantic model `ServerConfig`, `JobsConfig`,
  `sandbox.workdir_strategy`, `idempotency`, `workdir_ttl` (16.6).
- `pyproject.toml` — extra `[api]` (`fastapi`, `uvicorn[standard]`) (16.10).

**Depends on:** — (entry point).
**Deliverables / done:** Trust DB auto-approve jalan di pipeline; policy confirmation terkonsumsi;
async approval resolve/deny/timeout; event baru serializable.
**Testing:** pytest unit (no HTTP): `test_pipeline_trust_autopapprove`, `test_policy_confirmation`,
`test_async_approval_resolve_deny_timeout`, `test_event_to_dict_pending_approval`. cov ≥80.
**Risk:** menyentuh core pipeline → pastikan CLI/TUI/regression tetap hijau.

**Known limitation (M0, post-review):** `policy_needs_confirmation` (16.2) is fully
honored for DESTRUCTIVE tools and for the `AsyncCallbackApprovalHandler` (M0/M2 primary
path — it consults the handler for every un-trusted tool, so the step-3 prompt covers the
policy reason in a single prompt). It is **inert for SAFE/MODERATE tools when a SYNC handler**
(CLI/Callback) is configured: step 3 consults the handler, which auto-approves low-risk tools
without prompting, and the `approval_prompted` guard then suppresses step 4c. This is **not a
regression** (pre-M0 the flag was never consumed at all) and matches Q2 Option A. The robust fix
= evaluate policy CONFIRM before step 3 and fold the reason into one `_resolve_approval` call
(Option B, deferred — it reorders pipeline steps, overriding Q2 Option A; revisit if/when needed).

---

### M1 — MVP Interactive (SSE) + Extensibility
**Goal:** serve interactive SSE end-to-end (`/chat/stream` + sessions + health) dgn **concurrency
safety** (lock + per-session workdir) + **entry point built-in & customize**. Self-host minimal.

**Yang dibentuk (modul baru `koboi/server/`):**
- `koboi/server/__init__.py`, `app.py` — FastAPI app + **`create_app(config, *, extra_tools,
  extra_hooks, approval_handler, extra_middleware, extra_routes)`** factory + **`koboi serve`
  CLI** (16.19).
- `koboi/server/pool.py` — `AgentPool` (lazy/reuse/evict/`close()`) + **per-session `asyncio.Lock`**
  (16.13) + **per-session workdir** assign (16.14) + LRU idle eviction.
- `koboi/server/sse.py` — SSE encoder (`event_to_dict` → `data: {json}\n\n`, `[DONE]`).
- `koboi/server/routes/` — sessions (create/get/delete), `/chat/stream` (acquire lock →
  `run_stream()` → SSE), `/healthz`, **`/readyz`** (16.26).
- `koboi/server/middleware.py` — `request-id` middleware (16.26); integrasi log hygiene (16.17).
- `examples/server_built_in.py` + `examples/server_customize.py` — runnable (16.19).

**Depends on:** M0 (config + approval handler).
**Deliverables / done:** `koboi serve --config agent.yaml` jalan; `/chat/stream` SSE; 2 run di
session sama → serial (tak race); 2 session → workdir beda & `validate_path` tolak cross; `/healthz`
+`/readyz`; `create_app(extra_tools=[...])` aktif di served.
**Testing:** integration (httpx `AsyncClient` ke app): SSE happy, lock serial, workdir isolation,
readyz-down→503, create_app custom aktif.
**Risk:** lifecycle AgentCore di server async (close saat evict, no leak MCP); blocking RAG init
saat lazy create (catat utk M5 warm-up).

---

### M2 — HITL Interactive
**Goal:** full human-in-the-loop di path interaktif (real-time approval via SSE).

**Yang dibentuk:**
- `koboi/server/approvals.py` — registry `approval_id → Future` + timeout task + handling scope.
- Route `POST /v1/sessions/:id/approve {approval_id, decision, scope}` → resolve Future;
  `scope=always` → `trust_db.record_decision(always=True)`.
- Wire `AsyncCallbackApprovalHandler` (M0) ke registry (pool inject per interactive agent).
- SSE: pastikan stream tetap terbuka selama `pending_approval`.

**Depends on:** M0 (handler+event+Trust DB wiring), M1 (server/SSE).
**Deliverables / done:** destructive tool → `pending_approval` → `/approve` → approved/denied;
`scope=always` → call ke-2 auto-approve; timeout → deny.
**Testing:** integration — HITL happy / deny / always / timeout.

**Known test limitation (M2, post-implementation):** The full mid-stream HTTP flow
(SSE stream open + concurrent `POST /approve`) **can't be tested via httpx
ASGITransport** — `asyncio.create_task` inside anyio's task management doesn't
interleave correctly (the streaming response's background task is never scheduled
while the test code blocks on `aiter_lines`). The queue-bridge + coordinator +
handler integration IS validated **directly** (no HTTP, via
`test_server_approvals.py::TestQueueBridgeIntegration` — proven event sequence:
`ToolCallEvent → PendingApprovalEvent → resolve → ToolResultEvent → CompleteEvent`).
The `/approve` HTTP route logic is tested via 404 cases. In **production (uvicorn)**,
concurrent TCP connections work correctly. To test the full HTTP flow, either (a)
spin up a real uvicorn server in a thread + use real httpx TCP, or (b) use a
test framework that supports concurrent ASGI calls (e.g., `httpx.AsyncClient` with
separate transport instances per request — not yet validated).

---

### M3 — Auth & Tenancy (interface-ready)
**Goal:** autentikasi API key + ownership + abstraksi siap multi-tenant.

**Yang dibentuk:**
- `koboi/server/auth.py` — middleware API-key (keys file + env `${KOBOI_API_KEYS}` back-compat) (16.22).
- `koboi/cli/keys.py` (subcommand `koboi keys create|rotate|revoke|list`) → file keys (hash, bukan plaintext) (16.22).
- `koboi/server/ownership.py` — sidecar table `session_id→owner` + cek kepemilikan (16.5); kolom
  `owner` di `jobs`.
- `koboi/trust.py` — **`TrustStore` protocol** (bungkus `TrustDatabase`) utk future per-tenant (16.15).
- sqlite-enforce utk job/resume (16.16); infra rate-limit middleware per-key.

**Depends on:** M1.
**Deliverables / done:** 401 tanpa key; 403 akses session/job owner lain; 429 cap; `koboi keys`
mint/rotate usable; `TrustStore` interface ada.
**Testing:** auth integration + CLI tests (`koboi keys` create/revoke honored).
**Risk:** precedence env-vs-file; simpan **hash** key (bukan plaintext) di file.

---

### M4 — Jobs (Autonomous)
**Goal:** background job otonom end-to-end + **resume-on-startup** (durability tanpa external queue).

**Yang dibentuk:**
- `koboi/guardrails/approval.py` — **`AutonomousApprovalHandler`** (Trust DB + deny, no pause) (16.11).
- `koboi/server/jobs.py` — **`JobStore`** (SQLite `jobs`) + **`JobRegistry`** (in-memory
  task+buffer+status) + `run_job` + scheduler (slot global+per-tenant) (16.12).
- Event buffer in-memory capped (16.12) + **TTL GC** record+buffer+workdir (16.24).
- `koboi/server/resume.py` — **resume-on-startup** (scan JobStore: pending→requeue,
  running+non-terminal→`resume()`, running+terminal→reconcile).
- **Idempotency-Key dedup** (16.23) — col `idempotency_key` + ring cache window.
- Routes: `POST /jobs`, `GET /jobs`, `GET /jobs/:id`, `GET /jobs/:id/stream`, `POST /jobs/:id/cancel`.
- **Guardrails-job-active verification** (16.27); **dedicated session by default** utk job.

**Depends on:** M0, M1, M3 (owner utk job).
**Deliverables / done:** submit→202; poll→running→completed; stream replay+tail; attach-terminal
return result; cancel; resume-on-startup (kill→restart→lanjut); autonomous destructive→deny;
idempotent retry (key sama=job_id sama); workdir TTL persist across resume.
**Testing:** integration + **restart-resume test** (kill mid-job, restart, verify lanjut);
idempotency; workdir TTL.
**Risk:** resume idempotency (16.8 — destructive re-exec); partial-turn (16.7) → verifikasi di M5.

---

### M5 — Ops & Hardening (production-grade self-host)
**Goal:** siap produksi: error contract konsisten, graceful drain, observability (Langfuse) kaya,
state protocol-ization (SaaS-ready), RAG warm-up, load.

**Yang dibentuk:**
- **Error contract** (16.25): envelope `{error:{code,message,details,request_id}}`; `ErrorEvent`
  bawa code+retriable; job `failed` bawa `error_class`+`retriable`.
- **Drain** (16.18): SIGTERM handler — disconnect in-flight interactive, drain job sesuai
  `drain_seconds`, **flush Langfuse**, `agent.close()` semua, flush JobStore.
- **Langfuse enrichment** (16.21): tag trace `job_id`/`user`/`tenant`/`request_id`/`mode`; return
  `trace_id` di `complete`/job result; flush saat drain.
- **State protocol-ization** (16.20): protocol `SessionStore`/`LockProvider`/`EventBuffer`/
  `JobStore` (impl in-memory tetap default; slot Redis utk SaaS).
- **RAG warm-up / pre-build** (16.9); verifikasi **partial-turn consistency** (16.7); doc
  **idempotensi-resume** (16.8).
- Load test harness (`benchmarks/` atau `tests/load/`).

**Depends on:** M1–M4.
**Deliverables / done:** error konsisten lintas endpoint; graceful drain (no orphan); trace
enriched + trace_id returned; protocol siap swap SaaS; load: 50 job concurrent tanpa fatal
`database is locked`; cold-start job (RAG) < ambang.
**Testing:** drain test (SIGTERM mid-run), load bench, error-contract tests.
**Risk:** refactor state ke protocol — pastikan impl in-memory default tetap jalan (regression).

---

### Post-M5 (future, additive — tidak nge-blok self-host)
- **OpenAI-compat adapter** (native = source of truth; `/v1/chat/completions` + `choices[].delta`).
- **Webhook delivery** slot (`jobs.webhook`).
- **Artifact retrieval** (`GET /jobs/:id/artifacts/:name` baca per-session workdir).
- **SaaS:** multi-tenant enforcement (`TrustStore` per-tenant, per-tenant DB), control plane
  (signup/billing/key-pool), Redis-backed state impl (swap via protocol M5).

---

> **Rekomendasi mulai:** eksekusi **M0** dulu (no HTTP, testable, fondasi kedua mode), lalu M1.
> M0–M2 = self-host interaktif usable; M3–M4 = multi-user + background; M5 = production-grade.
---

## 21. Open Questions / Decisions

**Resolved (v0.6):** exec; two modes; jobs no-HITL; delivery; tenancy; same-session; no sync;
per-session workdir; built-in+customize; langfuse; self-host-first/SaaS-phased; **API keys=file+CLI**;
**idempotency=header+window**; **workdir=TTL**; **shape=native (OpenAI adapter later)**; + defaults
(error contract, guardrails-job-active, /readyz, drain-interactive).

**Masih terbuka (minor / future):**
1. `api_keys` kosong = tolak semua vs dev-allow localhost?
2. Default interactive approval scope: `once` (rec) vs `always`?
3. Ownership: sidecar table (rec) vs kolom di schema memory?
4. Event buffer: in-memory capped v1 (rec) vs durable table?
5. Orchestration + resume: tetap 409 v1 vs investasi multi-agent resume?
6. Webhook (future): terminal only vs +progress?
7. Artifact retrieval (deferred): `GET /jobs/:id/artifacts/:name` baca workdir?
8. Langfuse future: per-tenant project vs single-project + metadata tag?

> Semua tersisa bersifat minor/future; tak ada lagi fork architecture-shaping yang nge-blok M0.

---

## 22. Reference Index (codebase grounding)

| Fakta | Lokasi |
|---|---|
| `StreamEvent` union / `event_to_dict()` | `events.py:97` / `:128` |
| `AgentCore.run_stream/run/resume`; NOT concurrent-safe | `loop.py:484,459,463,123,262,321` |
| resume re-eksekusi tool call tanpa hasil | `loop.py:470-471` |
| `facade.run/run_stream/chat/resume`; orchestration raise | `facade.py:142,147,158,163,174` |
| `Config.from_yaml(session_id=)` / `ConfigBuilder` / `AgentAssembler.build` | `config.py:448,299` / `facade.py:686,849` |
| Pipeline approval/deny/abort/mode_blocked | `loop_pipeline.py:127,131,174,197` |
| `policy_needs_confirmation` / PolicyHook set | `chain.py:91` / `policy_hook.py:74` |
| Approval handlers / `_build_approval` | `guardrails/approval.py:36,45,100` / `facade.py:615` |
| `TrustDatabase` (dibangun, tak dipakai pipeline, global) | `facade.py:649,833,922`; `trust.py:40,44-51,54,78,119` |
| Sandbox workdir GLOBAL → per-session | `sandbox/restricted.py:117,126,163` |
| `StepJournal` API | `journal.py:64,137,156,178` |
| SQLite WAL+busy_timeout, per-instance conn, session_id, `add_tool_result` | `memory_sqlite.py:38-44,33,171` |
| Rate limiter per-instance/per-session, no tenant | `rate_limiter.py:15,25-27` |
| Journal auto-disable bila non-sqlite | `facade.py:769-771` |
| MCP client/close; LLM client shared concurrent-safe | `mcp/client.py:41`,`http_client.py:54`; `facade.py:223,241`; `orchestrator.py:111` |
| RetryClient no mid-stream retry / SubAgentManager cancel | `client.py:153` / `subagent.py:300-327,366` |
| `TaskManager` (todo, BUKAN job) / `notifications.py` desktop-only | `task.py:21` / `notifications.py:9` |
| **Langfuse = loop hook** (`LangfuseTracingHook`); sessionId/usage/model/tool-span; fail-open; `push_langfuse_scores` | `hooks/langfuse_hook.py:31,8,128,148-177,181`; `facade.py:388` |
| Plugin entry points | `plugins.py:25-50` |
| Registry decorators `@register_*` | `rag/registry.py:133,160,187`; `context/registry.py:35` |
| Custom tool/module import | `facade.py:473,506,527`; `tools/registry.py:216` |
| Config sections (no `server`/`jobs`) / pyproject extras (no `api`); `[tracing]=langfuse` | `config_models.py:169,272-291` / `pyproject.toml:34` |
| No `create_app()`/`koboi serve` yet (GAP) | grep: NONE → net-new (M1) |

---

## Changelog

**Post-M5 batch-4 (sprint 4) — G2 request-time mode + iteration knobs** (the Wave-2 "depth unlock"; tests green, 0 regressions):
- **G2 — per-request `mode` + `max_iterations` on `/v1/chat/stream` and `/v1/jobs`.** `ChatStreamRequest`/`JobSubmitRequest` gain optional `mode` + `max_iterations` (`schema.py`); validated at the route edge → 400 `invalid_mode` envelope (not FastAPI 422). Stamped per-request under `pool.session_lock` via `mode_manager.switch_mode()` + `agent._core.max_iterations`, save/restore in `finally` (the pooled agent is reused, so restore prevents a later `mode: null` request inheriting a prior request's mode). `app.py` (`_resolve_mode`/`_resolve_allowed_modes`, chat + jobs handlers), `jobs.py` (`run_job`/`_execute_job` + persisted `mode`/`max_iterations` columns so resume re-applies them).
- **Safe-scoping (the security framing that gates this feature):**
  1. Default HTTP allowlist = `{chat, plan, act, auto}` — `yolo` is **opt-in only** via `server.allowed_modes`. YOLO drops the rate limiter + approval gate + CHAT/PLAN mode block (only `PolicyHook` hardcoded safety remains), so it stays operator-gated.
  2. **Jobs always reject `yolo`** (`allow_yolo=False`) even when yolo is allowlisted — an autonomous (no-HITL) run must not drop the approval gate. The `AutonomousApprovalHandler` deny-by-default is not escapable via mode.
  3. `max_iterations` is **clamped** to `server.limits.max_iterations_cap` (default 25, ceiling — never rejected for being large); floor `ge=1`.
  4. Per-request binding (Option A refined to **per-request swap**, not per-session cache — the pool's "one agent per session, reused" invariant makes caching-at-create race-prone).
  5. Config-only path unchanged when the fields are absent (`None` ⇒ config default).
  6. Invalid/out-of-allowlist mode ⇒ 400 `invalid_mode`; invalid `server.allowed_modes` entry ⇒ raises at startup (fail loud).
- **Honesty note (Narrow scope):** `act` and `auto` are **enforcement-identical today** — `ModeHook` only distinguishes chat/plan (read-only via its hardcoded tool allowlist) from yolo (bypass). The declarative `ModeConfig.allow_file_write/allow_shell/permission_level` fields are dormant (zero readers). Over HTTP the effective knob is therefore: read-only (`chat`/`plan`) / full (`act`/`auto`) / `yolo` (opt-in bypass). Making each mode genuinely distinct = separate follow-up (touches `mode_hook` + `loop_pipeline`).
- **Deferred (G2 sub-knob):** per-request `tools` allowlist — no clean seam (tool registry is built at agent-build time; would need a runtime `pre_tool_use` hook = a 3rd tool-filtering source of truth vs `ModeHook`'s list). Ships as its own PR.
- **New config knobs:** `server.allowed_modes: [...]` (default unset ⇒ safe allowlist) and `server.limits.max_iterations_cap: 25`. See §18.

**Post-M5 batch-1 hardening** (low-effort/high-impact gotchas; tests green, ruff clean, 0 new mypy errors):
- **G7 — `/readyz` now probes the DB** (`SELECT 1` via `OwnershipStore.ping()`), not just reports the backend — a wedged/closed DB now yields 503 (was hardcoded `ok=True`). Resolves 16.26's aspirational claim. `health.py:57`, `ownership.py` (+`ping`).
- **G2 — durable job/ownership sidecar** when `memory.db_path` is set, even for ephemeral (`in_memory`) conversation backends → `resume_on_startup` can find jobs after restart. Opt-in via `memory.db_path`; `:memory:` only when the path is omitted (preserves test behavior). `app.py` (`_sidecar_db_path`); resolves 16.16's "sqlite-enforce" intent without a hard error.
- **G4 — `server.host`/`server.port` honored from YAML**: precedence CLI flag > YAML > default (8000). Schema default aligned 8080→8000; stale "auth lands in M3" warning corrected. `app.py` (`_resolve_bind`), `cli.py`, `config_models.py`.
- **G3\* — restart-interrupted jobs now `retriable=True`** + `error_class="InterruptedByRestart"` (was `retriable=False`) — clients can distinguish restart failures and resubmit. `jobs.py` (`resume_on_startup`); extends 16.25.

**Pre-existing (unrelated, out of scope for batch 1):** 10 mypy errors — route handlers annotated `-> Response` returning Pydantic/dict (app.py), and `LangfuseTracingHook.set_serving_metadata` attr (jobs.py:287). Present before this batch (diff line-ranges don't overlap any error line). `mypy` is a hard CI gate (the `lint` job) so CI is likely already red on these — candidate quick follow-up (typing-only fixes).

**Post-M5 batch-2 (sprint 2)** — design forks resolved via AskUserQuestion; tests green, ruff clean, 0 new mypy errors:
- **G6 — `/v1/chat/stream` Idempotency-Key (409-reject).** Same `(owner, session_id, idempotency_key)` within a TTL window → `409 duplicate_request` (no replay). In-memory `IdempotencyRegistry` (`koboi/server/idempotency.py`, configurable `server.idempotency.chat_ttl_seconds`, default 600s); checked after pre-checks and before the agent runs so duplicates 409 fast without consuming the session lock. Resolves 16.23's chat gap (jobs already deduped). Requires the client to also send `X-Session-Id` (a stable session is part of the dedup key).
- **G5a — `per_tenant_max` enforcement.** Each owner's running jobs are capped at `jobs.per_tenant_max` (default 5) → `429 too_many_jobs_per_tenant`. Enforced **only when API keys are configured** (dev/no-auth mode would collapse all owners to `dev`); counts running jobs (matches `active_count`). `JobRegistry.active_count_for_owner` + `app.py::submit_job`. Partially fills the M5 §16 per-tenant slot (real queue + pending-count still deferred to G5c).

**Post-M5 batch-3 (sprint 3)** — G5c split: tractable parts pulled forward, multi-process deferred; tests green, ruff clean, 0 new mypy errors:
- **G5c-a — Job TTL reaper.** Periodic in-process sweep reaps terminal jobs older than `jobs.ttl_seconds` (default 24h) from `JobStore` + `JobRegistry` (mirrors the workdir GC, 16.24). Bounds disk/memory; zero multi-process dependency. `JobStore.reap_terminal_older_than` + `JobRegistry.forget` + `app.py::_job_ttl_gc_loop`.
- **G5c-b — In-process queue backlog.** Beyond `max_concurrent`, jobs queue up to `queue_depth` (default 32) and run as slots free (done-callback drain) instead of `429`; the `429` now fires only when the queue is also full (`queue_full`). `JobRegistry.peek_admit`/`enqueue_pending`/`pop_pending` + `app.py::_start_job`/`_on_job_done`. Stays single-process.
- **Deferred (recorded OPEN, no code, no locked model):** **G1 multi-process** + the **external job queue** — out of v1 by locked decision #1 (in-process asyncio); the swap seams (`SessionStore`/`LockProvider`/`EventBuffer`/`TrustStore`) already exist. Concurrency model (session-affinity vs Redis-shared) and external-queue target (Redis vs Postgres-SKIP-LOCKED) left open for SaaS kickoff.

**Known edges:** `resume_on_startup` still starts all pending jobs immediately on restart (bypasses the queue); the pre-existing submit TOCTOU (check→start spans the `get_or_create` await) is inherited.

**v0.6 §20 expanded:** tambah **Phase Development Plan** detail — per fase (M0–M5 + post-M5):
modul/file yang dibentuk, komponen, dependensi, deliverable, cara testing, risk. Keputusan tak
berubah; ini perincian eksekusi.

**v0.5 → v0.6** (final checkpoint — 4 forks resolved + 4 defaults):
- **API keys** = keys file + `koboi keys` CLI (env back-compat) — fix 16.22.
- **Idempotency** = `Idempotency-Key` header + dedup window — fix 16.23.
- **Workdir** = keep tied to session/job TTL (24h GC) — resolves open Q #10; fix 16.24.
- **API shape** = koboi-native SSE+jobs primary; OpenAI-compat adapter additive later — resolves
  open Q #7.
- **Defaults:** error contract (16.25), guardrails-job-active (16.27), `/readyz`+request-id (16.26),
  drain-disconnects-interactive (16.18 expanded).
- Add decisions #12–#15; phasing distributed; remaining open Q reduced to 8 minor/future.

**v0.4 → v0.5:** Langfuse confirmed integrated (loop hook) + 16.21; self-host-first locked.

**v0.3 → v0.4:** extensibility (§6) + deployment models (§12); create_app(16.19)+state-protocol(16.20).

**v0.2 → v0.3:** per-session lock + per-session workdir; drop sync; tenancy interface-ready.

**v0.1 → v0.2:** add Job/autonomous mode; JobStore/Registry.

*End of requirements draft v0.6 (final). Ready for execution — M0 first.*

---

## Appendix: CF Tunnel deployment topology (post-M5 alignment)

### Topology

```
Browser (CF Pages FE)
  │  HTTPS + Bearer auth
  ▼
Cloudflare Edge  ←── CF Tunnel (outbound only, zero open ports on VPS)
  │  HTTP (tunnel private link)
  ▼
cloudflared (docker-compose service)
  │
  ▼
koboi:8080  (FastAPI + SSE)
```

TLS is terminated at the Cloudflare edge — no certs needed on the VPS. The tunnel authenticates
outbound from the VPS to Cloudflare; no inbound ports are opened.

### Why cloudflared replaces nginx

nginx was providing TLS (`:443`) and HTTP→HTTPS redirect for local self-host. CF Tunnel makes
both unnecessary: TLS is handled by Cloudflare, and the tunnel itself is the ingress. The
`docker-compose.yml` now ships a `cloudflared` service instead of `nginx`. `nginx.conf` is
kept in-repo as an optional/legacy path for users who prefer a local reverse proxy.

### CORS configuration

The default is **fail-closed**: no `server.cors` block = no `CORSMiddleware` = no cross-origin
access. A cross-origin browser FE (e.g. a Cloudflare Pages app at `https://app.yourdomain.com`)
requires an explicit `cors:` block in the YAML config:

```yaml
server:
  cors:
    allow_origins: ["https://app.yourdomain.com"]
    # allow_credentials: false  # keep false — auth uses Bearer header, not cookies
```

`allow_credentials` must remain `false` (the default). Auth is header-based Bearer, not
cookie-based, so wildcard CORS with `allow_credentials: false` is also safe if needed.

### SSE keepalive

Cloudflare's HTTP idle-response timeout is ~100 seconds. The `sse_stream` encoder emits a
comment frame (`": keepalive\n\n"`) every 15 seconds of generator silence. SSE clients ignore
comment frames per spec (RFC 8895 §9.1); Cloudflare resets its idle timer on each frame.

The primary silence risk is HITL approval waits (`APPROVAL_TIMEOUT = 120s`) and sparse
autonomous-job event phases — both exceed 100s. Normal LLM token streaming is real-time and
not a risk.

### Per-key rate limiting

CF WAF rate limits are per-IP only. Per-API-key rate limiting (to protect per-tenant quotas)
remains an **app-layer TODO** — the seams exist (`api_key_id` on `request.state`), but no
enforcement is wired yet.
