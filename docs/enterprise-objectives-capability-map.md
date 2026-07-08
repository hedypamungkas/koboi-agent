# Enterprise Objectives & Capability Map — koboi-agent

**Date:** 2026-07-07
**Method:** 2 agents — (1) completeness audit grounded in AWS Well-Architected (6 pillars) + SOC 2 Trust Services + Gartner Critical Capabilities + Google SRE + F500 procurement patterns; (2) infra deep-dive (HA/cluster/DR/observability/upgrades) verified at `file:line`.
**Companion to:** `docs/open-core-extension-strategy.md` (the mechanism/split) — **this doc is the "what" (objectives); that doc is the "how" (extension seams).** Read this first.
**One-line summary:** the earlier roadmap had the right *mechanism* and *feature split* but **under-articulated objectives** and a **wave order that inverts procurement dependencies**. This doc corrects both.

---

## TL;DR — the 4 corrections

1. **Objectives were under-articulated.** Enterprise ≠ "SSO + RBAC + a hosted version." A complete enterprise envelope is **12 categories / ~100 objectives** (catalogued in §1), grounded in Well-Architected/SOC2/Gartner/SRE. The earlier scope covered the *identity/governance* envelope and **systematically missed the *operational-reliability* (DR/backup/upgrade/SLA) and *commercial* (metering/billing) envelopes**, plus one security-depth miss (**KMS/CMEK**).
2. **Wave order was wrong.** The roadmap shipped *identity* (Wave 2) before the platform could be *operated or billed*. **Insert Wave 1.5 — Operational Reliability + Commercial Envelope** (DR, SLA, metering, KMS, migration, observability-depth) between state-externalization (Wave 1) and identity (Wave 2). These are procurement *prerequisites*, not polish.
3. **Core seam set grows from ~6 → ~16** (all additive, `trust.py:TrustStore` template). The infra agent found the P0 keystone **omitted `lock_provider` + `approval_registry`**, and 5 more seams are needed for HA to be real (lease/reaper, SchemaMigrator, HealthRegistry injection, MetricsSink-as-Protocol, WorkdirProvider, drain-lock-release) — plus KMS, Metering, Backup/retention seams from the commercial/reliability envelope.
4. **HA/cluster is now concrete, not hand-waved** — stateless API tier + Redis/Postgres + sticky routing + **distributed per-session lock with TTL+heartbeat = single-writer-per-session** + **fencing tokens** (the load-bearing correctness mechanism). The 3 hardest risks are named with mitigations (§5).

---

## 1. Enterprise Objectives Catalog (12 categories)

| Cat | Category | Key objectives (condensed) | Source framework |
|---|---|---|---|
| A | **Availability & Resilience** | HA (multi-node, no SPOF); DR (RPO/RTO/backup/restore/failover); failure isolation (bulkheads/circuit-breakers); graceful drain; idempotency; durable-exec/resume; **SLI/SLO/SLA + error budget** | AWS-Reliability, SOC2-Availability, SRE |
| B | **Scalability & Performance** | horizontal scaling; rlimits; per-tenant fair scheduling; load-testing/capacity model; backpressure/autoscaling; SSE fan-out | AWS-Performance, SRE |
| C | **Deployment & Operations** | multi-region/AZ; **air-gap/on-prem install**; K8s/Helm; IaC; zero-downtime upgrade; **schema migration + upgrade path**; config+secrets; runbooks/on-call; backup-restore ops; chaos/game-day | AWS-OpEx, SOC2-CC8 |
| D | **State & Data Management** | externalized state; distributed locking; durability; **encryption at rest**; encryption in transit; **KMS/CMEK/BYOK**; data residency; retention/archival; **erasure (GDPR Art.17)**; PII redaction; PITR; RAG/vector lifecycle | AWS-Reliability, SOC2-Confidentiality, GDPR |
| E | **Security & Identity** | **SSO/SAML/OIDC/JWT**; **RBAC/ABAC**; **SCIM provisioning**; API-key lifecycle; secrets mgmt; mTLS/private-endpoints; sandbox; guardrails; **vuln-mgmt/SBOM/pentest**; secure SDLC; **SIEM-grade + tamper-evident audit**; intrusion detection; **IR + breach-notification (≤72h)**; pentest program; sub-processor/supply-chain; training-opt-out | SOC2-CC6/CC7/CC9, GDPR Art.28/33, EO 14028 |
| F | **Multi-tenancy** | tenant isolation (compute/state/network); tenant lifecycle (onboard/offboard/migrate); per-tenant config/quota/RBAC; **per-tenant metering+billing**; per-tenant encryption keys; noisy-neighbor protection | SOC2-Security, enterprise-SaaS |
| G | **Observability** | `/metrics` (Prometheus); OTel tracing; structured logging; per-step agent traces; dashboards; **SLO alerting/paging**; **cost/token observability**; eval-as-observability; profiling; session-replay | SRE, AWS-Cost |
| H | **Governance & Trust** | HITL deny-by-default; graduated trust DB; AI-safety (OWASP LLM/NIST AI RMF/MITRE ATLAS); model-gov/eval-gates; bias/fairness/toxicity eval; explainability/provenance; **compliance attestation (SOC2/ISO/HIPAA/FedRAMP)**; DPA+sub-processor list; auditor evidence export; **independent attestation ("attested-safe")**; policy engine; retention/legal-hold | SOC2, NIST AI RMF, GDPR, OWASP LLM |
| I | **Integration & Extensibility** | tool/MCP ecosystem; multi-provider LLM; vector-DB/embedder pluggability; MCP client+server; webhooks/SIEM forwarders; **multi-language SDK (TS/Go)**; **OpenAI-compatible API**; migration tooling | adoption |
| J | **DX & Lifecycle** | CLI/TUI/playground; eval-as-code CI; bare-install quickstart; docs/runbooks; semver/deprecation; release mgmt; coverage gate; templates/vertical packs; marketplace | DX |
| K | **Cost / Commercial** | **metering (req/token/tool/agent)**; per-tenant quota; **billing/invoicing**; cost attribution/chargeback; cost-optimization (smart routing/caching); tiering; enterprise licensing (EULA); **cloud-marketplace listing** | AWS-Cost, commercial |
| L | **Sustainability** | efficiency/carbon reporting | AWS-Sustainability |

(Full per-objective detail + coverage ✅/🟡/❌ lives in the agent transcript; the condensed view above is the decision-useful cut.)

---

## 2. Coverage matrix (condensed — what koboi has today vs the gap)

| Category | Today | Procurement risk if unfixed |
|---|---|---|
| A. Availability/Resilience | 🟡 drain+idempotency+journal ✅; **HA/DR/SLA ❌** | **CRITICAL** — DR is procurement red-flag #1; no SLA = no contract |
| B. Scalability | ❌ can't add workers (AgentCore not concurrent-safe; SQLite single-writer) | HIGH |
| C. Deploy/Ops | 🟡 Dockerfile ✅; **no Helm/IaC/migration/runbook/air-gap-artifact ❌** | HIGH — "runnable" claim undermined |
| D. State/Data | 🟡 WAL+journal ✅; **encryption/KMS/residency/retention/erasure/PITR ❌** | CRITICAL (esp. **KMS/CMEK** — regulated buyers reject plaintext keys) |
| E. Security/Identity | 🟡 sandbox/guardrails/keys ✅; **SSO/RBAC/SCIM/SIEM-audit/mTLS/IR/SBOM ❌** | CRITICAL |
| F. Multi-tenancy | ❌ flat owner; no isolation/lifecycle/fair-share | HIGH |
| G. Observability | 🟡 Langfuse+diagnostics ✅; **no metrics/OTel/dashboards/SLO-alerts/cost-obs ❌** | HIGH — no SLO = no SLA |
| H. Governance/Trust | 🟡 HITL+trust-DB+policy ✅ (a genuine strength); **attestation/bias-eval/legal-hold ❌** | MEDIUM-HIGH |
| I. Integration | 🟡 providers/tools/MCP ✅; **no TS/Go SDK, no OpenAI-compat API ❌** | HIGH (adoption) |
| J. DX | 🟡 CLI/TUI/eval ✅; **bare-install broken (known #1 blocker)** | HIGH |
| K. Cost/Commercial | ❌ **no metering primitive at all** | **CRITICAL** — managed cloud unsellable/billable |
| L. Sustainability | ❌ | LOW |

---

## 3. The corrected wave sequencing (the structural fix)

The earlier `Wave 1→2→3→4` inverts procurement dependencies (it ships identity before the platform is operable/billable). Corrected order:

| Wave | Scope | Why this order |
|---|---|---|
| **Wave 0** | DX/install: fix bare-install (#1 known blocker), HITL demo | Adoption gate — nothing else matters if examples don't run |
| **Wave 1** | **State-externalization keystone** — the ~16 seams' foundation (state-backend injection + `lock_provider` + `approval_registry` + promote `protocols.py`) | Unlocks HA/scaling/multi-tenancy; everything below depends on it |
| **Wave 1.5 ⭐ NEW** | **Operational Reliability + Commercial Envelope**: DR/backup/restore+RPO/RTO; SLI/SLO/SLA+alerts; **metering+billing**; **KMS/CMEK (`KeyProvider`)**; schema-migration+upgrade-path; observability depth (Prometheus `/metrics`, OTel, dashboards, per-tenant cost) | **Procurement prerequisites.** You cannot sell a managed cloud or pass a F500 review without DR/SLA/metering — and they depend on Wave 1's state layer. Identity (old Wave 2) without an operable/billable platform is premature. |
| **Wave 2** | Identity envelope: SSO/SCIM, RBAC/ABAC, at-rest encryption, SIEM-grade+tamper-evident audit | The "who can do what" layer; needs Wave 1.5's KMS for key handling |
| **Wave 3** | Compliance + multi-tenancy depth: SOC2/ISO/HIPAA attestation **program** (not just a posture doc); tenant lifecycle/isolation/fair-share; retention/erasure/legal-hold/PITR; **air-gap install artifact** (deliver the sovereignty wedge) | The "prove it to auditors + isolate tenants" layer |
| **Wave 4** | Position + managed cloud: marketplace listing, TS/Go SDK, OpenAI-compat API, bias/fairness eval, SBOM/signed artifacts, attested-safe credential (Workday-Passport-style) | Commercialize + widen TAM |

> **The one-line correction:** add **Wave 1.5 (Operational Reliability + Commercial)** between Wave 1 and Wave 2. The current roadmap ships the identity envelope before the platform can be operated or billed — that inverts the actual procurement dependency order.

---

## 4. The expanded core seam set (additive; `trust.py:TrustStore` template)

The earlier seam roadmap listed ~6; the deep-dive grows it to **~16**, all additive (default = today's behavior, zero regression). Grouped by the wave that introduces them:

**Wave 1 (state keystone):**
1. **State-backend injection on `create_app`** — kwargs `session_store=/job_store=/ownership_store=/idempotency_registry=` + entry-point `koboi.state_backends`. *(P0)*
2. **`LockProvider` with TTL+renew** — `lock_provider=` kwarg; **mandatory TTL + heartbeat + fail-closed acquire** (was OMITTED from the original P0 keystone — the hardest thing to distribute, can't stay buried in `AgentPool`). *(P0)*
3. **`ApprovalRegistry` Protocol + `approval_registry=` kwarg** — HITL approvals must be externalizable (was OMITTED). *(P0)*
4. Promote `server/protocols.py` (`SessionStore`/`LockProvider`/`EventBuffer`) from decorative → runtime contract.

**Wave 1.5 (reliability + commercial):**
5. **`KeyProvider`/KMS Protocol** + `koboi.kms` entry-point — CMEK/BYOK (AWS KMS/GCP KMS/Vault/HSM). *(the security-depth miss)*
6. **`MetricsSink` Protocol** (real, not informal) + `koboi.metrics_sinks` — Prometheus `/metrics`, per-tenant cost.
7. **`MeteringSink` Protocol** + `koboi.metering_sinks` — per-req/token/tool/agent usage → billing.
8. **`SchemaMigrator` Protocol** (`upgrade()/current_revision()`) invoked from `lifespan` — enables Postgres+Alembic; OSS keeps additive default.
9. **`BackupProvider`/retention-erasure seam** — DR + GDPR Art.17.
10. **`JobStore` lease/reaper contract** (`acquire_lease/renew_lease/reap_expired`) or `JobReaper` hook — continuous cluster-wide reaping (today's `resume_on_startup` is single-node + startup-only).
11. **`WorkdirProvider` Protocol** — `pool.workdir_for()` is hardcoded local-disk; **silently blocks cross-node resume**. Enables RWX-PVC / object-store workdirs.
12. **`HealthRegistry` injection on `create_app`** — readiness must check Redis+PG+lock-provider, not just pool+db.
13. **Distributed-lock release in `_shutdown`** — today's drain relies on `asyncio.Lock` dying with the process; that breaks the moment locks are Redis-distributed.

**Wave 2 (identity):**
14. **`IdentityProvider` Protocol** + `koboi.auth_providers` — SSO/OIDC/SAML; returns `Principal(key_id/tenant_id/roles)`.
15. **`Authorizer` Protocol** — replaces hardcoded `_check_owner`/`_check_job_access`.
16. **`AuditSink` Protocol** + `koboi.audit_sinks` + **tamper-evidence (hash-chain/append-only)** — SIEM-grade; loop.py only calls `record()`.

**Cross-cutting:**
17. **`EntitlementService` OSS stub** (Grafana `OSSLicensingService` pattern) — `entitlements.enabled("feature")→False` pure-OSS, `True` when EE installed → runtime gating, one codebase/CI.
18. New entry-point groups: `koboi.hooks`, `koboi.sandbox_backends`, `koboi.approval_handlers` (~2 hrs, turns partial seams into full auto-register).

> Internal template to copy verbatim for every Protocol: `trust.py:59-79` `TrustStore` — *"defined now so a future multi-tenant store can be swapped in at the facade wiring point."* Same idiom already at `memory.py:12`.

---

## 5. The 3 hardest HA/cluster technical risks (with mitigations)

1. **Distributed per-session lock correctness under partition → double-execution of non-idempotent tool calls** (shell/write_file/git/web = real side-effects). **Mitigation: fencing tokens** — lock acquisition returns a monotonic token; every Postgres write carries `lock_token`; Postgres rejects writes whose token < the row's current. + fail-closed acquire (503 if Redis unreachable) + short TTL (30s) + heartbeat.
2. **SSE stream + stream-lifetime lock can't migrate cross-node** — live TCP is bound to one event loop; the dead-TTL window *is* the per-session downtime. **Mitigation:** short TTL (30-60s) + heartbeat renewer + journal-resume + client auto-reconnect. (Don't chase live migration — Temporal rehydrates workflows, doesn't migrate TCP.)
3. **Per-session workdir on local disk** silently defeats cross-node resume even when memory+journal are in Postgres. **Mitigation:** `WorkdirProvider` Protocol + RWX PVC (interactive) / object-store-sync (autonomous jobs).

**The concrete HA topology:** stateless API tier (N pods) + shared **Redis** (locks/idempotency/approvals/routing) + **Postgres** (jobs/memory/trust/audit/steps) + object store (RAG corpus); sticky LB on `X-Session-Id` + distributed lock = single-writer-per-session; lease-reaper for dead-node jobs; **active-passive multi-region** (per-session single-writer + streaming rules out active-active).

---

## 6. Procurement-blocker additions (top 12, ranked)

These are the objectives MISSING from the current scope that would terminate a F500 procurement call. (Each needs either an OSS seam or an EE artifact — noted.)

1. **DR + backup/restore + RPO/RTO artifact** (OSS `BackupProvider` + EE runbook) — red-flag #1.
2. **SLI/SLO/SLA + error-budget ops** (OSS `/metrics` + SLO catalog; EE SLA contract) — no SLA = unsellable cloud.
3. **Cost metering + billing pipeline** (OSS `MeteringSink` + EE Stripe/Chargebee) — cloud can't bill.
4. **KMS/CMEK/BYOK (`KeyProvider` Protocol)** (OSS Protocol + EE KMS impls) — regulated buyers reject plaintext keys.
5. **Schema migration (Alembic) + upgrade-path doc** (OSS core) — top-3 procurement question.
6. **IR + breach-notification (≤72h) runbook + DPA/sub-processor list** (EE + legal) — GDPR Art.28/33.
7. **Compliance attestation program** (SOC2 Type II → ISO 27001 → HIPAA-BAA → FedRAMP) (EE audit+legal) — posture doc ≠ auditor letter.
8. **Tenant lifecycle + isolation + noisy-neighbor fair-share** (EE) — "isolate by what?"
9. **Data retention + erasure + legal-hold + PITR** (OSS retention Policy + EE archival) — GDPR Art.5/17.
10. **Air-gap install artifact** (offline wheel + signed images + install guide) — the sovereignty *wedge* is currently undelivered; IBM watsonx + Glean have it.
11. **Observability depth** (Grafana dashboards + SLO alerts + per-tenant cost/token meter) — operators can't run it; no alerts = no on-call.
12. **SCIM 2.0 + tamper-evident audit + SIEM forwarders + OpenAI-compat API + SBOM/signed artifacts** (mixed OSS/EE) — each individually blocks a buyer segment.

---

## 7. OSS/enterprise line — confirmed (Line B), with one caveat

**Line B still holds:** ship Redis/Postgres state backends in **OSS** (adoption + sovereignty-wedge consistency + avoid "license to run production"); enterprise = SSO/RBAC/audit/encryption/multi-tenancy/compliance + managed cloud.

**Caveat the deep-dive surfaced:** several "enterprise" *objectives* still require an **OSS seam** so the enterprise repo can implement them without forking — notably `KeyProvider` (KMS), `MeteringSink` (billing), `SchemaMigrator`, `BackupProvider`, `MetricsSink`. **Contract in OSS core, implementation in enterprise repo** — the open-core principle holds, but the contract surface is bigger than the original 6 seams.

---

## 8. Next steps

1. **Adopt the corrected wave order** — insert Wave 1.5 (§3) into the roadmap. This is the single highest-leverage correction.
2. **Amend the P0 keystone** — add `lock_provider=` + `approval_registry=` (they were omitted); spec `LockProvider` with mandatory TTL+renew+fail-closed.
3. **First PR (unchanged priority, expanded scope):** P0 keystone (state injection + lock_provider + approval_registry + promote protocols.py) + `EntitlementService` stub + `import-linter` as-if-foss CI + LICENSE→Apache-2.0. Then Wave 1.5 seams.
4. **Decide the Line-B refinement** — confirm basic Redis/Postgres backends ship in OSS; advanced scale-out (multi-region HA, cross-cluster resume) in EE.
5. **Stand up the enterprise-repo skeleton** implementing the first backend (Redis state) to prove the contract round-trips — but only AFTER the Wave 1.5 reliability seams exist, or the "enterprise" repo would ship HA without DR/SLA/metering (the procurement-fail scenario).

*Framework citations (AWS Well-Architected, SOC 2 TSC, Gartner, Google SRE) live in the completeness-audit agent transcript.*

---

## Appendix A — Full per-objective gap matrix

The complete enterprise-objective coverage at HEAD `31b953f`, code-verified. **Legend:** ✅ covered today · 🟡 partial (some coverage, gaps) · ❌ gap (no seam, no artifact). **Tally: ~18 ✅ / ~25 🟡 / ~50 ❌.** The ❌ + 🟡 rows ARE the enterprise feature-gap backlog.

### A. Availability & Resilience
| ID | Objective | Status | What's needed |
|---|---|---|---|
| A1 | HA (multi-node, no SPOF) | 🟡 | Wave 1 state + K8s/Helm + session-affinity design |
| A2 | DR (RPO/RTO/backup/restore/failover) | ❌ | DR runbook + automated backup/restore + cross-region failover |
| A3 | Failure isolation (bulkheads/circuit-breakers) | 🟡 | outbound circuit breakers on LLM/vector calls |
| A4 | Graceful drain | ✅ | — |
| A5 | Idempotency | ✅ | — |
| A6 | Durable exec / resume | 🟡 | cross-cluster resume; promote task.py:112 DAG; checkpointer/time-travel |
| A7 | SLI/SLO/SLA + error budget | ❌ | SLO catalog + alert gates; SLA for cloud |

### B. Scalability & Performance
| ID | Objective | Status | What's needed |
|---|---|---|---|
| B1 | Horizontal scaling | ❌ | Wave 1 state layer (prerequisite) |
| B2 | rlimits | ✅ | — |
| B3 | Per-tenant fair scheduling | ❌ | distributed fair-share scheduler |
| B4 | Load testing / capacity model | ❌ | capacity test suite + sizing guide |
| B5 | Backpressure / autoscaling | ❌ | HPA + concurrency limits |
| B6 | SSE fan-out | 🟡 | EventBuffer Protocol (single-process today) |

### C. Deployment & Operations
| ID | Objective | Status | What's needed |
|---|---|---|---|
| C1 | Multi-region / AZ | ❌ | (deferred — document as gap) |
| C2 | Air-gap / on-prem install | 🟡 | offline wheel + signed images + install guide (**wedge is undelivered**) |
| C3 | K8s / Helm | 🟡 | Helm chart + operator eventually |
| C4 | IaC (Terraform/Pulumi) | ❌ | reference module |
| C5 | Zero-downtime upgrade | 🟡 | rolling/rollback path doc |
| C6 | Schema migration + upgrade path | ❌ | Alembic + major-version upgrade doc |
| C7 | Config + secrets at deploy | 🟡 | vault integration |
| C8 | Runbooks / on-call guide | ❌ | SRE ops guide per failure mode |
| C9 | Backup + restore ops | ❌ | tested restore drills |
| C10 | Chaos / game-day | ❌ | failure-injection harness |

### D. State & Data Management
| ID | Objective | Status | What's needed |
|---|---|---|---|
| D1 | Externalized state layer | 🟡 | P0 keystone (kwargs + entry-point) |
| D2 | Distributed locking | ❌ | `RedisLockProvider` |
| D3 | Data durability | 🟡 | replication |
| D4 | Encryption at rest | ❌ | Wave 2 |
| D5 | Encryption in transit | 🟡 | mTLS / private endpoints |
| D6 | KMS / CMEK / BYOK | ❌ | `KeyProvider` Protocol + KMS impls |
| D7 | Data residency / region pinning | ❌ | per-tenant region config |
| D8 | Retention & archival | ❌ | configurable retention + cold storage |
| D9 | Erasure (GDPR Art.17) | ❌ | tenant/subject erasure API + cascade delete |
| D10 | PII redaction in logs/audit | ❌ | redact at the `AuditSink` seam |
| D11 | PITR | ❌ | Postgres WAL archiving |
| D12 | RAG / vector lifecycle | 🟡 | corpus versioning / reindex |

### E. Security & Identity
| ID | Objective | Status | What's needed |
|---|---|---|---|
| E1 | SSO / SAML / OIDC / JWT | ❌ | `IdentityProvider` Protocol (P0) |
| E2 | RBAC / ABAC | ❌ | `Authorizer` Protocol (P1) |
| E3 | SCIM / JIT provisioning | ❌ | SCIM 2.0 endpoint |
| E4 | API-key lifecycle | ✅ | — |
| E5 | Secrets management | 🟡 | vault/KMS protocol |
| E6 | Network security / mTLS | ❌ | mTLS + private endpoints |
| E7 | Sandbox / workload isolation | ✅ | — (differentiator) |
| E8 | Guardrails (I/O, policy, injection) | ✅ | — |
| E9 | Vuln mgmt / SBOM / pentest | 🟡 | CycloneDX SBOM + scheduled pentest |
| E10 | Secure SDLC | 🟡 | commit signing |
| E11 | SIEM-grade + tamper-evident audit | ❌ | `AuditSink` Protocol + hash-chain |
| E12 | Intrusion / anomaly detection | ❌ | outbound-anomaly (exfil) detection |
| E13 | Incident response + breach notification (≤72h) | ❌ | IR runbook + 72h notification contract |
| E14 | Penetration testing program | ❌ | annual + post-major-change pentest |
| E15 | Sub-processor / supply chain | 🟡 | signed artifacts + sub-processor list |
| E16 | Privacy / training opt-out | ✅ | — (structural: self-host) |

### F. Multi-tenancy
| ID | Objective | Status | What's needed |
|---|---|---|---|
| F1 | Tenant isolation (compute/state/network) | ❌ | isolation design |
| F2 | Tenant lifecycle (onboard/offboard/migrate) | ❌ | lifecycle APIs + scripts |
| F3 | Per-tenant config/quota/RBAC | ❌ | tenant-scoped policy/limits/roles |
| F4 | Per-tenant metering & billing | ❌ | see K1/K3 |
| F5 | Per-tenant encryption keys | ❌ | depends on D6 |
| F6 | Noisy-neighbor protection | ❌ | distributed per-tenant rate/fair-share |

### G. Observability
| ID | Objective | Status | What's needed |
|---|---|---|---|
| G1 | `/metrics` Prometheus | ❌ | Wave 1.5 |
| G2 | OpenTelemetry tracing | ❌ | Wave 1.5 |
| G3 | Structured logging + aggregation | 🟡 | ship-to-sink (Loki/ELK/Splunk) |
| G4 | Per-step agent traces | ✅ | — |
| G5 | Dashboards / SLO panels | ❌ | Grafana dashboard JSON |
| G6 | Alerting / paging | ❌ | SLO-based alerts |
| G7 | Cost & token observability | 🟡 | per-tenant/per-run meter |
| G8 | Eval-as-observability | ✅ | — |
| G9 | Performance profiling | ❌ | flame graphs |
| G10 | Session replay / debug | 🟡 | replay UI |

### H. Governance & Trust
| ID | Objective | Status | What's needed |
|---|---|---|---|
| H1 | HITL / deny-by-default approvals | ✅ | — |
| H2 | Graduated trust DB | ✅ | — |
| H3 | AI safety (OWASP LLM / NIST AI RMF / MITRE ATLAS) | 🟡 | coverage matrix |
| H4 | Model governance / eval gates | 🟡 | deploy gate blocking on regression |
| H5 | Bias / fairness / toxicity eval | ❌ | scorers in eval registry |
| H6 | Explainability / provenance | 🟡 | attribution UI |
| H7 | Compliance attestation (SOC2/ISO/HIPAA/FedRAMP) | ❌ | audit **program** (not a posture doc) |
| H8 | DPA + sub-processor list | ❌ | publish + sign |
| H9 | Auditor evidence export | ❌ | evidence-pack exporter |
| H10 | Independent attestation ("attested-safe") | ❌ | attested-safe credential (Workday-Passport-style) |
| H11 | Policy engine | ✅ | — |
| H12 | Records retention / legal hold | ❌ | legal-hold mechanism |

### I. Integration & Extensibility
| ID | Objective | Status | What's needed |
|---|---|---|---|
| I1 | Tool / MCP plugin ecosystem | ✅ | — |
| I2 | Multi-provider LLM | ✅ | — |
| I3 | Vector DB / embedder pluggability | 🟡 | managed vector DB (Pinecone/Weaviate/pgvector) |
| I4 | MCP client + server | ✅ | — |
| I5 | Webhooks / SIEM forwarders | ❌ | AuditSink forwarders (Splunk/Sumo/S3) |
| I6 | Multi-language SDK | ❌ | TS/Go server client |
| I7 | OpenAI-compatible API | ❌ | `/v1/chat/completions` parity |
| I8 | Migration tooling | ❌ | LangChain/CrewAI importers |

### J. DX & Lifecycle
| ID | Objective | Status | What's needed |
|---|---|---|---|
| J1 | CLI / TUI / playground | ✅ | — |
| J2 | Eval-as-code in CI | ✅ | — |
| J3 | Bare-install quickstart | 🟡 | broken — **#1 known blocker** (Wave 0) |
| J4 | Docs / runbooks | 🟡 | SRE runbooks |
| J5 | Semver / deprecation policy | 🟡 | published deprecation policy |
| J6 | Release management | ✅ | — |
| J7 | Test-coverage gate | ✅ | — |
| J8 | Templates / vertical packs | 🟡 | vertical starter repo |
| J9 | Agent/tool/skill marketplace | ❌ | — |

### K. Cost / Commercial
| ID | Objective | Status | What's needed |
|---|---|---|---|
| K1 | Metering (req/token/tool/agent) | ❌ | `MeteringSink` Protocol + per-call recording |
| K2 | Per-tenant quota / rate limit | 🟡 | distributed quota |
| K3 | Billing / invoicing | ❌ | billing pipeline (Stripe/Chargebee) |
| K4 | Cost attribution / chargeback | ❌ | showback/chargeback |
| K5 | Cost optimization (smart routing/caching) | ❌ | resurrect `AgentDef.llm_config` + escalation |
| K6 | Tiering / discounts / committed-use | ❌ | enterprise tier SKU |
| K7 | Enterprise licensing (EULA) | 🟡 | draft EULA |
| K8 | Cloud-marketplace listing | ❌ | AWS/Azure/GCP listing |

### L. Sustainability
| ID | Objective | Status | What's needed |
|---|---|---|---|
| L1 | Efficiency / carbon reporting | ❌ | optional kWh/token |

> **How to read this as a backlog:** every ❌ is either an **OSS seam** to add (so the enterprise repo can implement it without forking) or an **enterprise-repo artifact** to ship — the "where" column in §6 + the seam list in §4 assign each. The ~50 ❌ rows are the enterprise feature gaps; the ~25 🟡 rows are partial gaps that need deepening.
