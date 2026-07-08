# Open-Core Extension Strategy for koboi-agent

**Date:** 2026-07-07
**Method:** 3 parallel agents — (1) map existing extension seams in core, (2) identify hardcoded gaps + minimal seams to add, (3) research open-core patterns across 9 projects (GitLab, Grafana, Temporal, Mattermost, Sentry, Supabase, PostHog, Metabase, Airbyte). All primary-sourced (curl/GitHub raw/Wayback; codebase seams verified at `file:line`).
**Companion to:** `docs/competitor-enterprise-benchmark.md` + project memory `enterprise-readiness-assessment`.

> ⚠️ **REFINED 2026-07-07 — read `docs/enterprise-objectives-capability-map.md` first.** That doc (the "what"/objectives) surfaced three corrections to this doc (the "how"/mechanism):
> 1. **Insert Wave 1.5 (Operational Reliability + Commercial: DR / SLA / metering / KMS / migration / observability-depth) between Wave 1 and Wave 2.** The wave order below inverts procurement dependencies — it ships identity before the platform is operable/billable.
> 2. **The P0 keystone must also include `lock_provider=` (with mandatory TTL + heartbeat + fail-closed acquire).** The per-session lock is the hardest thing to distribute and was omitted from the kwarg list below — it can't stay buried in `AgentPool`.
> 3. **The core seam set grows ~6 → ~16** (all additive): add `KeyProvider`(KMS), `MeteringSink`(billing), `SchemaMigrator`(Alembic), `BackupProvider`(DR/erasure), `MetricsSink`(real Protocol), `JobStore` lease/reaper, `WorkdirProvider`, `HealthRegistry` injection, distributed-lock release in `_shutdown`, `ApprovalRegistry` Protocol. Contract-in-core still holds; the contract surface is just bigger.

---

## TL;DR

- **Structure → Pattern A:** separate `koboi-agent-enterprise` repo + PyPI package, `depends on koboi-agent>=X`, registers its implementations against **Protocols declared in OSS core** via entry points. Core never imports enterprise.
- **License → Apache-2.0 core** (not current MIT; absolutely not AGPL/FSL/BUSL — koboi is a *library*, those kill adoption vs LangGraph/CrewAI MIT).
- **The OSS/enterprise line → ship Redis/Postgres state backends in OSS** (Temporal/Supabase playbook); enterprise = SSO/RBAC/audit/encryption/multi-tenancy/compliance + managed cloud. *(This is the one decision to confirm — see §2.)*
- **Core is ~55% extension-ready today.** The Protocols already exist (`MemoryBackend`, `TrustStore`, `server/protocols.py`); the gap is **wiring** (the composition root ignores them) + an **Auth seam** + a few **entry-point groups**. The work is additive, not a redesign.

---

## 1. The structure: Pattern A (separate package via entry points)

```
koboi-agent            (OSS, public, Apache-2.0)   ← defines Protocols + entry-point groups + OSS stubs
   ▲
   │  depends on (one-way)
   │
koboi-agent-enterprise (separate repo, proprietary/Source-Available)  ← implements Protocols, registers via entry points
```

- **Registration mechanism already exists** — `koboi/plugins.py` discovers `koboi.providers/guardrails/scorers/tools` via `importlib.metadata.entry_points()`. Enterprise adds new groups (`koboi.state_backends`, `koboi.auth_providers`, `koboi.rbac`, `koboi.audit_sinks`, `koboi.kms`, `koboi.tenants`, `koboi.entitlements`) and self-registers on import.
- **Pattern lineage:** Mattermost's `init()`-registration + Grafana's OSS-stub-interface, ported to Python. Every studied project with a separate enterprise artifact uses this shape.
- **Distribution:** enterprise as a private wheel (GitHub release / AWS CodeArtifact / self-hosted index) — `pip install koboi-agent-enterprise` on top of `pip install koboi-agent`.

**Why not Pattern B (same-repo `enterprise/` carve-out like PostHog/Metabase):** it contradicts your stated "separate enterprise repo" goal, exposes enterprise source to all OSS contributors, and complicates the license story. **Why not Pattern C (pure-OSS + cloud only, Temporal/Supabase):** it offers zero protection against a competitor wrapping the core — viable only if you commit to running a cloud business and accept fork-and-host risk.

---

## 2. The OSS/enterprise line — the one decision to confirm

Where you draw the line determines your moat. Two viable lines:

| Line | In OSS core | In enterprise repo | Precedent |
|---|---|---|---|
| **A (your initial instinct)** | single-node + SQLite | Redis/Postgres state, SSO, RBAC, audit, encryption, multi-tenancy | (minority) |
| **B (recommended)** | **single-node + SQLite + Redis/Postgres state backends** | SSO, RBAC, audit, encryption, multi-tenancy, compliance + managed cloud | **Temporal, Supabase, Grafana, Metabase, Mattermost** |

**Recommendation: Line B.** Reasons:
1. **Consistency with your wedge.** Your positioning is the *open control plane / data sovereignty / no lock-in* play (Grafana/Temporal/Ollama). Gating the ability to run production behind a license **contradicts** that wedge. Line B lets anyone self-host a real production deployment — the adoption engine.
2. **Avoids "license to run production" friction** — the exact struggle Sentry/FSL projects hit. Enterprises respect paying for *security/compliance/ops*, not for the basic ability to scale.
3. **The stickier moat is the security/compliance layer.** That's where Grafana, Metabase, AND Mattermost all draw the line — it's higher-value, harder to replicate, and procurement-critical.
4. **State-backends-in-OSS removes the awkward monetization friction** and maximizes the contributor/adoption flywheel.

**Middle path if you want both:** ship a *basic* Redis/Postgres backend in OSS (single-region, simple locks) and reserve *advanced* scale-out (multi-region HA failover, cross-cluster resume, sharding) for enterprise. Basic-in-OSS / advanced-in-enterprise is a common compromise.

> ⚠️ **This is the strategic call to make.** Everything below (the core seam roadmap) is identical regardless of A or B — the seam must live in core either way; only *where the implementation ships* differs.

---

## 3. What core must do FIRST — the seam roadmap (all additive, non-breaking)

Core extension-readiness today: **~55%.** Solid entry-point seams already exist for LLM providers, guardrails, tools, scorers. The Protocols for memory/trust/server-state are **already defined** but **decorative** — `create_app` and `AgentAssembler` hardcode the concrete impls and never consult them. Auth has no seam at all.

| Pri | Seam to add in core | Shape | Effort | Unlocks |
|---|---|---|---|---|
| **P0 (keystone)** | **State-backend injection on `create_app`** | Add kwargs `session_store=/job_store=/ownership_store=/idempotency_registry=/approval_registry=None` → default to current in-process classes; promote `server/protocols.py` to runtime contract; add `koboi.state_backends` entry-point | ~½ day | **HA + horizontal scaling + multi-tenancy + encryption-via-DB** (Redis/Postgres impls) |
| **P0** | **`IdentityProvider` Protocol + injection point** | `validate(request)→Principal` (carries `key_id/tenant_id/roles`); wrap current Bearer as `ApiKeyIdentityProvider` default; `create_app(identity_provider=None)` + `koboi.auth_providers` entry-point | ~1 day | **SSO/OIDC/SAML/JWT** + prerequisite for RBAC + real multi-tenancy |
| P1 | **`Authorizer` Protocol** | Replace hardcoded `_check_owner`/`_check_job_access` (`app.py:348`) with `Authorizer.can_access(principal, resource, action)`; default `OwnerAuthorizer` | ~1 day | RBAC (depends on P0 auth) |
| P1 | **`AuditSink` Protocol + entry-point** | Promote `AuditTrail.record()` to Protocol; default `SQLiteAuditSink`; `koboi.audit_sinks` (`loop.py` only calls `record()`) | ~½ day | SIEM / append-only audit |
| P1 | **`MemoryBackend` entry-point + async note** | Protocol exists (`memory.py:12`); add `koboi.memory_backends` discovery + document sync→`to_thread` | ~½ day | Postgres memory (multi-node) |
| P1 | **`EntitlementService` OSS stub** | `entitlements.enabled("feature")→False` default; `koboi.entitlements` group | ~½ day | Runtime gating of enterprise features without build flags (Grafana pattern) |
| Trivial | **New entry-point groups** | Add `koboi.hooks`, `koboi.sandbox_backends`, `koboi.approval_handlers` to `plugins.py:_GROUPS` | ~2 hrs | Hooks/sandbox/approval auto-register on `pip install` |

**Additive-by-default CONFIRMED:** every kwarg defaults to `None`→today's behavior; every entry-point group empty in core = no-op; every config knob defaults to current; Pydantic `extra="ignore"` preserves an `enterprise:` section today. **Zero regression** — `passthrough`/single-node/Bearer stays the default.

**The internal template to copy verbatim:** `trust.py:59-79` `TrustStore` — *"defined now so a future multi-tenant store can be swapped in at the facade wiring point without editing loop_pipeline.py or its tests."* That sentence is the open-core contract; the same pattern is already used at `memory.py:12`.

---

## 4. Non-negotiables (rules every successful open-core project follows)

1. **Core NEVER imports enterprise; enterprise imports core.** Universal across all 9 projects. **Enforce in koboi** with: an `import-linter` `forbid-contracts` rule failing CI if any `koboi.*` imports `koboi_enterprise.*`, **plus an `as-if-foss` CI job** (GitLab's trick) that uninstalls the enterprise package and runs the full suite.
2. **Contract-in-core:** if enterprise will implement it, the Protocol/ABC + entry-point group MUST be declared in OSS core. Core holds contracts + OSS stubs; enterprise holds real implementations.
3. **Additive-only — enterprise-uninstalled = pure OSS behavior.** Every enterprise-gated path has an OSS no-op stub.
4. **Per-file/per-directory license boundary, machine-checkable.** Explicit file headers + a dedicated LICENSE in the enterprise subtree (PostHog/Metabase/GitLab/Mattermost all do this).
5. **License matches adoption model, not defensiveness instinct.** Library/framework consumed by developers → permissive. End-user app run-as-is → AGPL viable. SaaS-defensive server → FSL/BUSL viable. **Wrong category kills adoption.**
6. **One canonical registration mechanism, documented** — extend the existing entry-point system; don't invent a parallel one.

---

## 5. The EntitlementService pattern (the one clever mechanism to steal)

From Grafana's `OSSLicensingService` (copy verbatim): every enterprise-gated code path in OSS calls `entitlements.enabled("sso")`. In pure-OSS it returns `False` (feature inert). When `koboi-agent-enterprise` is installed, its `EntitlementService` impl validates a license token and returns `True`. **This is how additive-only behavior is enforced at runtime without build flags or separate artifacts** — the same code path runs in both editions; only the entitlement result differs. It's the cleanest way to keep one codebase, one CI, zero `if ENTERPRISE:` branches scattered around.

---

## 6. License rationale — Apache-2.0 core

- **Switch from current MIT → Apache-2.0.** The explicit patent grant + patent-retaliation clause matters for an AI framework (model/patent exposure), and enterprise procurement teams are measurably more comfortable with Apache-2.0 (Supabase chose Apache over MIT for exactly this).
- **NOT AGPL** (Grafana/Metabase). AGPL's §13 network-use clause is catastrophic for a *library* developers embed into their own products — it would force every user building a commercial AI-agent product on koboi to open-source their whole app. Grafana/Metabase can use AGPL because they're *end-user applications*; koboi is a *library*. **AGPL hands LangGraph (MIT) and CrewAI (MIT) the entire adoption surface.**
- **NOT FSL/BUSL** (Sentry/HashiCorp). The "Competing Use" clause would make it illegal for a user to build substantially-similar functionality commercially — for koboi that means a user couldn't legally build a commercial AI-agent product *with* koboi. Kills your consumer/embedding model outright.
- **Enterprise repo:** proprietary (closed) or Source-Available (Mattermost-style "production use requires subscription").
- **The competitive moat is the enterprise features + managed cloud, not the core license.** Matches Temporal (MIT, moat=ops), Supabase (Apache, moat=managed+data-gravity), GitLab (MIT core, moat=proprietary EE).

---

## 7. Precedent — what to take from each

| Project | Take this | License |
|---|---|---|
| **Temporal** | Ship state backends (Cassandra/SQL) + multi-tenancy (namespaces) **in OSS**; cloud = ops/SSO/SLA | MIT |
| **Supabase** | "If the tool doesn't exist, build and open source it" — adoption-first; cloud = managed value | Apache-2.0 |
| **Grafana** | The `OSSLicensingService` stub pattern (→ EntitlementService); private peer-repo + build-tag import manifest | AGPL (app — not for koboi) |
| **Mattermost** | Separate private `enterprise` peer-repo + `init()` self-registration against OSS Go interfaces; per-file `LICENSE.enterprise` | AGPL + Source-Available EE |
| **GitLab** | The `as-if-foss` CI job (`rm -rf ee/` + full suite) to enforce no-core-deps-on-EE | MIT CE + proprietary EE |
| **PostHog** | Python same-repo `ee/` one-way import (`ee/settings.py` imports `posthog.settings`) — the closest Python analog if you ever pick Pattern B | MIT + custom EE |
| **Metabase** | The 7-line "router" LICENSE at repo root pointing to two licenses (cleanest legal mechanism) | AGPL + Commercial |
| **Sentry** | Cautionary: FSL's restrictions hurt the embedding use case — **don't** | FSL |
| **Airbyte** | Cautionary: ELv2 loses the OSI "open source" label, enterprise legal red-flags it — **don't** | ELv2 |

---

## 8. Next steps

1. **Confirm the OSS/enterprise line** (§2) — Line B recommended (state backends in OSS), middle-path (basic OSS / advanced enterprise) as fallback.
2. **Ship the P0 keystone as the first PR** — state-backend injection on `create_app` + promote `server/protocols.py` to runtime contract + `koboi.state_backends` entry-point. ~½ day, additive, unlocks the most.
3. **In the same PR or紧跟**: add the `EntitlementService` OSS stub + the `import-linter` `as-if-foss` CI guard + switch LICENSE to Apache-2.0. These three together make core genuinely open-core-ready.
4. **Then** P0 `IdentityProvider` (SSO seam) → P1 `Authorizer`/`AuditSink`/`MemoryBackend` entry-points → stand up the `koboi-agent-enterprise` skeleton repo that implements the first backend (Redis state) to prove the contract round-trips.
5. **Defer** the "advanced scale-out" (multi-region HA, cross-cluster resume) decision until the basic Redis backend proves out — that's where the Line A vs B middle-path gets resolved on real evidence.
