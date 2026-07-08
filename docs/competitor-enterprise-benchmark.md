# koboi-agent vs the Enterprise AI Agent Market — Competitive Benchmark

**Date:** 2026-07-07
**Method:** 6-agent research fleet + 1 peer-contributed deep-dive (Workday). All primary-sourced via `curl` on official/trust/pricing pages, Wayback Machine snapshots, and GitHub raw READMEs/LICENSE files. `WebSearch`/`WebFetch` were hard rate-limited until 2026-07-28, so items not confirmable from fetched primary sources are tagged `[unverified]`. OSS licenses independently verified against repo LICENSE files.
**Companion to:** the enterprise-readiness assessment (see project memory `enterprise-readiness-assessment`).

---

## TL;DR

- **koboi's defensible uniqueness:** it is the *only* framework in its OSS peer set (LangGraph, CrewAI, LlamaIndex, AutoGen/AG2) that ships a **complete runnable stack in OSS** — FastAPI/SSE server + CLI + Textual TUI + Bearer-key auth + async jobs + step-journal resume + RAG + input/output guardrails + **seccomp sandbox** + graduated Trust-DB HITL + eval-as-code + YAML portability — with **zero paid-tier gating of the server**. The peers are libraries first; their runnable server, sandbox, and enterprise controls are all commercial/cloud.
- **It is NOT enterprise-ready today** on two axes: (1) single-node, in-process control plane (no HA/scaling/multi-node state); (2) no SSO/RBAC, no at-rest encryption, no SIEM-grade audit, no multi-tenancy, no compliance posture. Both are *integration* problems — the seams (`server/protocols.py` Protocol stubs, hook-based observability, plugin registries) already exist.
- **Don't out-compliance Salesforce/Bedrock/Google (you'll lose).** Own the **open-control-plane wedge**: data sovereignty + no lock-in + guardrails/eval-as-code inspectability. Two-pronged: vs IBM/hyperscalers = *"sovereignty without the OpenShift/IBM-Consulting tax"*; vs SaaS black-boxes = *"trust via inspectability, evolvable toward independent attestation."*

---

## 1. The 2026 enterprise bar (table stakes — 80%+ of managed competitors ship these)

SSO/SAML+OIDC · SCIM+RBAC · SOC 2 Type II (+ ISO 27001 at parity) · HIPAA+BAA · GDPR+data residency · long-retention audit+export · data-training opt-out · agent guardrails (PII redaction, prompt-injection) · HITL/escalation · per-step observability/traces · multi-region HA+SLA · TLS + AES-256 at rest (+ CMEK/BYOK increasingly demanded).

**koboi clears:** guardrails, HITL/escalation, data-training opt-out (user's own deployment), per-step traces (journal). **koboi fails (~7/12):** SSO, RBAC, SOC2/HIPAA posture, at-rest encryption, SIEM audit, multi-region HA/SLA, data residency.

> FedRAMP, air-gapped/on-prem, BYO-cloud, and customer-managed keys remain *differentiators*, not table stakes (<30% of the field) — this is where the opening sits.

---

## 2. Competitive threat ranking (koboi's POV)

| Rank | Competitor | Why it's the threat |
|---|---|---|
| **#1** | **LangGraph (+ LangSmith)** | Deepest durable-exec (checkpointer/time-travel/Postgres); **LangSmith is the industry-default observability/eval platform**; named traction (Klarna, Uber, J.P. Morgan). Targets the same serious-engineering-team segment. |
| **#2** | **CrewAI (+ AMP Enterprise)** | The **only OSS peer with a shipping Enterprise SKU** — SSO/RBAC, Crew Studio, SAM-cert + FedRAMP High option, Fortune-500 traction — on the *exact* SSO/compliance/multi-tenancy axis koboi is absent. The buy-vs-build threat. |
| Watch | **Google ADK** | Closest OSS architectural analog (Apache-2.0, code-first, ADK 2.0 graph workflows+HITL) — but Gemini-first and still stabilizing (2.0 breaking changes). |
| Watch | **LiveKit Agents** | Genuine OSS-self-host peer (Apache-2.0 framework *and* server, 30k+ stars, MCP + LLM-judge test framework) — but voice-native, zero overlap with koboi's text/RAG/sandbox surface. Adjacent, not competitive today. |

---

## 3. Consolidated landscape (condensed; per-competitor verified essence)

### A. Managed enterprise platforms (vendor-locked, full-stack)

- **Salesforce Agentforce** — SaaS on Hyperforce; CRM-data-grounded agents + Einstein Trust Layer; full compliance matrix (SOC1/2/3, HIPAA-BAA, FedRAMP, ISO 27xxx/27701, PCI, HITRUST). Pricing: Flex Credits $500/100k + $2/conversation + $125/user/mo add-ons. *Beats koboi:* compliance/identity/CRM ecosystem. *koboi wins:* self-host/seccomp/eval/no-lock-in.
- **ServiceNow Now Assist / AI Agents** — SaaS; workflow-native (ITSM/HRSD/CSM) + Moveworks (closed Dec-2025, ~$2.85B) + Now Assist Control Tower. SOC2/ISO/HIPAA/FedRAMP Moderate. `[most product pages JS-gated → largely unverified]`. *Beats koboi:* workflow-graph grounding. *koboi wins:* workflow-agnostic + self-host.
- **Microsoft Copilot Studio + Foundry Agent Service + SK + AutoGen/AG2 + M365 Copilot** — broadest stack; **deepest compliance verified** (FedRAMP High + DoD IL5 + ISO 42001 + HITRUST, learn.ms compliance page); Entra ID + Purview governance. *Beats koboi:* compliance breadth + identity + M365 distribution. *koboi wins:* one batteries-included self-host stack (MS = assemble 3–4 products).

### B. Cloud-native agent services

- **AWS Bedrock Agents → AgentCore** ⚠️ *Agents Classic closes to new customers 2026-07-30; AgentCore is go-forward.* 30+ regions; FedRAMP High/HIPAA/ISO/SOC/CSA STAR L2; **"Automated Reasoning" = unique formal-logic guardrail**; managed Knowledge Bases. No on-prem. *Beats koboi:* managed scale + compliance + Automated Reasoning. *koboi wins:* self-host/BYO-cloud/no-lock-in/seccomp/eval-as-code.
- **Google Vertex Agent Builder + ADK + Agentspace** — **deepest compliance in the market** (FedRAMP High + CJIS + FIPS 140-2 + ISO 42001 + 60 frameworks); Model Armor (model-agnostic AI firewall); ADK (Apache-2.0) is the closest OSS analog but Gemini-first + ADK 2.0 breaking changes. *Beats koboi:* insurmountable compliance depth. *koboi wins:* true multi-provider + no GCP dependency + seccomp + YAML portability.
- **IBM watsonx Orchestrate + watsonx.governance** ⭐ — **REAL on-prem/air-gap via Cloud Pak for Data on OpenShift** (primary-sourced on live CPD page: "available for self-hosting, or as a managed service"); **watsonx.governance = 2026 Gartner MQ Leader for AI Governance**. SOC/ISO/HIPAA/HITRUST/FedRAMP. *This is THE enterprise bar for koboi's sovereignty wedge.* *Beats koboi:* turnkey on-prem + governance + compliance artifacts. *koboi wins:* "same sovereignty, ~100× less ceremony — no OpenShift cluster, no IBM Consulting, no licensing."

### C. Foundation-model vendors

- **OpenAI** — SaaS-only; **NO FedRAMP / NO on-prem / NO self-host** (confirmed); SOC2 + HIPAA-BAA. Agents SDK is open + multi-provider but **client-library-only, no server**. *Beats koboi:* frontier models + distribution. *koboi wins:* self-host air-gapped + multi-provider *server* + seccomp + eval.
- **Anthropic** — **multi-cloud (Bedrock + Vertex + MS Foundry GA 2026-06-29)** = the procurement shortcut koboi can't match; SOC2/HIPAA/ISO direct, no direct FedRAMP. Model is single-vendor (Claude). *Beats koboi:* multi-cloud distribution. *koboi wins:* true provider neutrality + self-host + seccomp + eval.
- **Google Gemini** — Workspace distribution (billion-user reach) + deepest compliance matrix. *Beats koboi:* distribution + compliance. *koboi wins:* provider-neutral + self-host anywhere + seccomp.

### D. OSS frameworks (koboi's closest peers — all libraries first)

- **LangChain / LangGraph** — MIT; low-level orchestration library; **durable exec is the headline** (checkpointer + `interrupt()`/resume + time-travel + Postgres/SQLite); **LangSmith = observability/eval gold standard (paid)**; no free OSS server (LangServe deprecated under restrictive license); no OSS sandbox (LangSmith Sandboxes = paid cloud). **Top threat.**
- **CrewAI** — MIT (verified; *not* Apache-2.0 despite rumors); library + local CLI, role-based DX (`Agent(role/goal/backstory)`), 55k stars; **only peer with shipping Enterprise SKU (AMP)** — SSO/RBAC, Crew Studio, Workload-Identity Secrets Manager, SAM/FedRAMP, Fortune-500; no free server, no OSS sandbox (relies on paid cloud VMs E2B/Daytona). **Top threat (buy-vs-build).**
- **LlamaIndex** — MIT; data/RAG-specialist library; **best-in-class RAG** (300+ connectors, 130+ parse formats) + **best-in-class OSS RAG eval metrics** (faithfulness/answer-relevancy/correctness + Ragas + native OTel); no server, no sandbox, no audit/Trust-DB. RAG/eval depth exceeds koboi.
- **Microsoft AutoGen / AG2** — AutoGen MIT but **maintenance mode** → Microsoft Agent Framework (MAF, MIT) is the enterprise successor; **AG2 = Apache-2.0 community fork**. **Only OSS peer with a real sandbox** (Docker + Jupyter code executors); OpenTelemetry-first distributed tracing; GroupChat pedigree. No production server (AutoGen Studio explicitly non-production).

### E. Enterprise agent startups (closed SaaS; deep-dive 2026-07-07)

- **Sierra** — SaaS; **FedRAMP High + ~$15B valuation** (strongest federal posture among pure-play agent startups).
- **Decagon** (~$4.5B, Series D $250M, Coatue/Index) — pure SaaS, **no sovereignty option** (no single-tenant/VPC/on-prem); SOC2 only (no FedRAMP/HIPAA surfaced). **Best-in-class observability** (Trace View + Watchtower always-on QA + Simulations + A/B testing) + **Agent Operating Procedures** (natural-language workflows) = its wedge; CX-vertical. *Beats koboi:* observability/eval-loop polish + vertical CX depth + funding. *koboi wins:* self-hostable/open + sandbox+seccomp + eval-as-code (customer-owned, not proprietary Watchtower).
- **Glean** (~$300M ARR / $7.2B Series F; $100M ARR confirmed Feb-2025) — **GENUINE single-tenant-on-customer-cloud** (AWS/Azure/GCP, multi-region) verified on security page; SOC2/HIPAA/GDPR. **The only documented customer-cloud sovereignty deployment among startups → the most direct threat to koboi's wedge.** Enterprise Context (Personal+Enterprise Graph, 100+ connectors); "Glean Protect for Agents" (AUP/DLP/action-validation). *BUT "your cloud" is Glean-operated, not customer-operated — no air-gap/self-host binary.* *Beats koboi:* grounding-layer depth + funding + the only real customer-cloud option. *koboi wins:* truly customer-operated self-host (air-gap possible), no SaaS lock-in, sandbox+seccomp, eval-as-code.
- **Aisera** — **acquired by Automation Anywhere (CONFIRMED, banner on every page).** ⚠️ **CORRECTION:** the "hybrid/on-prem/VPC capable" claim does **NOT** survive primary-source verification — Aisera's own security page is **SaaS-only** (SOC 2 of a SaaS, CSA STAR for "Aisera SaaS"); no VPC/on-prem/single-tenant path documented. Broadest compliance badges in cluster (SOC2/ISO27001/GDPR/CCPA/CSA STAR/HIPAA-BAA), TRAPS governance framework, Aisera Unify (A2A/MCP/AGNTCY), domain-tuned LLMs (IT/HR). **Acquisition uncertainty is the dominant risk** — no longer independent. *koboi wins on sovereignty: koboi's claim is real & documented; Aisera's is not.*

### F. Vertical / voice / niche

- **Workday (Sana + Workday Build + ASOR + Agent Passport)** — vertical HR/Finance; **Agent Passport = third-party-attested runtime monitoring** (Cisco AI Defense launch partner) vs **OWASP LLM Top 10 / NIST AI RMF / MITRE ATLAS**, allow/block/route + single-revocation kill-switch. The novel idea: *"the attestor is NOT the vendor who built the agent."* Platform name = "Workday Build" (not Illuminate); BYOK unverified. `[attestation model JS-gated; "agent passport" not in SEC EDGAR — target concept, not verified spec]`
- **Zendesk AI Agents** — SaaS; **FedRAMP Li-SaaS + ISO 42001** (rare AI-mgmt standard) + SOC2/ISO/HIPAA; **public pricing** ($55–$115/agent/mo). Compliance benchmark to aim at. Li-SaaS = Low Impact only.
- **LiveKit Agents** — **Apache-2.0 OSS (framework + media server)**, full self-host, 30k+ combined stars, MCP + LLM-judge test framework (parallels koboi eval-as-code). **OSS-self-host peer** — but voice-native, zero text/RAG/sandbox overlap. *Caveat:* turn-detection models are non-Apache (LiveKit Model License).
- **Vapi** — voice SaaS; SOC2/HIPAA/PCI on Scale; zero-data-retention option ($1K/mo). Not OSS.
- **Restack** — Apache-2.0 OSS, K8s-native autoscaling (down-to-zero), tiny ecosystem, SOC2 only.
- **Inkeep** — CX/ops niche; OSS claim defunct. Least relevant.

---

## 4. koboi vs field — scorecard

| Axis | koboi | OSS peers (LangGraph/CrewAI/LlamaIndex/AutoGen) | Managed platforms |
|---|---|---|---|
| Complete runnable OSS stack (server+CLI+auth+jobs) | ✅ **unique** | ❌ libraries only (paid server) | n/a (SaaS) |
| Sandbox for untrusted code | ✅ seccomp (syscall) | ⚠️ AutoGen Docker only; others none/paid | ⚠️ black-box cloud |
| HITL + graduated trust-DB | ✅ mature | ⚠️ basic (interrupt/human_input) | ✅ |
| Eval-as-code | ✅ eve-style `t` | ⚠️ LangSmith paid; LlamaIndex RAG-metrics; LiveKit LLM-judge | ⚠️ managed eval |
| Durable execution / resume | ⚠️ step-journal (additive) | ✅ LangGraph checkpointer/time-travel | ✅ |
| Multi-provider model freedom | ✅ OpenAI/Anthropic/CF | ⚠️ mostly any-model via client | ❌ vendor-locked |
| Observability/tracing | ❌ Langfuse-only | ✅ LangSmith / OTel | ✅ |
| SSO/RBAC/multi-tenancy | ❌ absent | ⚠️ CrewAI AMP only (paid) | ✅ |
| Compliance badges | ❌ zero | ❌ bring-your-own | ✅ deep |
| Self-host / air-gap / data sovereignty | ✅ **the wedge** | ⚠️ LiveKit/Restack/koboi only | ❌ IBM-only |

---

## 5. The two-pronged positioning

1. **vs IBM / hyperscalers (the sovereignty buyers):** *"The lightweight, portable, no-lock-in sovereignty play — the same data-sovereignty as watsonx-on-OpenShift, ~100× less ceremony (no OpenShift cluster, no IBM Consulting, no licensing)."* This is the Grafana/Temporal/Ollama position.
2. **vs SaaS black-boxes (the regulated buyers):** *"Trust through inspectability — guardrails and eval as repo-owned code, fully auditable, evolvable toward a Workday-Passport-style independent attestation credential — not a vendor-issued 'safe' label."*

**Target buyers:** regulated/data-sovereignty-sensitive (government, EU/GDPR, healthcare, finance) + platform teams who refuse vendor lock-in but still need production-grade agent governance.

---

## 6. Steal-able ideas (P-level exploration tickets)

1. **Open "attested-safe" runtime-monitoring layer** (from Workday Agent Passport) — extend koboi's `evals/t/` into a signed, standards-aligned (OWASP LLM Top 10 / NIST AI RMF / MITRE ATLAS) safety credential. koboi's eval infra is ~80% of the way there. *Highest-leverage differentiator vs SaaS black-boxes.*
2. **Formal-logic guardrail option** (from Bedrock Automated Reasoning) — complement koboi's regex/pattern guardrails with a deterministic proof-based check for high-stakes tool calls.
3. **LangGraph-class durable execution** — promote koboi's additive step-journal toward checkpointer/time-travel semantics (the `TaskManager` dormant DAG at `task.py:112` is the on-ramp).
4. **LangSmith-class observability** — koboi's biggest OSS-peer gap; add OTEL traces + a `/metrics` endpoint (reuses existing `TelemetrySnapshot`).
5. **LLM-judge test framework** (from LiveKit) — already conceptually present in koboi's eval; formalize as a first-class primitive.

---

## 7. What koboi must close (enterprise envelope)

The competitor benchmark doesn't change the enterprise-readiness roadmap — it *confirms* it. The managed platforms set the compliance bar koboi can't match (and shouldn't chase directly); the OSS peers confirm koboi's feature/sandbox/HITL depth is competitive-to-leading, with **observability + durable-exec** as the gaps vs LangGraph, and **SSO/RBAC/enterprise-tier** as the gap vs CrewAI. See the 4-wave path in the enterprise-readiness assessment:
- **Wave 1** externalize state (Redis/Postgres) — keystone [XL]
- **Wave 2** OIDC/SAML + RBAC + at-rest encryption + SIEM audit [L]
- **Wave 3** `/metrics` + OTEL + compliance posture/erasure [M]
- **Wave 4** position the wedge (air-gapped/on-prem + BYO-everything + eval/guardrails-as-code) [docs+marketing]

---

## Sources index (selected; full per-competitor citations in the fleet agent transcripts)

**Managed/platform:** salesforce.com/{agentforce,trust/compliance,hyperforce} · servicenow.com (JS-gated) · learn.microsoft.com/compliance/regulatory/offering-home · aws.amazon.com/bedrock/{agents,agentcore,guardrails} · cloud.google.com/{products/agent-builder,security/compliance,vertex-ai/sla} · ibm.com/products/{cloud-pak-for-data,watsonx-orchestrate,watsonx-governance}+ibm.com/cloud/compliance
**Foundation:** openai.com/enterprise(+enterprise-privacy) · anthropic.com/news/claude-for-enterprise · aws.amazon.com/bedrock/security-compliance · cloud.google.com/gemini · workspace.google.com/pricing
**OSS frameworks (LICENSE-verified):** github.com/langchain-ai/langgraph (MIT) · github.com/crewAIInc/crewAI (MIT) · github.com/run-llama/llama_index (MIT) · github.com/microsoft/autogen (MIT, maintenance-mode) · github.com/ag2ai/ag2 (Apache-2.0) · github.com/microsoft/agent-framework (MIT) · docs.langchain.com · crewai.com/pricing · docs.crewai.com · docs.llamaindex.ai
**Startups:** sierra.ai · decagon.ai/{security,blog} · glean.com/{platform/security,legal} · aisera.com
**Vertical/voice/niche:** blog.workday.com + SEC EDGAR Workday 10-K FY26 · zendesk.com/{service/ai/ai-agents,trust-center,pricing} · github.com/livekit/agents (Apache-2.0) + livekit.io/pricing · vapi.ai/pricing · restack.io · inkeep.com
