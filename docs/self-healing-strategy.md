# Self-Healing Strategy — Feasibility Study for koboi-agent

**Date:** 2026-07-15
**Scope (locked with user):** Comprehensive — infra + agent-loop + workflow; **headline gap = agent-level structured self-correction**.
**Recovery philosophy (locked):** **Configurable** — operator tunes the recovery-vs-handover threshold per config (YAML-driven; subsumes both "recover-first" and "conservative-handover").
**Deliverable:** this doc (`docs/self-healing-strategy.md`), matching the `docs/*-strategy.md` pattern.
**Method:** 4 parallel agents (3 `code-explorer` for orchestration/multi-agent/resilience-baseline + 1 competitor-research) + independent deep-dive of `loop.py`/`loop_pipeline.py`/`config_models.py`/`orchestration/`. The 3 code-explorers went idle without delivering report content (teammate-messaging channel failed to relay findings), so all code claims below were re-verified directly by the author with `grep`/`Read` and cite `file:line`. **Web search was hard rate-limited until 2026-07-28** across all backends (`WebSearch`, `web_search_prime`, `webReader`), so the competitor section draws on (a) koboi's own primary-sourced prior research (`docs/competitor-enterprise-benchmark.md`, `docs/agentic-vs-autonomous-strategy.md`) and (b) well-established academic patterns from training knowledge; framework-specific claims beyond what the prior docs verified are tagged `[unverified]`.

---

## 1. TL;DR

- **Your two assumptions are TRUE.** Multi-agent-in-one-config and advanced orchestration are both real and shipped (§2). The 2026-07-06 `agentic-vs-autonomous-strategy.md` claim that "zero graph/DAG classes exist" is **stale** — Phase 3 (2026-07-09) closed it.
- **koboi is NOT at zero self-healing.** It has infra retry/failover, crash-resume (loop **and** DAG graph-cursor), implicit ReAct error-feedback, doom-loop injection, grounding-abstention, handover, **and** mode-specific re-planning (deep_research re-plans on coverage gaps; planner re-plans on node failure up to `max_replans`). What it lacks is a **unified, bounded, configurable self-correction layer** that generalizes those scattered recoveries to the standard agent loop.
- **The precise gap:** no structured **reflection / self-critique loop** ("critique → revise → retry with a different approach"), no **automatic replan-on-failure** in the standard loop, no **backtracking**, terminal failures **abort (`raise`) instead of heal**, and **no configurability** for recovery-vs-handover — the exact knob you asked for.
- **The wedge fit:** "trustworthy unattended autonomy" demands self-healing that is **bounded** (turns + token budget), **grounding-anchored** (critique against retrieved context, not free-form), **observable** (reflection turns in the StepJournal), and **handover-bounded** (converge-or-handover, never fake recovery). This makes self-healing the **layer BELOW handover** — recover when you can, hand over honestly when you can't.
- **Recommendation:** ship an opt-in `ReflectionHook` + `self_healing:` config (P0), wire replan-on-failure into the existing `GroundingGuardrail`/`HandoverDetectionHook` seam (P1), generalize `max_replans` node-failure recovery to all DAG modes (P2). Phased in §7.

---

## 2. Assumption validation (the user's ask)

### 2a. "Multi-agent in one config" → **TRUE**

| Claim | Evidence |
|---|---|
| One YAML declares multiple agents | `orchestration.agents: list[dict]` inside `OrchestrationConfig` (`config_models.py:324-331`) — nested under `orchestration:`, which is why a top-level grep misses it |
| Per-agent model/provider freedom | `AgentConfig` (`config_models.py:20`) + `LLMConfig` (`config_models.py:54` — `provider`/`model`/`api_key`/`base_url` per agent); per-agent `llm_config` shipped via PR #9 |
| Delegation mechanism | `SubAgentManager` (`subagent.py:70`) + `delegate_tasks` tool (`tools/builtin/subagent.py:46`) |
| Shared infrastructure across sub-agents | `OrchestrationConfig.share_mcp: bool = True` (`config_models.py:331`) |
| Coordination model | **Central-orchestrator** (router → fan-out → synthesize), **not peer-to-peer messaging** — consistent with the `cross-instance-a2a-feasibility` memory |
| Concrete demos | `examples/25_subagent_delegation.{py,yaml}`, `examples/04_tool_use_multi.{py,yaml}` |

### 2b. "Advanced orchestration" → **TRUE**

| Claim | Evidence |
|---|---|
| 7 execution modes | `koboi/types.py:119-127` — `sequential`, `parallel`, `sequential+revision`, `parallel+revision`, `dag`, `dynamic`, `deep_research` |
| Conditional branching | `conditionals: list[dict]` (`types.py:160`) |
| Real subsystem files | `koboi/orchestration/`: `dag_scheduler.py`, `planner.py`, `workflow_graph.py`, `router.py`, `research.py`, `orchestrator.py`, `factory.py` |
| Demo configs | `configs/{dag_demo,conditional_demo,dynamic_demo,deep_research_demo,orchestrated,advanced_orchestrated}.yaml` |
| Demo scripts | `examples/{15_orchestration_keyword,16_orchestration_llm,24_config_driven_orchestration,dynamic_workflow_live,phase3_live_e2e,workflow_graph_demo}.py` |

**Verdict:** both assumptions hold. These are production-shaped, not stubs.

---

## 3. Self-healing baseline (what koboi already has)

| Level | Mechanism | Trigger | Evidence | Maturity |
|---|---|---|---|---|
| **Transport** | `RetryClient` exponential backoff | transient LLM HTTP errors | `client.py` | real |
| **Transport** | `ProviderPool` failover + circuit-breaker | provider failure | `llm/pool.py` | real (Tier 2, `multi-provider-modes`) |
| **Agent-loop (implicit)** | Tool denial/error fed back as `tool_result` → model can react next iteration (standard ReAct) | rate-limit / policy / mode-block / denied / tool exception | `loop_pipeline.py:88-112` (`_deny_or_skip`→`memory.add_tool_result("Error: …")`); broad `except … never break the turn` (`loop.py:237`) | real, but **unstructured** |
| **Agent-loop (targeted)** | Doom-loop detection + recovery injection | repeated-tool pattern | `loop_pipeline.py:327-343` (`DoomLoopHook` → `DOOM_LOOP_DETECTED` fan-out) | real, narrow |
| **Crash/resume** | `StepJournal` native writes + `_repair_interrupted_turn` + `--resume` | crash/redeploy mid-turn | `loop.py:523-581`, `loop.py:452-470` | real, library-level (rare) |
| **Crash/resume (graph)** | DAG graph-cursor resume | crash mid-graph | `dag_scheduler.py:37,118,188` | real (Phase 3) |
| **Workflow (mode-specific)** | deep_research re-plans on coverage gaps | coverage < threshold | `orchestration/CLAUDE.md:40` ("re-plans on gaps until `coverage_threshold`/`max_depth`/budget") | real, **deep_research only** |
| **Workflow (mode-specific)** | planner re-plans on node failure | node failure | `orchestration/CLAUDE.md:78` ("`max_replans` re-plans on node failure") | real, **planner only** |
| **Workflow (routing)** | LLM→keyword router fallback | LLM router unavailable/error | `router.py:215-245` | real |
| **Confidence** | `GroundingGuardrail` abstain | claim coverage < threshold | `guardrails/grounding.py:5,136` | real (opt-in) |
| **Confidence** | `HandoverDetectionHook` structural handover | low coverage / user-ask patterns | `hooks/handover_detection_hook.py:51-61` (ladder: answer ≥0.8 / abstain 0.5-0.8 / handover <0.5) | real (opt-in) |

**Key asymmetry:** recovery exists, but it is **scattered and mode-specific**. The deep_research/planner re-plan pattern is the closest thing to self-correction — and it is **not available in the standard `chat`/`act`/`auto` loop**.

---

## 4. The gap (precise, ranked)

1. **No structured reflection / self-critique loop.** When a tool fails or an output is weak, the agent does not systematically "critique → revise → retry with a *different* approach." It either lets the model implicitly react (unbounded, unobservable) or abstains/hands over. `grep` for `reflect|self_correct|self_critique|backtrack|critique` across `koboi/` returns **only semantic uses** ("mirrors/represents") — zero critique→revise loops. **This is the headline gap.**
2. **No automatic replan-on-failure in the standard loop.** The `max_replans` recovery is planner/DAG-scoped; a tool failure in `act`/`auto` does not trigger a plan revision — it surfaces an error string and hopes the model adapts.
3. **No backtracking / alternative-approach exploration.** No Tree-of-Thoughts-style "try A, on failure try B" with state rollback.
4. **Terminal failures abort, not heal.** `max_iterations` exhaustion *raises* `AgentMaxIterationsError` (`loop.py:654`); guardrail block *raises* `AgentGuardrailError` (`loop.py:325`); handover *raises* `AgentHandoverError` (`loop.py:300,353`). There is **no recovery step before the raise**.
5. **No unified workflow node-failure recovery.** Node-failure re-planning is planner-specific; a plain `dag`/`conditional` node failure has no generalized re-route / fallback-agent / sub-plan-retry.
6. **No configurability.** No YAML knob for "on failure: reflect ≤N times, then handover" — the exact **Configurable** philosophy you selected. The deep_research knobs (`coverage_threshold`, `max_depth`, `max_replans`) are the closest precedent but are mode-locked.

---

## 5. Competitor research — how peers handle self-healing

> **Sourcing caveat:** web was rate-limited until 2026-07-28. Below, **primary-sourced** claims come from koboi's own prior research (`docs/competitor-enterprise-benchmark.md`, 2026-07-07; `docs/agentic-vs-autonomous-strategy.md`, 2026-07-06 — both `curl`-verified against official/GitHub/arXiv sources). **Academic** claims are stable, well-cited 2023 results from training knowledge. **Framework-specific** claims beyond the prior docs are tagged `[unverified]` and must be re-checked when web returns.

### Level 1 — Transport / infra
- **LangGraph**: retry policies on tools/nodes + checkpointer-backed durability *(primary-sourced: `competitor-enterprise-benchmark.md:60`)*. `[unverified: exact retry-policy API]`
- **koboi parity**: `RetryClient` + `ProviderPool` failover/circuit-breaker — **competitive**.

### Level 2 — Agent-loop self-correction / reflection (the gap)
- **Reflexion** (Shinn et al., 2023): verbal reinforcement — after a failure, the agent generates a self-reflection (natural-language "what went wrong"), stores it in episodic memory, and retries with that reflection in context. Bounded by max trials. *The canonical pattern.*
- **Self-Refine** (Madaan et al., 2023): generate → feedback (same or critic model) → refine, iterated until a stopping criterion. No external memory — purely in-context.
- **CRITIC** (Gou et al., 2023): self-critique where the critique is *grounded in external tools* (search, code interpreter, doc) rather than free-form — reduces hallucinated self-assessment.
- **Self-RAG** (Asai et al., 2023): reflection tokens gate retrieval and self-evaluation — the model learns when to retrieve and when its answer is supported.
- **Tree-of-Thoughts** (Yao et al., 2023): explore multiple reasoning paths, evaluate, **backtrack** on dead-ends.
- **LangGraph**: reflection is expressed as a **graph** — a critique node + conditional edge routing back to the generator on failure *(primary-sourced trend; `[unverified: specific node API]`)*.
- **AutoGen / AG2**: multi-agent reflection — a "critic" agent reviews a "generator" agent's output; error feedback loops back to the model; GroupChat pedigree *(primary-sourced: `competitor-enterprise-benchmark.md:63`)*.
- **CrewAI**: task retry + delegation chains + fallback agents `[unverified]`.
- **OpenAI Agents SDK**: `max_turns` guard + tool-error retry + guardrails + handoffs `[unverified]`.

**Common SOTA patterns:** explicit reflection loops (critique→revise) over blind retry; **bounded convergence** (`max_reflection_turns` / budget); explicit **give-up conditions** (timeout, cost, confidence threshold, human handover); **backtracking** on dead-ends; critique **grounded in external evidence** (CRITIC/Self-RAG) to avoid hallucinated self-assessment.

**Failure modes peers hit:** reflection loops that never converge (cost blowup); compounding errors (bad critique → worse revision); over-reliance on LLM self-judgment (unreliable without grounding); latency/cost. These are the design constraints koboi's solution must respect.

### Level 3 — Workflow / graph recovery
- **LangGraph**: checkpointer + `interrupt()`/resume + time-travel + conditional edges = the deepest durable-exec story; node failure handled via graph topology + retry *(primary-sourced: `competitor-enterprise-benchmark.md:60`)*.
- **koboi parity**: graph-cursor crash resume + planner `max_replans` — **partial** (crash-resume is strong; node-failure re-route is planner-scoped, not generalized).

### Level 4 — Crash / state recovery
- **LangGraph**: checkpointer (Postgres/SQLite) is the headline, platform-tier-marketed *(primary-sourced)*.
- **koboi parity**: `StepJournal` + `--resume` at **library level** — **rare and competitive** (per `agentic-vs-autonomous-strategy.md:39`).

### What's adoptable for koboi's wedge
- **Reflexion-style bounded reflection** is the highest-fit pattern: it is model-agnostic, composes with koboi's existing hook chain, and its "store reflection in memory" maps onto `ConversationMemory`/`ProactiveMemory`. The **convergence boundary must be handover**, not silent infinite retry — this is where koboi's existing `HandoverDetectionHook` becomes the floor.
- **CRITIC-style grounding** of the critique (against retrieved context / tool output, not free-form) is essential for the "trustworthy" half of the wedge — it prevents the #1 failure mode (hallucinated self-assessment).
- **LangGraph's graph expression of reflection** is instructive but koboi should keep reflection in the **agent loop** (hook-injected), not require a graph — most koboi runs are `chat`/`act`/`auto`, not DAGs.
- **Generalize the existing `max_replans`** pattern: koboi already invented a bounded re-plan-on-failure primitive (for the planner) — the strategic move is to lift it into a reusable, loop-level, configurable facility rather than invent a new concept.

---

## 6. Proposed design — aligned with "trustworthy unattended autonomy" + Configurable

### 6.1 A `self_healing:` config section (opt-in, default off — no behavior change)
```yaml
self_healing:
  enabled: true
  reflection:
    enabled: true
    max_turns: 2            # bounded — converge or handover
    budget_tokens: 4000     # hard cost ceiling per turn
    grounding_anchored: true # critique must cite retrieved context / tool output
  on_tool_error: reflect    # reflect | retry | handover
  on_low_grounding: replan  # replan | abstain | handover  (layer BELOW handover)
  handover_after: exhausted # exhausted | low_confidence  (the Configurable knob)
```
This is the **Configurable** philosophy made concrete: an operator picks "recover-first, handover last" (`handover_after: exhausted`) or "conservative" (`handover_after: low_confidence`) per deployment.

### 6.2 `ReflectionHook` (Hook ABC, priority ~45 — business band)
- Fires on `POST_TOOL_USE` (tool error/weak result) and `POST_LLM_CALL` (low grounding).
- Injects a **structured critique→revise** context message (bounded by `max_turns` + `budget_tokens`), grounded in retrieved context / tool output when `grounding_anchored: true`.
- Records each reflection turn in the `StepJournal` (new `status="reflection"` step) → **observable + auditable**, reuses the native-write durability seam.
- Convergence: if `max_turns`/budget exhausted without a grounding-passing result → escalate per `handover_after` (handover via existing `HandoverDetectionHook`, or abstain via `GroundingGuardrail`). **Never silently claims recovery** — a reflection turn that doesn't improve grounding is recorded honestly and escalates.

### 6.3 Replan-on-failure (P1) — the layer below handover
- On repeated tool failure or low grounding, inject a "revise the plan" step before the existing abstain/handover fires.
- Wired into the `GroundingGuardrail`/`HandoverDetectionHook` seam: the ladder becomes **answer → reflect/replan → abstain → handover**, inserting recovery between "answer" and "abstain."

### 6.4 Workflow node-failure recovery (P2)
- Generalize planner `max_replans` to all DAG/conditional modes: node failure → re-route to a fallback agent / sub-plan retry, with the same bounded-turns + handover contract.
- Reuses `dag_scheduler.py` graph-cursor primitives + `router.py` fallback pattern.

### 6.5 Honest-convergence guarantees (the "trustworthy" contract)
- **Bounded**: `max_turns` + `budget_tokens` (no infinite reflection).
- **Grounding-anchored**: critique must cite evidence (prevents hallucinated self-assessment — the CRITIC lesson).
- **Observable**: reflection turns in the StepJournal + `RunResult.metadata['self_healing']` (reflection_count, recovered, handed_over).
- **Handover-bounded**: non-convergence → handover, never fake recovery. Aligns with the A5 honesty rule (never claim "knows when it doesn't know at X%" — that needs human PPI).

---

## 7. Phased roadmap

| Phase | Scope | Size | Outcome |
|---|---|---|---|
| **P0** | `ReflectionHook` MVP + `self_healing:` config (bounded, opt-in, default off). Turn today's *implicit* error-feedback into an *explicit, bounded, observable* reflect step on tool failure. | M | First general self-correction in the standard loop |
| **P1** | Replan-on-failure: insert reflect/replan between "answer" and "abstain" in the grounding/handover ladder. | M | Recovery layer below handover |
| **P2** | Generalize `max_replans` node-failure recovery to all DAG/conditional modes (re-route / fallback agent). | L | Unified workflow self-healing |
| **P3** | Backtracking / ToT-style alternative exploration (researchy; guard with budget + grounding). | XL | Frontier parity |
| **P4** | Eval: `t.*` self-healing scorers (`reflection_invoked`, `recovered_vs_handover`, `converged_within_budget`) + provenance via StepJournal. | M | Evidence the recovery is real, not theatrical |

**Sequencing logic:** P0 is the cheapest credible entry (generalizes a pattern koboi already invented for the planner). P1 connects it to the existing confidence ladder — this is where the "trustworthy" contract is enforced. P2 unifies the workflow half. P4 must ship alongside, not after — self-healing without eval is exactly the kind of unverifiable "it works" claim the `rag-prod-readiness-eval-gap` memory warns against.

---

## 8. Risks & failure modes (honest)

| Risk | Mitigation |
|---|---|
| Reflection loops never converge (cost blowup) | Hard `max_turns` + `budget_tokens`; handover on exhaustion |
| Compounding errors (bad critique → worse revision) | `grounding_anchored: true` + `GroundingGuardrail` gate before accepting a revised output |
| Fake recovery (claims fixed, isn't) | Handover boundary + reflection turns recorded in StepJournal for audit; `RunResult.metadata` exposes `recovered` honestly |
| Cost / latency regression | Opt-in, default off, per-config; `budget_tokens` ceiling |
| Reflection amplifies hallucination | Critique must cite retrieved context / tool output (CRITIC/Self-RAG lesson), not free-form |
| Interaction with doom-loop detector | Reflection must not become a new doom-loop vector — `DoomLoopHook` already detects repeated-tool patterns; reflection turns are distinct `status="reflection"` steps |

---

## 9. What this is NOT (honesty boundaries)

- **NOT "knows when it doesn't know at X%."** That requires human PPI calibration (like the A5 conformal on-ramp) and is never claimable from self-healing alone. Self-healing recovers; it does not certify.
- **NOT a replacement for handover.** It is the **layer below handover** — recover when you can, hand over honestly when you can't. The two compose; they don't compete.
- **NOT unbounded autonomy.** Bounded (turns + budget), observable (journal), and handover-bounded. "Unattended" yes; "unaccountable" no.
- **NOT externally verified on the competitor side (yet).** Framework-specific claims in §5 are `[unverified]` until web returns (2026-07-28); re-check before any external-facing claim. The academic patterns and the koboi-side evidence are solid.

---

## 10. Next actions

1. **Decide P0 scope** — confirm `ReflectionHook` + `self_healing:` config is the entry point (this doc's recommendation).
2. **Re-verify competitor §5** when web rate-limit resets (2026-07-28) — especially LangGraph reflection-node API, AutoGen critic-agent specifics, CrewAI task-retry, OpenAI Agents SDK `max_turns`/tool-error behavior.
3. **Optional live validation** of the §2 assumptions: `koboi graph configs/dag_demo.yaml --json` and `koboi validate configs/advanced_orchestrated.yaml` (no API key needed) to prove the topology parses end-to-end.
4. If P0 approved, spin a worktree (`worktree-self-healing-p0`) and implement `ReflectionHook` + `SelfHealingConfig` + `t.*` scorers per §6/§7.

---

## Sources
- **Primary-sourced (koboi prior research):** `docs/competitor-enterprise-benchmark.md` (2026-07-07, `curl`-verified) · `docs/agentic-vs-autonomous-strategy.md` (2026-07-06, 18-source/3-vote verified) — LangGraph durable exec/checkpointer (`:60`), AutoGen/AG2 Docker+GroupChat+OTel (`:63`), CrewAI role+Enterprise (`:61`), OpenAI Agents SDK client-only+handoffs/HITL (`:54`), arXiv 2603.07670/2606.04990/2603.22928/2606.12191.
- **Academic (training knowledge, stable 2023 results — re-verify DOIs when web returns):** Reflexion (Shinn et al.) · Self-Refine (Madaan et al.) · CRITIC (Gou et al.) · Self-RAG (Asai et al.) · Tree-of-Thoughts (Yao et al.).
- **Code (author-verified, `file:line`):** `koboi/types.py:119-127,160` · `koboi/config_models.py:20,54,324-331` · `koboi/loop.py:237,300,325,452-470,523-581,596,654` · `koboi/loop_pipeline.py:88-112,302-303,327-343` · `koboi/orchestration/{dag_scheduler.py:37,118,188, router.py:215-245, CLAUDE.md:40,78}` · `koboi/guardrails/grounding.py:5,136` · `koboi/hooks/handover_detection_hook.py:51-61` · `koboi/subagent.py:70` · `koboi/tools/builtin/subagent.py:46`.
