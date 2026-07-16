# Self-Healing (Agent Behavior) — Feasibility Study & Roadmap

**Status:** Research / feasibility (no code yet). **Date:** 2026-07-15. **Scope:** AI *agent-behavior* self-healing (recovery from bad tool output, malformed LLM responses, low confidence, stuck loops) — not infrastructure reliability. Infra-class mechanisms are inventoried but flagged.

> This is a **fresh** study produced independently of any prior session notes. All findings are verified against `main` (`b377cce`) with file:line evidence. It deliberately **does not** build on `docs/self-healing-strategy.md` (kept untouched); reconcile or delete that file separately.

---

## 1. TL;DR — verdict

- **Multi-agent/orchestration assumption: VALID.** All six orchestration modes (sequential/parallel/dag/conditional/dynamic/deep_research) are implemented and tested. Multi-agent-in-one-config is real but **central/hub-and-spoke only** — no peer-to-peer on `main`.
- **Self-healing gap is real but narrow.** koboi already has the *rungs* of a recovery ladder — transport retry, provider failover, tool-error→model feedback, doom-loop nudge, grounding abstention, handover-to-human, durable resume. What it **lacks** is **one unified, bounded, configurable reflection loop in `AgentCore`** that sits *below* handover and *above* retry, plus an **escalation policy** that orders those rungs. Two latent bugs also suppress the recovery that nominally exists.
- **Recommendation:** ship a **unified, bounded, tool-grounded reflection loop** (`ReflectionHook`) as the wedge, architected to compose into a declarative **escalation ladder** (`retry → reflect → replan → handover`). **Opt-in by default** (`self_healing:` config), fail-soft, sharing a `max_turns` budget with doom-loop. Ground the critique in the existing `GroundingGuardrail` + tool results — **never pure intrinsic self-critique**, which the literature shows is empirically weak.

---

## 2. Method

Three read-only explorations ran in parallel, all verified from code (not memory):

1. **Multi-agent/orchestration validation** — enumerated every execution mode, router, sub-agent path, config seam, and CLI/server surface.
2. **Current self-healing inventory** — grepped `reflect|self_heal|self_correct|critique|replan` and read every resilience/recovery path in `loop.py`, `loop_pipeline.py`, `tools/registry.py`, `harness/`, `guardrails/`, `hooks/`, `journal.py`, `orchestration/`.
3. **Competitor & academic research** — primary-source survey of LangGraph, AutoGen/AG2, CrewAI, OpenAI Agents SDK, Anthropic guidance, and the academic lineage (ReAct, Reflexion, Self-Refine, CRITIC, Self-RAG, LATS, ReWOO, Plan-and-Solve).

---

## 3. Part A — Current state (what koboi already has)

### 3.1 Inventory of existing "self-healing-like" mechanisms

| # | Mechanism | Category | Location | Default? | Maturity |
|---|-----------|----------|----------|----------|----------|
| 1 | `RetryClient` (exp backoff, honors `retry_after`) | retry (INFRA) | `koboi/client.py:123-179` | on | 4/5 |
| 2 | `HttpTransport.post` retry (429/5xx) | retry (INFRA) | `koboi/llm/http_transport.py:53-92` | on | 4/5 |
| 3 | `ProviderPool` failover + `CircuitBreaker` | failover (INFRA) | `koboi/llm/pool.py:71-264` | opt-in (`pools:`) | 4/5 |
| 4 | **Tool-error → string fed to LLM** (always-on) | tool-feedback | `koboi/tools/registry.py:197-242` | on | 4/5 |
| 5 | `DoomLoopDetector` + `DoomLoopHook` (nudge) | loop-guard | `koboi/harness/doom_loop.py`, `koboi/hooks/doom_loop_hook.py:34-71` | opt-in | 4/5 |
| 6 | `GroundingGuardrail` (claim-decompose + NLI → abstain) | grounding | `koboi/guardrails/grounding.py:109-173` | opt-in | 3/5 |
| 7 | `transfer_to_human` + `HandoverDetectionHook` | handover | `koboi/tools/builtin/handover.py`, `koboi/hooks/handover_detection_hook.py` | tool on / hook opt-in | 3/5 |
| 8 | `StepJournal` + `resume()` + `_repair_interrupted_turn` (idempotency-aware) | resume (INFRA-ish) | `koboi/journal.py`, `koboi/loop.py:546-604,684-703` | on (SQLite) | 4/5 |
| 9 | Orchestrator `max_replans` (dynamic mode) | re-plan | `koboi/orchestration/orchestrator.py:799-822` | **OFF (default 0)** | 3/5 ⚠ |
| 10 | `QualityEvaluator` revision loop | reflection (orchestration-only) | `koboi/orchestration/orchestrator.py:103-136,409-448` | opt-in, non-streamable | 3/5 |
| 11 | Deep-research coverage-gated iteration | re-plan | `koboi/orchestration/orchestrator.py:869-1142` | opt-in (one mode) | 3/5 |
| 12 | `max_iterations` hard stop | budget | `koboi/loop.py:619,676-678` | on | 4/5 |

### 3.2 The headline gap

> **No unified, configurable agent-behavior reflection loop exists in `AgentCore`.**

`grep -rniE "reflect|self_heal|self_correct|critique|replan" koboi/` returns only orchestration-layer re-planning, incidental uses of "reflect" (meaning "mirrors"), and one DB schema-self-heal comment. `AgentCore` (`koboi/loop.py`) has **no turn-level "did this go well? if not, critique & retry" step**. The behavior-level recovery that exists is scattered across independently-toggled features with **no shared config namespace** (`harness.doom_loop`, `guardrails.output`, `handover.detection`, `execution.max_replans`, `use_revision`) and **no composable policy**.

### 3.3 Two latent bugs that suppress existing recovery (must fix regardless of design)

- **`max_replans` never reliably fires on crashes.** `Orchestrator._run_single` (`orchestrator/orchestrator.py:466-472`) catches exceptions and returns `AgentResult(answer="Error: …")` but never sets `failed=True`. `max_replans` keys on `r.failed` (`:801`), so it only fires on the synthesis path that explicitly sets the flag (`:1308`). The code comments at `:1137-1139` acknowledge this. **Effect:** a crashed node is not re-planned.
- **`max_replans` is off by default** (`facade.py:1954`, default `0`). Even with the flag bug fixed, re-planning is opt-out-absent unless configured.
- **Empty LLM response is silently "complete".** `is_complete = not tool_calls` (`types.py:78`) means an empty content + no tool calls returns `success=True` with an empty answer (`loop.py:641`, `_process_output("")`). No re-ask, no repair — only the streaming `None` branch emits an `ErrorEvent` (`loop.py:751-755`).

### 3.4 What koboi does today for six concrete failures

| Failure | Today's behavior |
|---|---|
| Tool raises `ValueError` | `ToolRegistry.execute` catches, returns `"Error: …"` string as tool_result (`registry.py:240-242`); LLM may self-correct next turn. **No tool-level retry.** |
| Low grounding (0.4) | `GroundingGuardrail` → `abstain` → loop swaps output for a refusal (`loop.py:349-362`). **No reground/regenerate.** |
| Stuck repeating tool+args | *If* doom-loop configured: 3rd identical call → inject `[DOOM LOOP WARNING]` nudge. **Else:** loops until `max_iterations` → `AgentMaxIterationsError`. |
| Malformed tool_call (bad JSON) | `"Error: invalid arguments JSON: …"` fed back (`registry.py:202-204`). LLM may re-emit. |
| Provider 5xx | `HttpTransport` retry ×2 → `RetryClient` retry ×3 → (if pool) failover+circuit-break → else uncaught `LLMError`, turn fails. |
| Crash mid-turn → `--resume` | Marks interrupted steps, re-executes missing tool calls, **skips non-idempotent tools** (`loop.py:589-603`). Strongest existing self-heal. |

---

## 4. Part B — Competitor & research landscape

### 4.1 The single most important finding

> **Tool/grounding-grounded self-healing is proven; pure intrinsic self-correction is mostly hype.**

A parallel line of work (*Huang et al., "LLMs Cannot Self-Correct Reasoning Yet", ICLR 2024*) shows intrinsic self-correction **without an external signal often degrades or barely helps** — the wins concentrate where a **verifier** exists (tools, tests, retrieval, environment reward, a stronger judge). **koboi already owns a verifier** (`GroundingGuardrail`) and the terminal rung (handover). That is the entire reason a *grounded* reflection loop is the right wedge, and why a *navel-gazing* one should be avoided.

### 4.2 Taxonomy (12 categories, infra flagged)

1. Transport/execution retry (INFRA) · 2. Provider/model failover (INFRA) · 3. Turn/iteration budget · 4. Checkpoint resume/replay (INFRA-leaning) · 5. **Tool-error feedback-to-model** · 6. **Self-Refine** (intrinsic critic→revise) · 7. **Reflexion** (episodic reflection memo) · 8. **CRITIC** (tool-grounded critique) · 9. **Self-RAG** (reflection tokens for retrieve/ground) · 10. Tree/plan search (LATS/ToT) · 11. **Re-plan/re-route on failure** (ReAct, Plan-and-Solve, ReWOO) · 12. **Escalation-to-human**.

### 4.3 Framework matrix (what each production framework ships)

| Framework | retry | reflection | self-critique | re-plan/reroute | escalation | loop-guard | verification |
|---|---|---|---|---|---|---|---|
| **LangGraph** | `with_retry`/`with_fallbacks` | reflection-agent pattern | via critic node | `add_conditional_edges` + `Command` | durable exec + interrupts | `recursion_limit` | evaluator-optimizer |
| **AutoGen/AG2** | via client | **Reflection** design pattern | critic/reviewer agent | Handoffs pattern | **Intervention Handler** | max turns | executor agent |
| **CrewAI** | `max_retry_limit` | via critic agent | **Task `guardrail` fn → retry** | hierarchical/`Flow` | via delegation | `max_iter`/`max_execution_time` | Task guardrail |
| **OpenAI Agents SDK** | `ModelSettings.retry` + **compat-retry w/ rollback** | — | Output guardrail | **Handoffs** | Escalation agent (typed reason) | `max_turns` + **`error_handlers`** | Input/Output/**Tool guardrails** + **Tripwires** |
| **Anthropic guidance** | infra | evaluator-optimizer | same | orchestrator-workers/routing | human-in-the-loop | bounded augmented-LLM | tool-use + iterative refine |

**Notable for design borrow:** OpenAI SDK's `error_handlers` dict (controlled output per error class) and compatibility-retry-with-rollback; CrewAI's Task-guardrail-as-verifier-with-retry; AutoGen's Intervention Handler; LangGraph's conditional edges as the re-route primitive.

### 4.4 Academic anchors (mechanism in one line)

- **Reflexion** — after a failed trial, write a verbal "what went wrong" memo to an episodic buffer; prepend on retry. Verbal RL, no weight updates.
- **Self-Refine** — one LLM plays generator→feedback→refiner to a stop condition. No training.
- **CRITIC** — critique **using external tools** (search/code/calc/retrieval), revise from tool feedback. Beats intrinsic decisively.
- **Self-RAG** — emit reflection tokens to decide retrieve/relevant/supported; re-retrieve/regenerate.
- **ReAct** — reasoning + action interleaved; "handle exceptions by updating the plan." Foundational detect-and-adapt.
- **ReWOO / Plan-and-Solve** — plan up front, decouple from execution → cheap re-execution and clean replan.

---

## 5. Part C — Recommended design

### 5.1 Principle: ground the critique, bound the loop, compose the rungs

The design is **one bounded reflection loop** + **one declarative ladder**, layered onto seams that already exist.

```
                         ┌──────────────────────────────────────────┐
 failure detected ─────► │ ESCALATION LADDER (declarative policy)    │
                         │  1. retry            (have: RetryClient)  │
                         │  2. reflect          (NEW: ReflectionHook)│
                         │  3. replan           (have: max_replans*) │
                         │  4. handover         (have: handover)     │
                         └──────────────────────────────────────────┘
                               │ each rung bounded by shared max_turns
                               ▼
                         [ continue | give up → handover/refuse ]
```

`*` replan requires the `_run_single` failed-flag fix (§3.3).

### 5.2 The wedge: `ReflectionHook` (tool-grounded, bounded)

- **Class:** `koboi/hooks/reflection_hook.py` — subclasses `Hook` (`koboi/hooks/chain.py:241-244`).
- **Handles:** `[HookEvent.POST_TOOL_USE, HookEvent.POST_OUTPUT]` (both verified to exist).
- **Trigger conditions (configurable):**
  - `POST_TOOL_USE` → `ctx.tool_result` indicates failure (starts with `"Error:"`, or a structured marker) **and** the same tool/args has already failed ≥ N times this turn.
  - `POST_OUTPUT` → the **grounding verifier** returns coverage < `threshold` (reuse `GroundingGuardrail`'s NLI/claim-decompose — do not reinvent the verifier).
- **Action:** run a **side-LLM critique** ("tool X failed because …; args were …; suggest a corrected call / different tool" **or** "the answer is ungrounded on claim Z; suggest a reground query"). **Inject** the critique as a system/tool message into `ctx.messages` (same inject pattern as `DoomLoopHook`), then let the loop re-enter. Critique inputs are **redacted via `redact.py`** (never leak sensitive args to the critique prompt).
- **Bounding:** shares a **`max_turns` budget with doom-loop** (the reflection rung and doom-loop must not each be unbounded — otherwise reflection *amplifies* the loop). When the budget is exhausted, **fall through to the next ladder rung** (replan → handover), not an infinite retry.
- **Fail-soft:** any critique/verifier error **passes through** (mirrors `GroundingGuardrail`'s fail-soft contract) — reflection must never break a run that would otherwise succeed.

### 5.3 Config namespace (opt-in, inert-by-default)

```yaml
self_healing:
  enabled: true                 # opt-in; default false
  max_turns: 3                  # shared reflection+doom-loop budget per run
  fail_soft: true               # pass-through on critique error (default true)
  critic:
    provider: null              # null = reuse main client; or named provider from `providers:`
    model: null
  triggers:
    tool_error:
      enabled: true
      repeat_threshold: 2       # critique after Nth identical tool failure
    low_grounding:
      enabled: true
      threshold: 0.6            # below this coverage, reflect+reground (looser than guardrail abstain)
    # doom_loop and empty/max_iter are loop-guard rungs, not critique triggers (see §5.4)
  ladder:                       # declarative ordering; remove a rung to disable it
    - retry
    - reflect
    - replan
    - handover
```

- **Why a critic provider seam:** koboi already has `providers:` + `ProviderPool` + `resolve` (named-provider resolver). Routing the critic to a cheap/fast model (e.g. a mini) keeps reflection cost down — a key production concern (cf. CrewAI/SDK).
- **Inert-by-default:** matches koboi's established principle (proactive memory, sandbox, command-hooks are all opt-in). No surprise latency/$$ on existing configs.

### 5.4 Trigger-surface coverage (the four surfaces — porsioned)

Per the agreed scope, **all four** surfaces are covered, but with distinct roles:

| Surface | Role | Phase |
|---|---|---|
| **Tool errors** | **Primary reflection trigger** (POST_TOOL_USE) — grounded in the tool result | P0 feedback + P1 critique |
| **Low grounding** | **Primary reflection trigger** (POST_OUTPUT) — grounded via the existing verifier; turns today's *abstain* into *reground-and-retry* before refusing | P1 |
| **Doom-loop / repeated failure** | **Loop-guard rung** — upgrade today's opt-in *nudge* to optionally **force replan** (invoke orchestration planner) when the shared budget burns; reflection must be doom-loop-aware | P2 |
| **Empty/malformed + max_iter** | **Loop-guard + graceful-degrade rung** — fix silent-complete (re-ask on empty); summarize partial progress instead of hard-stop on `max_iterations` | P0 (empty) + P3 (graceful) |

Reflection is keyed on the two **grounded** triggers (tool-error, low-grounding) where critique has proven value. Doom-loop and empty/max_iter are **loop-guards/degrade**, not critique — consistent with the §4.1 "don't navel-gaze" principle.

### 5.5 Interaction with existing rungs (no reinvention)

- **Grounding guardrail** = the verifier. Reflection's low-grounding path calls into the same NLI/claim logic; the guardrail's `abstain` becomes the *terminal* outcome only *after* reflection's reground attempts are exhausted.
- **Handover** = terminal rung. Reflection feeds into handover's digest ("attempted reground ×N, still ungrounded") — the B4 warm-handoff summary already exists.
- **Doom-loop** = the budget enforcer. Reflection's `max_turns` is the *same* counter doom-loop watches; this prevents the classic failure where self-healing creates a faster doom-loop.
- **Resume** = orthogonal (infra). Reflection writes normal conversation rows, so `--resume` and `_repair_interrupted_turn` are unaffected; on resume, non-idempotent tools are still skipped.

---

## 6. Part D — Feasibility & risk assessment

### 6.1 What makes this *easy* (favorable seams)

- The **hook system** (15+ events, priority-sorted `HookChain`, `HookContext` with `tool_result`/`messages`/`metadata`/`abort`) is exactly the extension point a reflection loop needs — no core-loop surgery for the wedge.
- A **verifier already exists** (`GroundingGuardrail`); we are *composing*, not building, the hard part.
- A **critic client seam** already exists (`providers:` + `ProviderPool` + named-provider `resolve`).
- The **inject pattern** is already proven by `DoomLoopHook`/`HandoverDetectionHook`.
- **Fail-soft** and **redaction** conventions are already established (`redact.py`, grounding fail-soft).

### 6.2 Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Cost/latency blow-up** (extra LLM calls per turn) | High | Opt-in default; `max_turns` shared budget; cheap critic model via provider seam; per-trigger enable flags |
| **Reflection amplifies doom-loop** | High | Shared `max_turns` counter with doom-loop; reflection rung is itself doom-loop-aware; fall-through to replan/handover on budget exhaustion |
| **Intrinsic-critique hype** (no win without verifier) | High | Mandate a verifier — grounding/tools — for every critique trigger; **never** ship pure self-critique |
| **Non-idempotent tool double-fire on reflection-retry** | High | Respect `ToolDefinition.idempotent` (`registry.py:260-269`): skip re-exec / synthetic result on reflection-retry, exactly as `_repair_interrupted_turn` does on resume (`loop.py:589-603`) |
| **Sensitive data in critique prompt** | Medium | Route all critique inputs through `redact.py` |
| **Reflection in streaming (SSE)** | Medium | P1 ships for non-stream `run`; streaming reflection (re-buffering) deferred — mirrors the existing output-guardrail buffering pattern (`loop.py` `_process_output` G8) |
| **`_run_single` failed-flag bug masks replan** | Medium | Fixed in P0, before P2 replan relies on it |

### 6.3 Honest scope ceiling

This is **bounded reflection + ladder composition**, not autonomous self-improvement. It will not make the agent smarter than its model+verifier allow; it makes it **recover reliably within a declared budget and escalate cleanly when it can't** — which is exactly the "trustworthy unattended autonomy" posture.

---

## 7. Part E — Phased roadmap

Effort is rough (S=≤1 day, M=2-4 days, L=1 wk+) relative to this codebase's patterns. All phases are **opt-in / additive / fail-soft**. Each phase is independently shippable.

### P0 — Bug fixes + grounded tool-error feedback  *(effort: M, default: fixes on, feedback opt-in)*
**Unblocks and de-risks everything; ship first.**
- Fix `_run_single` to set `failed=True` on caught exceptions (`orchestration/orchestrator.py:466-472`).
- Make `max_replans` configurable with a sane non-zero default when re-plan is opted in (`facade.py:1954`).
- Fix empty-response silent-complete: empty content + no tool calls → re-ask (bounded) rather than `success=True` empty (`types.py:78`, `loop.py:641`).
- Upgrade tool-error feedback from a bare `"Error: …"` string to a **structured** "tool X failed because Y; consider Z" message, respecting `ToolDefinition.idempotent` (`tools/registry.py:197-242` → `loop_pipeline.py`).
- **Tests:** extend `test_journal`/orchestration-resume tests for the failed-flag; new `test_empty_response_reask`; new `test_structured_tool_error`.
- **Deliverable value:** immediate reliability gains; fixes recovery that nominally exists today.
- **STATUS (2026-07-16): SHIPPED on `worktree-self-healing-initiative`.** All four landed — `_run_single` failed-flag; `ExecutionConfig.max_replans` (default 0, opt-in); empty-response bounded re-ask (default-ON 1×, last-iteration-safe); structured `ToolPipelineResult.errored/error_kind/idempotent` + `execute_outcome()`. 4580 tests pass. **Known caveat (deferred to P2):** the now-live `max_replans>0` replan loop re-runs the *whole* planned graph (`results=[]` + fresh `graph_run_id`, no cursor-skip), so non-idempotent side-effecting tools in already-succeeded nodes can double-fire — keep dynamic nodes read-only or `max_replans=0` until P2.

### P1 — `ReflectionHook` (the wedge)  *(effort: M-L, opt-in)*
- New `koboi/hooks/reflection_hook.py` (handles `POST_TOOL_USE` + `POST_OUTPUT`).
- Low-grounding path: compose with `GroundingGuardrail` verifier → reground-and-retry before abstain.
- Tool-error path: critique repeated identical failures → corrected call suggestion.
- `self_healing:` config namespace wired in `facade.py` (build hook into the chain at priority ~60, post-business); critic via `providers:` resolver.
- Shared `max_turns` budget with doom-loop; fail-soft; `redact.py` on critique inputs.
- **Tests:** `test_reflection_hook` (low-grounding→reground, repeated-tool-error→critique, budget-exhaustion→fall-through, fail-soft-on-judge-error, non-idempotent-skip).
- **Demo:** `configs/self_healing_demo.yaml`.
- **Deliverable value:** closes the headline gap.
- **STATUS (2026-07-16): SHIPPED on `worktree-self-healing-initiative`.** `ReflectionHook` (priority 60, POST_TOOL_USE repeated-error critique + POST_OUTPUT low-grounding reground + SESSION_START reset); loop retry-seam (`_process_output` stashes `reflection_retry` → `_run_loop` honors via bounded `continue`; `run_stream` skips); `SelfHealingConfig` (opt-in); facade wiring (critic via `resolve_llm_spec`/`create_client`, fail-soft fallback). Verifier-grounded (GroundingGuardrail + P0-D), fail-soft, redacted. 14 tests + demo config; 4594 pass. Reground = self-correct against EXISTING context (critique names ungrounded claims). **Deferred:** (a) streaming parity (run_stream logs+skips); (b) query-reformulation/re-retrieval reground (P1b/P3); (c) skip the critic on the final iteration to avoid a wasted call (P2 -- needs hook↔agent `max_iterations` coupling); (d) shared `TurnBudget` with doom-loop (P2); (e) thread P0-D `errored`/`error_kind` onto POST_TOOL_USE `HookContext` so the hook needn't string-match `"Error:"` (P1b).

### P2 — Declarative escalation ladder + replan-on-failure  *(effort: M, opt-in)*
- `self_healing.ladder:` declarative policy mapping failure-class → ordered rungs.
- Upgrade doom-loop from *nudge* to optionally **force replan** (invoke orchestration planner / ReWOO-style re-plan) when the shared budget burns.
- Error-class tagging hook so the ladder can route (transient→retry, schema→reflect, grounding→reground, policy→escalate).
- **Per-node/subtree retry (resolves the P0 caveat):** cache succeeded-node results on replan so only the failed subtree re-runs under a continued `graph_run_id` (avoids double-firing non-idempotent tools); respect `ToolDefinition.idempotent`.
- **Tests:** ladder ordering, replan-on-failure, rung removal disables behavior.
- **Deliverable value:** turns scattered mechanisms into one coherent, auditable policy.
- **STATUS (2026-07-16): P2a SHIPPED on `worktree-self-healing-initiative`.** The single-agent escalation ladder: `FailureClassifierHook` (priority 5, tags failure_class from P0-D error_kind + live grounding threshold) + `LadderRouterHook` (priority 6, picks ONE rung/POST_OUTPUT turn: reflect-if-budget else handover; stamps `recovery_plan`) + shared per-run `RecoveryBudget` (router-owned, consumed by ReflectionHook on actual fire, reset on SESSION_START). One-line guards in ReflectionHook + HandoverDetectionHook honor the chosen rung (full back-compat when no router). Pipeline surfaces `error_kind` onto POST_TOOL_USE ctx. **Bonus bug fix:** `DoomLoopHook` now resets its detector on SESSION_START (was leaking history across runs). Config `self_healing.ladder`. 15 tests; 4609 pass. The previously-implicit ladder (handover silently won at coverage<0.5) is now EXPLICIT: reflect tries first, escalates to handover when the budget burns out.
  - **STATUS (2026-07-16): P2b SHIPPED** — orchestration per-node/subtree retry (resolves the P0 replan double-fire caveat). `AgentResult.had_non_idempotent_tool` (populated in `_run_single` via `idempotent=False OR risk_level∈{MODERATE,DESTRUCTIVE}` — the risk-level belt-and-suspenders catches the under-flagged DESTRUCTIVE builtins `run_shell`/`write_file`); `_run_dag_waves_with_flow(cached_results=...)` skips cached nodes (event parity, no re-run/re-record) + seeds downstream outputs; the replan loop re-runs the **failed subtree** (directly-failed + transitive downstream via `_downstream_closure`) and carries forward succeeded + side-effecting nodes. 6 tests; 4615 pass. Opt-in (`max_replans=0` default → zero behavior change).
  - **Still deferred:** (a) fold `doom_nudge` into the router (doom detection ahead of the router + node→orchestrator doom-signal via `AgentDoomLoopError`); (c) make the router's `handover` choice authoritative (fire unconditionally, not just when coverage<0.5); (d) **systemic:** set `idempotent=False` on the remaining side-effecting builtins (`delegate_tasks`, `memory_store` — currently SAFE-risk so the risk-level heuristic misses them) so both replan AND crash-resume are fully protected; (e) dynamic-graph durability (one `graph_run_id` across replans + wire the dormant `list_completed_nodes` cursor for `koboi run --resume`).

### P3 — Error taxonomy + graceful degrade  *(effort: M, opt-in)*
- Rule-based (+ optional side-LLM) error classifier stamping `HookContext` tags.
- Graceful degrade on `max_iterations`: summarize partial progress instead of `AgentMaxIterationsError` hard-stop (`loop.py:619,676-678`).
- **Deliverable value:** no more silent/abrupt dead-ends; improves the ladder's routing quality.
- **STATUS (2026-07-16): graceful degrade SHIPPED** — `AgentCore.graceful_max_iter` (opt-in, default off; independent of `self_healing.enabled`) + `_graceful_max_iter_summary()` side-LLM summary (fail-soft → last assistant msg → generic notice). The `_run_loop` tail returns a degraded `RunResult(success=True, metadata max_iter_degraded)` and `run_stream` yields `TextDelta`+`CompleteEvent` instead of `ErrorEvent`. The summary is routed through `_process_output` (output guardrails run on it + it's persisted, parity with every terminal answer; a block/handover falls back to the generic notice). 5 tests; 4620 pass. The **error-taxonomy** half of P3 was already delivered by P2a's `FailureClassifierHook` (tags `failure_class`); no separate side-LLM classifier was needed.
  - **Follow-up (P0-C gap found in review):** `empty_response_reask_limit` is not facade-plumbed (hardcoded default 1) — expose it via `self_healing` config like `graceful_max_iter`.

### P4 — Optional, higher-cost verifiers (opt-in, gated to `act`+)  *(effort: L)*
- **CRITIC-generalization:** run `calculator`/`web_search`/code tools to verify code/math/fact claims (generalizes grounding beyond RAG).
- **Self-consistency:** sample N completions, majority/structured-merge, for high-stakes outputs (multi-provider pool already supports parallel calls). Strictly opt-in — 3-5× spend.
- **Deliverable value:** measurable answer-quality gains where a verifier exists; keep off by default.

---

## 8. Part F — Alignment with koboi's vision

koboi's wedge is **"trustworthy unattended autonomy"** (autonomous-loop, not just workflow-graph). Self-healing is the load-bearing property that makes *unattended* trustworthy: an unattended agent that can't detect-and-recover-within-a-budget, or escalate cleanly, is not safe to leave alone. This design advances the vision by:

- **Composing** the rungs koboi already built (no throwaway), behind **one declarative policy**.
- **Grounding** every recovery in a verifier (grounding/tools) — empirically the only self-heal that works.
- **Bounding** everything with a shared budget + **escalating** to human on exhaustion — the definition of trustworthy autonomy.
- Staying **inert-by-default** and **fail-soft** — consistent with every other koboi subsystem.

---

## 9. Open questions (deferred to implementation)

1. **Critic model:** reuse main client by default, or mandate a separate cheap model in the `self_healing.critic` seam? (Recommend: default to main, allow named provider.)
2. **Streaming (SSE):** P1 non-stream only, or buffer-and-reflect for `/v1/chat/stream` day-one? (Recommend: non-stream first; streaming mirrors output-guardrail buffering later.)
3. **Orchestration scope:** does `ReflectionHook` fire inside orchestration nodes (per-`AgentCore`) and/or at the orchestrator synthesis layer? (Recommend: per-node first — the `QualityEvaluator` already owns synthesis revision.)
4. **Conformal thresholds:** ship runtime calibrated thresholds now, or keep the eval-only on-ramp? (Recommend: keep eval-only until a PPI dataset exists — never claim "knows when it doesn't know at X%".)
5. **Reconcile docs:** delete or merge the pre-existing `docs/self-healing-strategy.md` into this file.
