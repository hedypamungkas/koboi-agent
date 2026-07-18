# Deterministic Workflow Export/Import for koboi-agent: Managing LLM Non-Determinism for Enterprise Automation

*Deep research, 2026-07-14. Method: 7-agent fan-out (3 codebase analysis + 4 web research) running concurrently -> adversarial fact-check of 14 load-bearing claims (10 confirmed / 4 nuanced / 0 refuted; 2 verify agents lost to a gateway HTTP 429) -> synthesis. All load-bearing codebase claims were independently re-verified against source by the lead after synthesis (FORWARDABLE_LLM_KEYS, lossy Config.to_dict via LLMConfig extra='ignore', lossy `koboi graph --format json`, AgentDef fields, ScriptedClient, http_transport.post chokepoint, PR #37 ResearchContext.to_json/from_json, deep_research as execution.mode). Sourcing caveats in section 8.*

---

## 1. TL;DR

1. **"Deterministic LLM output" is not achievable on hosted API models in a guaranteed sense.** OpenAI's `seed` is explicitly Beta "best effort," monitored via `system_fingerprint` because backend changes "might impact determinism"; Anthropic exposes no `seed` at all and states "even with temperature of 0.0, the results will not be fully deterministic." (Verified — confirmed.) The honest framing is **variance reduction**, not determinism — except via two mechanisms: **exact-match output caching** and **deterministic-execution replay** that reuses recorded side-effect results (the Temporal/DBOS/Inngest archetype). (Verified — nuanced: a third path, deterministic *local* inference of open-weight models, exists but does not apply to hosted APIs.)

2. **There are four distinct "determinism layers" and enterprises routinely conflate them.** FORMAT (schema valid) is trivially achievable; DECISION (same branching) and PATH (same step sequence) are achievable only via recorded-trace replay or a static workflow graph; OUTCOME (byte-identical answer) is achievable only via output caching or full record-replay. Most vendor features marketed as "determinism" deliver only FORMAT or a narrow variance reduction.

3. **Claude Code "Dynamic Workflows" is export/import *of an orchestration scaffold*, not of determinism.** Claude LLM-authors a JS script (`agent()`/`pipeline()` primitives, `args` global), the runtime executes it live, and re-runs re-spawn fresh LLM subagents — so leaf outputs remain non-deterministic. Resume is session-scoped cache-replay that does not survive process exit. (Verified — confirmed.) The genuinely repeatable artifact is the *scaffold* (control flow), not the *outputs*.

4. **koboi already has 80% of the seams but they are disconnected.** `temperature`/`seed`/`top_p` are forwardable to providers (`FORWARDABLE_LLM_KEYS`), `response_format` plumbing exists end-to-end, `DagScheduler.persist_plan` writes durable resume rows, `ResearchContext.to_json()/from_json()` round-trips (PR #37), and `ScriptedClient`/`MockClient` prove the LLM-client seam supports deterministic replay. But: no `WorkflowDefinition` envelope, no `AgentDef.to_dict()/from_dict()`, lossy `Config.to_dict()` (drops all determinism knobs via `extra='ignore'`), lossy `koboi graph --format json`, no production replay transport, and no capture-from-run path.

5. **The strategic wedge beyond Claude Code is "capture a successful autonomous run, freeze it into a replayable workflow."** This combines Archetype A (code-as-workflow) + Archetype C (recorded-trace/event-log) + Archetype D (captured-program): export bundles the graph definition *and* the recorded LLM/tool I/O, so re-execution is byte-deterministic without re-calling the model. This is the only archetype that makes agentic LLM-call replay genuinely deterministic, and no surveyed competitor ships it as a first-class export.

6. **Recommended path:** Horizon 1 — faithful serialization (`WorkflowDefinition` + `AgentDef.to_dict/from_dict` + non-lossy `to_dict`); Horizon 2 — determinism layering (per-node pinning + structured outputs + optional `ReplayClient`); Horizon 3 — capture-from-run + byte-deterministic replay. Reuses `DagScheduler`, `WorkflowGraph`, the `steps` journal, and the `eval/t` surface.

---

## 2. The Determinism Problem for Enterprise Agent Adoption

### 2.1 Why non-determinism blocks day-to-day automation

Enterprise automation demands **repeatability contracts**: an accounts-payable clerk must trust that "process this invoice" produces the same ledger entry today and tomorrow, an SRE runbook must produce the same diagnosis for the same stack trace, and an audit trail must be reconstructable. LLM agents break this contract in four compounding ways:

- **Token-level sampling noise** — even at `temperature=0`, floating-point non-associativity combined with dynamic batching and GPU kernels whose reduction orders vary with batch size perturbs logits, so argmax can flip (OpenAI: "the content generated from a model is non-deterministic").
- **Backend/infra drift** — providers change routing, batching, and weights; OpenAI surfaces this via `system_fingerprint`, which "represents the backend configuration that the model runs with" and is used "to understand when backend changes have been made that might impact determinism."
- **Model version drift** — even dated snapshots are sunset on fixed timelines (OpenAI's documented history: gpt-4-0314, gpt-4-0613, gpt-3.5-turbo-0613, etc.); Chen et al. 2023 (arXiv:2307.09009) measured significant capability shifts in gpt-3.5-turbo and gpt-4 between March and June 2023.
- **Control-flow sensitivity** — in an agentic loop, a single differing token can change the tool selected, which changes the next observation, which changes the entire downstream path. Small token variance is amplified into large path divergence.

Customers "getting different results for the same instructions" is therefore not a bug to fix but a property of the substrate that must be *engineered around*.

### 2.2 The four-layer determinism taxonomy

To talk precisely, distinguish four layers — each achievable by different techniques:

| Layer | Definition | Achievable on hosted LLM APIs? | Primary technique |
|---|---|---|---|
| **FORMAT** | Output always conforms to a declared schema/grammar (valid JSON, valid enum, parseable structure) | **Yes** | Structured/constrained outputs (OpenAI Structured Outputs, Outlines, xgrammar, `lm-format-enforcer`) |
| **DECISION** | The agent makes the same discrete choice given the same input (same tool, same conditional branch, same classification) | **Partially** — narrowed by pinning + FORMAT, not guaranteed; fully achievable only if the decision is made by deterministic code over recorded inputs | Sampling pinning + structured outputs + recorded-input replay |
| **PATH** | The agent follows the same sequence of steps/tool calls | **Yes, conditionally** — via a static workflow graph (PATH fixed by code) OR recorded-trace replay (PATH fixed by event log) | Workflow-as-code (Archetype A) + recorded-trace replay (Archetype C) |
| **OUTCOME** | Byte-identical final answer for byte-identical input | **Yes, narrowly** — via exact-match output cache OR full record-replay that reuses recorded LLM responses | Exact-match output cache; deterministic-execution replay |

**Key insight:** FORMAT is cheap and always worth doing. DECISION and PATH are the layers automation actually depends on (a runbook that branches differently is a different run). OUTCOME is the strongest but the most expensive (requires recording and replaying every LLM call). The enterprise honest answer is: **"we give you PATH + DECISION determinism via a pinned workflow graph and recorded LLM I/O; OUTCOME determinism via optional output caching; FORMAT always."** No hosted LLM can promise OUTCOME without recording/replay.

---

## 3. Academic and Production Approaches to Managing Determinism

Ranked from weakest (variance-reducer / illusion) to strongest (true reproducibility). Each entry states what it **guarantees**, what it **does not**, the evidence, and the verify verdict.

### Tier 1 — Variance reducers (narrow the distribution; do NOT collapse it)

#### 3.1 Sampling pinning (temperature=0, seed, top_p)
- **Guarantees:** Narrows the sampling distribution. OpenAI `seed` is documented (official SDK) as Beta "best effort": "repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter to monitor changes in the backend."
- **Does NOT guarantee:** Bit-identical output. GPU floating-point/batching perturbs upstream logits; `system_fingerprint` changes invalidate reproducibility; seed pins only the downstream sampling RNG.
- **Cross-provider:** Anthropic Messages API exposes **no `seed`** (only `temperature`/`top_p`/`top_k`) and notes "even with temperature of 0.0, the results will not be fully deterministic." Cloudflare (OpenAI-compatible) accepts seed.
- **Verify verdict:** **Confirmed** (against openai-python and anthropic-sdk-python source).

#### 3.2 Self-consistency (Wang et al. 2022, arXiv:2203.11171)
- **Guarantees:** Reduces *final-answer* variance for tasks with a verifiable discrete answer, by sampling many paths (temperature 0.5–0.7) and majority-voting.
- **Does NOT guarantee:** Token-level determinism; helps only when per-sample accuracy > ~50%; the majority answer can still flip across independent self-consistency runs.
- **Verify verdict:** **Confirmed.** Empirically: GSM8K +17.9%, SVAMP +11.0%, AQuA +12.2%.

#### 3.3 Model version pinning (dated snapshots)
- **Guarantees:** Reduces cross-version drift within the snapshot's lifetime.
- **Does NOT guarantee:** Durable reproducibility — providers forcibly sunset snapshots (gpt-4-0613, gpt-3.5-turbo-0613, etc.) on fixed timelines; replacement/alias models behave differently.
- **Verify verdict:** **Nuanced.** Thesis confirmed (OpenAI's deprecation history + Chen et al. 2023), BUT the specific model names "gpt-5.5 / gpt-5.4-mini / gpt-5.4-nano" and exact dates cited in raw research **could not be verified** against OpenAI's official pages (current flagship per docs is gpt-5.6; "gpt-5.4-mini" appears to be an internal gateway alias, not an official OpenAI name). Pinning buys a bounded window, not indefinite reproducibility.

### Tier 2 — Illusion-of-determinism (marketed as determinism; is not)

#### 3.4 Structured / constrained outputs (FORMAT only)
- **Guarantees:** The output *parses* against a JSON Schema / grammar / regex. "The model will always generate responses that adhere to your supplied JSON Schema" (OpenAI).
- **Does NOT guarantee:** Content correctness or reduced behavioral variance. OpenAI itself warns "this can result in hallucinations if the input is completely unrelated to the schema." The JSON always parses; the values inside still vary run-to-run.
- **Verify verdict:** **Confirmed** as FORMAT-only (Outlines arXiv:2307.09702 guarantees "the structure of the generated text," explicitly structural not semantic).

#### 3.5 Prompt / prefix caching (KV reuse)
- **Guarantees:** Cost/latency reduction (OpenAI: cache writes 1.25× uncached input rate; Anthropic: cache hits 10% of base input price).
- **Does NOT guarantee:** Identical output. OpenAI verbatim: "the model computes a new response from the cached prompt prefix, so otherwise identical nondeterministic requests are not guaranteed to return identical output." The cached object is the *input prefix KV*, not the response; the completion is regenerated.
- **Verify verdict:** **Confirmed** (against OpenAI, Anthropic, Gemini caching docs). Conflating prompt caching with output caching is the canonical illusion-of-determinism error.

#### 3.6 Semantic cache (GPTCache-style)
- **Guarantees:** Higher cache hit rate / lower latency vs exact match.
- **Does NOT guarantee:** Determinism or correctness. Returns a cached answer to a *near-neighbor* query (embedding similarity); GPTCache's own README admits "you may encounter false positives during cache hits." Embedding-model drift and approximate-NN indices (Hnswlib/FAISS) add further non-determinism.
- **Verify verdict:** **Confirmed.** (Refinement: a frozen cache + pinned deterministic embedding model + exact NN is *locally* reproducible, but that reproducibility is not correctness.)

### Tier 3 — Guards and proxies (reduce probability of unacceptable output; do not make generation deterministic)

#### 3.7 Output verification / guardrails / LLM-as-judge
- **Role:** Post-hoc checks that verify, retry, or branch on failure. Convert an unverified output into an accepted/rejected one, bounding the probability of an unacceptable result.
- **Verify verdict:** Plausible / supporting (arXiv:2502.06193). Complementary to, not a substitute for, true determinism levers.

#### 3.8 Eval / golden-master / snapshot testing (the determinism PROXY)
- **Guarantees:** A regression gate — that the *distribution* of behavior has not drifted past a threshold across versions.
- **Does NOT guarantee:** Any specific future run. "You cannot make the model deterministic, but you can freeze a corpus of (input, acceptable-output) cases and gate regressions against it."
- **Production instances:** OpenAI cookbook routes users to promptfoo ("a more portable, code-oriented workflow for maintaining, running, and extending evaluations over time"); LangSmith "Create a Dataset from Existing Runs"; Langfuse Datasets. Combined with N-shot statistics it bounds variance.
- **Verify verdict:** **Confirmed.**

### Tier 4 — True reproducibility (byte-identical for identical input)

#### 3.9 Exact-match output cache
- **Guarantees:** Byte-identical output for byte-identical (previously-seen) input, *without calling the model*. The cached artifact is the full response, so replay is exact by construction (contrast with prompt caching, which caches only the prefix).
- **Does NOT guarantee:** Outputs for *novel* inputs (only reproduces what you have seen); correctness of the cached answer.
- **Verify verdict:** **Nuanced.** The mechanism is genuine, but "one of only two techniques" is too strong *as a universal statement*: deterministic local inference of open-weight models (fixed seed + `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` + pinned weights + identical hardware) is a third path. **For hosted API LLMs (which koboi targets), caching and replay are indeed the only guaranteed mechanisms.**

#### 3.10 Deterministic-execution replay (Temporal / DBOS / Inngest) — the deepest pattern
- **Guarantees:** The workflow *re-execution* is deterministic. The workflow code is constrained to be deterministic; every non-deterministic operation (LLM call, wall-clock, randomness, network I/O) is isolated behind an Activity/step whose result is **recorded once and reused, not recomputed, on replay**. Temporal docs (verbatim): "When a Workflow calls an Activity, the Activity runs once, its result is recorded in the Event History. During replay, that result is reused, not recomputed." Temporal explicitly lists "LLM invocations" under Activities.
- **Engineering discipline:** Workflow code must NOT call Date.now()/Math.random()/native I/O directly — violations surface as non-determinism errors on replay. "The determinism is a property of the harness, not the model."
- **Verify verdict:** **Confirmed** (against docs.temporal.io, fetched directly). Nuance: the three engines differ in recovery granularity — Temporal does full event-sourced replay from the start of history; Inngest replays with step.memoization; DBOS resumes from the last completed step checkpoint — but all three reuse recorded side-effect results.

#### 3.11 Workflow-capture (autonomous run → frozen workflow)
- **Role:** The *bridge* from Tier 3 to Tier 4. Record a successful one-off agentic run (its trace + tool-call sequence + LLM I/O), then freeze it into a reusable, replayable workflow. Observability tools provide the CAPTURE half (LangSmith "Create a Dataset from Existing Runs," Langfuse Datasets); durable-execution engines provide the FREEZE-INTO-REPLAY half.
- **Why it matters for koboi:** This is the Claude-Code-plus feature — capture is what turns a one-off autonomous success into a durable automation asset.

---

## 4. The Export/Import Landscape

### 4.1 Claude Code Dynamic Workflows (mechanism + limits)

A dynamic workflow is a **JavaScript script that Claude LLM-authors** and a dedicated runtime executes in an isolated background environment. The canonical structure (from code.claude.com/docs/en/workflows, verified verbatim):

```js
export const meta = {
  name: 'audit-routes',
  description: 'Audit every route handler for missing auth checks',
}

const found = await agent('List every .ts file under src/routes/.', { schema: {...} })
const audits = await pipeline(found.files, file => agent(`Audit ${file}...`, { label: file }))
return audits.filter(Boolean)
```

- **Two orchestration primitives:** `agent(prompt, opts)` spawns one LLM subagent; `pipeline(list, fn)` runs one agent per list item. Input via a global `args`.
- **No direct filesystem/shell access from the script itself** — all I/O delegated to subagents; the script only coordinates.
- **Export/share UX:** Every run writes its script to `~/.claude/projects/...`. To reuse, run `/workflows`, select the run, press `s`, and save to `.claude/workflows/` (project-shared, git-distributed) or `~/.claude/workflows/` (personal). Saved workflows become `/<name>` slash-commands in autocomplete.
- **GA** on Claude Code v2.1.154+, all paid plans + Anthropic API + Bedrock + Vertex + Microsoft Foundry.

**Limits (load-bearing):**
1. **NOT deterministic.** Each `agent()` leaf re-invokes the LLM every run (leaf outputs non-deterministic); authoring itself is LLM-generated (same prompt → possibly different script) until saved; resume is session-scoped cache-replay of completed agents that does NOT survive process exit ("If you exit Claude Code while a workflow is running, the next session starts the workflow fresh").
2. **Runtime+provider coupling** — scripts only run inside the Claude Code workflow runtime via private `agent()`/`pipeline()` primitives; every subagent uses the session's model. Not portable to other agent frameworks.
3. **Scale caps** — up to 16 concurrent agents; 1,000 agents total per run.
4. **No mid-run user input** — only permission prompts can pause a run.
5. **LLM-authored JS is a review/security surface** (mitigated by per-run approval + "View raw script").

(All mechanism claims above: **Verified — confirmed**, against Anthropic's official docs.)

**What is repeatable is the orchestration scaffold (control flow), not the outputs.** The script-as-code is the determinism axis Claude Code chose; it does not attempt OUTCOME determinism.

### 4.2 Four archetypes of agentic-workflow serialization

Cross-framework survey decomposes every product into one **definition archetype** + optionally one **runtime archetype**:

| Archetype | Definition form | Examples | Determinism on re-run | Portability / shareability |
|---|---|---|---|---|
| **A — Code-as-workflow** | Imperative/declarative code (Python/TS) | Temporal, Airflow, DBOS, Inngest, CrewAI, Mastra, OpenAI Agents SDK, LangGraph-definition | Deterministic *if* paired with C-style recorded results; otherwise LLM nodes re-execute non-deterministically (LangGraph) | Low–medium: bound to the language/SDK |
| **B — Declarative JSON/YAML graph** | Portable nodes + connections + params | n8n, Make, Zapier; LangGraph `get_graph().to_json()` (viz-only) | **Lowest** — all nodes (incl. AI) re-run fresh; no memoization | **Highest** — language-agnostic, marketplace-ready |
| **C — Recorded-trace / event-log** | Append-only event history / checkpoints | Temporal Event History, DBOS Postgres checkpoints, Inngest State store, Mastra snapshots, LangGraph checkpoints | **Highest** — recorded side-effect results are reused, never recomputed (the *only* archetype that makes LLM-call replay deterministic) | **Lowest** — bound to exact code+language that produced it |
| **D — Captured-program** | Exported session bundling intent + recorded LLM/tool I/O | Claude Code dynamic workflows (partial: script + session-scoped cache) | Medium — scaffold deterministic once saved; leaf outputs re-sampled | Medium — script is portable text, but runtime + provider coupled |

### 4.3 The determinism × portability tradeoff

Ranked by **replay determinism** (best → worst): **C > A+C > D >> B**.
Ranked by **portability/shareability** (best → worst): **B >> D > A > C**.

**No surveyed product makes the graph/branching itself deterministic on replay by constraining the LLM.** They all push the determinism boundary to the *leaves* — every non-deterministic call is wrapped as an Activity/step/snapshot whose recorded result is replayed. LangGraph is the counter-example that proves the rule: because it does NOT record/reuse leaf results, its LLM-node replay is non-deterministic by default.

**Implication:** to make an agentic export replay-deterministic, the export must bundle BOTH the definition (code or graph) AND the recorded leaf I/O trace. This is Archetype A+C combined — the Temporal gold standard — and is the design wedge recommended for koboi in §6.

---

## 5. koboi Current State (Grounded)

### 5.1 Workflow-definition surface — scattered, no envelope object

There is **no single serializable workflow-definition object**. The canonical graph is implicitly the union of:

- **`AgentDef`** (`koboi/types.py:133-152`) — the closest thing to a node spec: `name`, `system_prompt`, `description`, `keywords`, `depends_on` (DAG edges), `conditionals` (list of `{to, when}`), `llm_config`, `rag_config`, `tools_config`, `interrupt_after`. Plain `@dataclass` with **no serialize/deserialize methods** (no `asdict`/`to_dict`/`to_json`).
- **`WorkflowGraph` / `CompiledGraph`** (`koboi/orchestration/workflow_graph.py:46-49, 70-93`) — graph state in **PRIVATE** attributes (`_nodes`, `_edges`, `_conditionals`, `_node_defs`, `_deps`). No public getters; no `to_dict`/`to_json`/`to_yaml`. A graph built in Python code **cannot be exported today**.
- **`DagScheduler`** (`koboi/orchestration/dag_scheduler.py:38-54`) — holds `deps`/`conditionals`/`interrupt_nodes` in memory; `persist_plan()` (`:111-134`) writes durable resume rows to the SQLite `steps` table but is **write-only** (no `load_plan`); rows record node names + wave order, NOT edges/conditionals/prompts.
- **Raw YAML `orchestration:` dict** — `Config.from_yaml` returns it verbatim (`config.py:446-447`).
- **Transient `PlanResult`/`PlanStep`** (`planner.py:23-38`) — dynamic-mode graphs, LLM-planned per query, held only in memory (`_dynamic_blueprints`, `_agents_map`); **never serialized** except as partial node/wave rows.

### 5.2 Determinism knobs — present but inert / unpersisted

| Knob | Status in koboi | Evidence |
|---|---|---|
| `temperature` | First-class, threaded to all providers when set | `RetryClient.temperature` (`client.py:54,86`); adapters (`openai_adapter.py:45,53,81-82`; `anthropic_adapter.py:36,45,69-70`) |
| `seed`, `top_p`, `top_k` | In `FORWARDABLE_LLM_KEYS`; forwarded to OpenAI/Cloudflare; **`seed` dropped for Anthropic** | `config.py:86-103`; `_PROVIDER_EXTRA_KEYS` (`factory.py:30-58`) |
| `response_format` (JSON Schema) | Plumbing exists end-to-end; used by `QualityEvaluator` + `planner.plan_or_skip`; **ABSENT on agent nodes** (`AgentCore.run` calls `complete` with no `response_format`) | `client.py:127,132`; `openai_adapter.py:21-30`; `anthropic_adapter.py:210-221`; `planner.py:42-65`; `orchestrator.py:53-61` |
| `max_tokens` / `reasoning_effort` / `thinking` | First-class; per-agent materialized via `client_builder` | `config.py:98-101`; `factory.py:42-44`; `facade.py:1668-1674` |
| Per-agent LLM client override | Supported — dedicated client built when `llm_config` has knobs beyond `max_context_tokens` | `factory.py:127-141, 229-236`; `facade.py:1664-1674` |
| Prompt caching (`cache_control`) | **NOT wired anywhere** (`grep` across `koboi/llm/` returned zero; the only "ephemeral" hit is `loop.py` proactive-memory) | — |

**Critical serialization bug:** `Config.to_dict()` (`config.py:250-254`) returns `schema.model_dump()` and **drops every determinism knob** (`seed`, `top_p`, `response_format`, `reasoning_effort`, etc.) because `LLMConfig` uses `extra='ignore'` (`config_models.py:54-70`). A round-trip `YAML → Config → to_dict → YAML` loses all determinism settings. The values survive in `config.raw` (`config.py:247`) but `raw` is undocumented as an export source.

### 5.3 Durability / resume — crash-recovery, not replay

- **`koboi run --resume`** (`cli_commands.py:174-183`) does **rehydrate-and-CONTINUE, NOT deterministic replay.** `resume()` (`loop.py:616-635`) marks prior "running" steps "interrupted," re-executes missing tool calls (skipping non-idempotent tools with a synthetic result at `:517-536`), then issues a FRESH LLM call. Two resumes of the same interrupted session can diverge.
- **`steps` journal** (`journal.py:66-150`) stores status + token counts + redacted tool args; it does **NOT** store raw LLM response content, generation params actually sent, or logprobs. Not a deterministic replay log.
- **DAG orchestration** persists a durable graph plan (`dag_scheduler.py:111-134`) — but it is completion-cursor recovery, not full plan reconstruction. **Plain sequential/parallel orchestration has NO durability/resume at all.**
- **No loader** reconstructs a runnable workflow from `graph_plan` rows: `persist_plan` is write-only; `list_completed_nodes` returns `{node_id: output}`, not topology/edges/agent defs.

### 5.4 Existing export surface — lossy, not importable

- **`koboi graph --format json`** (`cli_commands.py:103`) is **LOSSY**: emits only nodes + `depends_on` edges; OMITS conditional edges, execution mode, router config, and all per-agent `llm_config`/`rag_config`/`tools_config`/`system_prompt`. Visualization aid, not a portable format.
- **`koboi diagnostics`** (`diagnostics.py:22-122`) exports a sanitized ZIP (config.json redacted via `_sanitize_config`; messages.json; telemetry; tools; hooks). Not importable; `koboi sessions` is text-only. No `cmd_import` anywhere.
- **`ResearchContext.to_json()/from_json()`** (`orchestration/research.py:218-284`, PR #37) **round-trips** (verified by `tests/orchestration/test_deep_research.py:168-194`) — this IS the existing export/import seam for deep-research run state. Pair with `research:` + `websearch:` config sections to reproduce the recipe.

### 5.5 Replay doubles — test-scoped only

- **`MockClient`** (`tests/conftest.py:16-50`) and **`ScriptedClient`** (`koboi/eval/t/mock.py:11-40`) serve predetermined `AgentResponse` lists by index. They prove the LLMClient seam (`koboi/llm/base.py`) supports a drop-in replay client, and that **the orchestration engine is deterministic GIVEN fixed LLM responses**. But they are test-scoped only — not production-wired as a record/replay transport.

### 5.6 Skills + evals — already portable+deterministic-shaped

- **`skills/*`** are `SKILL.md` with `agentskills.io` frontmatter.
- **`evals/**/*.eval.py`** are portable Python test files with module-level `MOCK_RESPONSES` for deterministic, API-key-free runs.

These are the closest existing analogues to a portable workflow format inside koboi.

### 5.7 Deep-research path (PR #37) — half-built for portability

- `deep_research` is an **execution.mode**, not an `AgentMode` (`modes.py:13-21`); dispatched as a separate path (`orchestrator.py:1161-1164`).
- `_run_deep_research` is an **LLM-driven adaptive loop** (plan → DAG waves → coverage judge → re-plan → synthesis) — inherently non-deterministic. Cannot be expressed as a static `WorkflowGraph`.
- **Resume** (`orchestrator.py:884-914`) loads a journaled `ResearchContext` and skips plan+research, going straight to synthesis. So "import and continue researching" is NOT supported today — only "import and finalize the report."
- **Per-call web provenance is incomplete:** `CountingSearchProvider`/`CountingFetchProvider` (`websearch/providers/counting.py:21-46`) meter budget but do NOT record actual search queries/results or fetched content. The only record is coarse `SearchEvent`/`FetchEvent` (query/url only).

### 5.8 Precise gaps for export/import + determinism

1. No `WorkflowDefinition` envelope dataclass bundling nodes + edges + conditionals + mode + router + determinism profile.
2. No `AgentDef.to_dict()/from_dict()` (inverse of `facade._parse_agent_defs` at `facade.py:1580-1607`).
3. `WorkflowGraph`/`CompiledGraph` expose no public accessors and no serialize.
4. `koboi graph --format json` lossy.
5. `Config.to_dict()` lossy for all determinism knobs (`extra='ignore'`); no `to_yaml()`.
6. No per-call generation-param provenance in the journal.
7. No runtime replay/cache transport (ScriptedClient is eval-only).
8. Dynamic-mode `PlanResult` transient and non-portable.
9. No structured session export/import (`koboi export <session>` / `koboi import`).
10. No capture-from-run path (autonomous run → frozen workflow).

---

## 6. Proposed Design: Deterministic Workflow Export/Import for koboi (beyond Claude Code)

The goal is the Claude-Code feature (export/import agentic workflows) **plus** genuine PATH/OUTCOME determinism via record-replay. The design combines Archetype A (code-as-workflow, via koboi's existing graph model) + Archetype C (recorded-trace, via the journal) + Archetype D (captured-program, via capture-from-run).

### 6.1 The `WorkflowDefinition` envelope — portable, declarative, versioned, provider-agnostic

Add a new dataclass in `koboi/types.py` (near `AgentDef`) that becomes the single serializable artifact:

```python
@dataclass
class DeterminismProfile:
    """Workflow-wide determinism defaults; per-node llm_config overrides."""
    temperature: float | None = 0.0        # default pin
    seed: int | None = None                # OpenAI/Cloudflare only; warn on Anthropic
    top_p: float | None = None
    response_format: dict | None = None    # optional per-node JSON schema
    model_pin: str | None = None           # dated snapshot, e.g. "gpt-5-2025-08-07"
    replay_mode: str = "live"              # live | cache | replay

@dataclass
class WorkflowDefinition:
    schema_version: str                    # "1.0"
    name: str
    description: str
    execution_mode: str                    # dag | conditional | sequential | parallel | dynamic
    full_graph: bool = False
    router: dict | None = None             # router_type + config
    nodes: list[AgentDef] = field(default_factory=list)
    determinism: DeterminismProfile = field(default_factory=DeterminismProfile)
    provenance: dict | None = None         # source run_id, captured_at, koboi_version
```

**Schema example (exported YAML):**

```yaml
schema_version: "1.0"
name: invoice-approval
description: "Classify → validate → route invoice for payment"
execution_mode: dag
router: { type: keyword }
determinism:
  temperature: 0.0
  seed: 42              # note: ignored on Anthropic; export validator warns
  model_pin: "gpt-5-2025-08-07"
  replay_mode: live
nodes:
  - name: classify
    system_prompt: "Classify the invoice type..."
    llm_config:
      temperature: 0.0
      response_format: { type: json_schema, json_schema: { ... } }
    tools_config: { allowed: [calculator] }
  - name: validate
    system_prompt: "Validate fields against ledger rules..."
    depends_on: [classify]
    conditionals:
      - { to: route_payment, when: "{{validate.status}} == 'ok'" }
      - { to: route_exception, when: "{{validate.status}} == 'error'" }
  - name: route_payment
    depends_on: [validate]
    interrupt_after: true
provenance:
  source_run_id: "run_20260714_001"
  captured_at: "2026-07-14T10:30:00Z"
  koboi_version: "0.14.0"
```

### 6.2 (a) Serialization — fill the seams

| Work item | Seam (file:symbol) | Effort |
|---|---|---|
| `AgentDef.to_dict()/from_dict()` | `koboi/types.py:133-152`; reference field mapping from `facade._parse_agent_defs` (`facade.py:1580-1607`) | Low |
| `WorkflowGraph.to_dict()/to_json()/to_yaml()` + `from_dict()/from_yaml()` + public `@property` accessors for `_nodes/_edges/_conditionals` | `koboi/orchestration/workflow_graph.py:46-49, 70-93` | Medium |
| Fix `Config.to_dict()` lossiness: merge `raw['llm']` ∩ `FORWARDABLE_LLM_KEYS` back into the dump, OR change `LLMConfig` to `extra='allow'` | `config.py:250-254`; `config_models.py:54-70` | Low |
| `Config.to_yaml()` — dump `self._data` directly (already preserves everything post env-interpolation + provider-ref expansion) | `config.py:247` (raw property) | Low |
| `PlanResult.to_dict()/from_dict()` for dynamic-mode portability | `koboi/orchestration/planner.py:23-38` | Medium |
| `WorkflowDefinition.to_yaml()/from_yaml()` — the new round-trip | new module `koboi/orchestration/workflow_definition.py` | Medium |

### 6.2 (b) Capture-from-run — the Claude-Code-plus feature

The bridge from autonomous → deterministic. Three-step pipeline:

1. **Capture** — record a successful autonomous run. Reuse `koboi/diagnostics.py:22 collect_diagnostics` (already bundles config.json/messages.json/telemetry/tools/hooks) as the base; generalize into a portable session-export format. Extend `koboi/journal.py:66 record_step` with two additive columns: `generation_params_json` (seed/temperature/top_p actually sent) and `response_hash`/`response_json` (the raw LLM response). The additive ALTER pattern (`memory_sqlite.py:17-55 ensure_steps_table`) already adds columns without breaking existing DBs. This turns the journal into a deterministic replay log keyed by `(session, turn, step)`.

2. **Extract** — derive a `WorkflowDefinition` from the captured trace. For a deep-research run, serialize `ResearchContext.to_json()` (`research.py:234-284`) + the `research:`/`websearch:` config sections + the per-round `PlanResult` DAGs. For a DAG/conditional run, read `DagScheduler.deps`/`.conditionals` + the captured `AgentDef` configs. For an autonomous single-agent run, the "workflow" is the tool-call graph implied by the journal (a linear/branching sequence of tool uses).

3. **Freeze** — write the `WorkflowDefinition` (+ optional recorded LLM/tool I/O bundle for replay mode) to `.koboi/workflows/<name>.yaml` (project-shared, git-distributed) or `~/.koboi/workflows/<name>.yaml` (personal), mirroring Claude Code's save UX.

### 6.2 (c) Layering determinism

Stack the techniques from §3, strongest last:

1. **Pinning** (PATH/DECISION narrowing): a `DeterminismProfile` (workflow-wide) consumed by `_agent_client_builder` (`facade.py:1668-1674`) so every node's dedicated client is built with pinned sampling. `seed` flows through `extract_extra_params` (`config.py:106`) → `_PROVIDER_EXTRA_KEYS` filter (`factory.py:30-58`). **Validator warns on Anthropic** (seed unsupported) and on unpinned `model_pin` (bounded-window caveat).

2. **Structured node outputs** (FORMAT guarantee): optionally pass a per-node `response_format` (JSON schema) into `AgentCore.run` → `client.complete` so conditional predicates (`orchestrator._eval_conditional` at `orchestrator.py:453-486`) match structured fields instead of free-form text. The `response_format` plumbing already exists end-to-end (`client.py:127` → adapters); this is "wire it on the agent-node path, not just planner/evaluator."

3. **Exact-match output cache** (OUTCOME guarantee, Tier 4): generalize `ScriptedClient` (`koboi/eval/t/mock.py:11-40`) into a production `ReplayClient`/`CachedTransport` wrapping `HttpTransport.post` (`http_transport.py:52`, the single body→response chokepoint used by all providers). Memoize/replay on a stable hash of the request body. The orchestration engine is already deterministic given fixed responses (proven by MockClient tests). Two modes:
   - `replay_mode: cache` — memoize live responses; subsequent identical calls return the cached response (real model first time, byte-identical after).
   - `replay_mode: replay` — load a recorded I/O bundle from the export; no live calls at all (byte-deterministic, offline, API-key-free — the same property that makes `evals/**/*.eval.py` deterministic).

4. **Deterministic-replay of non-LLM steps** (Temporal/DBOS archetype): non-LLM tool calls are already deterministic given fixed inputs; the journal records their redacted args + status. On replay, re-execute deterministic tools and reuse recorded LLM responses (per §3.10 — quarantine non-determinism behind a recorded activity). This is exactly the seam `_repair_interrupted_turn` (`loop.py:479-537`) already uses for crash-recovery, generalized to full-session replay.

### 6.2 (d) Import / share / replay UX + CLI

Register new subparsers in `koboi/cli.py:213` (next to `graph`); bodies in `koboi/cli_commands.py`:

```
koboi export <session> --format yaml|json [--with-replay-bundle]
  # Config.from_yaml → _parse_agent_defs → WorkflowDefinition → file
  # --with-replay-bundle also writes the recorded LLM/tool I/O (ReplayClient input)

koboi import <bundle> [--name <name>] [--scope project|user]
  # read WorkflowDefinition → ConfigBuilder.orchestration(agents=list[dict]) (config.py:669-684) → Config → KoboiAgent
  # ConfigBuilder.orchestration already accepts agent dicts — the import side is half-built

koboi run <config> --workflow <name> [--replay-mode live|cache|replay] [--input <args-json>]
  # invoke an imported workflow with structured args (the args-global analog)

koboi workflows list|show|delete    # manage .koboi/workflows/ + ~/.koboi/workflows/
```

The `--scope project|user` mirrors Claude Code's `.claude/workflows/` vs `~/.claude/workflows/` distinction. Workflows checked into git are shareable across teams (the enterprise shareability axis).

### 6.2 (e) Reuse of koboi assets

| Asset | Role in the design |
|---|---|
| `DagScheduler` (`dag_scheduler.py:38-54`) | Add `load_plan(db_path, graph_run_id) -> (waves, deps, conditionals)` classmethod + `DagScheduler.from_plan(...)` to reconstruct a runnable graph from `graph_plan` rows (closes the export/import gap). |
| `WorkflowGraph` (`workflow_graph.py`) | Add `serialize()`/`deserialize()` — the natural portable-workflow-format seam (LangGraph-shaped API). |
| `steps` journal (`journal.py:66`, `memory_sqlite.py:17-55`) | Extend with `generation_params_json` + `response_hash`/`response_json` → becomes the deterministic replay log. |
| `ResearchContext.to_json/from_json` (`research.py:234-284`) | The existing deep-research export artifact. Pair with `research:`/`websearch:` config for full recipe portability. |
| `ScriptedClient`/`MockClient` (`eval/t/mock.py`, `conftest.py`) | Template for the production `ReplayClient`/`CachedTransport`. |
| `eval/t` surface | Exported workflows with `replay_mode: replay` are runnable as evals (deterministic, API-key-free) — unifies the workflow + eval formats. |
| `diagnostics.py collect_diagnostics` | Base for capture-from-run (already bundles config/messages/telemetry). |
| `_emit_research_hook` (`orchestrator.py:210-222`) | Capture the control-plane LLM-call trace (plan/coverage/synthesis) as an audit trail of planning decisions. |
| `StreamEvent` provenance stream (`events.py:117-149`) | `SearchEvent`/`FetchEvent`/`SourceEvent`/`CoverageEvent` reconstruct WHAT a run did independently of LLM content. To close per-call provenance, instrument `CountingSearchProvider`/`CountingFetchProvider` to record query+results into `ResearchContext`. |

---

## 7. Roadmap

### Horizon 1 — Foundational serialization (faithful round-trip)
*Impact: High (unblocks everything). Effort: Low–Medium. No new deps.*

- H1.1 Fix `Config.to_dict()` lossiness (merge `FORWARDABLE_LLM_KEYS` back in); add `Config.to_yaml()`. — `config.py:250-254`, `config_models.py:54-70`
- H1.2 `AgentDef.to_dict()/from_dict()` with schema version. — `types.py:133-152`
- H1.3 `WorkflowGraph`/`CompiledGraph` public accessors + `to_dict/from_dict`. — `workflow_graph.py:46-93`
- H1.4 Extend `koboi graph --format json` to emit conditionals + execution_mode + router + per-agent llm_config (converts the lossy viz into a valid partial export). — `cli_commands.py:103`
- H1.5 `WorkflowDefinition` envelope + `to_yaml/from_yaml`. — new `orchestration/workflow_definition.py`
- H1.6 `koboi export <session> --format yaml|json` + `koboi import <bundle>` + `koboi workflows list`. — `cli.py:213`, `cli_commands.py`

**Dependencies:** none (additive). Ship as a minor release.

### Horizon 2 — Determinism layering (PATH + FORMAT guarantees)
*Impact: High (the enterprise ask). Effort: Medium. No new deps.*

- H2.1 `DeterminismProfile` workflow-level section; consumed by `_agent_client_builder`. — `facade.py:1668-1674`
- H2.2 Per-node `response_format` wired on the agent-node path (not just planner/evaluator); conditional predicates match structured fields. — `loop.py:564`, `orchestrator.py:453-486`
- H2.3 Export validator: warn on Anthropic+seed, warn on unpinned model, warn on `sliding_window` (non-deterministic LLM summarization — see §5/§8).
- H2.4 Production `ReplayClient`/`CachedTransport` at `http_transport.py:52` with `replay_mode: cache` (memoize live responses). Generalizes `ScriptedClient`.

**Dependencies:** H1. Beneficial: PR #23/#26 (memory) for stable per-session state; PR #37 (deep-research) for `ResearchContext` portability.

### Horizon 3 — Capture-from-run + byte-deterministic replay (OUTCOME guarantee)
*Impact: Strategic (the wedge beyond Claude Code). Effort: High.*

- H3.1 Extend `steps` journal with `generation_params_json` + `response_hash`/`response_json` (additive ALTER). — `journal.py:66`, `memory_sqlite.py:17-55`
- H3.2 `DagScheduler.load_plan(...)` + `DagScheduler.from_plan(...)` to reconstruct runnable graphs from `graph_plan` rows.
- H3.3 Capture-from-run: generalize `collect_diagnostics` into a portable export that bundles `WorkflowDefinition` + recorded LLM/tool I/O (`--with-replay-bundle`).
- H3.4 `replay_mode: replay` path — load the bundle into `ReplayClient`; no live LLM calls (byte-deterministic, offline).
- H3.5 Dynamic-mode portability: `PlanResult.to_dict/from_dict` + "freeze plan" path in `_run_dynamic` (`orchestrator.py:640-744`).
- H3.6 Instrument `CountingSearchProvider`/`CountingFetchProvider` to record per-call provenance into `ResearchContext` (closes the deep-research audit gap).

**Dependencies:** H1 + H2. PR #37 merged (for `ResearchContext` + deep-research hooks). PR #23/#26 merged (for journal + session_meta stability).

### Impact × Effort summary

| Horizon | Capability | Impact | Effort | Dependency |
|---|---|---|---|---|
| H1 | Faithful workflow serialization + export/import CLI | High | Low–Med | none |
| H2 | Determinism pinning + structured nodes + output cache | High | Med | H1 |
| H3 | Capture-from-run + byte-deterministic replay | Strategic | High | H1+H2, PR #37 |

---

## 8. Caveats

- **Sourcing limitations.** Web-search and web-reader MCPs returned HTTP 429 (quota exhausted until 2026-07-28) during the underlying research; primary-source verification for some claims relied on direct HTTP fetches of vendor docs (Anthropic, OpenAI, Temporal) and GitHub READMEs. Third-party comparisons (Cursor, Devin, etc.) could not be independently retrieved and are not asserted here.
- **Model-name verification gap.** The specific OpenAI model names "gpt-5.5 / gpt-5.4-mini / gpt-5.4-nano" and exact deprecation dates (2026-06-11, 2026-12-11, 2026-10-23) cited in raw research **could not be verified** against OpenAI's official deprecation/models pages (current flagship per docs is gpt-5.6; "gpt-5.4-mini/-nano" appear to be internal gateway/proxy aliases). The *thesis* of model-version pinning (bounded window, forced sunsets, Chen et al. 2023 drift) is confirmed; the specifics are not. Do not cite those names/dates externally.
- **"LLM-42" citation unverified.** The arXiv:2601.17768 (LLM-42) source cited in raw research for the floating-point/batching non-determinism mechanism is **unauthenticated** (GitHub returns 0 results; could not be independently surfaced). The underlying mechanism is well-established in ML-systems literature (PyTorch randomness docs, NVIDIA determinism docs) and in OpenAI's own best-effort `seed` caveat; rely on those, not on "LLM-42."
- **"Only two techniques" framing.** For hosted API LLMs (koboi's target), output caching and record-replay are indeed the only guaranteed reproducibility mechanisms. As a universal statement, this omits a third path — deterministic *local* inference of open-weight models with `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` + pinned weights + identical hardware. This caveat does not affect the recommendations (koboi uses hosted providers) but should not be overclaimed externally.
- **Recovery-granularity nuance.** "Replay determinism (Temporal/DBOS/Inngest)" groups three engines that share the archetype but differ in mechanism: Temporal does full event-sourced replay from the start of history; Inngest uses step.memoization; DBOS resumes from the last completed step checkpoint. The shared property (recorded side-effect results reused, not recomputed) is what matters; the grouping is loose.
- **Point-in-time claims.** Claude Code dynamic workflows are GA on v2.1.154+ (verified); the feature surface (primitives, scale caps, resume semantics) may evolve. OpenAI/Anthropic API surfaces (seed, system_fingerprint, prompt-caching pricing) are accurate as of July 2026 and may change.
- **Hidden non-determinism in koboi itself.** `sliding_window` context strategy summarizes old messages via an LLM call (`context/manager.py:337-367`); the cached summary is stable per-session (survives resume) but a fresh re-run produces a different summary → different prompt window → different output. A `DeterminismProfile` should warn or force `smart_truncation` (rule-based, deterministic) for replay-grade exports.
- **Anthropic seed unsupported.** Any `DeterminismProfile` with `seed` set will be silently dropped on Anthropic (no `seed` in the Messages API). The export validator must warn; cross-provider determinism contracts are unavoidably provider-conditional.

---

## 9. Sources

Deduplicated, one-line annotations.

**Determinism — academic & vendor primary**
- https://arxiv.org/abs/2203.11171 — Wang et al. 2022, Self-Consistency (variance-reducer via majority vote over sampled paths; verified confirmed).
- https://arxiv.org/abs/2307.09009 — Chen et al. 2023, "How is ChatGPT's behavior changing over time?" (independent evidence of cross-version drift).
- https://arxiv.org/abs/2307.09702 — Outlines (Willard & Louf); constrained decoding guarantees structure not content.
- https://platform.openai.com/docs/guides/structured-outputs — Structured Outputs guarantee schema validity, not content correctness.
- https://platform.openai.com/docs/guides/prompt-caching — Prompt caching caches the prefix KV; "otherwise identical nondeterministic requests are not guaranteed to return identical output" (verified confirmed).
- https://platform.openai.com/docs/api-reference/chat/create — `seed` (Beta, "best effort") + `system_fingerprint` field semantics (verified confirmed against openai-python source).
- https://platform.openai.com/docs/guides/text-generation — "the content generated from a model is non-deterministic."
- https://platform.openai.com/docs/deprecations — OpenAI deprecation history (model-pinning bounded window; specific gpt-5.5/5.4-mini names unverified).
- https://docs.anthropic.com/en/api/messages — Anthropic Messages API: no `seed`; "even with temperature of 0.0, the results will not be fully deterministic" (verified confirmed against anthropic-sdk-python source).
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching — Anthropic prompt caching framed purely as cost/latency (no determinism claim; verified).
- https://ai.google.dev/gemini-api/docs/caching — Gemini context caching; zero occurrences of "deterministic"/"identical."
- https://github.com/zilliztech/GPTCache — Semantic cache; "you may encounter false positives during cache hits"; exact-match is a strict subset (verified confirmed).
- https://pytorch.org/docs/stable/notes/randomness.html — `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG` (the third-path caveat for local open-weight models).

**Determinism — replay/event-sourcing engines (primary docs)**
- https://docs.temporal.io/workflows — "Activity runs once, its result is recorded... reused, not recomputed"; lists "LLM invocations" under Activities (verified confirmed, fetched directly).
- https://docs.temporal.io/develop/python/determinism — Workflow determinism constraints; Date.now()/random/network-as-non-Activity are replay hazards (verified confirmed).
- https://docs.dbos.dev/quickstart — Durable workflows recover "from where it left off"; resume-from-checkpoint semantics.
- https://docs.dbos.dev/ai/debugging.md — DBOS `fork`: restart from a completed step, deterministically reproducing state.
- https://docs.dbos.dev/explanations/portable-workflows.md — `portable_json` cross-SDK serialization (data portable, definition stays in-language).
- https://www.inngest.com/docs-markdown/learn/inngest-steps — Step memoization by id; non-deterministic side effects go inside `step.run()` (could not be re-fetched live; JS-rendered site).

**Production patterns / observability / eval**
- https://github.com/langchain-ai/langsmith-sdk/blob/main/python/README.md — "Create a Dataset from Existing Runs" (the capture primitive).
- https://langfuse.com/docs/evaluation/dataset-runs/datasets — Langfuse Datasets (test sets + pre-deployment testing).
- https://langfuse.com/docs/prompt-management/get-started — Prompt versioning as a pinned release artifact.
- https://github.com/open-telemetry/semantic-conventions-genai — OTel GenAI semantic conventions (spans/metrics/events, vendor-neutral audit substrate).
- https://github.com/instructor-ai/instructor — Pydantic-validated structured output with retry (FORMAT-contract guard).
- https://github.com/noamgat/lm-format-enforcer — Constrained decoding (format only).
- https://github.com/mlc-ai/xgrammar — Grammar-constrained decoding (format only).
- https://www.promptfoo.dev/ — Eval/regression-gate tooling (OpenAI cookbook-recommended).

**Claude Code dynamic workflows (verified)**
- https://code.claude.com/docs/en/workflows — "Orchestrate subagents at scale with dynamic workflows" (primary; JS script + `agent()`/`pipeline()` + `args` + constraints; verified confirmed verbatim).
- https://claude.com/blog/introducing-dynamic-workflows-in-claude-code — Launch blog; "tens to hundreds of parallel subagents"; GA announcement.

**Competitor workflow frameworks**
- https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/graph.py — `get_graph()` returns a `DrawableGraph` (structure export, viz-only, not replay).
- https://python.langchain.com/docs/concepts/persistence/ — LangGraph checkpointers + `get_state_history` (time-travel); LLM nodes re-execute non-deterministically by default.
- https://mastra.ai/docs/workflows/snapshots.md — Serializable complete execution state.
- https://mastra.ai/docs/workflows/time-travel.md — Re-execute from any step, reconstructing prior results from snapshot.
- https://docs.crewai.com/concepts/flows — Event-driven Python flows + checkpointing + replay.
- https://docs.n8n.io/workflows/export-import/ — Portable JSON workflows; includes credential names/IDs (strip before sharing); no replay determinism.
- https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html — Code-as-DAG; Python source is source of truth.
- https://openai.github.io/openai-agents-python/sessions/ — Session state (SQLite/Redis/SQLAlchemy); conversation persistence, not event-sourced replay.

**koboi codebase (internal; from digest)**
- `koboi/types.py:133-152` — `AgentDef` dataclass (9 fields, no serialize).
- `koboi/orchestration/workflow_graph.py:46-93` — `WorkflowGraph`/`CompiledGraph` private fields, no serialize.
- `koboi/orchestration/dag_scheduler.py:38-172` — `DagScheduler`; `persist_plan` write-only, `list_completed_nodes`.
- `koboi/config.py:86-112, 247, 250-254, 446-447, 669-684` — `FORWARDABLE_LLM_KEYS`, `raw`, lossy `to_dict`, orchestration property, `ConfigBuilder.orchestration`.
- `koboi/config_models.py:54-70` — `LLMConfig` `extra='ignore'` (root cause of lossy export).
- `koboi/facade.py:1580-1607, 1664-1674` — `_parse_agent_defs` (YAML→AgentDef only), `_agent_client_builder`.
- `koboi/llm/factory.py:30-58, 127-141` — `_PROVIDER_EXTRA_KEYS`, `_has_client_overrides`.
- `koboi/llm/http_transport.py:52` — `HttpTransport.post` (single body→response chokepoint for a replay layer).
- `koboi/client.py:54, 86, 127` — `RetryClient.temperature`, `complete(response_format=...)`.
- `koboi/loop.py:129, 479-537, 563-564, 616-635` — `response_schema`, `_repair_interrupted_turn`, `complete` call site, `resume`.
- `koboi/journal.py:66-150` — `record_step` (status/tokens/redacted args; no raw response/params).
- `koboi/memory_sqlite.py:17-55, 79-99` — `ensure_steps_table` (additive ALTER pattern), `research_context` table.
- `koboi/diagnostics.py:22-139` — `collect_diagnostics` ZIP + `_sanitize_config`.
- `koboi/cli_commands.py:103, 174-183, 234-273` — lossy `cmd_graph`, `--resume`, text-only `cmd_sessions`.
- `koboi/eval/t/mock.py:11-40` — `ScriptedClient` (eval-only replay double).
- `tests/conftest.py:16-50` — `MockClient` (test-only replay double).
- `koboi/orchestration/research.py:151-284` — `SourceStore`, `ResearchContext.to_json/from_json` (PR #37 export seam).
- `koboi/orchestration/orchestrator.py:210-222, 453-486, 830-1069, 1161-1164` — `_emit_research_hook`, `_eval_conditional`, `_run_deep_research`, deep_research dispatch.
- `koboi/websearch/providers/counting.py:21-46` — `CountingSearchProvider`/`CountingFetchProvider` (meter budget, do not record results).
- `koboi/context/manager.py:218, 337-367` — `smart_truncation` (deterministic), `sliding_window` (LLM-summarized, non-deterministic).
