# Vercel `eve` vs koboi-agent: Deep Dive & Gap Analysis

> Research date: 2026-06-23
> Source: https://github.com/vercel/eve (cloned) + https://vercel.com/eve
> eve repo: created 2026-06-16, ~2.2k⭐, Apache-2.0, TypeScript, single runtime dep (`nitro`), **beta / pre-1.0**
> All eve claims below are grounded in the repo's `docs/` and `packages/eve/src/` tree.

---

## 1. What `eve` actually is

`eve` is a **filesystem-first framework for *durable backend* AI agents that "run anywhere."** The defining sentence from its own docs: *"An eve session is a durable conversation. It can run for days and survives process restarts and redeploys without any work on your part."*

### The 7 defining ideas

| # | Idea | What it means in eve |
|---|------|----------------------|
| 1 | **Filesystem = authoring interface** | An agent is a directory on disk: `agent/instructions.md` (required, always-on system prompt), `agent/agent.ts` (model), `tools/`, `skills/`, `channels/`, `connections/`, `schedules/`, `subagents/`, `hooks/`. Names are **derived from file paths** (no `name:` fields). `npx eve init` scaffolds it. |
| 2 | **Durability by default** | Built on the open-source [Workflow SDK](https://workflow-sdk.dev/). A **turn is a durable workflow** that **checkpoints at each step** (one model call + its tool calls). Crash/redeploy/timeout → resume from last completed step; completed steps are **replayed, never re-run**. **Parked work** (human approval, interactive OAuth, long subagent) **suspends with zero compute** and resumes on the trigger. |
| 3 | **Two-context trust boundary** | **App runtime** (trusted: your code, `process.env`, secrets, model calls, durable state) **vs. Sandbox** (isolated `/workspace`, *no* `process.env`, *no* secrets, network controlled by policy). Only `bash`/`read_file`/`write_file`/`glob`/`grep` run in the sandbox, and even those live in the app runtime and **proxy** into it. The model never holds a credential. |
| 4 | **Exactly one sandbox, multiple backends** | Every agent gets one sandbox. Backends: `vercel()` (microVM), `docker()`, `microsandbox()` (VM), `justbash()` (pure-JS interpreter), `defaultBackend()` (best-available). Features: `bootstrap` (template-scoped setup, cached) + `onSession` (per-session, can read `ctx.session.auth`), network policy (`allow-all`/`deny-all`/allow-list+subnets), **credential brokering** (inject auth header at the network firewall so the secret stays out of the sandbox process). |
| 5 | **The harness** | Default agent loop + ~11 built-in tools, all **overridable** (author a file at the same slug, spread the default) or **disableable** (`disableTool()` sentinel). Compaction at `thresholdPercent` (0.9 default) **preserves framework tool state**: resets read-before-write tracking, re-injects the active todo list. |
| 6 | **Skills = load-on-demand markdown** (`load_skill`) | Agentskills.io standard (koboi already aligns here). Per-agent scoped. `ctx.getSkill(id)` lazily reads packaged sibling files from the sandbox. **`defineDynamic`** resolves per-session instructions/skills/tools from `ctx.session.auth` (tenant/team/plan/feature-flag). |
| 7 | **Evals, channels, connections, schedules are first-class files** | `defineEval` + integrated `t` driver/assert/judge; `channels/` (Slack/Discord/Telegram/Teams/Twilio/GitHub/Linear/HTTP) with fail-closed auth + constant-time signature compare; `connections/` (MCP + OpenAPI) with `connection_search` dynamic discovery + per-step token caching (never serialized to durable state); `schedules/` cron as files. Plus: docs ship **inside the package** (`node_modules/eve/docs`, version-matched) so coding agents read exact docs. |

### Coding principles worth noting (from `AGENTS.md`)
- **Wrap third-party deps**; aim for `nitro` as the *only* runtime dependency (vendor the rest). Avoids supply-chain exposure.
- **Derive names from file paths**; **name definitions for the protocol they target** (`defineMcpClientConnection`, not `defineConnection`).
- **Pre-1.0: prefer breaking changes** over backwards-compat — no legacy fallback.
- **Comment why, not what.** Test tiers: unit / integration / scenario (real subprocess/HTTP) / e2e (fixture-owned `eve eval`).

---

## 2. Architecture map: eve vs koboi-agent

| Capability | `eve` | koboi-agent | Gap for koboi |
|---|---|---|---|
| **Language / paradigm** | TypeScript, *filesystem-authored* agent dir, compiles to manifest | Python, **YAML-config-driven** | Philosophical (see §3.1) |
| **Durability / resume** | ✅ Workflow SDK, step checkpointing, parked work, survives restart/redeploy | ❌ None — `loop.py` is a plain async loop; SQLite only stores **conversation memory**, not workflow/turn state | **Critical** |
| **Sandbox / trust boundary** | ✅ Isolated `/workspace`, no secrets/env, network policy, credential brokering, 4 backends | ❌ `shell.py` runs `subprocess.run` **directly in the host with full `os.environ`**; relies on `PolicyHook` deny-patterns/sensitive-paths gating only | **Critical (security)** |
| **Built-in tool surface** | bash/read_file/write_file/glob/grep/web_fetch/web_search/todo/ask_question/agent/load_skill/connection_search; override/disable by file | 9 builtin tools (calculator/fs/shell/web/memory/search/git/subagent/task) via `@tool()` + `__init__.py` registration | Medium — no override/disable convention, no HITL `ask_question` |
| **Human-in-the-loop** | ✅ `needsApproval` (always/once/never/predicate) **durably pauses**; `ask_question` parks until answered | Partial — approval handlers + rate limiter in guardrails, but **not durable** (in-memory pause) | High |
| **Skills** | `load_skill`, per-agent, `ctx.getSkill`, dynamic via `defineDynamic` | ✅ agentskills.io-aligned, TF-IDF routing (EN+ID stopwords), `SkillPersistenceHook` re-injects after compaction, `disable_model_invocation`/`user_invocable`/`disallowed_tools` | Low — koboi is at parity / ahead on routing & ID support |
| **Eval** | `defineEval` + `t` (drive **and** assert inline), 3 surfaces (run-level / `t.check` / `t.judge` LLM-as-judge), **gate vs soft**, `eve eval --strict` CI exit code | `EvalRunner.run_case/run_suite`, scorers (bfcl/deepeval/gaia/ragas/swe_bench/skill), config/case-driven | High DX gap, but **koboi has richer benchmark scorers** |
| **Channels** | ✅ 8 platform channels + custom, fail-closed auth, constant-time sig compare | ❌ CLI / TUI / library only | High (for "backend agent" positioning) |
| **Connections / MCP** | MCP + OpenAPI, `connection_search` dynamic discovery, OAuth, per-step token cache (never persisted) | MCP client (stdio+HTTP) + server, but no dynamic cross-connection discovery / token-caching guarantee | Medium |
| **Schedules** | ✅ cron as files | ❌ (has `TaskManager`, not cron) | Medium |
| **Subagents** | Copy of agent, shares parent sandbox+tools, **fresh history + own durable session + own sandbox** | ✅ `SubAgentManager` for parallel delegation | Low-Medium (no own sandbox/durable session per subagent) |
| **Context control** | Compaction w/ tool-state preservation (read-before-write reset, todo re-inject); workspace-as-context (inspect via tools, not pasted) | 4 strategies (truncation/smart_truncation/key_facts/sliding_window) + RAG + SkillPersistenceHook | Medium (no read-before-write / todo re-inject on compact) |
| **Hooks** | Observe-only stream-event subscribers (`*` wildcard), fire **after durable record**, can't inject context | 19 hooks, `HookEvent` (15 events), **can modify context** (`pre_ctx`), priority bands | koboi hooks are **more powerful** (can inject); eve's are safer/cleaner |
| **RAG** | (not core; connections + sandbox workspace instead) | ✅ Full pipeline: chunker/retriever/augmentation, hybrid search, relevance gate | **koboi ahead** |
| **Multi-provider LLM** | Via AI SDK + provider strings (`anthropic/...`, `openai/...`) | ✅ OpenAI / Anthropic / Cloudflare, RetryClient, registry | koboi ahead (Cloudflare) |
| **Modes / trust graduation** | Sandbox policy + approval + auth | ✅ chat/plan/act/auto/**yolo** + TrustDatabase graduated perms | koboi ahead (richer mode model) |
| **Docs-in-package** | ✅ `node_modules/eve/docs` version-matched | ❌ docs only in repo | Low |
| **Orchestration** | (delegated to subagents + experimental `Workflow` tool — model-authored JS) | ✅ router (keyword/LLM/hybrid) + orchestrator + dynamic agent builder | koboi ahead on routing |

---

## 3. Gap analysis (grouped, with evidence)

### 3.1 CRITICAL — Durability (`eve`'s signature; koboi has none)
**Evidence (koboi):** `grep -niE "checkpoint|durable|resume|step.?boundary|workflow" koboi/loop.py koboi/loop_pipeline.py` → only a docstring hit. Memory backend (SQLite WAL) stores **conversation**, not **turn/step workflow state**.
**What koboi can't do today that eve can:** resume an agent turn after a crash/redeploy; durably pause for human approval / OAuth / a long subagent and resume later with zero compute; replay completed steps without re-running side effects.
**Why it matters:** A "configurable AI agent framework" aimed at backend agents that *can't survive a redeploy mid-turn or pause-for-approval durably* is limited to short-lived interactive use. This is the single biggest capability gap.

### 3.2 CRITICAL — Sandbox / trust boundary (security)
**Evidence (koboi):** `shell.py` calls `subprocess.run(...)` in the host process; `_build_env()` does `os.environ.copy()` and passes it through. The only guard is `PolicyHook`'s `COMMAND_DENY_PATTERNS` + `SENSITIVE_PATHS` (deny-listing), and **even YOLO mode bypasses approval/rate-limit** (only PolicyHook's hardcoded safety remains).
**What eve does instead:** shell/file tools **proxy into an isolated sandbox** with *no* `process.env`, *no* secrets, a `/workspace` fs, and a network policy; secrets reach hosts only via **credential brokering at the firewall** (sandbox process never sees them).
**Why it matters:** koboi's model can execute model-authored commands in the host environment with the **same env vars as the app** — i.e. an agent running `env` or a crafted command can exfiltrate `OPENAI_API_KEY`, DB URLs, etc. Deny-listing is a weak control vs. isolation. This is the highest-risk gap.

### 3.3 HIGH — Eval DX
**Evidence (koboi):** `EvalRunner.run_case(case: EvalCase) -> EvalResult`; rich scorer lib but authoring is **config/case-driven** (separate input/run/checks). **eve:** one `async test(t)` function that **both drives the agent and asserts inline** — `t.send(...)`, `t.completed()`, `t.calledTool("x")`, `t.check(t.reply, includes("..."))`, `t.judge.autoevals.closedQA("...")`, with **gate vs soft** severity and `eve eval --strict` → non-zero CI exit.
**Why it matters:** eve's model makes evals **feel like tests** (familiar, low-friction, CI-native). koboi has stronger *scorers* but a higher-friction *authoring/CI* story.

### 3.4 HIGH — Channels (delivery)
koboi is CLI/TUI/library. eve agents are addressable over 8 channels with platform-grade auth. For "backend agent" positioning, the delivery layer is missing.

### 3.5 MEDIUM — Connections dynamic discovery + token-caching guarantee
koboi has MCP but not: `connection_search` (discover tools across connections on demand → qualified callable names), nor the security invariant that **connection tokens are cached per step and never serialized to durable state**.

### 3.6 MEDIUM — Schedules, per-session durable `defineState`, compaction tool-state preservation, HITL `ask_question`, tool override/disable-by-file, docs-in-package, subagent-own-sandbox.

### 3.7 Philosophy — Filesystem-authoring vs YAML-config
eve: *author an agent as a directory, compile & run*. koboi: *configure an agent via YAML*. Both are valid; koboi's stance ("configurable framework") is a deliberate differentiator. **Don't adopt wholesale** — selectively steal the *convention-over-config* ergonomics (e.g. file-derived tool names, override/disable-by-file) without abandoning YAML.

### 3.8 Where koboi is AHEAD of eve (keep / double down)
- **Hooks that can inject/modify context** (`pre_ctx`) — eve hooks are observe-only. koboi's are strictly more powerful.
- **TF-IDF skill routing with Indonesian stopwords** — unique regional advantage; eve has no routing engine.
- **RAG pipeline** (chunker/retriever/hybrid/relevance-gate) — eve has none in core.
- **Multi-provider incl. Cloudflare** + RetryClient.
- **Mode model** (chat/plan/act/auto/yolo) + graduated `TrustDatabase`.
- **Orchestration router** (keyword/LLM/hybrid).
- **Benchmark scorer breadth** (bfcl/gaia/swe_bench/ragas/deepeval).

---

## 4. Recommendations (prioritized, opinionated)

Ranked by **impact × (1/effort)** and fit with koboi's identity. "Adopt" does not mean "copy TypeScript into Python."

### 🥇 P0 — Isolated execution surface + secret hygiene (security, medium effort)
The fastest path to closing the security gap without a full microVM:
1. Stop forwarding full `os.environ` to shell tools — pass an **explicit allow-list** (or empty) env.
2. Add a `Sandbox` **abstraction** in front of `shell.py`/`filesystem.py` with: isolated working dir, no secret env, output truncation/timeout (already partly there), and a pluggable backend (`subprocess`-restricted default → `docker` → optional microVM). Mirror eve's `SandboxBackend` adapter shape.
3. Add a **network policy** concept (allow-list/deny-all) for shell egress.
4. Even a deny-egress + no-env default would eliminate the worst exfiltration paths. Credential brokering is a later enhancement.
ROI: closes the highest-risk gap with the least architectural disruption.

### 🥈 P1 — Integrated eval authoring surface (`t` model) (high impact, medium effort, best ROI)
koboi already has the scorer engine. Wrap it in an eve-style authoring surface:
- New loader for `.eval.py` files exporting one `async test(t)`; `t.send/respond/calledTool/usedNoTools/completed`, `t.check(value, matcher)`, `t.judge.<scorer>` (reuse existing `deepeval`/`ragas` scorers), **gate vs soft** + `atLeast(threshold)`, and `--strict` → CI exit code.
- Keep `EvalCase`/config path as the non-code alternative. This is mostly a **new authoring + runner adapter**, not a rebuild.
ROI: makes evals test-shaped and CI-native; leverages koboi's existing scorer breadth (a real advantage over eve).

### 🥉 P2 — Durable execution (highest impact, highest effort; phase it)
The signature capability. Don't try to build a Workflow SDK from scratch in one shot:
- **Phase A (low effort):** **step journal** — serialize each turn's model+tool steps to the existing SQLite WAL backend; on agent start, offer `--resume <session>`. Replays completed steps, re-runs the interrupted one. This alone gives crash-recovery.
- **Phase B (medium):** **durable pause/resume** for approval & HITL — park a turn, persist the resume handle, resume on callback. Removes in-memory approval fragility.
- **Phase C (high):** full workflow backends. Consider standing on a Python durable-execution lib (Temporal / DBOS / Restate) rather than reimplementing replay semantics — but Phase A is 80% of the user-visible value at 20% of the cost.
ROI: transforms koboi from "interactive agent" into "long-lived backend agent" — the core of eve's pitch.

### P3 — Selective feature adoptions (medium impact, medium effort)
- **Channels:** add an HTTP + one platform channel (Slack) with fail-closed auth + constant-time signature compare. Unblocks "backend agent" delivery.
- **Connections:** add `connection_search`-style dynamic discovery across MCP/OpenAPI connections + a **per-step token cache, never persisted to memory/durable state** security invariant.
- **Compaction tool-state preservation:** reset read-before-write tracking + re-inject active todo list on compact (koboi already re-injects skills via `SkillPersistenceHook` — extend the same hook).
- **HITL `ask_question` tool** (park until user answers) + **`needsApproval` predicates** for tools.
- **Schedules** (cron) + **`defineState`** (durable per-session state primitive).
- **Tool override/disable-by-file** convention (even within the YAML world: `tools: { write_file: { override: ... } }`).

### P4 — Cheap wins (low effort)
- **Docs-ship-in-package** — bundle `docs/` into the wheel so coding agents read version-matched docs (koboi's own CLAUDE.md would benefit).
- **Steal eve's coding principles:** wrap third-party deps behind koboi-owned surfaces; "comment why not what"; test tiers (unit/integration/scenario/e2e).
- **`toModelOutput`** — let a tool project a summary to the model while full output goes to hooks/channels.

---

## 5. Bottom line

eve's three load-bearing ideas that koboi lacks, in priority order: **(1) durable execution**, **(2) an isolated sandbox trust boundary**, **(3) test-shaped, CI-native evals**. Of these, **#3 has the best ROI** (koboi already owns the scorer layer), **#2 is the most urgent** (active security exposure), and **#1 is the largest strategic lift** but defines what "backend agent" means.

koboi should **not** abandon YAML-config for filesystem-authoring, nor trade away its real advantages (context-mutating hooks, ID-localized routing, RAG, multi-provider, richer scorers, mode/trust model). The winning move is to **steal eve's durability, isolation, and eval DX while keeping koboi's configuration-first identity and its stronger hooks/RAG/scorers.**

---

## Sources
- `vercel/eve` repo (cloned): `README.md`, `AGENTS.md`, `SKILL.md`, `docs/concepts/{execution-model-and-durability,default-harness,context-control,security-model}`, `docs/{sandbox,skills,tools/overview}.mdx`, `docs/guides/hooks.md`, `docs/evals/overview.mdx`, `packages/eve/src/` tree (`compiler/`, `execution/durable-session-migrations/`, `harness/`, `evals/`, `runtime/`, `setup/`).
- https://vercel.com/eve (product page)
- koboi-agent: `koboi/loop.py`, `koboi/loop_pipeline.py`, `koboi/tools/builtin/shell.py`, `koboi/eval/`, `CLAUDE.md`, `docs/skills-architecture-research.md`
