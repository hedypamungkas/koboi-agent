# Trustworthy Unattended Autonomy — koboi-agent positioning

> **The wedge:** *durability × sandbox isolation × deny-by-default approval × CI-native eval.*
> No peer agent framework combines all four at the library level.

koboi-agent is a YAML-driven, async-Python, multi-provider AI agent framework. Its
defensible position is **not** feature breadth (RAG is in-process, MCP auth is
static-Bearer or OAuth2 client-credentials only) — it is the integration of capabilities that make an agent
safe to leave running unattended, shipped together at the framework core.

## The five moats

1. **Crash/redeploy resume.** `StepJournal` (`koboi/journal.py`) eagerly writes a
   `running` step marker to SQLite (WAL) *before* each LLM call, so a SIGKILL or
   redeploy leaves a state `koboi run --resume <session>` can rehydrate and
   continue — re-executing only the tool calls whose results were never persisted
   (`AgentCore.resume` + `_repair_interrupted_turn` in `koboi/loop.py`). Writes
   are loop-native (not hooks), so durability cannot be bypassed.
   **Proof:** `python benchmarks/crash_recovery/run.py` (correctness rate + wall-clock).
   LangGraph markets "durable execution" only at the platform/LangSmith tier.

2. **Seccomp HARD network isolation without a container** (`koboi/sandbox/`).
   `preexec_fn` builds and loads the syscall filter between `fork` and `exec`, so
   it persists across `execve` — `python3`/`bash`/`curl` all inherit the deny list
   on `connect`/`connectat`/`sendto`/`sendmsg`. Combined with rlimits
   (`cpu`/`as_mb`/`fsize_mb`/`nofile`), a symlink-safe `validate_path`, a PATH
   allowlist, and a secret-stripped `build_safe_env`. Achieved on Linux + the
   `python3-seccomp` system package — no container per run. (See
   `koboi/sandbox/CLAUDE.md`; the CI `seccomp:` job proves HARD egress deny.)

3. **Self-hostable server with the C3 contract** (`koboi/server/`). FastAPI app
   exposing `/v1/chat/stream` (interactive SSE + HITL) and `/v1/jobs` (autonomous
   background jobs), behind Bearer keys, per-session ownership, idempotency keys,
   and a graceful drain. **C3:** autonomous destructive jobs are *refused at
   execution unless* `sandbox.backend='restricted'`, and the
   `AutonomousApprovalHandler` is deny-by-default on destructive tools without a
   Trust-DB rule. Most agent frameworks are libraries, not servers.

4. **CI-native evaluation as code** (`koboi/eval/t/`). The eve-style `t` authoring
   DSL drives an agent and records outcome-aware assertions
   (`calledTool`/`toolWasBlocked`/`retrievedChunk`/`blocked`/`warned`/
   `activatedSkill`/`completed`) with gate/soft severity, mock-deterministic via
   `ScriptedClient` (no API key on every commit), routed through 12 built-in
   scorers. Run: `koboi eval-test evals/ --mock --strict`.

5. **Supply-chain-hardened Skills** (`koboi/skills/`). agentskills.io-aligned,
   3-tier progressive disclosure with budget-aware discovery, plus an H3
   shell-injection deny-list on SKILL.md `` !`command` `` preprocessing and
   `disable_model_invocation`/`user_invocable`/`disallowed_tools` fields. The
   "ClawHavoc" case (~1,200 malicious skills infiltrating a marketplace) is a real,
   documented attack surface — koboi's deny-list is on-trend, not over-engineered.

## What koboi is (and isn't)

koboi is an **autonomous-loop** framework (AutoGPT / Claude-Code family): the LLM
decides each step inside `AgentCore._run_loop`. It is *not* a workflow-graph
engine (LangGraph / CrewAI Flows / Mastra) — multi-agent coordination is routing +
fan-out, not a DAG. Closing the workflow-graph half (a durable graph runtime on
top of this loop, seeded from the dormant `TaskManager` dependency graph) is the
next phase. See `docs/architecture.md` for the subsystem map.

## Competitive landscape (point-in-time, 2026-07)

| Framework | What it owns | Where koboi's wedge still wins |
|---|---|---|
| OpenAI Agents SDK | Sandbox Agents (container) | No library-level crash/resume |
| LangGraph | Durable execution (platform tier) | No CI-native eval DSL; no library-level resume |
| CrewAI | Crews + Flows dual-pillar | Memory/durability not library-level crash-resume |
| Mastra | TS graph engine + HITL | Different language segment (TypeScript) |
| Claude Code | Agentic coding product | A product, not a self-hostable library |

koboi's edge is the **integration** of durability + eval + isolation + serving +
skills at the framework core, with Python 3.10+ multi-provider breadth.

## Honest limitations

- **Single-node.** Hot state is in-process (`pool.py`, `jobs.py`, `idempotency.py`);
  the `protocols.py` seams exist for a future Redis/Postgres swap. Multi-node HA is
  a scale claim, not part of the unattended-autonomy wedge.
- **RAG is in-process** (no vector DB; fs-only document source). Adequate for the
  autonomy wedge; a production-RAG tier is a separate track.
- **MCP auth is static-Bearer or OAuth2 client-credentials** (token refresh + 401
  recovery in `koboi/mcp/auth.py`); authorization-code / user-delegated flows
  aren't supported yet, so remote MCP servers requiring interactive user consent
  still need a manually minted token.

*Star counts and competitor version numbers are point-in-time snapshots; GitHub
restricted the stargazers API on 2026-06-30, so growth-rate comparisons are
currently impaired.*
