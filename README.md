# koboi-agent

[![CI](https://github.com/hedypamungkas/koboi-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/hedypamungkas/koboi-agent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hedypamungkas/koboi-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/hedypamungkas/koboi-agent)
[![PyPI version](https://img.shields.io/pypi/v/koboi-agent)](https://pypi.org/project/koboi-agent/)
[![Python](https://img.shields.io/pypi/pyversions/koboi-agent)](https://pypi.org/project/koboi-agent/)
[![License: MIT](https://img.shields.io/pypi/l/koboi-agent)](https://github.com/hedypamungkas/koboi-agent/blob/main/LICENSE)
[![Docker](https://github.com/hedypamungkas/koboi-agent/actions/workflows/docker.yml/badge.svg)](https://github.com/hedypamungkas/koboi-agent/actions/workflows/docker.yml)

Configurable AI agent framework for **trustworthy unattended autonomy**. YAML-driven config, async Python 3.10+, multi-provider LLM (OpenAI, Anthropic, Cloudflare).

## Why koboi: durable, sandboxed, evaluable

koboi-agent's defensible position is the integration of five assets that are rare **at the library level** (no peer agent framework combines all five):

- **Crash/redeploy resume** — the SQLite `StepJournal` eagerly writes a `running` marker *before* each LLM call (WAL), so a SIGKILL/redeploy leaves a resumable state; `koboi run --resume <session>` rehydrates and continues, re-executing **only the missing tool calls**. Reproducible proof + wall-clock: `python benchmarks/crash_recovery/run.py`. (LangGraph markets "durable execution" only at the platform/LangSmith tier.)
- **Seccomp HARD network isolation without a container** — the restricted sandbox denies egress at the syscall layer (`connect`/`connectat`/`sendto`/`sendmsg`, inherited across `execve`) plus rlimits + PATH allowlist + secret-stripped env, on Linux + the `python3-seccomp` system package. No peer ships this without spinning up a container.
- **Self-hostable REST/SSE + autonomous-jobs server with a real security contract** — `koboi serve` exposes interactive SSE chat (human-in-the-loop approvals) + autonomous background jobs behind Bearer keys, per-session ownership, idempotency, and a graceful drain. The **C3 contract**: autonomous destructive jobs are *refused unless* `sandbox.backend='restricted'`, and approvals are deny-by-default without a Trust-DB rule.
- **CI-native agent evaluation you treat like code** — the eve-style `t` authoring DSL (`koboi eval-test`) drives an agent and asserts outcomes (`calledTool`/`toolWasBlocked`/`retrievedChunk`/`blocked`/`warned`/`activatedSkill`/`completed`) with mock determinism (no API key burned on commit) and gate/soft severity, routed through 12 built-in scorers.
- **Supply-chain-hardened Skills** — agentskills.io-aligned, 3-tier progressive disclosure, with a shell-injection deny-list on SKILL.md `!cmd` preprocessing (the "ClawHavoc" ~1,200-malicious-skills marketplace attack is a real, documented threat).

Try the HITL flow on a bare install — `python examples/hitl_client.py` (httpx-only; auto-resolves `pending_approval` events) against `koboi serve configs/hitl_demo.yaml`.

➡️ Full positioning & competitive analysis: [docs/trustworthy-unattended-autonomy.md](docs/trustworthy-unattended-autonomy.md)

## Features

- **Multi-provider LLM**: OpenAI, Anthropic, Cloudflare Workers AI
- **YAML-driven config** with `${ENV_VAR}` interpolation
- **Built-in tools**: calculator, filesystem, shell, web, memory, search, git, subagent, task
- **Hook lifecycle**: 15 event types for logging, guardrails, telemetry
- **RAG pipeline**: chunking (fixed/sentence/paragraph/semantic), retrieval (keyword/semantic/hybrid), augmentation
- **Guardrails**: input/output validation, rate limiting, approval workflows, policy engine
- **Multi-agent orchestration**: keyword/LLM/hybrid routing; sequential, parallel, DAG, conditional, and dynamic (LLM-planned) execution
- **Context management**: truncation, smart truncation, key facts, sliding window
- **Sandboxed execution**: pluggable passthrough/restricted backends (per-session workdir, network/rlimit isolation)
- **MCP** client (stdio + HTTP) and server support
- **HTTP/SSE server & jobs**: `koboi serve` — interactive SSE chat (HITL) + autonomous background jobs; API keys, ownership, idempotency, durable resume
- **Evaluation**: BFCL, GAIA, SWE-bench, RAGAS, DeepEval scorers
- **Terminal UI** (Textual): chat, command palette, diff view, session management

## Quickstart

### Install

```bash
pip install koboi-agent            # bare install: --help, validate, run, sessions, keys, eval, eval-test, graph, diagnostics, init-zsh
# Extras (optional):
#   pip install koboi-agent[tui]   # interactive `koboi chat` (Textual TUI)
#   pip install koboi-agent[api]   # `koboi serve` (HTTP/SSE server; `koboi keys` works on bare install)
#   pip install koboi-agent[dev,tui,api]  # everything (contributors)
```

### Set your API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### Run the CLI

Most commands work on a bare install (no extras needed):

```bash
koboi validate configs/simple_chat.yaml     # check a config without running the agent
koboi run configs/simple_chat.yaml -m "What is 2 + 2?"     # one-shot query (plain output)
koboi run configs/simple_chat.yaml --print  # streaming JSON lines (pipe-friendly)
koboi keys create                           # mint an API key (for `koboi serve`)
koboi graph configs/dag_demo.yaml           # render an orchestration DAG (Mermaid; --format json)
```

Interactive chat needs the `[tui]` extra:

```bash
pip install koboi-agent[tui]
koboi chat configs/simple_chat.yaml         # Textual TUI; or `--print` for JSON lines (no extra)
```

### Run programmatically

```python
import asyncio
from koboi import KoboiAgent

async def main():
    async with KoboiAgent.from_config("configs/simple_chat.yaml") as agent:
        result = await agent.run("What is 2 + 2?")
        print(result.content)

asyncio.run(main())
```

## Serving (HTTP/SSE)

Run koboi as a stateless HTTP service: **interactive SSE chat** (with human-in-the-loop
approvals) and **autonomous background jobs** (durable resume). Requires the `[api]` extra.

```bash
pip install -e ".[api]"
koboi keys create                               # mint an API key (Bearer auth)
koboi serve configs/server_deploy.yaml --host 0.0.0.0 --port 8080
```

Then:

```bash
# interactive SSE chat (stream tokens + HITL approvals)
curl -N -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"message":"What is 2+2?"}' http://localhost:8080/v1/chat/stream

# autonomous job (202 + poll / SSE replay)
curl -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"message":"Summarize the Q3 report"}' http://localhost:8080/v1/jobs
```

Two paths, same composition: **`koboi serve <config>`** (built-in) or
**`create_app(config, extra_tools=..., extra_hooks=..., approval_handler=...)`**
(customize by code). See `configs/server_simple.yaml` / `configs/server_deploy.yaml`,
`koboi/server/CLAUDE.md`, and `docs/rest-sse-requirements.md`. Self-host deploy via the
bundled `Dockerfile` + `docker-compose.yml` (Cloudflare Tunnel).

### Container customization (3 tiers)

The published image (`ghcr.io/hedypamungkas/koboi-agent:<version>`) is a **base layer** — all three customization paths work without rebuilding koboi:

- **Mount a YAML config** — `docker run -e KOBOI_CONFIG=/app/agent.yaml -v agent.yaml:/app/agent.yaml …` (built-in path, zero code).
- **Mount an extensions dir** — `docker run -e KOBOI_EXTENSIONS_DIR=/app/ext -v ext/:/app/ext …` (custom tools / RAG retrievers via `tools.custom` / `rag.custom_modules`; the dir is auto-added to `sys.path`).
- **Derive a new image** — `FROM ghcr.io/hedypamungkas/koboi-agent:<version>` for full `create_app(extra_tools=…, extra_routes=…)` composition.

See [`examples/docker/`](examples/docker) for runnable, LLM-free proofs of each tier.

## Configuration

Agents are configured via YAML. Key sections:

```yaml
agent:
  name: "my-agent"
  system_prompt: "You are helpful."
  max_iterations: 10
  mode: "chat"  # chat | plan | act | auto | yolo

llm:
  provider: "openai"        # openai | anthropic | cloudflare
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
  base_url: "${OPENAI_BASE_URL:}"

tools:
  builtin: [calculator, web_search, memory_store, memory_recall]
  custom:
    - module: "my_tools"

context:
  strategy: "sliding_window"  # noop | truncation | smart_truncation | key_facts | sliding_window
  max_context_tokens: 8000

rag:
  enabled: true
  chunker: "paragraph"       # fixed | sentence | paragraph | semantic
  retriever: "keyword"       # keyword | semantic | hybrid
  top_k: 10
  documents:
    - path: "./data/sample/product_catalog.md"

guardrails:
  input:
    max_length: 10000
  rate_limit:
    max_calls_per_minute: 20

harness:
  doom_loop:
    consecutive_identical_threshold: 3
  telemetry: true
  carryover: true
```

See `configs/` for full examples and `.claude/skills/yaml-config.md` for the complete schema.

## Testing

```bash
pytest                        # all tests
pytest tests/test_config.py   # single file
pytest -k "hook"              # by keyword
pytest --cov=koboi            # with coverage
```

## Examples

`examples/` contains 32 numbered scripts covering every feature, plus `server_built_in.py` / `server_customize.py` (HTTP serving), `hitl_client.py` (HITL client), and workflow-graph demos (`workflow_graph_demo.py`, `dynamic_workflow_live.py`, `phase3_live_e2e.py`):

| Range | Features |
|-------|----------|
| 01-04 | Basic chat and tool use |
| 05-08 | Context management, RAG, and guardrails |
| 09-10 | MCP client/server |
| 11-14 | Policy, hooks, skills, custom tools |
| 15-16 | Multi-agent orchestration |
| 17 | Anthropic provider |
| 18-20 | Harness (telemetry, doom loop, carryover) |
| 21-24 | Evaluation, production setup, SWE-bench, config-driven orchestration |
| 25-28 | Subagent delegation, task management, benchmarks, custom RAG |
| 29-32 | Skills (enhanced), eval-test, tool selection, sandbox + resume |
| server_* | `koboi serve` (built-in) and `create_app()` (customize) |
| hitl_client / workflow_graph_demo / dynamic_workflow_live / phase3_live_e2e | HITL client + DAG/workflow-graph demos |

Examples use `click` + `rich` (in the `[tui]` extra), so install that first:

```bash
pip install -e ".[tui]"                        # examples need click + rich
python examples/01_simple_chat.py              # automatic mode
python examples/01_simple_chat.py -m interactive  # interactive mode
# Server examples need [api]: pip install -e ".[api]"
# Bare-install-safe (no extras): 27, 29, 31, 32, hitl_client.py
```

## Architecture

For a detailed architecture overview (agent loop lifecycle, hook system, tool pipeline, extension points), see **[docs/architecture.md](docs/architecture.md)**.

`KoboiAgent` (`facade.py`) is the single entry point. It assembles:

- **AgentCore** (`loop.py`) -- async agent loop
- **RetryClient** (`client.py`) -- LLM HTTP transport with retry
- **ToolRegistry** (`tools/`) -- tool registration and execution
- **HookChain** (`hooks/`) -- lifecycle event dispatch (15 events)
- **ContextManager** (`context/`) -- context window strategies
- **AugmentationStrategy** (`rag/`) -- RAG pipeline
- **Guardrails** (`guardrails/`) -- input/output validation
- **PolicyEngine** (`harness/`) -- rule-based tool filtering
- **SkillRegistry** (`skills/`) -- skill discovery
- **ModeManager** (`modes.py`) -- chat/plan/act/auto/yolo modes
- **TrustDatabase** (`trust.py`) -- graduated permissions
- **Sandbox** (`sandbox/`) -- passthrough/restricted execution backends (per-session workdir, network/rlimit isolation)
- **StepJournal** (`journal.py`) -- per-iteration step journal for crash/redeploy resume
- **Server** (`server/`) -- FastAPI HTTP/SSE serving (interactive chat + autonomous jobs)
- **Orchestrator** (`orchestration/`) -- multi-agent coordination
- **SubAgentManager** (`subagent.py`) -- parallel sub-agent delegation
- **MCP clients** (`mcp/`) -- external tool servers

All subsystems are configured from a single YAML file via `Config` (`config.py`).

## License

MIT
