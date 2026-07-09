# CLAUDE.md -- koboi-agent

## What this is
Configurable AI agent framework. YAML-driven config, async Python 3.10+, multi-provider LLM (OpenAI, Anthropic, Cloudflare).

## Quick commands
- Install:     `pip install -e ".[dev,tui]"`
- Test all:    `pytest`
- Test single: `pytest tests/test_config.py -k "test_from_yaml"`
- Test by tag: `pytest -k "hook"`
- Coverage:    `pytest --cov=koboi --cov-report=term-missing`
- Run CLI:     `koboi chat configs/simple_chat.yaml`
- Run example: `python examples/01_simple_chat.py`
- Serve HTTP:  `koboi serve configs/server_deploy.yaml`  (needs `[api]` extra: `pip install -e ".[api]"`)
- API keys:    `koboi keys create`                       (Bearer auth; `koboi keys list|revoke|rotate`)
- Resume:      `koboi run --resume <session>`            (`koboi sessions <config>` lists persisted sessions)
- Eval (eve):  `koboi eval-test evals/ --mock --strict`
- Benchmarks:  `pytest tests/benchmarks/ -o python_files="bench_*.py" --benchmark-only`  (perf regression gate; see docs/performance-benchmarking.md)

## Directory map
```
koboi/              Main package (~188 .py files)
  config.py         Config + ConfigBuilder -- YAML loading, ${VAR:default} interpolation
  config_models.py  Pydantic v2 schema validation for config
  facade.py         KoboiAgent -- single entry point, assembles all subsystems
  cli.py            Console-script entry (`koboi`): argparse dispatcher routing serve/keys/validate/run/chat/sessions/eval/eval-test/diagnostics/init-zsh; bare-install works for all no-TUI commands (bodies in cli_commands.py; interactive `chat` lazy-imports tui.app)
  cli_commands.py   Core (no-extra) command bodies for validate/run/chat-print/sessions/eval/eval-test/diagnostics/init-zsh -- stdlib print() output, returns exit codes
  loop.py           AgentCore -- async agent loop, hook integration
  loop_pipeline.py  ToolExecutionPipeline -- 8-step tool execution flow
  client.py         RetryClient -- LLM HTTP transport with exponential backoff
  events.py         StreamEvent union type for streaming
  types.py          All dataclasses: RunResult, ToolDefinition, EvalCase, etc.
  exceptions.py     AgentError hierarchy (LLMError hierarchy lives in llm/base.py)
  memory.py         In-memory ConversationMemory + MemoryBackend protocol
  memory_sqlite.py  SQLite-backed memory backend (WAL mode); also hosts the `steps` journal table
  journal.py        StepJournal -- per-iteration step journal for crash/redeploy resume (P2-A)
  tokens.py         Token estimation helpers
  modes.py          AgentMode enum (chat/plan/act/auto/yolo), ModeManager
  trust.py          TrustDatabase for graduated permissions
  logger.py         AgentLogger
  plugins.py        Entry-point plugin discovery (koboi.providers, koboi.tools, etc.)
  subagent.py       SubAgentManager for parallel sub-agent delegation
  task.py           TaskManager for structured task tracking
  diagnostics.py    Session diagnostic export
  notifications.py  Notification system
  llm/              LLM providers: base ABC, OpenAI adapter, Anthropic adapter, factory, auth, registry, http_transport, pool (ProviderPool/failover), resolve (named-providers resolver)
  tools/            Tool registry + builtin/ (calculator, filesystem, shell, web, memory, search, git, subagent, task)
  hooks/            Hook system: chain.py (HookEvent enum, Hook ABC, HookChain) + registry.py + 19 specialized hooks
  context/          Context window strategies: truncation, smart_truncation, key_facts, sliding_window
  rag/              RAG pipeline: chunker (fixed/sentence/paragraph/semantic), retriever (keyword/semantic/hybrid), augmentation, registry
  guardrails/       Input/output guardrails, rate limiter, audit trail, approval handlers, registry
  harness/          Telemetry, carryover state, doom loop detection, policy engine, env hygiene (env.py)
  sandbox/          Pluggable subprocess/fs isolation backends (passthrough default, restricted); reuses ComponentRegistry
  server/           FastAPI HTTP/SSE serving layer: app, jobs, pool, auth, ownership, idempotency, approvals, keys_cli, schema, sse, health, middleware, protocols
  orchestration/    Multi-agent: router (keyword/LLM/hybrid), orchestrator, factory, dynamic agent builder
  mcp/              MCP client (stdio + HTTP) and server
  skills/           Skill discovery and registry (agentskills.io standard) with budget, invocation control, dynamic context
  eval/             Evaluation: runner, config, registry, regression, loaders/, scorers/, t/
  tui/              Terminal UI (Textual): app, screens/ (9), widgets/ (12)
tests/              ~170 test files, asyncio_mode="auto", shared conftest.py with MockClient
configs/            21 YAML agent configs
examples/           32 numbered example scripts (01-32) + server_built_in/server_customize, with matching YAMLs
evals/              Sample eve-style `t` eval files (*.eval.py) -- run via `koboi eval-test`
skills/             4 skill definitions: code_review, customer_service, hotel_receptionist, search_and_summarize
mcp_servers/        1 MCP server example: todo_server.py
data/               Sample documents for RAG demos (Acme Corp)
benchmarks/         BFCL benchmark data (DO NOT read benchmarks/results.json -- 183MB)
docs/               Architecture overview, REST/SSE requirements, performance benchmarking, skills/eve research, strategy audits
```

## Code conventions
- Module docstrings: `"""koboi/path -- short description."""`
- Import order: stdlib -> third-party -> koboi.*, blank lines between groups
- TYPE_CHECKING guard used where circular imports exist (e.g., loop.py imports logger, context, rag types)
- Type hints: `list[dict] | None` (modern union syntax, not `Optional[List[dict]]`)
- Private helpers: `_` prefix (e.g., `_safe_eval`, `_walk_resolve`)
- Hook pattern: subclass `Hook` ABC, implement `handles() -> list[HookEvent]` and `async execute(ctx: HookContext) -> HookContext`
- Tool registration: `@tool()` decorator with inline JSON Schema `parameters` dict
- Config env vars: `${VAR}` and `${VAR:default}` interpolation in YAML strings
- Tests: pytest with `asyncio_mode="auto"`, `MockClient` in conftest.py, fixtures: `mock_client`, `tool_registry`, `memory`, `simple_config`

## Gotchas
- `benchmarks/results.json` is 183MB -- never read it
- `koboi_memory.db` is SQLite WAL-mode, 3 files (.db, .db-shm, .db-wal); it also holds a `steps` table (P2-A journal, additive via `CREATE TABLE IF NOT EXISTS`)
- `sandbox:` YAML section drives `koboi/sandbox/`; default `passthrough` preserves pre-P0b behavior, opt into `restricted` for cwd/env/PATH/network/rlimit isolation. `restricted` network is SOFT by default (token-scan blocks curl/wget/nc but NOT interpreters like `python3 -c 'import urllib'` or `bash /dev/tcp`); set `sandbox.network_isolation: seccomp` for HARD syscall-layer egress deny (Linux + `python3-seccomp` system package; `_HAS_SECCOMP` gate; falls back to soft with a warning). `server_deploy.yaml`/`e2e_full.yaml` enable seccomp by default. `KOBOI_SANDBOX_DIR` is still honored as a back-compat fallback. Subprocess tools (`run_shell`, `git_*`, filesystem) declare `deps=["sandbox"]` and read `_deps["sandbox"]`; the facade always wires a (passthrough-or-better) sandbox
- `journal:` YAML section (default `enabled: true`) drives `koboi/journal.py`; auto-disabled when `memory.backend != sqlite` (it borrows the SQLite connection). Loop writes are native (not hooks) so durability can't be bypassed
- `koboi run --resume <session>` rehydrates-and-continues an interrupted session; `koboi sessions <config>` lists persisted sessions. Resume is unsupported in orchestration mode (v1)
- `.agent_memory.json` is a runtime artifact at project root
- `Config.from_yaml()` raises `FileNotFoundError`, not a generic error
- `AgentCore` is in `loop.py`, not `core.py`
- `HookEvent` values are snake_case strings: `"pre_tool_use"`, `"post_llm_call"`, etc. (15 total)
- `SkillDefinition` new fields: `disable_model_invocation`, `user_invocable`, `disallowed_tools`
- `build_discovery_prompt()` accepts `budget_chars` param; `SkillRegistry` defaults to 8000
- `activate_skill()` preprocesses `` !`command` `` blocks (shell injection); set `run_shell=False` to disable
- `SkillPersistenceHook` (priority 45) re-injects activated skills after POST_COMPACT
- Skill scorers: only `skill_trigger_accuracy` in eval/ (`skill_routing_accuracy` + `skill_token_overhead` were removed)
- The `facade.py` `_build_*` functions are module-level, not class methods
- TUI entry point: `koboi.tui.app:main` (setuptools script)
- Tool functions can be sync or async -- sync runs in thread via `asyncio.to_thread`
- Tool return type must be `str` -- the registry calls `str(result)`
- `HookChain` hooks sorted by priority: 0-19 infra, 20-39 security, 40-59 business, 60-79 post, 80-100 cleanup
- RAG/Context/Guardrail components use registry pattern with `@register_*` decorators
- Plugin entry points: `koboi.providers`, `koboi.tools`, `koboi.guardrails`, `koboi.scorers`
- YOLO mode (`/mode yolo`) bypasses rate limiting, approval, and mode blocks -- but PolicyHook's hardcoded safety (sensitive paths, dangerous commands) is always enforced via `pre_ctx.abort` check in pipeline
- `server:` + `jobs:` + `sandbox:` + `journal:` YAML sections drive `koboi/server/`, `koboi/sandbox/`, `koboi/journal.py`. `koboi serve` needs the `[api]` extra (`fastapi`, `uvicorn`); `koboi keys` mints/rotates Bearer keys (file: `~/.koboi/keys.json`, hashed). Config is read via dotted-path `config.get("server", ...)` (Pydantic `ServerConfig`/`JobsConfig` are cosmetic validation only)
- AgentCore is NOT concurrent-safe → the server wraps each session in its own `asyncio.Lock` (per-session, not per-core); `pool.session_lock` is the seam to install per-run state under
- Per-request mode/iteration knobs (G2): `/v1/chat/stream` + `/v1/jobs` bodies accept `mode` + `max_iterations`; validated against `server.allowed_modes` (default = chat/plan/act/auto; **yolo is opt-in**), **jobs always reject yolo**, clamped to `server.limits.max_iterations_cap` (default 25). Stamped per-request under the session lock, restored in `finally`
- Output guardrail (`loop.py` `_process_output`) honors `GuardrailResult.action`: `block`/`deny`/`abort` raises `AgentGuardrailError` (denies); `warn`/absent prepends a warning. When any output guardrail is configured, `run_stream` **buffers TextDeltas** until the check passes (G8) — otherwise blocked tokens would stream before the guardrail runs
- Graceful drain (`_shutdown`): cancels in-flight interactive stream tasks, flushes Langfuse **off-loop + concurrently** (`asyncio.to_thread`+`gather`; `agent.close()` does NOT flush), then closes jobs/pool/store, under `wait_for(server.timeouts.drain_seconds)`
- `Idempotency-Key` header: 409-reject on `/v1/chat/stream` (same owner+session+key within `server.idempotency.chat_ttl_seconds`); dedup on `/v1/jobs` (same key → same job_id, replay-friendly)
- `AutonomousApprovalHandler` (jobs) is deny-by-default on destructive tools without a Trust-DB rule; autonomous jobs additionally **require `sandbox.backend='restricted'`** (passthrough refused at execution, C3)
