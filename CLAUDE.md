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
- Resume:      `koboi run <config> --resume <session>`    (`koboi sessions <config>` lists persisted sessions; `--delete <id>` deletes one)
- MCP serve:   `koboi mcp-serve configs/simple_chat.yaml` (expose this agent's tools as a stdio MCP server; SAFE-only by default, `--allow NAME`/`--allow-all` to escalate)
- Eval (eve):  `koboi eval-test evals/ --mock --strict`
- Benchmarks:  `pytest tests/benchmarks/ -o python_files="bench_*.py" --benchmark-only`  (perf regression gate; see docs/performance-benchmarking.md)

## Directory map
```
koboi/              Main package (194 .py files)
  config.py         Config + ConfigBuilder -- YAML loading, ${VAR:default} interpolation
  config_models.py  Pydantic v2 schema validation for config
  facade.py         KoboiAgent -- single entry point, assembles all subsystems
  cli.py            Console-script entry (`koboi`): argparse dispatcher routing serve/keys/validate/run/chat/sessions/eval/eval-test/graph/diagnostics/init-zsh; bare-install works for all no-TUI commands (bodies in cli_commands.py; interactive `chat` lazy-imports tui.app)
  cli_commands.py   Core (no-extra) command bodies for validate/run/chat-print/sessions/eval/eval-test/graph/diagnostics/init-zsh -- stdlib print() output, returns exit codes
  loop.py           AgentCore -- async agent loop, hook integration
  loop_pipeline.py  ToolExecutionPipeline -- 8-step tool execution flow
  client.py         RetryClient -- LLM HTTP transport with exponential backoff
  events.py         StreamEvent union type for streaming
  types.py          All dataclasses: RunResult, ToolDefinition, EvalCase, etc.
  exceptions.py     AgentError hierarchy (LLMError hierarchy lives in llm/base.py)
  memory.py         In-memory ConversationMemory + MemoryBackend protocol
  memory_sqlite.py  SQLite-backed memory backend (WAL mode); also hosts the `steps` journal table
  journal.py        StepJournal -- per-iteration step journal for crash/redeploy resume (P2-A)
  proactive_memory.py  ProactiveMemory -- opt-in proactive long-term memory (auto-extract D + semantic recall C + core-memory block B); `memory.proactive` config
  redact.py         Shared secret redaction (value-shape + key-name masking) used by journal/jobs/diagnostics
  tokens.py         Token estimation helpers; optional tiktoken BPE via `[tokenizer]` extra (chars/3 fallback)
  modes.py          AgentMode enum (chat/plan/act/auto/yolo), ModeManager
  trust.py          TrustDatabase for graduated permissions
  logger.py         AgentLogger
  plugins.py        Entry-point plugin discovery (koboi.providers, koboi.tools, etc.)
  subagent.py       SubAgentManager for parallel sub-agent delegation
  task.py           TaskManager for structured task tracking
  diagnostics.py    Session diagnostic export
  notifications.py  Notification system
  _extensions_path.py  Adds `KOBOI_EXTENSIONS_DIR` to `sys.path` (container "mount an extensions dir" tier -- see README Container customization)
  llm/              LLM providers: base ABC, OpenAI adapter, Anthropic adapter, factory, auth, registry, http_transport, pool (ProviderPool/failover), resolve (named-providers resolver)
  tools/            Tool registry + builtin/ (calculator, filesystem, shell, web, memory, search, git, subagent, task)
  hooks/            Hook system: chain.py (HookEvent enum, Hook ABC, HookChain) + registry.py + 20 specialized hooks
  context/          Context window strategies: truncation, smart_truncation, key_facts, sliding_window
  rag/              RAG pipeline: chunker (fixed/sentence/paragraph/semantic), retriever (keyword/semantic/hybrid), augmentation, registry
  guardrails/       Input/output guardrails, rate limiter, audit trail, approval handlers, registry
  harness/          Telemetry, carryover state, doom loop detection, policy engine, env hygiene (env.py)
  sandbox/          Pluggable subprocess/fs isolation backends (passthrough default, restricted); reuses ComponentRegistry
  server/           FastAPI HTTP/SSE serving layer: app, jobs, pool, auth, ownership, idempotency, approvals, keys_cli, schema, sse, health, middleware, protocols
  orchestration/    Multi-agent: router (keyword/LLM/hybrid), orchestrator (sequential/parallel/dag/conditional/dynamic), factory, dynamic agent builder, dag_scheduler (wave-parallel DAG), planner (LLM plan-or-skip), workflow_graph (programmatic builder)
  mcp/              MCP client (stdio + HTTP) and server
  skills/           Skill discovery and registry (agentskills.io standard) with budget, invocation control, dynamic context
  eval/             Evaluation: runner, config, registry, regression, loaders/, scorers/, t/
  tui/              Terminal UI (Textual): app, screens/ (9), widgets/ (12)
tests/              ~224 test files, asyncio_mode="auto", shared conftest.py with MockClient
configs/            28 YAML agent configs
examples/           33 numbered example scripts (01-33) + server_built_in/server_customize, hitl_client, a command-hook forwarder (_command_hook_forwarder), and workflow demos (dynamic_workflow_live, phase3_live_e2e, workflow_graph_demo); matching YAMLs
evals/              Sample eve-style `t` eval files (*.eval.py) -- run via `koboi eval-test`
skills/             4 skill definitions: code_review, customer_service, hotel_receptionist, search_and_summarize
mcp_servers/        1 MCP server example: todo_server.py
data/               Sample documents for RAG demos (Acme Corp)
benchmarks/         BFCL benchmark data (DO NOT read benchmarks/results.json -- 183MB)
docs/               Architecture overview, REST/SSE requirements, performance benchmarking, custom command-hooks guide, trustworthy-unattended-autonomy positioning, one-pager, skills/eve research, strategy audits
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
- MCP exposure/management: `koboi mcp-serve <config>` exposes the agent's tools as a stdio MCP server; **SAFE-only by default** (the bridge calls `ToolRegistry.execute()` which bypasses the risk/approval/audit pipeline), `--allow NAME` adds a MODERATE tool, `--allow-all` is the only way to expose DESTRUCTIVE. The server layer has runtime MCP management at `POST/GET/DELETE /v1/sessions/{id}/mcp/servers` + `POST /v1/sessions/{id}/mcp/servers/{id}/reconnect` (in-process, session-scoped, not persisted across restart/eviction). `BaseMCPClient.is_connected()/transport/tool_names/endpoint/name` + `KoboiAgent.mcp_status()` feed the TUI **f2** MCP-status screen.
- `koboi_memory.db` is SQLite WAL-mode, 3 files (.db, .db-shm, .db-wal); it also holds a `steps` table (P2-A journal, additive via `CREATE TABLE IF NOT EXISTS`)
- `sandbox:` YAML section drives `koboi/sandbox/`; default `passthrough` preserves pre-P0b behavior, opt into `restricted` for cwd/env/PATH/network/rlimit isolation. `restricted` network is SOFT by default (token-scan blocks curl/wget/nc but NOT interpreters like `python3 -c 'import urllib'` or `bash /dev/tcp`); set `sandbox.network_isolation: seccomp` for HARD syscall-layer egress deny (Linux + `python3-seccomp` system package; `_HAS_SECCOMP` gate; falls back to soft with a warning). `server_deploy.yaml`/`e2e_full.yaml` enable seccomp by default. `KOBOI_SANDBOX_DIR` is still honored as a back-compat fallback. Subprocess tools (`run_shell`, `git_*`, filesystem) declare `deps=["sandbox"]` and read `_deps["sandbox"]`; the facade always wires a (passthrough-or-better) sandbox
- `journal:` YAML section (default `enabled: true`) drives `koboi/journal.py`; auto-disabled when `memory.backend != sqlite` (it borrows the SQLite connection). Loop writes are native (not hooks) so durability can't be bypassed
- `koboi run --resume <session>` rehydrates-and-continues an interrupted session; `koboi sessions <config>` lists persisted sessions. Plain sequential/parallel orchestration resume is unsupported; DAG mode persists a durable graph plan + per-node completion records (`dag_scheduler.py` graph-cursor-resume primitives to the `steps` table)
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
- `hooks:` YAML section declares external-command hooks (no Python in the agent): `allow_exec` default-deny gate + `on_event:` entries, each spawning a command (`uv`/`uvx`-friendly) that exchanges JSON over stdio. Wired by `_build_command_hooks()` in `facade.py` (not the `_REGISTRY` pattern). See `docs/custom-hooks.md` for the wire protocol and `examples/33_command_hook_messaging.py` for a runnable demo
- `jobs.webhooks` fires an HTTP POST (fire-and-forget, retried on 5xx/network error) to each matching URL on a job's terminal status (`completed`/`failed`/`timed_out`/`cancelled`); `secret` HMAC-SHA256-signs the body (`X-Koboi-Signature` header). Operator-configured URLs only, never tenant-supplied
- Mode-block runs BEFORE approval in `ToolExecutionPipeline` (`loop_pipeline.py`) -- an approved tool is never retroactively mode-blocked, and a Trust-DB allow rule cannot bypass chat/plan mode-blocking. `ModeManager.is_tool_allowed()` is the single source of truth shared by the pipeline gate and `ModeHook`
- **Proactive long-term memory** (`memory.proactive`, opt-in, inert by default): `ProactiveMemory` (`proactive_memory.py`) makes KV memory proactive. **D extract** = `ProactiveExtractionHook` (SESSION_END, priority 65) side-LLM-extracts durable facts → redacts → stores in the KV `_MemoryStore` + maintains a core block. **C recall** = each turn, embed the user msg, cosine-rank KV facts (`SemanticRetriever._cosine_similarity`, NOT the corpus-coupled retriever), inject top-N **ephemerally** into the system msg in `loop._get_managed_messages` (NOT persisted as a conversation row — survives compaction because it's re-added each turn). **B core block** = bounded always-in-context summary in `session_meta`. Recall needs a real **`embedding:` model** (gpt-4o-mini can't embed; set `embedding:` or it falls back to the chat client and fails). Recall cache is invalidated whenever extract stores new facts.
- **Redaction** (`redact.py`): shared value-shape + key-name masking reused by `journal.py`/`server/jobs.py`/`diagnostics.py`. In the step journal it's **fail-safe** (`_safe_redact` masks args wholesale on any error → never aborts the durability write) and `_redact_nested` is **depth-capped** (`_REDACT_MAX_DEPTH=32`, guards `RecursionError` on untrusted nested args).
- **Schema self-heal**: `SQLiteMemory.list_sessions`/`delete_session`/`fork_session` call `_ensure_schema_on(conn)` on a raw connection (adds steps/tasks/session_meta tables + the `owner` column via `_migrate_add_owner`) — safe to point at a pre-existing/older DB. `delete_session` clears messages+steps+session_meta+tasks+sessions.
- **Tokenizer extra**: `pip install koboi-agent[tokenizer]` → `tiktoken>=0.7` for accurate OpenAI token counts; `tokens.make_tokenizer(provider, model)` returns a BPE counter (OpenAI only) wired into `ContextManager.tokenizer`, else the chars/3 heuristic. CI installs `.[dev,tui,api]` (no tokenizer) → tiktoken tests `importorskip`.
- **`ToolDefinition.idempotent: bool = True`** (`types.py`, set via `@tool(..., idempotent=)`): `False` marks side-effecting tools that must not silently double-fire on crash-resume — `_repair_interrupted_turn` skips re-execution (records a synthetic result) for non-idempotent tools. Default `True` preserves prior behavior.
- New memory/context knobs: `memory.retention.max_messages` (cap stored rows, default None=unbounded), `memory.owner` (tenant tag on stored rows, schema prep for multi-tenancy), `context.safety_margin` (headroom reserved inside `manage()` so one large response can't push an over-budget payload; default 0).
- **Session REST surface** (`server/app.py`): `GET /v1/sessions` (list, owner-scoped + fails closed 401 when auth on but no caller identity), `POST /v1/sessions/{id}/fork` (rolls back DB+owner rows on any `get_or_create` failure), `DELETE /v1/sessions/{id}` (clears DB rows AND holds `pool.existing_session_lock` so a concurrent `/chat/stream` can't re-insert orphaned rows). `create_app` accepts externalized-state injection kwargs (`session_store`/`job_store`/`event_buffer`/`idempotency_store`/`ownership_store`/`approval_registry`, each default None → in-process impl).

