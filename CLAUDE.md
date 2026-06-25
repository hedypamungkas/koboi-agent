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

## Directory map
```
koboi/              Main package (~150 .py files)
  config.py         Config + ConfigBuilder -- YAML loading, ${VAR:default} interpolation
  config_models.py  Pydantic v2 schema validation for config
  facade.py         KoboiAgent -- single entry point, assembles all subsystems
  loop.py           AgentCore -- async agent loop, hook integration
  loop_pipeline.py  ToolExecutionPipeline -- 8-step tool execution flow
  client.py         RetryClient -- LLM HTTP transport with exponential backoff
  events.py         StreamEvent union type for streaming
  types.py          All dataclasses: RunResult, ToolDefinition, EvalCase, etc.
  exceptions.py     AgentError and LLMError hierarchies
  memory.py         In-memory ConversationMemory + MemoryBackend protocol
  memory_sqlite.py  SQLite-backed memory backend (WAL mode)
  tokens.py         Token estimation helpers
  modes.py          AgentMode enum (chat/plan/act/auto/yolo), ModeManager
  trust.py          TrustDatabase for graduated permissions
  logger.py         AgentLogger
  plugins.py        Entry-point plugin discovery (koboi.providers, koboi.tools, etc.)
  subagent.py       SubAgentManager for parallel sub-agent delegation
  task.py           TaskManager for structured task tracking
  diagnostics.py    Session diagnostic export
  notifications.py  Notification system
  llm/              LLM providers: base ABC, OpenAI adapter, Anthropic adapter, factory, auth, registry
  tools/            Tool registry + builtin/ (calculator, filesystem, shell, web, memory, search, git, subagent, task)
  hooks/            Hook system: chain.py (HookEvent enum, Hook ABC, HookChain) + 19 specialized hooks
  context/          Context window strategies: truncation, smart_truncation, key_facts, sliding_window
  rag/              RAG pipeline: chunker (fixed/sentence/paragraph/semantic), retriever (keyword/semantic/hybrid), augmentation, registry
  guardrails/       Input/output guardrails, rate limiter, audit trail, approval handlers, registry
  harness/          Telemetry, carryover state, doom loop detection, policy engine
  orchestration/    Multi-agent: router (keyword/LLM/hybrid), orchestrator, factory, dynamic agent builder
  mcp/              MCP client (stdio + HTTP) and server
  skills/           Skill discovery and registry (agentskills.io standard) with budget, invocation control, dynamic context
  eval/             Evaluation: runner, config, registry, regression, loaders/, scorers/, t/
  tui/              Terminal UI (Textual): app, screens/ (8), widgets/ (13)
tests/              ~108 test files, asyncio_mode="auto", shared conftest.py with MockClient
configs/            12 YAML agent configs
examples/           30 numbered example scripts (01-30) with matching YAMLs
evals/              Sample eve-style `t` eval files (*.eval.py) -- run via `koboi eval-test`
skills/             2 skill definitions: code_review, search_and_summarize
mcp_servers/        1 MCP server example: todo_server.py
data/               Sample documents for RAG demos (Acme Corp)
benchmarks/         BFCL benchmark data (DO NOT read benchmarks/results.json -- 183MB)
docs/               Architecture overview, TUI design docs
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
- `koboi_memory.db` is SQLite WAL-mode, 3 files (.db, .db-shm, .db-wal)
- `.agent_memory.json` is a runtime artifact at project root
- `Config.from_yaml()` raises `FileNotFoundError`, not a generic error
- `AgentCore` is in `loop.py`, not `core.py`
- `HookEvent` values are snake_case strings: `"pre_tool_use"`, `"post_llm_call"`, etc. (15 total)
- `SkillDefinition` new fields: `disable_model_invocation`, `user_invocable`, `disallowed_tools`
- `build_discovery_prompt()` accepts `budget_chars` param; `SkillRegistry` defaults to 8000
- `activate_skill()` preprocesses `` !`command` `` blocks (shell injection); set `run_shell=False` to disable
- `SkillPersistenceHook` (priority 45) re-injects activated skills after POST_COMPACT
- Skill scorers: `skill_trigger_accuracy`, `skill_routing_accuracy`, `skill_token_overhead` in eval/
- The `facade.py` `_build_*` functions are module-level, not class methods
- TUI entry point: `koboi.tui.app:main` (setuptools script)
- Tool functions can be sync or async -- sync runs in thread via `asyncio.to_thread`
- Tool return type must be `str` -- the registry calls `str(result)`
- `HookChain` hooks sorted by priority: 0-19 infra, 20-39 security, 40-59 business, 60-79 post, 80-100 cleanup
- RAG/Context/Guardrail components use registry pattern with `@register_*` decorators
- Plugin entry points: `koboi.providers`, `koboi.tools`, `koboi.guardrails`, `koboi.scorers`
- YOLO mode (`/mode yolo`) bypasses rate limiting, approval, and mode blocks -- but PolicyHook's hardcoded safety (sensitive paths, dangerous commands) is always enforced via `pre_ctx.abort` check in pipeline
