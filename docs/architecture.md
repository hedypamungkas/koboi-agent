# Architecture Overview

This document explains how koboi-agent works internally. It covers the agent loop lifecycle, hook system, tool pipeline, configuration, extension points, and all major subsystems. Read it top-to-bottom for a complete picture, or jump to any section.

For quickstart and installation, see [README.md](../README.md). For dev setup and contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md). For directory layout and code conventions, see [CLAUDE.md](../CLAUDE.md).

---

## Subsystem Dependency Graph

`KoboiAgent` (`koboi/facade.py`) is the public entry point. It delegates to either `AgentCore` (single-agent mode) or `Orchestrator` (multi-agent mode), both assembled from a YAML config by `AgentAssembler`.

```
YAML Config
     |
     v
KoboiAgent (facade.py)
     |
     +---> AgentCore (loop.py)             [single-agent mode]
     |       |--- RetryClient (client.py)
     |       |--- ConversationMemory / SQLiteMemory
     |       |--- ToolRegistry (tools/)
     |       |--- HookChain (hooks/)
     |       |--- ContextManager (context/)
     |       |--- AugmentationStrategy (rag/)
     |       |--- input_guardrails[], output_guardrails[]
     |       |--- RateLimiter, AuditTrail, ApprovalHandler
     |       |--- SkillRegistry (skills/)
     |       +--- ToolExecutionPipeline (loop_pipeline.py)
     |
     +---> Orchestrator (orchestration/)   [multi-agent mode]
             |--- BaseRouter (keyword / LLM / hybrid)
             |--- AgentFactory / DynamicAgentBuilder
             +--- QualityEvaluator

     Shared: ModeManager (modes.py), TrustDatabase (trust.py)
```

`AgentAssembler.build()` runs 14 steps in dependency order:

| Step | Method | Produces |
|------|--------|----------|
| 1 | `build_logger()` | `AgentLogger` |
| 2 | `build_client()` | `RetryClient` (provider, model, API key, retries) |
| 3 | `build_memory()` | `SQLiteMemory` or `ConversationMemory` |
| 4 | `build_tools()` | `ToolRegistry` (builtins + custom + MCP) |
| 5 | `build_mcp()` | `list[MCPClient]` (connects MCP servers, registers tools) |
| 6 | `build_context()` | `ContextManager` subclass or `None` |
| 7 | `build_rag()` | `AugmentationStrategy` or `None` |
| 8 | `build_guardrails()` | `(input_guardrails, output_guardrails, rate_limiter, audit_trail)` |
| 9 | `build_approval()` | `ApprovalHandler` (CLI, callback, or auto) |
| 10 | `build_policy()` | `PolicyEngine` (from config rules) |
| 11 | `build_skills()` | `SkillRegistry` (scans for SKILL.md files) |
| 12 | `build_mode_manager()` | `ModeManager` (CHAT/PLAN/ACT/AUTO) |
| 13 | `build_trust_db()` | `TrustDatabase` (if graduated_permissions enabled) |
| 14 | `build_hooks()` | `HookChain` (LoggingHook + conditional hooks) |

After assembly, `_setup_subagent()` and `_setup_tasks()` wire optional sub-agent and task management tools.

---

## Agent Loop Lifecycle

`AgentCore.run()` (`koboi/loop.py`) is the heart of the framework. It runs a prepare-call-execute loop until the LLM produces a final answer or `max_iterations` is exhausted.

```
AgentCore.run(user_message)
  |
  v
_prepare_run()
  |-- SESSION_START hook
  |-- Input guardrails check
  |-- PRE_INPUT hook
  |-- Augment memory (RAG)
  |-- Add user message to memory
  +-- Get tool definitions
  |
  v
FOR i in 0..max_iterations-1:
  |
  v
_prepare_iteration(i)
  |-- PRE_COMPACT hook
  |-- Context management (truncate / summarize)
  |-- POST_COMPACT hook
  +-- Augment messages for LLM (RAG on-the-fly)
  |
  v
PRE_LLM_CALL hook --> LLM call --> POST_LLM_CALL hook
  |
  +--> [ACTIVATE_SKILL: name]? --> activate skill, continue loop
  |
  +--> response.is_complete?
  |      YES --> output guardrails --> POST_OUTPUT hook
  |            --> SESSION_END hook --> return RunResult
  |
  +--> response.tool_calls?
         YES --> ToolExecutionPipeline (8-step per tool)
               --> store results in memory --> continue loop
  |
  v
(max_iterations exhausted) --> raise AgentMaxIterationsError
```

**`run_stream()`** follows the same logic but yields `StreamEvent` objects (`IterationEvent`, `TextDeltaEvent`, `ToolCallEvent`, `ToolResultEvent`, `CompleteEvent`, `ErrorEvent`) instead of collecting results internally.

Key termination conditions:
- `response.is_complete` -- LLM returned content with no tool calls
- `max_iterations` exhausted -- raises `AgentMaxIterationsError`
- `final_response is None` (streaming only) -- yields `ErrorEvent`

---

## Hook System

The hook system implements the Observer pattern. Hooks subscribe to lifecycle events and can inspect, modify, or abort the agent loop.

### Hook ABC

```python
class Hook(ABC):
    priority: int = 50  # lower runs first

    @abstractmethod
    def handles(self) -> list[HookEvent]: ...

    @abstractmethod
    async def execute(self, ctx: HookContext) -> HookContext: ...
```

For a step-by-step guide to creating hooks, see `.claude/skills/creating-hooks.md`.

### HookEvent values (15 total)

```
SESSION_START       session_start       Session begins (before input validation)
SESSION_END         session_end         Session ends (after output or max iterations)
PRE_INPUT           pre_input           Before input guardrails
POST_OUTPUT         post_output         After output guardrails pass
PRE_COMPACT         pre_compact         Before context window management
POST_COMPACT        post_compact        After context window management
PRE_LLM_CALL        pre_llm_call        Before LLM API call
POST_LLM_CALL       post_llm_call       After LLM response received
PRE_TOOL_USE        pre_tool_use        Before tool execution
POST_TOOL_USE       post_tool_use       After tool execution
DOOM_LOOP_DETECTED  doom_loop_detected  Repeated action pattern detected
PRE_ROUTING         pre_routing         Before orchestration routing
POST_ROUTING        post_routing        After orchestration routing
AGENT_DISPATCHED    agent_dispatched    Sub-agent dispatched (orchestration)
AGENT_COMPLETED     agent_completed     Sub-agent completed (orchestration)
```

### HookContext

Every hook receives a `HookContext` dataclass with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `event` | `HookEvent` | Which event triggered this hook |
| `agent` | `AgentInfo \| None` | Model name, agent name, iteration |
| `tool_name` | `str \| None` | Tool being called (PRE/POST_TOOL_USE) |
| `tool_arguments` | `str \| None` | Tool arguments JSON |
| `tool_result` | `str \| None` | Tool result (POST_TOOL_USE) |
| `messages` | `list[dict] \| None` | Messages being sent to LLM |
| `user_message` | `str \| None` | Original user input |
| `llm_response` | `Any` | LLM response object |
| `metadata` | `MetadataBag` | Typed metadata (see below) |
| `abort` | `bool` | Set `True` to halt the chain |
| `inject_messages` | `list[str]` | Messages to inject into conversation |
| `hook_outcomes` | `list[tuple]` | Audit trail of hook results |

`MetadataBag` is a `dict` subclass with typed property accessors: `mode_blocked`, `policy_decision`, `guardrail_blocked`, `doom_loop_detected`, `rag_strategy`, `skills_detected`, `context_managed`, etc.

### Priority conventions

| Range | Purpose | Examples |
|-------|---------|----------|
| 0-19 | Infrastructure | `LoggingHook` (priority 0) |
| 20-39 | Security | `PolicyHook` (priority 25) |
| 40-59 | Business logic | Default (priority 50) |
| 60-79 | Post-processing | `AuditHook` (priority 80) |
| 80-100 | Cleanup | -- |

### Emit mechanics

`HookChain.emit(ctx)` iterates hooks sorted by priority for the given event. Each hook's `execute()` receives and returns the context (hooks can mutate it). If any hook raises, `ctx.abort` is set and the chain breaks. Injected messages (`ctx.inject_messages`) are added to conversation memory.

### Event timeline

```
SESSION_START
  |
  v
PRE_INPUT
  |
  v
[iteration loop]
  |
  PRE_COMPACT --> POST_COMPACT
  |
  PRE_LLM_CALL --> POST_LLM_CALL
  |
  +-- [tool calls]
  |     PRE_TOOL_USE --> POST_TOOL_USE  (per tool)
  |     [if doom loop] DOOM_LOOP_DETECTED
  |
  +-- [orchestration]
        PRE_ROUTING --> POST_ROUTING
        AGENT_DISPATCHED --> AGENT_COMPLETED
  |
  v
POST_OUTPUT
  |
  v
SESSION_END
```

---

## Tool Pipeline

`ToolExecutionPipeline` (`koboi/loop_pipeline.py`) encapsulates the 8-step tool execution flow shared by `run()` and `run_stream()`.

```
ToolExecutionPipeline.execute_tool_call(tc)
  |
  1. Rate limiter check ---------> blocked? return skip
  2. Risk level lookup (SAFE / MODERATE / DESTRUCTIVE)
  3. Approval handler check ------> denied? return skip
  4. PRE_TOOL_USE hook -----------> abort/mode_blocked? return skip
  5. Mode block check (from hook metadata)
  6. ToolRegistry.execute(name, args_json)
  7. POST_TOOL_USE hook
  8. Record result in memory + audit
```

### The `@tool()` decorator

Tools are registered with `@tool()` from `koboi/tools/registry.py`:

```python
@tool(
    name="my_tool",
    description="Does something useful",
    parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    risk_level=RiskLevel.SAFE,
    timeout=30.0,
)
async def my_tool(x: str) -> str:
    return f"Result: {x}"
```

- `parameters` is a JSON Schema dict
- `risk_level` gates approval: `SAFE` (no approval), `MODERATE` (configurable), `DESTRUCTIVE` (always requires approval)
- `timeout` overrides the default tool timeout
- `deps` lists dependency names injected from `ToolRegistry.set_dep()`
- Sync functions are auto-wrapped in `asyncio.to_thread`
- Return type must be `str` (the registry calls `str(result)`)

For a full guide, see `.claude/skills/creating-tools.md`.

---

## Configuration System

`Config` (`koboi/config.py`) loads and resolves YAML configuration.

### Loading

```python
config = Config.from_yaml("configs/my_agent.yaml")       # from file
config = Config.from_dict({"agent": {"name": "test"}})    # from dict
config = Config.from_string("agent:\n  name: test")       # from string
```

### Features

- **`${VAR:default}` interpolation** -- environment variables with optional defaults, applied recursively to all string values
- **`extends` inheritance** -- `extends: [base.yaml, overrides.yaml]` with deep merging and circular-reference detection
- **Pydantic validation** -- optional schema validation via `config_models.py`
- **`ConfigBuilder`** -- fluent API for programmatic construction: `.agent().llm().tools().build()`

### Config sections (18)

| Section | Controls |
|---------|----------|
| `agent` | Name, system prompt, max iterations, mode |
| `llm` | Provider, model, API key, base_url, timeout, retries, temperature |
| `tools` | Builtin list, custom modules, per-tool overrides |
| `context` | Strategy, max_context_tokens, keep_last |
| `rag` | Chunker, retriever, augmentation, documents |
| `guardrails` | Input/output checks, rate limits, approval mode |
| `harness` | Telemetry, carryover, doom loop, notifications |
| `tracing` | Langfuse integration |
| `policy` | Allow/deny/confirm rules |
| `skills` | Search paths |
| `mcp` | MCP server connections |
| `memory` | Backend (sqlite/in_memory), db_path |
| `orchestration` | Router type, agents, execution mode |
| `sandbox` | Backend (passthrough/restricted), workdir strategy, network, rlimits |
| `journal` | Step journal (enabled, record_tool_calls) — crash/redeploy resume |
| `server` | HTTP/SSE serving: host/port, auth, pool, timeouts, allowed_modes, idempotency |
| `jobs` | Autonomous jobs: max_concurrent, queue_depth, ttl, resume_on_startup |
| `keybindings` | TUI key overrides |

For the complete YAML schema reference, see `.claude/skills/yaml-config.md`.

---

## Serving Layer (HTTP/SSE)

`koboi/server/` exposes the agent over HTTP/SSE via a FastAPI app (`create_app()` /
`koboi serve`). Two execution modes share one `AgentPool` (per-session `KoboiAgent` +
per-session `asyncio.Lock` — AgentCore is not concurrent-safe — + per-session sandbox
workdir):

- **Interactive** (`POST /v1/chat/stream`) — SSE stream of `run_stream()` events with
  human-in-the-loop approvals (`POST /v1/sessions/{id}/approve`). Per-request `mode` +
  `max_iterations` knobs (G2), `Idempotency-Key` dedup.
- **Autonomous jobs** (`POST /v1/jobs`) — background runs with no HITL, deny-by-default
  `AutonomousApprovalHandler`, restricted sandbox mandatory, durable `resume_on_startup`.

Cross-cutting: API-key (Bearer) auth + per-session ownership, `/healthz` + `/readyz`,
request-id middleware, graceful drain (cancel in-flight streams + off-loop Langfuse
flush), and M5 `Protocol` seams (`SessionStore`/`LockProvider`/`EventBuffer`) for a
future Redis/SaaS state swap.

Driven by the `server:` + `jobs:` config sections; requires the `[api]` extra
(`fastapi`, `uvicorn`). See `koboi/server/CLAUDE.md` for routes/conventions/gotchas and
`docs/rest-sse-requirements.md` for the design spec.

---

## Extension Points

koboi-agent provides 10 extension points, all following a consistent pattern: define a class implementing an ABC, register it with a registry.

| Extension | ABC | Registry | How to Register |
|-----------|-----|----------|-----------------|
| Hooks | `Hook` | `HookChain` | Subclass, add to `_REGISTRY` or `agent.add_hook()` |
| Tools | `@tool()` | `ToolRegistry` | YAML `tools.custom`, `agent.add_tool()`, or plugin |
| LLM Providers | `LLMClient` | `ProviderRegistry` | `ProviderDescriptor`, plugin `koboi.providers` |
| Context Strategies | `ContextManager` | `ComponentRegistry` | `@register_context_strategy()`, `context.custom_modules` |
| RAG Chunkers | `BaseChunker` | `ComponentRegistry` | `@register_chunker()`, `rag.custom_modules` |
| RAG Retrievers | `BaseRetriever` | `ComponentRegistry` | `@register_retriever()`, `rag.custom_modules` |
| RAG Augmentation | `AugmentationStrategy` | `ComponentRegistry` | `@register_augmentation()`, `rag.custom_modules` |
| Guardrails | `BaseGuardrail` | `GuardrailRegistry` | `GuardrailRegistry.register()`, plugin `koboi.guardrails` |
| Eval Scorers | `BaseScorer` | `ScorerRegistry` | Plugin `koboi.scorers` |
| Plugins | N/A | `entry_points` | Declare in `pyproject.toml` under `koboi.*` groups |

### Plugin entry points

External packages can register components via `pyproject.toml` entry points:

```toml
[project.entry-points."koboi.providers"]
my_provider = "my_package.provider:create_descriptor"

[project.entry-points."koboi.tools"]
my_tools = "my_package.tools:register"
```

`koboi/plugins.py` discovers and loads these at startup.

### Custom modules via YAML

Context strategies, chunkers, retrievers, and augmentation strategies can be loaded via config:

```yaml
context:
  strategy: "my_strategy"
  custom_modules: ["my_package.context"]

rag:
  custom_modules: ["my_package.rag"]
```

---

## Context Management

`ContextManager` (`koboi/context/manager.py`) manages the message window to fit within token limits.

### Template method

```python
class ContextManager(ABC):
    @abstractmethod
    def _build_result(self, system_msgs, non_system) -> tuple[list[dict], str]: ...

    def manage(self, messages, max_tokens) -> list[dict]:
        # 1. Estimate tokens
        # 2. If within budget, return as-is
        # 3. Call _build_result() for strategy-specific selection
        # 4. Run ensure_tool_integrity() cleanup
```

### `ensure_tool_integrity()`

A 5-pass cleanup that fixes message sequences before sending to the LLM:
1. Collect valid tool_call IDs from assistant messages
2. Remove orphaned tool results (whose parent was removed)
3. Fix assistant messages with missing tool results
4. Merge consecutive same-role messages
5. Ensure first non-system message is `user`

### Built-in strategies

| Strategy | Behavior |
|----------|----------|
| `noop` | Pass through, no management |
| `truncation` | Keep last N messages |
| `smart_truncation` | System prompt + first user message + last N |
| `key_facts` | Extract tool results into compact facts, discard old messages |
| `sliding_window` | Summarize old messages via LLM, keep recent |

---

## RAG Pipeline

The RAG pipeline has three stages: chunking, retrieval, augmentation.

```
Documents (files)
     |
     v
Chunker (fixed / sentence / paragraph)
     |
     v
Chunks --> Retriever (keyword / semantic / hybrid)
               |
               v
          RetrievalResult (top_k chunks with scores)
               |
               v
          AugmentationStrategy
            |-- InMemory: augment user message before storing in memory
            +-- OnTheFly: augment last user message before each LLM call
```

### Chunkers

| Chunker | Strategy |
|---------|----------|
| `FixedSizeChunker` | Fixed-size windows with sentence-boundary snapping |
| `SentenceChunker` | Split on sentence boundaries |
| `ParagraphChunker` | Heading-aware paragraph merging |

### Retrievers

| Retriever | Strategy |
|-----------|----------|
| `KeywordRetriever` | TF-IDF cosine similarity |
| `SemanticRetriever` | Embedding-based cosine (requires LLM client for embeddings) |
| `HybridRetriever` | Reciprocal Rank Fusion of keyword + semantic |

### Augmentation

| Strategy | When it augments |
|----------|-----------------|
| `InMemoryAugmentation` | Before storing user message in memory |
| `OnTheFlyAugmentation` | Before each LLM call (with caching) |

Configuration:

```yaml
rag:
  enabled: true
  chunker: "paragraph"
  retriever: "keyword"
  augmentation: "on_the_fly"
  top_k: 3
  documents:
    - path: "./data/sample/product_catalog.md"
```

---

## Guardrails and Safety

The safety model has four layers: guardrails, policy engine, approval handler, and trust database.

### Guardrails

`BaseGuardrail` defines the interface: `check(content) -> GuardrailResult`. `PatternGuardrail` adds regex-based pattern matching. Built-in guardrails:

- `InputGuardrail` -- injection detection, length limits
- `OutputGuardrail` -- content filtering, sensitive data detection

Guardrails are composed via `GuardrailRegistry` and configured in YAML:

```yaml
guardrails:
  input:
    max_length: 10000
    block_patterns: ["ignore previous", "system prompt"]
  output:
    block_patterns: ["password", "secret"]
  rate_limit:
    max_calls_per_minute: 20
```

### PolicyEngine

`PolicyEngine` (`koboi/harness/policy.py`) evaluates tool calls against rules:

- **Hardcoded protections**: sensitive paths (`/etc`, `~/.ssh`), dangerous commands (`rm -rf`, `dd`)
- **User-defined rules**: glob patterns for tools/paths, actions (allow/deny/confirm), risk-level fallback
- **First-match-wins**: rules evaluated in order, first match determines outcome

### ApprovalHandler

Three variants:
- `auto` -- auto-approve safe tools, deny destructive ones
- `CLI` -- interactive terminal prompts for moderate/destructive tools
- `callback` -- programmatic approval via async callback

### TrustDatabase

`TrustDatabase` (`koboi/trust.py`) provides graduated permissions:
- SQLite-backed storage of "always allow" decisions
- Glob pattern matching for tool+argument combinations
- TTL support for temporary permissions
- Learns from user decisions over time

---

## Harness Subsystems

Three harness subsystems run as hooks, providing observability and resilience.

### TelemetryCollector

`koboi/harness/telemetry.py` -- Tracks iteration records, compaction records, permission records. Calculates a health score from configurable weights (success rate, tool efficiency, doom loop frequency, compaction frequency).

### CarryoverState

`koboi/harness/carryover.py` -- Persists goals, artifacts, verified work, and work log across context compaction events. When the context window is compressed, `CarryoverHook` re-injects the carryover state as a context message so the agent doesn't lose track of its objectives.

### DoomLoopDetector

`koboi/harness/doom_loop.py` -- Detects three patterns:
- **Consecutive identical**: same tool call N times in a row
- **Repeating pattern**: circular sequence of tool calls
- **Error retry**: same error produced N times

When detected, emits `DOOM_LOOP_DETECTED` hook event with recovery hints.

---

## Multi-Agent Orchestration

When `orchestration.enabled: true` in YAML, `KoboiAgent` bypasses `AgentCore` and delegates to `Orchestrator`.

```
User query
     |
     v
Router (keyword / LLM / hybrid)
     |
     v
RoutingDecision (selected agents, confidence, method)
     |
     +---> AgentFactory.create agents from config
     |
     v
Orchestrator.run()
     |
     +-- [sequential] --> Agent1 --> Agent2 --> Agent3
     |
     +-- [parallel]   --> Agent1 + Agent2 + Agent3 (asyncio.gather)
     |
     v
[optional] QualityEvaluator --> revision loop (max_revisions)
     |
     v
OrchestratorResult (final_answer, agent_results)
```

### Routers

| Router | Strategy |
|--------|----------|
| `KeywordRouter` | Matches query against per-agent keyword lists |
| `LLMRouter` | LLM-based routing with JSON response, falls back to keyword |
| `HybridRouter` | Keyword first, LLM for confirmation/additions |

### Agent creation

- `AgentFactory` -- creates pre-configured agents from `AgentDef` list in YAML
- `DynamicAgentBuilder` -- builds specialist agents on-the-fly for unknown domains using LLM-generated blueprints

### Quality evaluation

`QualityEvaluator` scores agent answers via LLM. If score falls below threshold, the orchestrator triggers a revision loop (up to `max_revisions`).

---

## LLM Providers

### LLMClient ABC

`koboi/llm/base.py` defines the provider interface:

```python
class LLMClient(ABC):
    @property
    def model(self) -> str: ...
    async def complete(self, messages, tools) -> AgentResponse: ...
    async def complete_stream(self, messages, tools) -> AsyncIterator[StreamEvent]: ...
    async def get_embeddings(self, text) -> list[float] | None: ...
    async def close(self) -> None: ...
```

### ProviderRegistry

`ProviderDescriptor` is a frozen dataclass declaring provider metadata: `name`, `env_key_api`, `env_key_base_url`, `default_model`, `factory`. `ProviderRegistry` stores descriptors and resolves env vars with fallback chains.

### Built-in providers

| Provider | Adapter | Default model |
|----------|---------|---------------|
| OpenAI | `openai_adapter.py` | `gpt-4o-mini` |
| Anthropic | `anthropic_adapter.py` | `claude-sonnet-4-20250514` |
| Cloudflare | `openai_adapter.py` (Workers AI) | `@cf/meta/llama-3.1-70b-instruct` |

### RetryClient

`RetryClient` (`koboi/client.py`) wraps any `LLMClient` with exponential backoff, retryable error classification (rate limits, server errors), and placeholder key detection.

---

## Memory and Persistence

### MemoryBackend protocol

`koboi/memory.py` defines the interface:

```python
class MemoryBackend(Protocol):
    def add_user_message(self, content): ...
    def add_assistant_message(self, content, tool_calls=None): ...
    def add_tool_result(self, tool_call_id, result): ...
    def add_context_message(self, content, label=""): ...
    def get_messages(self) -> list[dict]: ...
    def clear(self) -> None: ...
```

### Implementations

| Backend | Storage | Use case |
|---------|---------|----------|
| `ConversationMemory` | In-memory list | Ephemeral sessions, testing |
| `SQLiteMemory` | SQLite WAL-mode | Persistent sessions, cross-restart durability |

`SQLiteMemory` adds session management (`list_sessions`, `delete_session`, `fork_session`) and persists messages to a SQLite database.

### AuditTrail

`SQLiteAuditTrail` (`koboi/guardrails/audit.py`) logs every tool execution (name, arguments, result, risk level, timestamp) to SQLite for post-hoc analysis.

---

## Skills System

Skills provide progressive disclosure of capabilities: discovery (metadata only), activation (body loaded on-demand), resources (lazy-loaded files).

### SKILL.md format

```markdown
---
name: code_review
description: Reviews code for quality issues
compatibility: ">=0.1.0"
allowed_tools: [read_file, search_files]
---

## Instructions
When activated, review the provided code for...
```

### Discovery and activation

1. `SkillRegistry` scans project, user, and plugin paths for `SKILL.md` files
2. Discovery metadata is injected into the system prompt as a skill listing
3. When the LLM responds with `[ACTIVATE_SKILL: code_review]`, `AgentCore._activate_skill()` loads the SKILL.md body into memory as a context message
4. Skill resources are lazy-loaded on first access

---

## Interaction Modes

`AgentMode` (`koboi/modes.py`) controls what the agent can do:

| Mode | Tools | Permission | Use case |
|------|-------|------------|----------|
| `CHAT` | Read-only | Auto-approve safe | Conversation |
| `PLAN` | Read-only + planning | Auto-approve safe | Planning |
| `ACT` | All | Approval required | Execution |
| `AUTO` | All | Configurable | Full autonomy |

`ModeHook` enforces restrictions by setting `metadata.mode_blocked` on `PRE_TOOL_USE` events when a tool is not allowed in the current mode.

---

## MCP Integration

koboi-agent supports the Model Context Protocol for external tool servers.

### Transports

| Transport | Class | Connection |
|-----------|-------|------------|
| stdio | `MCPClient` | Subprocess (stdin/stdout) |
| HTTP | `StreamableHTTPMCPClient` | HTTP streaming |

### Tool bridging

`register_mcp_tools(mcp_client, tools)` discovers tools from an MCP server and registers them in `ToolRegistry`, translating between MCP tool schemas and the internal `ToolDefinition` format.

Configuration:

```yaml
mcp:
  servers:
    - name: "my_server"
      command: ["python", "my_server.py"]
      # or: url: "http://localhost:8080/mcp"
```

---

## Evaluation Framework

`EvalRunner` (`koboi/eval/runner.py`) executes eval cases against an agent and collects scored results.

### Components

| Component | Purpose |
|-----------|---------|
| `EvalRunner` | Executes cases, collects `EvalResult` |
| `EvalConfig` | Suite configuration (cases, scorers, settings) |
| `ScorerRegistry` | Extensible scorer registration |
| `LoaderRegistry` | Extensible data loader registration |
| `RegressionTracker` | Compares against baseline scores |

### Built-in scorers

`ToolUsage`, `KeywordPresence`, `OutputLength`, `IterationEfficiency`, `HealthScore`, `LLMJudge`, `Cost` -- plus framework-specific scorers for BFCL, GAIA, SWE-bench, RAGAS, and DeepEval.

---

## Data Types and Exceptions

### Core dataclasses (`koboi/types.py`)

`RunResult`, `AgentResponse`, `ToolCall`, `ToolDefinition`, `TokenUsage`, `GuardrailResult`, `AuditEntry`, `RateLimitConfig`, `RoutingDecision`, `AgentResult`, `OrchestratorResult`, `EvalCase`, `EvalScore`, `EvalResult`, `SkillDefinition`, `MCPToolInfo`

### Stream events (`koboi/events.py`)

`TextDeltaEvent`, `ToolCallEvent`, `ToolResultEvent`, `CompleteEvent`, `ErrorEvent`, `IterationEvent`, `RoutingDecisionEvent`, `AgentDispatchEvent`, `AgentResultEvent`, `OrchestrationCompleteEvent`

### Error hierarchy (`koboi/exceptions.py`)

```
AgentError
  +-- AgentMaxIterationsError
  +-- AgentGuardrailError
  +-- AgentToolError
  +-- AgentTimeoutError
  +-- AgentStreamError
  +-- AgentAbortedError

LLMError
  +-- LLMConnectionError
  +-- LLMAuthenticationError
  +-- LLMRateLimitError (retry_after)
  +-- LLMServerError
  +-- LLMInvalidRequestError
  +-- LLMResponseParseError
```

---

## Where to Go Next

| I want to... | Read this |
|--------------|-----------|
| Add a new tool | `.claude/skills/creating-tools.md` |
| Add a new hook | `.claude/skills/creating-hooks.md` |
| Understand YAML config | `.claude/skills/yaml-config.md` |
| Add an LLM provider | `CONTRIBUTING.md` > "Adding an LLM provider" |
| Run examples | `examples/README.md` |
| See config examples | `configs/CLAUDE.md` |
| Work on the TUI | `koboi/tui/CLAUDE.md` |
| Serve over HTTP/SSE | `koboi/server/CLAUDE.md` |
| Extend the eval framework | `koboi/eval/CLAUDE.md` |
| Write tests | `tests/conftest.py` and `.claude/rules/tests.md` |
