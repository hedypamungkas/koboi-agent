---
name: yaml-config
description: YAML config schema reference for koboi-agent
---

# YAML Config Schema

## agent (required)
```yaml
agent:
  name: str              # Required. Agent identifier.
  system_prompt: str     # System prompt (supports multiline with |)
  description: str       # Optional description
  max_iterations: int    # Default: 10
  mode: str              # "chat" | "plan" | "act" | "auto" | "yolo" (default: "chat")
  theme: str             # TUI theme name (default: "koboi-dark")
```

## llm (required)
```yaml
llm:
  provider: str          # "openai" | "anthropic" | "cloudflare" (default: "openai")
  model: str             # Required. Model identifier.
  api_key: str           # Supports ${ENV_VAR} interpolation
  base_url: str          # Provider base URL override
  timeout: float         # HTTP timeout in seconds (default: 120.0)
  max_tokens: int        # Max generation tokens. Optional -- omitted from the
                         #   request when unset (OpenAI/Cloudflare send no cap);
                         #   Anthropic falls back to 4096 (its API requires it).
  temperature: float     # Optional temperature override
  max_retries: int       # LLM-level retries (default: 3)
  retry_backoff_base: float  # Backoff base (default: 2.0)
  auth_token: str        # Secondary auth token (e.g. Anthropic OAuth)
  auth_type: str         # Auth type (default: "api_key")
  embedding_model: str   # For semantic RAG (default: "text-embedding-3-small")
  api_version: str       # API version (e.g. "2023-06-01" for Anthropic)

  # --- Forward-as-is generation params (optional) ---------------------------
  # Any of these, when present, are merged verbatim into the provider request
  # body. Sampling + response shaping:
  top_p: float                  # Nucleus sampling
  top_k: int                    # (Anthropic / some gateways)
  frequency_penalty: float      # OpenAI-compatible
  presence_penalty: float       # OpenAI-compatible
  stop: [str]                   # Stop sequences
  seed: int                     # Best-effort determinism (OpenAI-compatible)
  response_format: dict         # e.g. {"type": "json_object"}
  logit_bias: dict              # Token bias
  logprobs: bool                # Return logprobs
  top_logprobs: int             # How many logprobs to return
  verbosity: int                # OpenAI-compatible
  # Reasoning models:
  reasoning_effort: str         # "low" | "medium" | "high" (OpenAI o-series)
  max_completion_tokens: int    # OpenAI o-series cap (suppresses max_tokens)
  thinking: dict                # Anthropic: {"type":"enabled","budget_tokens":N}
```
- Only the allowlisted keys above are forwarded; infra keys (`provider`/`model`/`api_key`/`base_url`/`temperature`/`max_tokens`/`timeout`/retries/`auth_*`) are handled separately and never leak into the body.
- **OpenAI o-series:** set `max_completion_tokens` (not `max_tokens`); when both are present, `max_tokens` is dropped automatically (the API rejects the pair). Set `reasoning_effort` to control thinking depth.
- **Anthropic `thinking`:** requires `max_tokens` > `budget_tokens` -- forwarded as-is, so satisfy the constraint yourself (the provider rejects it otherwise).

**Per-agent overrides (orchestration):** under `orchestration.agents[*].llm`, any key above (plus `provider`/`model`/`api_key`/`base_url` to route an agent to a different model) overrides the top-level `llm:` block for that agent only; `max_context_tokens` tunes that agent's context window. Agents without an `llm:` block share the orchestrator's single client. See `configs/reasoning_model.yaml` for a worked example.

## tools
```yaml
tools:
  builtin: [calculator, web_search, memory_store, memory_recall, ...]
  custom:
    - module: "my_module"       # Module to scan for @tool-decorated functions
      function: "my_tool"       # Optional: specific function name
  defaults:                     # Default config for all tools
    timeout: 30
    max_output: 10000
    env_passthrough: false      # Escape hatch: pass full env to subprocess tools
    env_allowlist: [CARGO_HOME] # Extra env-var name/glob patterns to allow through
    env_blocklist: [MY_CUSTOM_*] # Extra env-var name/glob patterns to strip
  overrides:                    # Per-tool config overrides (key = registered tool name)
    run_shell:                  # use the real name; legacy "shell" alias also works
      timeout: 60
      max_output: 20000
  disabled: [delegate_tasks]    # DENYLIST: remove a tool from LLM view AND execution
  groups: [math, file]          # HIDE: only advertise these groups (tools stay callable)
```

## context
```yaml
context:
  strategy: str              # "noop" | "truncation" | "smart_truncation" | "key_facts" | "sliding_window"
  max_context_tokens: int    # Default: 8000
  keep_last: int             # Min messages to keep in truncation
  summarization_truncation: int  # Char limit per message in sliding window summary
```

## rag
```yaml
rag:
  enabled: bool
  chunker: str              # "fixed" | "sentence" | "paragraph"
  chunk_size: int           # For fixed chunker (default: 400)
  max_chunk_size: int       # For sentence/paragraph chunkers
  overlap: int              # For fixed chunker (default: 50)
  retriever: str            # "keyword" | "semantic"
  top_k: int                # Results to retrieve (default: 3)
  augmentation: str         # "in_memory" | "on_the_fly"
  documents:
    - path: "./path/to/doc.md"
```

## guardrails
```yaml
guardrails:
  input:
    detect_injection: bool  # Default: false
    max_length: int         # Max input length
  output:
    detect_sensitive: bool  # Default: false
  rate_limit:
    max_calls_per_session: int  # Default: 100
    max_calls_per_minute: int   # Default: 20
    rate_window_seconds: float  # Default: 60.0
  approval:
    handler: str            # "auto" | "cli" | "callback"
    graduated: bool         # Enable graduated trust
    trust_db_path: str      # Default: "koboi_trust.db"
```

## harness
```yaml
harness:
  telemetry: bool           # Enable telemetry collection
  carryover: bool           # Enable cross-session state
  carryover_limits:
    max_log_entries: int    # Default: 50
    max_goals: int          # Default: 10
    max_artifacts: int      # Default: 20
    max_verified: int       # Default: 20
  doom_loop:
    consecutive_identical_threshold: int  # Default: 3
    repeating_pattern_window: int         # Default: 6
    repeating_pattern_threshold: int      # Default: 2
    enable_recovery: bool                 # Default: true
    adaptive_threshold: bool              # Default: false
    task_complexity_hint: str             # "auto" | "simple" | "moderate" | "complex"
    error_retry_threshold: int            # Default: 3
  health_weights:
    loop_health: float          # Default: 0.20
    tool_success_rate: float    # Default: 0.20
    context_efficiency: float   # Default: 0.15
    compaction_fidelity: float  # Default: 0.15
    permission_friction: float  # Default: 0.15
    doom_penalty: float         # Default: 0.15
```

## tracing
```yaml
tracing:
  provider: "langfuse"
  public_key: str
  secret_key: str
  base_url: str
```

## policy
```yaml
policy:
  rules:
    - name: str           # Rule name
      tool: str           # Tool name glob pattern
      pattern: str        # Argument regex pattern
      action: str         # "allow" | "deny" | "confirm"
      risk_levels: [str]  # Filter by risk level
```

## skills
```yaml
skills:
  search_paths: ["./skills"]
```

## mcp
```yaml
mcp:
  servers:
    - command: "python"
      args: ["mcp_servers/todo_server.py"]
```

## memory
```yaml
memory:
  backend: str            # "sqlite" | "in_memory" (default: "sqlite")
  db_path: str            # Default: "koboi_memory.db"
  session_id: str         # Optional session identifier
```
