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
  max_tokens: int        # Max generation tokens (default: 4096)
  temperature: float     # Optional temperature override
  max_retries: int       # LLM-level retries (default: 3)
  retry_backoff_base: float  # Backoff base (default: 2.0)
  auth_token: str        # Secondary auth token (e.g. Anthropic OAuth)
  auth_type: str         # Auth type (default: "api_key")
  embedding_model: str   # For semantic RAG (default: "text-embedding-3-small")
  api_version: str       # API version (e.g. "2023-06-01" for Anthropic)
```

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
  overrides:                    # Per-tool config overrides
    shell:
      timeout: 60
      max_output: 20000
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
