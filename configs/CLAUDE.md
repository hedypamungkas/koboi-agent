# configs/ -- YAML agent configurations

## Available configs
```
simple_chat.yaml          Minimal chat agent (no tools)
sales_agent.yaml          Sales assistant with tools, RAG, guardrails, tracing
customer_service.yaml     Customer service agent
coding_agent.yaml         Code-focused agent with filesystem/shell/git tools
reasoning_model.yaml      OpenAI o-series reasoning model (max_completion_tokens + reasoning_effort; documents all forward-as-is llm: params)
sandbox_restricted.yaml   Restricted sandbox + SQLite journal/resume demo (P0b + P2-A)
rag_agent.yaml            RAG with document Q&A
orchestrated.yaml         Multi-agent orchestrator
advanced_full.yaml        Comprehensive config exercising most subsystems
advanced_orchestrated.yaml Advanced multi-agent orchestration with sub-agents
anthropic_chat.yaml       Anthropic provider example
anthropic_apikey.yaml     Anthropic with API key auth
anthropic_oauth.yaml      Anthropic with OAuth token
cloudflare_worker.yaml    Cloudflare Workers AI
eval_suite.yaml           Evaluation configuration
swe_bench.yaml            SWE-bench evaluation
benchmark_baseline.yaml   Retrieval baseline benchmark
benchmark_eval.yaml       Benchmark evaluation config
e2e_full.yaml             Full end-to-end (server + RAG + skills + tools)
server_simple.yaml        Minimal HTTP/SSE server (passthrough sandbox)
server_deploy.yaml        Production server (restricted sandbox, per-session workdir, server:/jobs:/tracing:)
```
**Server configs** (`server_simple.yaml`, `server_deploy.yaml`) drive `koboi serve`.
`server_deploy.yaml` is the Dockerfile `CMD` and the reference for `server:`/`jobs:`/`sandbox:`/`tracing:`.

## Schema reference
See `.claude/skills/yaml-config.md` for the full config schema.

## Env var interpolation
All string values support `${VAR}` and `${VAR:default}` syntax:
```yaml
api_key: "${OPENAI_API_KEY}"
base_url: "${OPENAI_BASE_URL:http://localhost:8080/v1}"
```

## Top-level sections
`agent`, `llm`, `tools`, `context`, `rag`, `guardrails`, `tracing`, `harness`, `policy`, `skills`, `mcp`, `memory`, `orchestration`, `sandbox`, `journal`, `server`, `jobs`, `keybindings`
