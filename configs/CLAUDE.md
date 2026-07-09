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
dag_demo.yaml             Multi-agent DAG workflow (research -> draft -> review -> publish)
conditional_demo.yaml     Conditional workflow (sentiment routes POSITIVE/NEGATIVE to a branch)
dynamic_demo.yaml         Dynamic workflow (LLM plans the graph per query; plan-or-skip)
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
hitl_demo.yaml            Human-in-the-loop approval demo (loopback; delete_file/run_shell trigger approvals)
server_simple.yaml        Minimal HTTP/SSE server (passthrough sandbox)
server_deploy.yaml        Production server (restricted sandbox, per-session workdir, server:/jobs:/tracing:)
command_hook_notify.yaml  External-command hook demo (`hooks:` section; see docs/custom-hooks.md)
jobs_webhooks.yaml        Job-webhook demo (`jobs.webhooks`; HMAC-signed terminal-status callbacks)
```
**Server configs** (`server_simple.yaml`, `server_deploy.yaml`) drive `koboi serve`.
`server_deploy.yaml` is the production reference for `server:`/`jobs:`/`sandbox:`/`tracing:`; the Dockerfile `CMD` defaults to `server_simple.yaml` (override at run time via `KOBOI_CONFIG=/app/configs/server_deploy.yaml`).

## Schema reference
See `.claude/skills/yaml-config.md` for the full config schema.

## Env var interpolation
All string values support `${VAR}` and `${VAR:default}` syntax:
```yaml
api_key: "${OPENAI_API_KEY}"
base_url: "${OPENAI_BASE_URL:http://localhost:8080/v1}"
```

## Top-level sections
`agent`, `llm`, `providers`, `pools`, `tools`, `context`, `rag`, `embedding`, `guardrails`, `tracing`, `harness`, `policy`, `skills`, `mcp`, `memory`, `subagent`, `orchestration`, `sandbox`, `journal`, `server`, `jobs`, `hooks`, `eval`, `keybindings`
