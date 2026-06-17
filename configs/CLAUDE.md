# configs/ -- YAML agent configurations

## Available configs
```
simple_chat.yaml        Minimal chat agent (no tools)
sales_agent.yaml        Sales assistant with tools, RAG, guardrails, tracing
customer_service.yaml   Customer service agent
coding_agent.yaml       Code-focused agent with filesystem/shell/git tools
rag_agent.yaml          RAG with document Q&A
orchestrated.yaml       Multi-agent orchestrator
anthropic_chat.yaml     Anthropic provider example
anthropic_apikey.yaml   Anthropic with API key auth
anthropic_oauth.yaml    Anthropic with OAuth token
cloudflare_worker.yaml  Cloudflare Workers AI
eval_suite.yaml         Evaluation configuration
swe_bench.yaml          SWE-bench evaluation
```

## Schema reference
See `.claude/skills/yaml-config.md` for the full config schema.

## Env var interpolation
All string values support `${VAR}` and `${VAR:default}` syntax:
```yaml
api_key: "${OPENAI_API_KEY}"
base_url: "${OPENAI_BASE_URL:http://localhost:8080/v1}"
```

## Top-level sections
`agent`, `llm`, `tools`, `context`, `rag`, `guardrails`, `tracing`, `harness`, `policy`, `skills`, `mcp`, `memory`, `orchestration`
