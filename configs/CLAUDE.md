# configs/ -- YAML agent configurations

## Available configs
```
simple_chat.yaml          Minimal chat agent (no tools)
sales_agent.yaml          Sales assistant with tools, RAG, guardrails, tracing
customer_service.yaml     Customer service agent
cs_confidence_handover.yaml  Confidence-aware CS with handover (A1-A3 + B1/B1.5 + B4; the confidence ladder demo)
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
swe_bench.yaml            SWE-bench evaluation
benchmark_baseline.yaml   Retrieval baseline benchmark
benchmark_eval.yaml       Benchmark evaluation config
e2e_full.yaml             Full end-to-end (server + RAG + skills + tools)
hitl_demo.yaml            Human-in-the-loop approval demo (loopback; delete_file/run_shell trigger approvals)
server_simple.yaml        Minimal HTTP/SSE server (passthrough sandbox)
server_deploy.yaml        Production server (restricted sandbox, per-session workdir, server:/jobs:/tracing:)
command_hook_notify.yaml  External-command hook demo (`hooks:` section; see docs/custom-hooks.md)
jobs_webhooks.yaml        Job-webhook demo (`jobs.webhooks`; HMAC-signed terminal-status callbacks)
proactive_memory.yaml     Proactive long-term memory demo (D extract + C recall + B core block; `memory.proactive:` + `embedding:`)
web_brave.yaml            Brave Search provider demo (`websearch.search.provider: brave`)
web_firecrawl.yaml        Firecrawl search+fetch provider demo (`websearch.search.provider: firecrawl`)
deep_research_demo.yaml   Deep research demo (`execution.mode: deep_research`; coverage-gated cited web research)
workflow_export_demo.yaml  Deterministic workflow export demo (sentiment-routing DAG; `orchestration.determinism` + node `output_schema`; `koboi export`/`import` + `run --workflow`)
self_healing_demo.yaml    Self-healing demo (bounded reflection + escalation ladder + graceful degrade + CRITIC; P0-P4; `examples/38_self_healing_demo.py`)
a2a_instance_x.yaml        Cross-instance A2A -- instance X (agent A; `peers:` outbound to peer C; `call_peer_agent` tool)
a2a_instance_y.yaml        Cross-instance A2A -- instance Y (agent C; accepts peer calls via `POST /v1/peer/invoke`, `inbound_tokens`)
a2a_dag_remote.yaml        Cross-instance A2A -- DAG with a REMOTE node (`orchestration.agents[].endpoint: peerY`); run alongside a2a_instance_y.yaml
aegis_ops_full.yaml       "Full sample" -- exercises all 32 KoboiConfig top-level sections in one DAG-orchestrated customer-ops scenario (providers/pools failover, RAG+rerank, proactive memory, self_healing, handover, restricted sandbox, MCP, media, peers, server/jobs, hooks, eval); see `examples/39_aegis_ops_full_demo.py` for the (also documented) orchestration-mode feature-coverage gaps it surfaced
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
`agent`, `mode`, `llm`, `providers`, `pools`, `tools`, `context`, `rag`, `embedding`, `guardrails`, `tracing`, `harness`, `policy`, `skills`, `mcp`, `memory`, `subagent`, `orchestration`, `sandbox`, `journal`, `server`, `jobs`, `hooks`, `eval`, `keybindings`, `websearch`, `research`, `handover`, `peers`, `media`, `self_healing`

### Notable sub-sections (recently added)
- `memory.proactive` — opt-in long-term memory: `enabled` (master), `extract` (D: auto-extract facts at SESSION_END), `recall` (C: semantic recall + inject top-N each turn), `core_block` (B: always-in-context summary); `top_k`/`min_score`/`max_facts` tune recall. Recall needs a dedicated `embedding:` model.
- `memory.retention.max_messages` — cap stored message rows (oldest pruned); None = unbounded.
- `memory.owner` — tenant/owner tag stamped on stored rows (schema prep for multi-tenancy).
- `context.safety_margin` — tokens of headroom reserved inside `manage()` so one large response can't push an over-budget payload (default 0).
- `rag.rerank` — `bool | dict`. `true` (legacy) wraps the retriever in the heuristic keyword-overlap `RerankerRetriever`; a **dict** `{provider: jina|cohere|local, api_key, model, base_url, timeout, fetch_multiplier, score_threshold}` selects a true cross-encoder (`koboi/rag/rerank.py`). `provider` defaults `jina`; HTTP backends need `api_key`, `local`/BGE needs the `[rerank-local]` extra. Fail-soft (any hiccup → base results).
- `rag.stopwords` / `rag.stemmer` — lexical-retriever normalization (Keyword/BM25/Hybrid). `stopwords: true|en|id` (id = ~80 function words); `stemmer: id` (Sastrawi via the `[indo-nlp]` extra; `True` is invalid for stemmer). Applied to both index and query tokens.
- `grounding_check` (output guardrail) — opt-in runtime faithfulness (Wave 2 A3): `name: grounding_check` under `guardrails.output` with `provider`/`model`/`threshold` (default 0.8); decomposes the answer into claims, NLI-checks each vs retrieved context, and abstains (refuses) when coverage < threshold. Fail-soft.
- `handover.detection` — opt-in structural handover (B1.5): `enabled`, `coverage_threshold` (A3 grounding coverage below this triggers handover; default 0.5), `ask_patterns` (regexes for explicit "talk to a human" requests). Pairs with `grounding_check` — without it the coverage trigger is inert (only explicit user-ask fires) and `facade.py` logs a build-time warning.
- `handover.digest` — opt-in warm-handoff summary (B4): `enabled` generates a side-LLM case-card summary attached to `HandoverEvent.summary`.
- `handover.webhooks` — HMAC-signed mid-conversation callbacks on `handover.requested` (see `docs/channel-bridge.md`).
- `self_healing:` — opt-in bounded reflection + escalation ladder (P0-P4): `enabled` (master, default false), `max_turns` (shared reflect/doom budget, default 3), `fail_soft` (default true), `graceful_max_iter` (opt-in `max_iterations`-exhaustion summary, independent of `enabled`), `empty_response_reask_limit` (default 1), `critic_llm` (named `providers:` ref or None = reuse agent client), `triggers.tool_error.repeat_threshold` (default 2) / `triggers.low_grounding.threshold` (default 0.6) / `triggers.tool_verification` (CRITIC, P4), `ladder` (failure_class → rung ordering, empty = built-in default), `self_consistency` (`n_samples`/`max_concurrency`/`modes`, P4). See `docs/self-healing-feasibility.md`, `configs/self_healing_demo.yaml`.
- `orchestration.determinism` — `{temperature, seed, top_p, model_pin, replay_mode}`: workflow-level determinism default; a per-node `determinism:` overrides via `DeterminismProfile.merge` (node wins). Drives the cache/replay tiers. No `seed` on Anthropic.
- `orchestration.agents[].output_schema` — structured output (JSON schema) on a node; `force_response_format_with_tools` forces it even with tools (Gap A+B).
- `replay:` — `{mode: live|cache|replay, cache_dir}` set by `koboi run --replay-mode`; `replay` is offline (raise-on-miss, no API key).
- `peers` — opt-in cross-instance agent-to-agent (A2A): `enabled`, `allow_private_network` (default false = strict SSRF; true permits same-org localhost/private-net peers), `inbound_tokens` (plaintext tokens accepted FROM peers, hashed at load), `peers[]` (`{name, url, token, agent_name, org, timeout}`). The `call_peer_agent` tool fans out to peers (`POST /v1/peer/invoke`); `orchestration.agents[].endpoint: <peer_name>` makes a node REMOTE (a `RemoteAgentProxy`). P3 self-observing org-claim: `org`/`org_secret` (shared HMAC across same-org instances)/`public_base_url` (advertised in the card); when `org_secret` is set, each peer's `GET /.well-known/agent-card` is fetched at startup + hourly and HMAC-verified — only verified peers are callable ("verified-only", replacing assume-same-org). See `configs/a2a_*.yaml`.
- `media:` — opt-in multimodal generation (`MediaConfig`): `enabled`, per-modality `image`/`video`/`music`/`speech`/`transcription` (`{provider: surplus|mock, surplus: {api_key, base_url, model}}`), `budget` (`max_cost_usd`/`max_images`/`max_video_seconds`/`max_music_seconds`), `storage` (`backend: local|r2|s3`; r2/s3 need `[media-cloud]`), `profiles` (ModelProfile overrides), `custom_modules` (`@register_*` providers). Inert unless `enabled`.
- `research.media` / `research.capabilities` — Deep Research auto-multimedia-briefing (tokens: image/video/music/speech).

