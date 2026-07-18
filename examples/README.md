# Koboi Agent Examples

Examples for trying out all koboi-agent features. From the simplest to production-ready.

## Prerequisites

```bash
# 1. Copy .env.example to .env and fill in your API key
cp .env.example .env

# 2. Install dependencies
#    Most numbered examples use the Rich UI + Click via examples/conftest.py, so they need
#    the [tui] extra (rich + click). The koboi package itself does NOT need these -- its CLI
#    is argparse-based. Bare-install-safe examples (no extras) are listed below.
pip install -e ".[tui]"
#    Server examples (server_built_in.py, server_customize.py) additionally need [api]:
pip install -e ".[api]"
#    Bare-install-safe examples (no extras, `pip install koboi-agent` only):
#       27_benchmark_suite.py, 29_skills_enhanced.py, 31_tool_selection.py,
#       32_sandbox_and_resume.py, hitl_client.py, ../benchmarks/crash_recovery/
```

## How to Run

Each example supports **dual mode**:

```bash
# Automatic mode (default) -- runs predefined questions with Rich UI
python examples/01_simple_chat.py

# Interactive mode -- free chat with the agent
python examples/01_simple_chat.py -m interactive

# Verbose output
python examples/01_simple_chat.py -v
```

Every example can also be run via CLI + YAML config:

```bash
koboi chat examples/01_simple_chat.yaml
```

## Examples

### Tier 1: Fundamentals

| # | Example | Features | How to Run |
|---|---------|----------|------------|
| 01 | [Simple Chat](01_simple_chat.py) | Basic chat without tools | `python examples/01_simple_chat.py` |
| 02 | [Config Showcase](02_config_showcase.py) | Load & compare all YAML configs | `python examples/02_config_showcase.py` |
| 03 | [Single Tool](03_tool_use_single.py) | Calculator tool, memory table | `python examples/03_tool_use_single.py` |
| 04 | [Multi Tool](04_tool_use_multi.py) | Calculator + Memory tools | `python examples/04_tool_use_multi.py` |
| 05 | [Context Management](05_context_management.py) | 5 context strategies compared | `python examples/05_context_management.py` |
| 06 | [RAG Basics](06_rag_basics.py) | Document Q&A with RAG | `python examples/06_rag_basics.py` |
| 07 | [Guardrails](07_guardrails.py) | Input/output guardrails, rate limit | `python examples/07_guardrails.py` |

### Tier 2: Integration

| # | Example | Features | How to Run |
|---|---------|----------|------------|
| 08 | [RAG Advanced](08_rag_advanced.py) | RAG configuration comparison | `python examples/08_rag_advanced.py` |
| 09 | [MCP Client](09_mcp_client.py) | Connect to MCP server | `python examples/09_mcp_client.py` |
| 10 | [MCP Server](10_mcp_server.py) | Build a custom MCP server | `python examples/10_mcp_server.py` |
| 11 | [Policy Engine](11_policy_engine.py) | Rules allow/deny/confirm | `python examples/11_policy_engine.py` |
| 12 | [Custom Hooks](12_custom_hooks.py) | Hook ABC, HookChain, abort/inject, agent integration | `python examples/12_custom_hooks.py` |
| 13 | [Skills Discovery](13_skills_discovery.py) | Skill discovery & activation | `python examples/13_skills_discovery.py` |
| 14 | [Custom Tools](14_custom_tools.py) | 3 ways to register custom tools | `python examples/14_custom_tools.py` |
| 15 | [Orchestration Keyword](15_orchestration_keyword.py) | Multi-agent with KeywordRouter | `python examples/15_orchestration_keyword.py` |
| 16 | [Orchestration LLM](16_orchestration_llm.py) | LLM/Hybrid router, dynamic agents | `python examples/16_orchestration_llm.py` |

### Tier 3: Advanced

| # | Example | Features | How to Run |
|---|---------|----------|------------|
| 17 | [Anthropic Provider](17_anthropic_provider.py) | Multi-provider (OpenAI vs Anthropic) | `python examples/17_anthropic_provider.py` |
| 18 | [Harness Telemetry](18_harness_telemetry.py) | Metrics & health score | `python examples/18_harness_telemetry.py` |
| 19 | [Doom Loop Detection](19_doom_loop_detection.py) | Unproductive loop detection | `python examples/19_doom_loop_detection.py` |
| 20 | [Carryover State](20_carryover_state.py) | State persistence across compaction | `python examples/20_carryover_state.py` |
| 21 | [Eval Suite](21_eval_suite.py) | Evaluation suite (12 default scorers) | `python examples/21_eval_suite.py` |
| 22 | [Full Production](22_full_production.py) | All features + custom ProfilingHook | `python examples/22_full_production.py` |
| 23 | [SWE Bug Hunter](23_swe_bug_hunter.py) | Complex multi-feature: RAG + tools + skills + hooks | `python examples/23_swe_bug_hunter.py` |
| 24 | [Config-driven Orchestration](24_config_driven_orchestration.py) | YAML-based multi-agent with specialist routing | `python examples/24_config_driven_orchestration.py` |
| 25 | [Subagent Delegation](25_subagent_delegation.py) | Parallel subagent spawning, lifecycle management | `python examples/25_subagent_delegation.py` |
| 26 | [Task Management](26_task_management.py) | Task tracking, dependencies, structured workflow | `python examples/26_task_management.py` |
| 27 | [Benchmark Suite](27_benchmark_suite.py) | Performance benchmarking across subsystems | `python examples/27_benchmark_suite.py` |
| 28 | [Custom RAG Registry](28_custom_rag_registry.py) | Custom chunker/retriever via registry decorators | `python examples/28_custom_rag_registry.py` |
| 29 | [Skills Enhanced](29_skills_enhanced.py) | Skill discovery + activation, persistence, budget | `python examples/29_skills_enhanced.py` |
| 30 | [Eval `t` Tests](30_eval_test.py) | eve-style `t` eval authoring (CI-native, mock mode) | `python examples/30_eval_test.py` |
| 31 | [Tool Selection](31_tool_selection.py) | Tool selection + secret hygiene | `python examples/31_tool_selection.py` |
| 32 | [Sandbox + Resume](32_sandbox_and_resume.py) | Restricted sandbox (soft + seccomp) + step-journal resume | `python examples/32_sandbox_and_resume.py` |
| 33 | [Command Hook Messaging](33_command_hook_messaging.py) | Declarative `hooks:` YAML -- external-command hook forwards LLM responses | `python examples/33_command_hook_messaging.py` |
| 34 | [Modern RAG Pipeline](34_rag_modern_pipeline.py) | BM25 + rewriting + filtering + reranking + caches | `python examples/34_rag_modern_pipeline.py` |
| 35 | [Confidence + Handover](35_confidence_handover_demo.py) | Confidence-aware CS + human handover (grounding, transfer_to_human) | `python examples/35_confidence_handover_demo.py` |
| 36 | [Workflow Export/Import](36_workflow_export_import.py) | Deterministic workflow bundle (`koboi export`/`import`) | `python examples/36_workflow_export_import.py` |
| 37 | [Workflow Cache + Replay](37_workflow_cache_capture_replay.py) | Capture response cache + offline `replay` (no API key) | `python examples/37_workflow_cache_capture_replay.py` |
| 38 | [Self-Healing Demo](38_self_healing_demo.py) | Bounded reflection loop, escalation ladder, graceful degrade, CRITIC verification | `python examples/38_self_healing_demo.py --mock` |
| 39 | [Aegis Ops -- Full Sample](39_aegis_ops_full_demo.py) | "Full sample": all 32 `KoboiConfig` top-level sections in one DAG-orchestrated scenario (providers/pools, RAG+rerank, proactive memory, self_healing, handover, sandbox, MCP, media, peers, server/jobs, hooks, eval) -- also documents the orchestration-mode feature-coverage gaps it surfaced | `python examples/39_aegis_ops_full_demo.py` |

### Server

| Example | Features | How to Run |
|---------|----------|------------|
| [Built-in Server](server_built_in.py) | Run the built-in SSE server (zero code) | `python examples/server_built_in.py` |
| [Customize Server](server_customize.py) | Extend the server by code (Path B) | `python examples/server_customize.py` |
| [HITL Client](hitl_client.py) | httpx-only HITL approval client (bare install) | `python examples/hitl_client.py` |
| [Workflow Graph Demo](workflow_graph_demo.py) | Programmatic DAG builder (no YAML, no planner) | `python examples/workflow_graph_demo.py` |
| [Dynamic Workflow Live](dynamic_workflow_live.py) | Live-LLM dynamic + DAG workflow e2e | `python examples/dynamic_workflow_live.py` |
| [Phase 3 Live E2E](phase3_live_e2e.py) | Live-LLM validation of Phase 3 capabilities | `python examples/phase3_live_e2e.py` |
| [Command Hook Forwarder](_command_hook_forwarder.py) | Standalone script invoked by example 33's `hooks:` entry | `uv run examples/_command_hook_forwarder.py OUTFILE` |
| [A2A Fan-out](a2a_fanout.py) | Cross-instance agent-to-agent: remote orchestration nodes via `call_peer_agent` / `/v1/peer/invoke` | `python examples/a2a_fanout.py` |

## Feature Coverage

| Feature | Examples |
|---------|----------|
| Simple chat | 01 |
| Config inspection (no API key needed) | 02 |
| Tool use (single/multi) | 03, 04 |
| Context management (5 strategies) | 05 |
| RAG (chunking, retrieval, augmentation) | 06, 08 |
| Guardrails (input, output, rate limit) | 07 |
| MCP (client + server) | 09, 10 |
| Policy engine | 11 |
| Custom hooks (Hook ABC, lifecycle, abort/inject) | 12 |
| Skills system | 13 |
| Custom tools | 14 |
| Multi-agent orchestration | 15, 16, 24 |
| Config-driven orchestration | 24 |
| Workflow graph (programmatic DAG builder) | workflow_graph_demo |
| Dynamic workflow (live LLM planning) | dynamic_workflow_live, phase3_live_e2e |
| DAG / conditional orchestration (configs) | dag_demo, conditional_demo, dynamic_demo |
| Cross-instance A2A (remote orchestration nodes) | a2a_fanout |
| Multi-provider LLM | 17 |
| Harness (telemetry, doom loop, carryover) | 18, 19, 20 |
| Evaluation suite | 21 |
| Full production (with custom hooks) | 22 |
| SWE Bug Hunter (complex multi-feature) | 23 |
| Subagent delegation (parallel subtasks) | 25 |
| Task management (dependencies, workflow) | 26 |
| Benchmark suite (performance testing) | 27 |
| Custom RAG registry (BM25, WordCount) | 28 |
| Skills (enhanced: activation, persistence, budget) | 29 |
| Eval `t` authoring (CI-native, mock mode) | 30 |
| Tool selection + secret hygiene | 31 |
| Sandbox isolation (soft + seccomp) + step-journal resume | 32 |
| Declarative external-command hooks (`hooks:` YAML) | 33 |
| Modern RAG pipeline (BM25, rewriting, filtering, reranking, caches) | 34 |
| Confidence-aware + human handover | 35 |
| Deterministic workflow export/import | 36 |
| Workflow cache + capture + offline replay | 37 |
| Self-healing (reflection, escalation ladder, graceful degrade, CRITIC) | 38 |
| HTTP/SSE server (built-in + code-customized) | server_built_in, server_customize |

## Notes

- Examples 02, 12 (standalone part), 19, and 20 **do not require an API key** (config inspection or simulation only)
- All examples use sample data from `data/sample/` (Acme Corp)
- Interactive mode: type `quit` to exit
- For the Anthropic provider, set `ANTHROPIC_API_KEY` in `.env`
- Shared utilities are in `conftest.py` (setup, dual-mode helpers, agent creation)
