# Koboi Agent Examples

Examples for trying out all koboi-agent features. From the simplest to production-ready.

## Prerequisites

```bash
# 1. Copy .env.example to .env and fill in your API key
cp .env.example .env

# 2. Install dependencies
pip install -e .
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
| 21 | [Eval Suite](21_eval_suite.py) | Evaluation with 6 scorers | `python examples/21_eval_suite.py` |
| 22 | [Full Production](22_full_production.py) | All features + custom ProfilingHook | `python examples/22_full_production.py` |
| 23 | [SWE Bug Hunter](23_swe_bug_hunter.py) | Complex multi-feature: RAG + tools + skills + hooks | `python examples/23_swe_bug_hunter.py` |
| 24 | [Config-driven Orchestration](24_config_driven_orchestration.py) | YAML-based multi-agent with specialist routing | `python examples/24_config_driven_orchestration.py` |
| 25 | [Subagent Delegation](25_subagent_delegation.py) | Parallel subagent spawning, lifecycle management | `python examples/25_subagent_delegation.py` |
| 26 | [Task Management](26_task_management.py) | Task tracking, dependencies, structured workflow | `python examples/26_task_management.py` |
| 27 | [Benchmark Suite](27_benchmark_suite.py) | Performance benchmarking across subsystems | `python examples/27_benchmark_suite.py` |
| 28 | [Custom RAG Registry](28_custom_rag_registry.py) | Custom chunker/retriever via registry decorators | `python examples/28_custom_rag_registry.py` |

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
| Multi-provider LLM | 17 |
| Harness (telemetry, doom loop, carryover) | 18, 19, 20 |
| Evaluation suite | 21 |
| Full production (with custom hooks) | 22 |
| SWE Bug Hunter (complex multi-feature) | 23 |
| Subagent delegation (parallel subtasks) | 25 |
| Task management (dependencies, workflow) | 26 |
| Benchmark suite (performance testing) | 27 |
| Custom RAG registry (BM25, WordCount) | 28 |

## Notes

- Examples 02, 12 (standalone part), 19, and 20 **do not require an API key** (config inspection or simulation only)
- All examples use sample data from `data/sample/` (Acme Corp)
- Interactive mode: type `quit` to exit
- For the Anthropic provider, set `ANTHROPIC_API_KEY` in `.env`
- Shared utilities are in `conftest.py` (setup, dual-mode helpers, agent creation)
