# koboi-agent

Configurable AI agent framework. YAML-driven config, async Python 3.10+, multi-provider LLM (OpenAI, Anthropic, Cloudflare).

## Features

- **Multi-provider LLM**: OpenAI, Anthropic, Cloudflare Workers AI
- **YAML-driven config** with `${ENV_VAR}` interpolation
- **Built-in tools**: calculator, filesystem, shell, web search, memory, git, subagent, task
- **Hook lifecycle**: 15 event types for logging, guardrails, telemetry
- **RAG pipeline**: chunking (fixed/sentence/paragraph/semantic), retrieval (keyword/semantic/hybrid), augmentation
- **Guardrails**: input/output validation, rate limiting, approval workflows, policy engine
- **Multi-agent orchestration**: keyword/LLM/hybrid routing, sequential/parallel execution
- **Context management**: truncation, smart truncation, key facts, sliding window
- **MCP** client (stdio + HTTP) and server support
- **Evaluation**: BFCL, GAIA, SWE-bench, RAGAS, DeepEval scorers
- **Terminal UI** (Textual): chat, command palette, diff view, session management

## Quickstart

### Install

```bash
pip install -e ".[dev,tui]"
```

### Set your API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### Run the CLI

```bash
koboi chat configs/simple_chat.yaml
```

### Run programmatically

```python
import asyncio
from koboi import KoboiAgent

async def main():
    async with KoboiAgent.from_config("configs/simple_chat.yaml") as agent:
        result = await agent.run("What is 2 + 2?")
        print(result.content)

asyncio.run(main())
```

## Configuration

Agents are configured via YAML. Key sections:

```yaml
agent:
  name: "my-agent"
  system_prompt: "You are helpful."
  max_iterations: 10
  mode: "chat"  # chat | plan | act | auto

llm:
  provider: "openai"        # openai | anthropic | cloudflare
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
  base_url: "${OPENAI_BASE_URL:}"

tools:
  builtin: [calculator, web_search, memory_store, memory_recall]
  custom:
    - module: "my_tools"

context:
  strategy: "sliding_window"  # noop | truncation | smart_truncation | key_facts | sliding_window
  max_context_tokens: 8000

rag:
  enabled: true
  chunker: "paragraph"       # fixed | sentence | paragraph
  retriever: "keyword"       # keyword | semantic
  top_k: 3
  documents:
    - path: "./data/sample/product_catalog.md"

guardrails:
  input:
    max_length: 10000
  rate_limit:
    max_calls_per_minute: 20

harness:
  doom_loop:
    consecutive_identical_threshold: 3
  telemetry: true
  carryover: true
```

See `configs/` for full examples and `.claude/skills/yaml-config.md` for the complete schema.

## Testing

```bash
pytest                        # all tests
pytest tests/test_config.py   # single file
pytest -k "hook"              # by keyword
pytest --cov=koboi            # with coverage
```

## Examples

`examples/` contains 28 numbered scripts covering every feature:

| Range | Features |
|-------|----------|
| 01-04 | Basic chat and tool use |
| 05-08 | Context management and RAG |
| 09-10 | MCP client/server |
| 11-14 | Policy, hooks, skills, custom tools |
| 15-16 | Multi-agent orchestration |
| 17 | Anthropic provider |
| 18-20 | Harness (telemetry, doom loop, carryover) |
| 21-24 | Evaluation, production setup, SWE-bench, config-driven orchestration |
| 25-28 | Subagent delegation, task management, benchmarks, custom RAG |

Run any example:

```bash
python examples/01_simple_chat.py              # automatic mode
python examples/01_simple_chat.py -m interactive  # interactive mode
```

## Architecture

For a detailed architecture overview (agent loop lifecycle, hook system, tool pipeline, extension points), see **[docs/architecture.md](docs/architecture.md)**.

`KoboiAgent` (`facade.py`) is the single entry point. It assembles:

- **AgentCore** (`loop.py`) -- async agent loop
- **RetryClient** (`client.py`) -- LLM HTTP transport with retry
- **ToolRegistry** (`tools/`) -- tool registration and execution
- **HookChain** (`hooks/`) -- lifecycle event dispatch (15 events)
- **ContextManager** (`context/`) -- context window strategies
- **AugmentationStrategy** (`rag/`) -- RAG pipeline
- **Guardrails** (`guardrails/`) -- input/output validation
- **PolicyEngine** (`harness/`) -- rule-based tool filtering
- **SkillRegistry** (`skills/`) -- skill discovery
- **ModeManager** (`modes.py`) -- chat/plan/act/auto modes
- **TrustDatabase** (`trust.py`) -- graduated permissions
- **Orchestrator** (`orchestration/`) -- multi-agent coordination
- **SubAgentManager** (`subagent.py`) -- parallel sub-agent delegation
- **MCP clients** (`mcp/`) -- external tool servers

All subsystems are configured from a single YAML file via `Config` (`config.py`).

## License

MIT
