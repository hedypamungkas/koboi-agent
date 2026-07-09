# koboi/orchestration/ -- Multi-agent routing and coordination

## What this is
Routes a query to one or more specialist agents, runs them sequentially or in parallel,
optionally revises low-quality answers, and combines results. Can also build specialist
agents on the fly for unknown domains. Each agent is a standalone `AgentCore` (`koboi/loop.py`).

## Key files
```
router.py         BaseRouter ABC + KeywordRouter/LLMRouter/HybridRouter (return RoutingDecision)
orchestrator.py   Orchestrator (seq/parallel/+revision) + QualityEvaluator; run()/run_stream()
factory.py        AgentFactory (builds an AgentCore per agent) + DynamicAgentBuilder
_utils.py         extract_json() -- brace-balanced JSON extraction from LLM text
__init__.py       Re-exports BaseRouter, the 3 routers, Orchestrator, QualityEvaluator, AgentFactory, DynamicAgentBuilder
```

## Extension API -- adding a router
There is NO registry or decorator for routers. To add one:
1. Subclass `BaseRouter` (router.py) and implement `async def route(self, query: str) -> RoutingDecision`.
2. Return `RoutingDecision(query, agents, confidence, method, reasoning, domain_label=None)`.
   `agents` is the list of agent names to dispatch to. `agents` must be non-empty and
   `confidence` in `[0,1]` -- `__post_init__` raises otherwise.
3. Pass the instance to `Orchestrator(client=..., router=<your instance>)`.

Built-in routers:
- `KeywordRouter(agent_defs=None)` -- substring match on `AgentDef.keywords`; broadcasts to all
  known agents at confidence 0.3 when nothing matches.
- `LLMRouter(client, fallback=None, enable_dynamic=True, agent_defs=None)` -- prompts the LLM, parses
  JSON, falls back to KeywordRouter on ANY failure (parse miss, brace-bearing description, network).
  `valid_names` defaults to `{hr,sales,finance}` unless `agent_defs` is given.
- `HybridRouter(client, confidence_threshold=0.5, enable_dynamic=True, agent_defs=None)` -- keyword
  first; when confidence >= threshold it also asks the LLM and merges any domains keyword missed.

## Running agents
- `Orchestrator.run(query, mode="sequential") -> OrchestratorResult`. Modes: `"sequential"`,
  `"parallel"`. With `use_revision=True` + an `evaluator`, execution mode becomes `"<mode>+revision"`.
- `Orchestrator.run_stream(query, mode=...)` -> async generator of events (`RoutingDecisionEvent`,
  `AgentDispatchEvent`, `AgentResultEvent`, `TextDeltaEvent`, `OrchestrationCompleteEvent`).
- `QualityEvaluator(client, threshold=0.6).evaluate(query, answer) -> (score, feedback, needs_revision)`.
- Agent construction: `AgentFactory.create_agent(name, client)` (Acme-Corp hr/sales/finance/general
  only) or config-driven `create_configured_agent(agent_def, ...)` / `create_all_configured(...)`.
- Dynamic agents: set `Orchestrator(enable_dynamic=True)`; the router emits `"dynamic"` and
  `DynamicAgentBuilder` generates a system prompt + retrieves chunks into an `AgentBlueprint`.

## Conventions
- `RoutingDecision`, `AgentResult`, `OrchestratorResult`, `AgentBlueprint`, `AgentDef` live in `koboi/types.py`.
- `RoutingDecision.method` is a `Literal["keyword","llm","hybrid(keyword)","hybrid(llm)","hybrid(keyword+llm)"]`.
- Parallel runs use `asyncio.gather(return_exceptions=True)` / `asyncio.wait(FIRST_COMPLETED)`; a
  failing agent becomes `AgentResult(answer="Error: ...", failed=True)` and never crashes the run.
- Multi-agent answers are synthesized via the orchestrator's shared `client` (LLM call), falling back
  to `=== Answer from <NAME> Agent ===` concatenation on failure.

## Gotchas
- Revision is NOT supported in streaming mode: `run_stream` logs a warning and runs direct when
  `use_revision` is set; the legacy revision path (`_run_with_revision_legacy`) is explicitly non-streamable.
- `KeywordRouter` without `agent_defs` uses a hardcoded Acme-Corp map (hr/sales/finance). For custom
  agents pass `agent_defs` with populated `AgentDef.keywords`, or use `LLMRouter`/`HybridRouter`.
- `LLMRouter._build_prompt` f-string-interpolates agent descriptions into the template, then `.format()`
  runs at route time. A description containing literal `{`/`}` raises and silently falls back to keyword.
- `enable_dynamic` must be True on BOTH the router (to emit `"dynamic"`) and the Orchestrator (to
  resolve it via `DynamicAgentBuilder`); otherwise the target is dropped or unresolved.
- Factory/dynamic agents use the **keyword** retriever only (hardcoded `retriever_registry.get("keyword")`).
- `AgentFactory.create_agent` is Acme-Corp-specific (hardcoded prompts + `koboi.rag.sample_documents`).
  Use `create_configured_agent` for config-driven agents.
- Per-agent LLM client: `create_configured_agent` builds a dedicated client only when `client_builder`
  is supplied AND `agent_def.llm_config` has keys beyond `max_context_tokens` (or is a `providers:` string
  ref); otherwise the shared orchestrator client is reused.
- See `docs/agentic-vs-autonomous-strategy.md`: orchestration here is seq/parallel/+revision only (no DAG).
