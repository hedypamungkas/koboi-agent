# koboi/orchestration/ -- Multi-agent routing and coordination

## What this is
Routes a query to one or more specialist agents, runs them, and combines results. Execution
modes: `sequential`, `parallel`, `dag` (dependency-ordered, wave-parallel), `conditional`
(output-predicate branching), `dynamic` (an LLM plans the graph per query, plan-or-skip), and
`deep_research` (coverage-gated, cited web research — plan → search → fetch → coverage eval →
drill → synthesize). Optionally revises low-quality answers, and can build specialist agents on the fly for unknown
domains. Each agent is a standalone `AgentCore` (`koboi/loop.py`).

## Key files
```
router.py         BaseRouter ABC + KeywordRouter/LLMRouter/HybridRouter (return RoutingDecision)
orchestrator.py   Orchestrator (sequential/parallel/dag/conditional/dynamic/deep_research) +
                  QualityEvaluator; run()/run_stream(); _run_deep_research loop + _synthesize_research
factory.py        AgentFactory (builds an AgentCore per agent) + DynamicAgentBuilder
remote_proxy.py   RemoteAgentProxy -- an AgentDef with `endpoint: <peer_name>` becomes a proxy that
                  POSTs a peer's /v1/peer/invoke; duck-types as a node (`await node.run(query) -> RunResult`).
                  Graph modes only (seq/parallel/dag/conditional); dynamic/deep_research rebuild local
                  agents per-query so `endpoint` is ignored there. Peer failure -> RunResult(content="Error:...")
dag_scheduler.py  DagScheduler -- topological wave grouping from AgentDef.depends_on; persists a
                  durable graph plan + per-node completion to the `steps` table (graph-cursor-resume primitives)
planner.py        plan_or_skip() + plan_research() -- one LLM call (response_format) decides
                  needs_workflow + extracts the step graph; simple requests skip the workflow.
                  Exports PlanResult / PlanStep (+ search_queries)
research.py       deep_research engine primitives: ResearchBudget (hard caps), SourceStore (numbered
                  [n] citations), ResearchContext (per-run state, journable), CoverageEvaluator
                  (LLM judge), RESEARCH_TOOLS_CONFIG, synthesis/coverage prompts
workflow_graph.py WorkflowGraph -- ergonomic programmatic builder (add_node/add_edge/
                  add_conditional_edges/compile().invoke()), LangGraph-shaped, over DagScheduler + Orchestrator
_utils.py         extract_json() -- brace-balanced JSON extraction from LLM text
__init__.py       Re-exports BaseRouter, the 3 routers, Orchestrator, QualityEvaluator, AgentFactory,
                  DynamicAgentBuilder, DagScheduler, PlanResult, PlanStep, plan_or_skip
```

## deep_research mode (`execution.mode: deep_research`)
Coverage-gated, cited web research (GPT-Researcher shape). `Orchestrator._run_deep_research`:
1. `plan_research` (planner.py) — multi-step queries get a step graph; simple queries take a fast
   direct-answer fallback (`_research_direct_answer`, also persists its answer session-tagged).
2. Per-node DAG waves — each node is an `AgentCore` with `web_search`/`web_fetch` wired to the
   configured `websearch` providers, wrapped in `CountingProvider` budget proxies. Findings flow
   into a `SourceStore` as numbered citations `[n]`.
3. `CoverageEvaluator` — one LLM judge/round → `(score, follow_ups, coverage_map)`. Iterates
   (re-plans on gaps) until `coverage_threshold` or `max_depth`/budget. Deterministic safety net:
   generates generic follow-ups if the judge returns none on low coverage (prevents shallow stops).
4. `_synthesize_research` → cited report; `_verify_citations` strips unresolvable `[n]`; Sources footer.
5. Persistence — `ResearchContext` (query + sources + coverage + `final_report`) journaled to the
   `research_context` SQLite table (session-tagged) → `GET /v1/sessions/{id}` + `koboi run --resume`.

Orchestrator `__init__` deep_research params: `research` (caps/threshold dict), `websearch_conf`
(provider config), `sandbox` (node tool dep), `session_id` (tags persisted rows). Production
quality bar + smoke scenarios: `docs/deep-research-smoke.md`.

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
- `Orchestrator.run(query, mode="sequential") -> OrchestratorResult`. Modes:
  - `"sequential"` / `"parallel"` -- run the routed agents in order, or via `asyncio.gather`.
  - `"dag"` -- dependency-ordered, wave-parallel. Edges come from `AgentDef.depends_on`;
    `dag_scheduler.waves()` groups nodes into topological levels (parallel within a level,
    sequential across levels). Pass `dag_scheduler=DagScheduler(agents_map, deps=...)`. With
    `full_graph=True` the whole configured graph runs instead of the routed subset.
  - `"conditional"` -- `dag` plus output-predicate branching: `dag_scheduler.conditionals` maps a
    source to `[{to, when: {contains|regex: value}}]`; only matching branches run.
  - `"dynamic"` -- `planner.plan_or_skip()` makes one LLM call: simple requests answer directly
    (`needs_workflow=False`); multi-step requests get an extracted step graph run as dag waves.
    `max_replans` re-plans on node failure.
  - With `use_revision=True` + an `evaluator`, sequential/parallel becomes `"<mode>+revision"`.
- `Orchestrator.run_stream(query, mode=...)` -> async generator of events (`RoutingDecisionEvent`,
  `AgentDispatchEvent`, `AgentResultEvent`, `TextDeltaEvent`, `OrchestrationCompleteEvent`).
- `QualityEvaluator(client, threshold=0.6).evaluate(query, answer) -> (score, feedback, needs_revision)`.
- Agent construction: `AgentFactory.create_agent(name, client)` (Acme-Corp hr/sales/finance/general
  only) or config-driven `create_configured_agent(agent_def, ...)` / `create_all_configured(...)`.
- Dynamic agents: set `Orchestrator(enable_dynamic=True)`; the router emits `"dynamic"` and
  `DynamicAgentBuilder` generates a system prompt + retrieves chunks into an `AgentBlueprint`.
- Programmatic graph (no YAML, no planner): `WorkflowGraph().add_node(...).add_edge(...).compile().invoke(query, client)`.

## Conventions
- `RoutingDecision`, `AgentResult`, `OrchestratorResult`, `AgentBlueprint`, `AgentDef` live in `koboi/types.py`.
- **Determinism / structured output (workflow export)**: `orchestration.determinism`
  (`temperature`/`seed`/`top_p`/`model_pin`/`replay_mode`) is a workflow-level default merged per-node via
  `DeterminismProfile.merge` (node wins) in `facade._parse_agent_defs`; per-node `output_schema` +
  `force_response_format_with_tools` set structured output (Gap A+B). See `koboi/workflows/CLAUDE.md`.
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
- Remote nodes (A2A): `AgentDef.endpoint: <peer_name>` makes a node a REMOTE peer agent --
  `create_configured_agent` returns a `RemoteAgentProxy` (no local client/rag/tools; the peer runs those)
  that POSTs the peer's `/v1/peer/invoke`. Applies to seq/parallel/dag/conditional only (dynamic/deep_research
  rebuild local agents per-query, so `endpoint` is ignored). Peer failure -> `RunResult(content="Error:...")`;
  the orchestrator detects it via the `"Error:"` prefix sniff. The peer registry is threaded from `peers:` config.
- Execution modes beyond seq/parallel: `dag` (dependency-ordered, wave-parallel via `DagScheduler`
  from `AgentDef.depends_on`), `conditional` (output-predicate branching via
  `dag_scheduler.conditionals`), `dynamic` (LLM `plan_or_skip` extracts the graph per query). Demos:
  `dag_demo.yaml`, `conditional_demo.yaml`, `dynamic_demo.yaml`; render any with `koboi graph <config>`.
- DAG durability: `dag_scheduler` writes `graph_plan` + `graph_node_complete` rows to the `steps`
  table (graph-cursor-resume primitives). The plain seq/parallel path is not journaled, so
  `koboi run --resume` does not apply to sequential/parallel orchestration.
