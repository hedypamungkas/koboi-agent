# koboi/hooks/ -- Hook system

## What this is
Observer-pattern lifecycle hooks. Every stage of the agent loop emits events that hooks can intercept, modify, or abort.

## Key files
```
chain.py              HookEvent enum, HookContext dataclass, Hook ABC, HookChain
registry.py           Declarative hook registration (HookEntry, build_hook_chain, register_hook)
callback_hook.py      CallbackHook -- wraps a plain function as a Hook
command_hook.py       CommandHook -- YAML-driven external command hook (uv/uvx; JSON over stdio; see docs/custom-hooks.md)
builtin.py            LoggingHook (all events), AuditHook (tool events)
mode_hook.py          Mode-aware tool filtering and system prompt injection
guardrail_hook.py     Input/output guardrail integration
context_hook.py       Context window management trigger
read_before_write_reset_hook.py  P3b tool-state preservation (read-before-write reset on compaction)
skill_persistence_hook.py  Re-injects activated skills after POST_COMPACT (priority 45)
policy_hook.py        Policy engine enforcement
telemetry_hook.py     Telemetry collection
langfuse_hook.py      Langfuse tracing integration
carryover_hook.py     Cross-session state carryover
doom_loop_hook.py     Repeated-action detection and abort
task_hook.py          Task management lifecycle
task_persistence_hook.py   Persists task state across compaction
proactive_extraction_hook.py  SESSION_END auto-extract of durable facts -> KV store (proactive memory D-side; priority 65)
handover_detection_hook.py    Structural handover detection (B1.5; PRE_INPUT user-ask patterns + POST_OUTPUT low A3-grounding coverage) -> sets ctx.metadata['handover_requested']; the emit-site raises AgentHandoverError
subagent_hook.py      Sub-agent lifecycle tracking
notification_hook.py  Notification dispatch
rich_subagent_hook.py Rich TUI sub-agent display
rich_task_hook.py     Rich TUI task display
failure_classifier_hook.py  FailureClassifierHook -- tags ctx.metadata['failure_class'] from the P0-D error_kind + live grounding threshold (priority 5, self-healing P2a)
typecheck_hook.py           TypecheckHook -- parses run_typecheck ruff/mypy/pyright output into ctx.metadata['typecheck_diagnostics'] + refines tool_error_kind to typecheck_failed (priority 4, self-healing Wave 2.4; fail-soft)
ladder_router_hook.py       LadderRouterHook -- picks ONE recovery rung per POST_OUTPUT turn (reflect-if-budget else handover), stamps ctx.metadata['recovery_plan'] (priority 6, self-healing P2a)
reflection_hook.py          ReflectionHook -- verifier-grounded reflection loop: POST_TOOL_USE repeated-error critique + POST_OUTPUT low-grounding reground + optional CRITIC tool-verification (priority 60, self-healing P1/P4)
```

## HookEvent values (15)
```
SESSION_START, SESSION_END, PRE_INPUT, POST_OUTPUT,
PRE_COMPACT, POST_COMPACT, PRE_LLM_CALL, POST_LLM_CALL,
PRE_TOOL_USE, POST_TOOL_USE, DOOM_LOOP_DETECTED,
PRE_ROUTING, POST_ROUTING, AGENT_DISPATCHED, AGENT_COMPLETED
```

## How to create a hook
1. Create `koboi/hooks/<name>_hook.py`
2. Import from `koboi.hooks.chain`: `Hook`, `HookContext`, `HookEvent`
3. Subclass `Hook`, implement `handles() -> list[HookEvent]` and `async execute(ctx: HookContext) -> HookContext`
4. Set `ctx.abort = True` to halt the chain
5. Set `ctx.inject_messages` to inject text into the conversation
6. Register in `koboi/hooks/registry.py` via `HookEntry` for auto-discovery

## How hooks are wired
The module-level `build_hook_chain()` function in `registry.py` creates the `HookChain` from config. Always includes `LoggingHook` (priority 0). Other hooks added based on config predicates.
`AgentCore.run()` calls `hook_chain.emit()` at each lifecycle point.

## Priority ranges
- 0-19: Infrastructure (LoggingHook = 0)
- 20-39: Security (PolicyHook = 25)
- 40-59: Business logic (default = 50)
- 60-79: Post-processing
- 80-100: Cleanup (AuditHook = 80)

## Custom command hooks (`hooks:` YAML section)
Declarative external-command hooks -- no Python in the agent. Declared under the
top-level `hooks:` section (`allow_exec` default-deny gate + `on_event:` entries);
each entry spawns a command per `events` and exchanges JSON over stdio. Wired by
`_build_command_hooks()` in `facade.py` (not via `_REGISTRY`). Security is layered
(allow_exec gate + sandbox isolation + fail-safe). Full protocol + security model +
server-path notes: `docs/custom-hooks.md`. Runnable demo: `examples/33_command_hook_messaging.py`.
