# koboi/tools/ -- Tool registry, @tool() decorator, dependency injection

## What this is
The tool layer. `ToolRegistry` holds tool definitions + handlers, exposes them to the
LLM as OpenAI function specs, and executes calls (sync tools run in a thread via
`asyncio.to_thread`; async tools awaited). 11 built-ins live in `builtin/` -- see
`.claude/rules/builtin-tools.md` for those. THIS doc covers the registry, the `@tool()`
decorator, `RiskLevel`, and dependency injection.

## Key files
```
registry.py   ToolRegistry + @tool() decorator + register_decorated + apply_tool_selection + truncate_text
state.py      ToolState -- per-session mutable state (read-before-write path set)
__init__.py   Re-exports ToolRegistry, tool, register_decorated
builtin/      11 built-in tools (calculator/filesystem/shell/web/memory/search/git/subagent/task/ingest/handover) -- see builtin/CLAUDE.md
```

## Public surface (verified against registry.py / types.py)
- `ToolRegistry(default_timeout=None)`:
  - `register(name, description, parameters, fn, risk_level=SAFE, timeout=None, group=None, idempotent=True)`
  - `get_definition(name) -> ToolDefinition | None`
  - `list_tools() -> dict[str, ToolDefinition]`
  - `get_definitions() -> list[dict]` -- OpenAI function-spec list (filtered by active groups)
  - `get_risk_level(name) -> RiskLevel | None`; `__contains__(name)` membership check
  - `execute(name, arguments_json) -> str` -- async; always calls `str(result)`
  - `set_dep(name, value)` / `get_dep(name)` -- dependency store
  - `set_tool_config(defaults, overrides)` / `get_tool_config(tool_name)` -- merged config
  - `set_active_groups(groups)`, `keep_only(names)`, `disable(names)` -- selection
- `@tool(name, description, parameters, risk_level=SAFE, timeout=None, group=None, deps=None, idempotent=True)`
  attaches `fn._tool_def` (a `ToolDefinition` incl. the `idempotent` flag) + `fn._tool_deps`; does NOT register by itself.
  Set `idempotent=False` for side-effecting tools that must not silently double-fire on crash-resume.
- `register_decorated(registry, module)` -- scans a module for `_tool_def`-tagged functions,
  wraps them (injecting `_deps` / `_tool_config`), then registers them.
- `apply_tool_selection(registry, tools_config)` -- single chokepoint for
  `tools.defaults` / `overrides` / `disabled` / `groups`; shared by facade + orchestration.
- `RiskLevel` (defined in `koboi.types`): `SAFE` / `MODERATE` / `DESTRUCTIVE`.

## How to add a custom tool
1. Decorate a function with `@tool(...)`; supply an inline JSON-Schema `parameters` dict and
   a `risk_level`. The function must return `str`.
2. (Optional) Set `deps=["sandbox", ...]` and/or declare `_deps: dict` and `_tool_config: dict`
   params on the function -- `_wrap_with_deps` injects them at call time from the registry.
3. Register via `register_decorated(my_registry, my_module)` (built-ins: `builtin.register_all(registry)`).
4. Configure/select via `apply_tool_selection(registry, tools_config)`.

## Conventions
- Tool functions return `str`; the registry calls `str(result)` on every return.
- Sync tools run via `asyncio.to_thread`; async tools are awaited directly.
- Per-tool timeout: `@tool(timeout=...)` overrides `ToolRegistry(default_timeout=...)`.
- Group-tagged tools (`group="file"`) can be hidden from the LLM with `set_active_groups([...])`
  while remaining executable.

## Gotchas
- **MCP tools default to `RiskLevel.SAFE` but are NOT forced SAFE**: `register_mcp_tools()`
  (`mcp/base.py`) accepts a per-server `risk_level` and a per-tool `risk_resolver`. Elevate a whole
  server via `mcp.servers[].risk_level: moderate|destructive` (G3), or opt into name-based per-tool
  inference with `risk_heuristic: true` (`default_risk_heuristic`). See `koboi/mcp/CLAUDE.md`.
- **Unknown args are silently stripped**: `execute()` keeps only keys present in
  `parameters.properties`; an arg missing from the schema never reaches the handler.
- **`policy.rules` arg matching is name-keyed**: a rule's `argument_patterns` keys must equal
  the tool's actual arg name (it does a substring scan of the JSON, then fnmatch); a rule keyed
  on `command` silently no-matches a tool whose arg is named `cmd`.
- **Per-tool override aliasing is 1:1 only**: `shell`->`run_shell` is aliased; `git` is N:1
  (git_status/git_log/git_diff), so `tools.overrides.git` is ambiguous and dropped.
- **`disable()` is the real denylist** (removes from both LLM view AND execution);
  `set_active_groups()` only hides from the LLM. Use `disabled: [...]` to truly drop a tool.
- **`ToolState` must be injected per-session**: call `set_dep("tool_state", ToolState())`; without
  it the filesystem tool falls back to a module-global read-path set shared across sessions.
