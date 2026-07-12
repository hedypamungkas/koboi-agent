---
globs: ["koboi/tools/builtin/**/*.py"]
---

# Builtin tool conventions

- Each tool file uses `@tool()` decorator from `koboi.tools.registry`
- `parameters` is a JSON Schema dict with `type`, `properties`, `required`
- Risk levels: `RiskLevel.SAFE` (default), `RiskLevel.MODERATE`, `RiskLevel.DESTRUCTIVE`
- Tool functions return `str` (the registry calls `str(result)`)
- Sync functions are auto-wrapped in `asyncio.to_thread`
- Register new tools in `__init__.py:register_all()`
- Tool config can be passed via `set_config()` if the module defines it
- Dependency injection: declare `_deps` and `_tool_config` params, injected at call time

## Built-in tools (10)
- `calculator.py` -- math expression evaluator (SAFE)
- `filesystem.py` -- read/write/list files (MODERATE)
- `shell.py` -- execute shell commands (DESTRUCTIVE)
- `web.py` -- web search and fetch (SAFE)
- `memory.py` -- persistent memory store/recall (SAFE)
- `search.py` -- regex search in files (SAFE)
- `git.py` -- git operations (MODERATE)
- `subagent.py` -- spawn parallel sub-agents (MODERATE)
- `task.py` -- structured task management (SAFE)
- `ingest.py` -- fetch a URL + chunk into the live knowledge corpus (MODERATE)
