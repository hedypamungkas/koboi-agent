# koboi/tools/builtin/ -- Built-in tool implementations

## What this is
The 11 shipped tools, each a `@tool()`-decorated function in its own module and registered by
`register_all()` (`__init__.py`, called from the facade). Sync tools run in a thread via
`asyncio.to_thread`; the registry calls `str(result)` on every return. See the parent
`koboi/tools/CLAUDE.md` for the registry, `@tool()`, `RiskLevel`, and dependency-injection mechanics.

## Tools
```
calculator.py   calculate                                SAFE        math expression evaluator
filesystem.py   list_files / read_file / write_file / delete_file   MODERATE   read/write/list/delete files
shell.py        run_shell                                DESTRUCTIVE execute shell commands
web.py          web_search / web_fetch                   SAFE        web search + fetch (backends in koboi/websearch/)
memory.py       memory_store / memory_recall             SAFE        persistent KV memory
search.py       grep_search / glob_find                  SAFE        regex/glob search in files
git.py          git_status / git_log / git_diff          MODERATE    git operations
subagent.py     delegate_tasks                           MODERATE    parallel sub-agent delegation
task.py         task_create / task_list / task_get / task_update / task_add_dependency   SAFE   structured task management
ingest.py       ingest_url                               MODERATE    fetch a URL + chunk into the live corpus (W3; needs rag.live + a fetch provider)
handover.py     transfer_to_human                        SAFE        yield the conversation to a human operator (B1; raises AgentHandoverError)
```

## Conventions
- One tool (or one cohesive group) per module; `register_all(registry)` imports + registers them all.
- `parameters` is an inline JSON-Schema dict (`type`/`properties`/`required`); the handler returns `str`.
- Risk levels: `SAFE` (default, skips approval) / `MODERATE` / `DESTRUCTIVE`.
- Dependency injection: declare `deps=[...]` on `@tool()` and a `_deps: dict` param; injected at
  call time. `ingest_url` uses `deps=["fetch_provider", "live_corpus"]`; filesystem/shell/git use
  `deps=["sandbox"]`.

## Gotchas
- **`transfer_to_human` does not pause on a Future** -- it raises `AgentHandoverError`, which
  propagates out of the run so `pool.session_lock` releases; the server turns it into a
  `HandoverEvent` / `awaiting_human` status. Awaiting a Future would deadlock the human's next
  `/chat/stream`.
- **`ingest_url` needs `rag.live: true`** -- the facade wires the `LiveCorpus` + `fetch_provider`
  deps only then; without it the tool returns an error string.
- **Mode-blocking is name-keyed** (`modes.py`): non-read-only tools are blocked in chat/plan unless
  allowlisted in `mode.read_only_tools` or the agent runs in act+.
