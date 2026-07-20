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

## Built-in tools (17)
- `calculator.py` -- math expression evaluator (SAFE)
- `filesystem.py` -- list/read/write/edit/patch/delete files (`list_files`/`read_file` SAFE; `write_file`/`edit_file`/`apply_patch`/`delete_file` DESTRUCTIVE, idempotent=False). `edit_file` = exact-string replace (unique match or `replace_all`), atomic temp+`os.replace` write; `apply_patch` = unified-diff patch (single file, multi-hunk, content-matched so `@@` line drift is tolerated, all-or-nothing atomic swap; parser in `_patch.py`); `read_file` takes optional `offset`/`limit` (1-based line range, numbered output)
- `shell.py` -- execute shell commands (DESTRUCTIVE)
- `web.py` -- web search and fetch (SAFE)
- `memory.py` -- persistent memory store/recall (SAFE)
- `search.py` -- regex search in files (SAFE)
- `git.py` -- git operations: `git_status`/`git_log`/`git_diff` (SAFE, read-only, on the chat/plan allowlist) + Wave-3 write tools `git_add`/`git_commit`/`git_checkout` (MODERATE; commit is idempotent=False with a `-c user.name/email` identity fallback) and `git_push` (DESTRUCTIVE, idempotent=False, no force flag)
- `subagent.py` -- spawn parallel sub-agents (MODERATE)
- `task.py` -- structured task management (SAFE)
- `ingest.py` -- fetch a URL + chunk into the live knowledge corpus (MODERATE)
- `handover.py` -- `transfer_to_human`, yield the conversation to a human operator (SAFE)
- `media.py` -- `generate_image`/`generate_video`/`generate_music`/`generate_speech`/`transcribe_audio` + async `submit_media_job`/`check_media_job` (MODERATE/DESTRUCTIVE; opt-in `media:`)
- `peer.py` -- `call_peer_agent`: cross-instance A2A fan-out via POST to a peer's `/v1/peer/invoke` (SAFE; needs `peers:` config + `peer_registry` dep)
- `repo_map.py` -- `repo_map`: directory tree + best-effort symbol outline (SAFE, read-only, on the chat/plan allowlist). Python files get real `ast.parse`-based top-level function/class signatures; other languages get a best-effort (not language-aware) regex scan. Bounded by `max_depth`/`max_entries`; skips `.git`/`node_modules`/`__pycache__`/etc.
- `github.py` -- GitHub PR tooling: `github_create_pr`/`github_update_pr` (DESTRUCTIVE, idempotent=False) + `github_list_prs`/`github_get_pr` (SAFE, read-only, on the chat/plan allowlist). In-process `httpx` client (needs `github: {enabled: true, token: ${GITHUB_TOKEN:}}`); bypasses sandbox network tiers (like `web_fetch`); token must never ride subprocess env.
- `background_shell.py` -- `submit_background_shell` (DESTRUCTIVE, idempotent=False) + `check_background_shell`/`kill_background_shell` (SAFE/MODERATE). Opt-in via `agent.background_shell.enabled`. Live process runs OUTSIDE the approval/policy pipeline for its whole lifetime once started -- `max_lifetime_seconds` (default 1800) is the mitigating cap. In-memory job registry (`koboi/harness/background_shell.py`), not durable across a restart.
- `typecheck.py` -- `run_typecheck`: run ruff/mypy/pyright on a validated path (SAFE, read-only, on the chat/plan allowlist). Fixed checker allowlist -- NEVER a user-supplied command (no injection surface, unlike `run_shell`); path is `shlex.quote`-d. Non-zero output is prefixed `[exit code: N]` (same token as `run_shell` so the Wave 2.3 signal still works as a fallback). When `self_healing.enabled`, `TypecheckHook` (`koboi/hooks/typecheck_hook.py`, priority 4) parses the output into structured `{file,line,severity,message}` diagnostics on `ctx.metadata["typecheck_diagnostics"]` and refines `error_kind` to `typecheck_failed`; ReflectionHook then names the first failing file:line. Override checker via `tools.overrides.run_typecheck.checker` or the per-call `checker` arg.
