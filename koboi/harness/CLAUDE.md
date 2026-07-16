# koboi/harness/ -- Runtime safety, observability, and resume-state primitives

## What this is
Cross-cutting concerns that keep the agent loop safe and observable at runtime:
a policy permission engine, doom-loop detection, secret-hygiene env filtering
for subprocesses, compaction-surviving carryover state, and session telemetry.
Plain dataclasses + sync classes (no async, no plugin registry); driven into the
loop by hooks in `koboi/hooks/` and config sections `policy:` / `harness:`.

## Key files
```
__init__.py       Re-exports PolicyEngine/PolicyRule/PolicyAction, DoomLoop*, CarryoverState, TelemetryCollector, PolicyAudit*, build_safe_env, configure_env_defaults
policy.py         PolicyEngine.evaluate(tool_name, arguments, risk_level) -> PolicyDecision; PolicyAction (ALLOW/CONFIRM/DENY); hardcoded SENSITIVE_PATHS + COMMAND_DENY_PATTERNS (checked first, non-overridable)
policy_audit.py   PolicyAuditLog -- JSONL audit trail; SHA-256-hashed args (first 16 hex); buffered flush (buffer_size=10)
telemetry.py      TelemetryCollector -- session metrics; weighted health_score() (0-100); report()/summary()
carryover.py      CarryoverState -- metadata that survives compaction; to_context_message()/from_context_message() round-trip
doom_loop.py      DoomLoopDetector -- detects consecutive_identical/repeating_pattern/error_retry; record()+check() (sync, stateful, cap 200)
env.py            build_safe_env(tool_config) -> sanitized dict for subprocess env=; block-list WINS over allow-list; configure_env_defaults()
utils.py          is_tool_error(result_str) -- doom-loop/telemetry error detection
```

## How to extend
No ABC or `@register_*` decorator here -- extension is via config + constructor args.

- **Policy rules**: `engine.add_rule(PolicyRule(name=..., action=PolicyAction.CONFIRM, tool_pattern="git_*", risk_levels=[RiskLevel.DESTRUCTIVE], argument_patterns={"command": "push *"}))`. First-match-wins, AFTER the hardcoded deny checks.
- **Doom-loop sensitivity**: `DoomLoopDetector(DoomLoopConfig(consecutive_identical_threshold=3, error_retry_threshold=3, adaptive_threshold=True, task_complexity_hint="complex"))`.
- **Telemetry weighting**: `TelemetryCollector(health_weights={...})` -- keys default to loop_health/tool_success_rate/context_efficiency/compaction_fidelity/permission_friction/doom_penalty; unknown keys fall back via `.get(key, default)`.
- **Env allow/block**: per-tool YAML `tools.defaults.env_allowlist` / `env_blocklist` (globs, case-insensitive); escape hatches `tools.defaults.env_passthrough: true` or `KOBOI_ENV_PASSTHROUGH=1`.

## How it's wired
- `PolicyEngine` is built by `_build_policy(config)` in `facade.py`, handed to `PolicyHook` (priority 25, `PRE_TOOL_USE`) via `hooks/registry.py`. DENY -> `ctx.abort`; CONFIRM -> `policy_needs_confirmation` metadata; decisions logged to `PolicyAuditLog`.
- `build_safe_env()` is the single leak-closing seam shared by `shell.py`, `git.py`, `skills/registry.py` (skill `!cmd` preprocessing), and both sandbox backends. `configure_env_defaults()` (called once from `tools/registry.py`) lets all leak sites share one config.
- `DoomLoopDetector`, `CarryoverState`, `TelemetryCollector` are each owned by their matching hook (`doom_loop_hook`/`carryover_hook`/`telemetry_hook`), instantiated in `hooks/registry.py`.

## Conventions
- Module docstrings: `"""koboi/harness/<file> -- <role>."""`; `from __future__ import annotations` first, then stdlib, then `koboi.*`.
- Dataclasses for all value types (`PolicyRule`, `PolicyDecision`, `DoomLoopConfig`, `TelemetrySnapshot`, `CarryoverState`); sync methods (the loop calls them from async hooks without await).
- Sensitive constants are module-level ALL_CAPS tuples/lists (`SENSITIVE_PATHS`, `COMMAND_DENY_PATTERNS`, `SECRET_BLOCKLIST`, `DEFAULT_ENV_ALLOWLIST`).

## Gotchas
- **Hardcoded safety is non-overridable.** `evaluate()` checks `SENSITIVE_PATHS` + `COMMAND_DENY_PATTERNS` FIRST, before any user rule -- YOLO mode and Trust-DB rules can't bypass them. The deny list blocks interpreter-exec vectors (`python3 -c`, `perl -e`, `bash -c`, `/dev/tcp`, base64-into-shell), not just `curl|bash`.
- **Bypass-resistant via `shlex` tokenization** (issue #45/#46, `_split_tokens`/`_interpreter_deny_reason`/`_rm_deny_reason`/`_sensitive_path_reason` in `policy.py`): scans the WHOLE token stream, not just anchored substrings, so flag-order variants (`python3 -W ignore -c`, `bash -ic`), stdin redirection (`python3 <<< code`), quote-split evasion (`cat /etc/pass''wd`), prefix-anchored globs (`/etc/pass*`), and long-form `rm --recursive --force` are all caught (the old regex-only match missed all of these). Same hardening applies to the shell tool's own `_check_command_blocked` gate.
- **`policy.rules` YAML only matches an arg literally named `command`.** `_build_policy()` (facade.py) hardcodes `argument_patterns={"command": pattern}` for every rule; it never fires on tools whose arg is `path`/`query`/etc. Gate those by building the `PolicyEngine` directly.
- **`evaluate()` takes `arguments` as a STRING** (the JSON-serialized tool args), not a dict -- it `json.loads` internally to match `argument_patterns`.
- **Block-list WINS over allow-list in env filtering.** `KOBOI_DB_TOKEN` matches the `KOBOI_*` allow-glob but is still stripped by `*_TOKEN`. Order: passthrough check -> allow-list keep -> block-list drop.
- **`PolicyAuditLog` hashes args** (SHA-256, 16 hex chars) -- raw args never reach disk; flush at `buffer_size`, call `close()` at session end.
- **`DoomLoopDetector` is sync + stateful** -- one instance per session; `reset()` clears the 200-entry history.
- See main `CLAUDE.md` "Gotchas" for the YOLO-vs-PolicyHook interaction and approval-before-ModeHook ordering.
