# koboi/sandbox/ -- Pluggable subprocess/fs isolation backends

## What this is
Isolation layer for tool execution (shell, filesystem, git). A sandbox anchors tool
paths to a workdir and can contain subprocesses (cwd/env/PATH/network/rlimits). Driven
by the `sandbox:` YAML section; uses the ComponentRegistry pattern shared with other
koboi subsystems. Default `passthrough` preserves pre-P0b behavior; opt into
`restricted` for serving/jobs.

## Key files
```
base.py        BaseSandbox ABC + SandboxResult dataclass (returncode/stdout/stderr/timed_out)
passthrough.py PassthroughBackend (name="passthrough") -- no isolation; behavior-preserving default
restricted.py  RestrictedProcessBackend (name="restricted") -- cwd/env/PATH/network/rlimit containment; validate_path()
registry.py    ComponentRegistry: @register_sandbox(name, description) decorator + build_sandbox(conf) + register_builtin_sandboxes()
__init__.py    Re-exports register_sandbox, build_sandbox, BaseSandbox; calls register_builtin_sandboxes() at import
```

## How it's wired
- `sandbox:` YAML → `_build_sandbox()` in the facade → `build_sandbox(conf)` → registry
  lookup by `backend` name → instance. Subprocess tools (`run_shell`, `git_*`,
  filesystem) declare `deps=["sandbox"]` and read `_deps["sandbox"]`; the facade always
  wires a (passthrough-or-better) sandbox.
- A **typo'd backend** raises `ValueError` (fail-loud) rather than silently falling
  back to passthrough. Absent/empty/`passthrough` → passthrough.

## Restricted backend (`restricted.py`)
- **Path containment**: `validate_path(path)` anchors relative paths to the workdir and
  rejects anything resolving outside it (`PermissionError`). Defense-in-depth:
  `session_id` is validated at the server route boundary AND in `workdir_for()`.
- **Network** (three layers):
  - *Soft deny* (default): token-scan deny of obvious egress binaries (`curl`/`wget`/etc. via
    `network_binaries`). SOFT -- does NOT block interpreters (`python3 -c 'import urllib'`)
    or shell builtins (`bash /dev/tcp`). NOTE: pip/npm/git are NOT in the deny scan --
    under plain `deny` they fetch from anywhere.
  - *Soft allowlist* (Wave 3, `network: allowlist` + `network_allowlist: [host globs]`):
    scanned binaries (the deny set PLUS `pip`/`pip3`/`npm`/`pnpm`/`yarn`/`git`) may only
    reference allowlisted hosts (`scheme://host` URLs + `git@host:` SSH forms); violations
    return rc 126. **Intent-limiting, NOT enforcement**: it constrains hosts WRITTEN in the
    command (`--index-url http://evil`, `git clone https://evil`) -- a command with no host
    tokens passes (default-index fetches are unscannable), and interpreters still bypass it.
    Realistic host set for a coding job: `pypi.org`, `files.pythonhosted.org`,
    `registry.npmjs.org`, `github.com`, `codeload.github.com`, `*.githubusercontent.com`.
    Proxy env vars are stripped like under deny. `sandbox.network` values are fail-closed
    (`allow`/`deny`/`allowlist`; a typo raises at config load -- previously any non-"deny"
    string silently meant allow). Hard per-destination enforcement awaits a container/netns
    backend.
  - *Hard* (`network_isolation: seccomp` | `seccomp_strict`): syscall-layer deny of
    `connect`/`sendto`/`sendmsg`/`sendmmsg` via a seccomp filter applied in the
    child (preexec_fn) that persists across execve. Blocks interpreters + builtins too.
    Linux-only + requires the `python3-seccomp` system package (`apt install python3-seccomp`
    (Debian/Ubuntu)); gated by `_HAS_SECCOMP`. The installer is FAIL-CLOSED (issue #51): a
    per-syscall `add_rule` failure or unresolvable deny action raises rather than loading a
    default-ALLOW filter. Two modes:
    - `seccomp` (legacy, back-compat): if unavailable, degrades to soft with a one-time
      warning. `server_deploy.yaml` / `e2e_full.yaml` enable this by default.
    - `seccomp_strict` (opt-in): if unavailable, RAISES at construction (fail-closed at boot)
      -- use when the operator has arranged Linux + bindings (or the Dockerfile build stage)
      and wants the sandbox to refuse boot rather than silently degrade.
- **rlimits**: optional cpu/as_mb/fsize_mb/nofile caps (applied in the child via preexec_fn;
  the seccomp filter shares the same preexec_fn, which is now built whenever rlimits OR
  seccomp is active -- not rlimits-only).
- **workdir strategy**: `shared` or `per_session` (the server uses per-session:
  `{workspace_root}/{session_id}`, GC'd at `server.workdir_ttl_seconds`).

## Conventions / gotchas
- `sandbox:` is the YAML section; `KOBOI_SANDBOX_DIR` is honored as a back-compat fallback.
- **Autonomous jobs require `restricted`** (`passthrough` refused at job execution -- C3).
- `register_sandbox` is the extension point for custom backends (e.g. a future Docker/container
  backend). Backends implement `run()`, `validate_path()`, `build_env()` (see BaseSandbox ABC).
- **In-process HTTP tool calls bypass all three network tiers** (Wave 4): `web_fetch`,
  `call_peer_agent`, and `github_*` make their outbound request via `httpx` directly in Python,
  never through a subprocess -- the sandbox's `run()`-based scanning never sees them. Trust for
  these is config-level (operator-set `api_base`/`peers:`), not the subprocess network scanner.
