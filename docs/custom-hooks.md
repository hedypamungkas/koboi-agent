# Custom command hooks (`hooks:`)

koboi lets you declare **external executable hooks in YAML** — a separate script
triggered on a lifecycle event, with **no Python code in the agent**. The command
runs as a subprocess, receives the event context as JSON on stdin, and (when
awaited) can return mutations. This is the Claude-Code / git-hook / pre-commit
model, and it pairs naturally with `uv`/`uvx` for zero-install, portable hooks.

Primary use case: forward every LLM response to a messaging channel (WhatsApp /
Telegram / Slack) without touching agent code.

## YAML schema

```yaml
hooks:
  allow_exec: true                 # REQUIRED gate; default false (default-deny)
  command_timeout: 10              # default per-invocation seconds
  on_event:
    - name: forward-to-whatsapp
      command: ["uvx", "my-wa-forwarder"]   # list -> shell=False; or a shell string
      events: ["post_output", "session_end"] # HookEvent names (see koboi/hooks/chain.py)
      fire_and_forget: true         # default true (observe, zero latency); false = full control
      priority: 60
      timeout: 15                   # per-hook override (else command_timeout)
      abort_on_error: false         # crash/timeout/non-2 -> abort? default false (fail-safe)
      pass_messages: false          # include ctx.messages (can be MB-scale)
      pass_metadata: false          # include ctx.metadata (may be non-serializable)
      cwd: null
```

## Protocol

**koboi → script (stdin, JSON):**
```json
{
  "event": "post_output",
  "iteration": 3,
  "agent": {"model": "gpt-4o-mini", "agent_name": "..."},
  "tool_name": null,
  "tool_arguments": null,      // already a JSON STRING when present -- parse it yourself
  "tool_result": null,
  "user_message": "...",
  "llm_response": {"content": "...", "tool_calls": [{"id":"...","name":"...","arguments":"..."}], "is_complete": true}
}
```
`messages` and `metadata` are included only when `pass_messages` / `pass_metadata` are set.

**script → koboi (stdout JSON + exit code):**
```json
{"abort": false, "inject_message": "...", "inject_messages": ["a","b"], "modified_tool_result": "..."}
```
| exit code | meaning |
|---|---|
| `0` | OK — apply any returned JSON (abort / inject / modified_tool_result) |
| `2` | explicit abort → the action is blocked (Claude-Code convention) |
| other / crash / timeout | error → fail-safe: log + continue (abort only if `abort_on_error: true`) |

> **`tool_arguments` is a JSON string, not an object** — the #1 integration gotcha. koboi sends it raw; your script must `json.loads()` it. Never f-string payload fields into a shell command — read them from stdin.

## `fire_and_forget`

- **`true` (default)** — observe / side-effect hooks (messaging, logging, webhooks).
  Spawned off-loop via `asyncio.to_thread` and **not awaited** → **zero SSE latency**;
  mutations are ignored (the script can't influence the current step). This is the
  right choice for the messaging use case.
- **`false`** — full control. Awaited; the script can `abort` (block the action),
  `inject` messages, or (on `post_tool_use`) `modified_tool_result`. Pays the
  subprocess latency in the hot path — use only when you need it.

Effective control surface today: `abort` works at **pre_input** and **pre_tool_use**;
`inject_message(s)` works at **all events**; `modified_tool_result` works at
**post_tool_use**. (`fire_and_forget: true` hooks can do none of these — they only
produce side effects.)

## Security model (layered)

1. **Default-deny gate** — `allow_exec` defaults to `false`; declared hooks are
   skipped with a warning until you opt in. Prevents "I cloned a config and it ran
   code."
2. **Sandbox isolation** — every command runs through the wired sandbox backend's
   `run()` + `build_env()` (scrubs `*_KEY` / `*_SECRET` / `*_TOKEN` env vars). Pair
   with `sandbox.backend: restricted` for cwd/env/network/rlimit containment.
3. **Fail-safe** — crash / timeout / non-zero exit → continue by default
   (`abort_on_error: true` makes a control hook fail-closed).
4. **Pinning encouraged** — pin `uvx tool@version` / `uv run --with pkg==x`; not
   enforced in v1.

**Known gaps (documented, not hidden):**
- **Network is soft-leaky by design.** The restricted sandbox's default (soft)
  network mode blocks `curl`/`wget`/`nc` tokens but **not** Python interpreters, so
  a `uvx`/`uv run` hook using `httpx`/`requests` has egress. `sandbox.network_isolation:
  seccomp` hard-blocks all egress (Linux-only) → messaging hooks **break**; koboi
  warns at startup when command hooks + seccomp coincide. Domain-allowlisting is the
  real future hardening.
- **macOS**: the seccomp hard layer is Linux-only; on Darwin isolation is cwd/env
  scoping only.

## Server behavior (SSE `/v1/chat/stream` + jobs `/v1/jobs`)

Both paths call `agent.run_stream()` under a per-session lock; hooks fire inline.

- **No blocking**: the sync `sandbox.run` is always offloaded via `asyncio.to_thread`,
  so one slow hook never freezes the event loop or other sessions.
- **No SSE latency** for messaging hooks (`fire_and_forget: true`).
- **Jobs** mandate `sandbox.backend: restricted`; messaging works under the default
  soft mode, breaks under seccomp (see warning above).
- **Resume / idempotency**: side-effecting hooks may **double-fire** on job resume
  or request retry — make your hook idempotent (dedupe by `session_id` + iteration +
  event, or an idempotency key the hook emits).
- **Attacker-controlled data**: `user_message`, `tool_result`, and `tool_arguments`
  are tenant/user-influenced and flow into your script's stdin. **Sanitize.** Prefer
  list-form `command` (shell=False, no interpolation) and pass dynamic data via stdin,
  not argv.

## Pitfalls

- `tool_arguments` is a JSON **string** (parse it; don't re-encode).
- `messages` can be MB-scale → only enable `pass_messages` on low-frequency events.
- `metadata` may hold non-serializable values → koboi filters it; opt in only if needed.
- `shell: true` + `passthrough` sandbox lacks process-group kill on timeout → prefer
  list-form `command` (and the `restricted` backend for reliable timeout kills).
- Fire-and-forget hooks hold strong task refs internally (no GC mid-run); on hard
  process exit a spawned child may be orphaned (bounded by `timeout`).
- Parsed stdout is capped (~64 KiB) before `json.loads` to bound memory.

## Example

See `examples/33_command_hook_messaging.py` (self-contained, mock LLM) and
`examples/_command_hook_forwarder.py` (the external script), plus the illustrative
`configs/command_hook_notify.yaml`.
