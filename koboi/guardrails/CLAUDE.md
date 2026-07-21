# koboi/guardrails/ -- Content safety, rate limiting, approval, and audit

## What this is
Safety layer around the agent loop: validates user input (prompt-injection patterns),
screens agent output (secret/key leaks), rate-limits tool calls, gates destructive
tools behind human/autonomous approval, and records an audit trail. Driven by the
`guardrails:` YAML section (input/output slots); approval handlers and audit trails are
wired by the facade and the server.

## Key files
```
base.py           BaseGuardrail ABC (async check) + PatternGuardrail (regex-driven base)
input.py          InputGuardrail -- prompt-injection (16 patterns, incl. Bahasa Indonesia) + length check (action "block")
output.py         OutputGuardrail -- secret/PII leak screen (action "warn")
grounding.py       GroundingGuardrail -- runtime faithfulness (claim-decompose + NLI vs retrieved context; action "abstain"; A3; fail-soft)
scope.py          ScopeGuardrail -- output scope guard (relevance-gated LLM judge keeps a specialized agent on-task; action "abstain"; fail-soft)
rate_limiter.py   RateLimiter -- per-session/per-tool/per-minute caps (RateLimitConfig)
audit.py          AuditTrail (in-memory) + SQLiteAuditTrail (WAL-persistent)
approval.py       ApprovalHandler base + CLI/Callback/AsyncCallback/Autonomous handlers
approval_types.py ApprovalRequest/Response/Outcome + ApprovalCallback contract
registry.py       GuardrailRegistry (factory map) + register_builtin_guardrails()
tui_approval.py   Back-compat re-export of TUIApprovalHandler from koboi.tui.approval
__init__.py       Re-exports public surface; calls register_builtin_guardrails() at import
```

## Extension API
- ABC: `BaseGuardrail` with `async check(self, content: str) -> GuardrailResult` (base.py).
  `GuardrailResult` (from `koboi.types`): `passed: bool`, `reason: str`, `action: str`
  (default `"block"`), `sanitized_content: str | None`.
- For regex guardrails, subclass `PatternGuardrail` and set `PATTERNS`
  (list of `(regex, description)`) + `DEFAULT_ACTION`; `check_patterns()` returns a
  result on the first match.
- Registration is a METHOD CALL (no decorator): `GuardrailRegistry.register("name", lambda **kw: MyGuardrail(**kw))`. Build from config via
  `GuardrailRegistry.from_config([{"name": ..., ...}])`.
- Built-in factories (registered at import): `injection_detector` (InputGuardrail),
  `content_filter` (OutputGuardrail), `grounding_check` (GroundingGuardrail — opt-in runtime
  faithfulness, Wave 2 A3), `scope_check` (ScopeGuardrail — opt-in output scope guard).
  `RateLimiter`/`AuditTrail` are NOT registered -- they are constructed directly by the facade.

## Conventions
- `check()` is async on guardrails; `should_approve()` may be sync (`bool`) OR async
  (`Awaitable[bool]`) -- the pipeline awaits when it is a coroutine function.
- Approval handlers audit each decision to an optional `audit_trail` set on the instance.
- `GuardrailResult.action`: `block`/`deny`/`abort` raise `AgentGuardrailError`; `warn`
  (or absent) prepends a warning and continues. Input guardrails raise on any
  `passed=False` regardless of action.

## Gotchas
- **No registration decorator** -- unlike sandbox/context/RAG, guardrails use a plain
  method call (`GuardrailRegistry.register`), not `@register_*`.
- **`AsyncCallbackApprovalHandler` and `AutonomousApprovalHandler` are NOT in
  `__init__.py`** -- import them directly from `koboi.guardrails.approval`.
- **`AsyncCallbackApprovalHandler` is fail-closed**: timeout (120s default) or callback
  error denies the tool call.
- **`AutonomousApprovalHandler` denies destructive tools by default** unless a Trust DB
  allow-rule matches; it uses a job-scoped `auto_approve_tools` set rather than seeding
  the Trust DB (the DB is shared across pooled agents, so a seeded rule would leak to
  chat sessions). Autonomous jobs additionally require `sandbox.backend='restricted'`
  (passthrough refused at execution).
- **Output screening buffers streamed tokens** -- when any output guardrail is
  configured, `run_stream` holds TextDeltas until `_process_output` passes (loop.py).
- **`ScopeGuardrail` is relevance-gated, not always-on**: a free deterministic pre-pass
  (`_looks_suspicious`) decides whether to spend an LLM judge call at all -- normal replies
  cost 0 extra calls; only flagged replies get classified `ON_SCOPE`/`OFF_SCOPE`/`INJECTION`.
  `OFF_SCOPE`/`INJECTION` swaps the reply for `deflection_text` (same `action="abstain"` swap
  `GroundingGuardrail` uses). Opt-in `fail_closed: true` routes judge-unavailable/error to
  `action="handover"` instead of fail-soft pass. `last_verdict` feeds the audit trail and
  `RunResult.metadata["verdict"]`.
- **Input `reason` is surfaced in the raised error**; output `reason` is logged
  server-side only (the error carries just the guardrail name) so a leaky reason cannot
  re-leak via the error frame.
- See `koboi/server/CLAUDE.md` for the autonomous-job approval path.
