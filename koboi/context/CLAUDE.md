# koboi/context/ -- Context window management strategies

## What this is
Shrinks the conversation when it exceeds the token budget. Pluggable strategies
(truncation, smart truncation, key-facts extraction, LLM-summarized sliding window)
share one ABC. Reuses `koboi.rag.registry.ComponentRegistry` for the same
`@register_*` pattern as RAG/guardrails, not a local registry.

## Key files
```
manager.py     ContextManager ABC + NoopContextManager + 4 @register'd strategies + ensure_tool_integrity()
registry.py    @register_context_strategy decorator, build_context() factory, load_custom_context_modules()
__init__.py    Re-exports the 6 manager classes only (no registration side-effect on import)
```

## Extension API -- add a custom strategy
1. Subclass `ContextManager` and implement the two abstract members:
   - `@property _strategy_name(self) -> str` -- uppercase label for logs (e.g. `"MY_STRATEGY"`)
   - `async def _build_result(self, system_msgs: list[dict], non_system: list[dict]) -> tuple[list[dict], str]`
     returns `(kept_messages, log_detail)`
2. Decorate it: `@register_context_strategy("my_strategy", description="...")` (from `koboi.context.registry`).
3. Put the class in a module and list it under YAML `context.custom_modules: [my_pkg.strategies]`
   (the import fires the decorator). `keep_last` / `summarization_truncation` from config are forwarded
   as kwargs automatically; `logger` / `client` are injected only if your `__init__` accepts them.

Do NOT override `manage()`: the concrete base method splits system vs non-system,
calls `_build_result` only when over budget, then runs `ensure_tool_integrity()` on the result.

## How it's wired
- YAML section `context:` → `_build_context()` (facade.py) → `build_context(strategy, ...)`.
- The facade explicitly runs `import koboi.context.manager` before `build_context` so the
  module-level decorators fire. **Importing the package (`__init__.py`) alone does NOT register built-ins.**
- Registry instance: `context_registry = ComponentRegistry("context_strategy")` (imported from `koboi.rag.registry`).
- Factory: `build_context(strategy, *, logger=None, client=None, **kwargs) -> ContextManager | None`.
- The loop calls `context_manager.manage(messages, max_context_tokens)` each iteration and feeds real
  usage back via `context_manager.last_actual_tokens = response.usage.prompt_tokens`.

## Registered strategies
```
truncation        TruncationManager        keep_last=6 -- last N non-system + system prompt
smart_truncation  SmartTruncationManager   keep_last=6 -- system + FIRST user + last N
key_facts         KeyFactsManager          keep_last=4 -- old tool results collapsed into one system msg + recent
sliding_window    SlidingWindowManager     keep_last=4, summarization_truncation=200 -- LLM summary + recent (needs client)
```

## Conventions
- YAML keys: `strategy` (default `"noop"`), `keep_last`, `summarization_truncation`, `custom_modules`, `max_context_tokens` (default 8000).
- `strategy: noop` → facade returns None (no manager); `NoopContextManager` also exists but is not registry-built.
- `manage()` is a passthrough while under budget; it only applies the strategy when `estimate_tokens(messages) > max_tokens`.
- `_effective_tokens()` returns `max(estimate_tokens(messages), self.last_actual_tokens)`.

## Gotchas
- **Unknown / typo'd strategy returns None, not an error**: `build_context` logs a warning and returns
  None, so context management silently disables. The facade short-circuits `noop` before that path.
- **`sliding_window` without a `client` silently skips summarization** (`if old and self.client`) and just
  keeps recent + system -- no summary, no warning.
- **`sliding_window._summarize` swallows all exceptions** (`except Exception`), keeping the previous
  summary -- summarization failures are invisible.
- **`smart_truncation` drops middle messages**: only system + first user + last `keep_last` survive; key
  facts in middle user turns are lost at compaction time.
- **`ensure_tool_integrity` injects synthetic messages** to keep the sequence API-valid: a
  `"[continuing analysis]"` user message is prepended if the first non-system role is not `user`, and
  orphaned tool results / empty `tool_calls` are stripped or repaired.
