"""koboi/hooks/langfuse_hook.py -- Langfuse tracing via Hook system.

Implements LangfuseTracingHook that sends trace/span/generation data to a
Langfuse server. Uses the existing Hook system -- zero changes to harness or loop.

Compatible with Langfuse SDK v2 + Server v2.

Fail-open: if langfuse SDK or credentials are missing, hook becomes no-op.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from koboi.hooks.chain import Hook, HookContext, HookEvent

_logger = logging.getLogger(__name__)

_LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse
    _LANGFUSE_AVAILABLE = True
except ImportError:
    pass


class LangfuseTracingHook(Hook):
    """Hook that sends tracing data to a Langfuse instance (SDK v2 API)."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        session_id: str | None = None,
        environment: str | None = None,
        release: str | None = None,
    ):
        self._public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        self._secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")
        self._base_url = base_url or os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3300")
        self._session_id = session_id
        self._environment = environment or os.environ.get("LANGFUSE_ENV", "local")
        self._release = release

        self._client: Any = None
        self._trace: Any = None
        self._trace_id: str | None = None
        self._spans: dict[str, Any] = {}
        self._generations: dict[str, Any] = {}
        self._timings: dict[str, float] = {}

        if _LANGFUSE_AVAILABLE and self._public_key and self._secret_key:
            try:
                self._client = Langfuse(
                    publicKey=self._public_key,
                    secretKey=self._secret_key,
                    host=self._base_url,
                )
            except TypeError:
                self._client = Langfuse(
                    public_key=self._public_key,
                    secret_key=self._secret_key,
                    host=self._base_url,
                )

    @property
    def trace_id(self) -> str | None:
        return self._trace_id

    @property
    def available(self) -> bool:
        return self._client is not None

    def get_client(self) -> Any:
        """Return the underlying Langfuse client, or None if unavailable."""
        return self._client

    @staticmethod
    def from_env() -> LangfuseTracingHook:
        return LangfuseTracingHook()

    def handles(self) -> list[HookEvent]:
        return list(HookEvent)

    async def execute(self, ctx: HookContext) -> HookContext:
        if not self._client:
            return ctx
        try:
            self._dispatch(ctx)
        except Exception as e:
            _logger.warning("Langfuse dispatch error: %s", e)
        return ctx

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                _logger.warning("Langfuse flush error: %s", e)

    # --- Dispatch ---

    def _dispatch(self, ctx: HookContext) -> None:
        handler = {
            HookEvent.SESSION_START: self._on_session_start,
            HookEvent.SESSION_END: self._on_session_end,
            HookEvent.PRE_LLM_CALL: self._on_pre_llm_call,
            HookEvent.POST_LLM_CALL: self._on_post_llm_call,
            HookEvent.PRE_TOOL_USE: self._on_pre_tool_use,
            HookEvent.POST_TOOL_USE: self._on_post_tool_use,
            HookEvent.PRE_COMPACT: self._on_pre_compact,
            HookEvent.POST_COMPACT: self._on_post_compact,
            HookEvent.DOOM_LOOP_DETECTED: self._on_doom_loop,
        }.get(ctx.event)
        if handler:
            handler(ctx)

    # --- Session lifecycle ---

    def _on_session_start(self, ctx: HookContext) -> None:
        self._trace = self._client.trace(
            name="Agent Run",
            sessionId=self._session_id,
            metadata={"environment": self._environment, "release": self._release},
        )
        self._trace_id = getattr(self._trace, 'trace_id', None) or getattr(self._trace, 'id', None)

    def _on_session_end(self, ctx: HookContext) -> None:
        self.flush()
        self._trace = None
        self._spans.clear()
        self._generations.clear()
        self._timings.clear()

    # --- LLM calls ---

    def _on_pre_llm_call(self, ctx: HookContext) -> None:
        if not self._trace:
            return
        key = f"llm_{ctx.iteration}"
        self._timings[key] = time.time()
        parent = self._get_or_create_iter_span(ctx.iteration)
        self._generations[key] = parent.generation(
            name=f"LLM Call #{ctx.iteration + 1}",
            input=_truncate(ctx.messages, 4000),
        )

    def _on_post_llm_call(self, ctx: HookContext) -> None:
        key = f"llm_{ctx.iteration}"
        gen = self._generations.pop(key, None)
        if not gen:
            return
        self._timings.pop(key, None)
        usage = None
        model = None
        output = ""
        if ctx.llm_response:
            resp = ctx.llm_response
            output = resp.content or ""
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
            if ctx.agent is not None:
                model = ctx.agent.model
        gen.end(
            output=_truncate(output, 2000),
            usage=usage,
            model=model,
        )

    # --- Tool calls ---

    def _on_pre_tool_use(self, ctx: HookContext) -> None:
        if not self._trace:
            return
        key = f"tool_{ctx.tool_name}_{ctx.iteration}"
        self._timings[key] = time.time()
        parent = self._get_or_create_iter_span(ctx.iteration)
        self._spans[key] = parent.span(
            name=f"Tool: {ctx.tool_name}",
            input=ctx.tool_arguments,
        )

    def _on_post_tool_use(self, ctx: HookContext) -> None:
        key = f"tool_{ctx.tool_name}_{ctx.iteration}"
        span = self._spans.pop(key, None)
        if not span:
            return
        self._timings.pop(key, None)
        span.end(output=(ctx.tool_result or "")[:2000])

    # --- Compaction ---

    def _on_pre_compact(self, ctx: HookContext) -> None:
        pass

    def _on_post_compact(self, ctx: HookContext) -> None:
        pass

    # --- Doom loop ---

    def _on_doom_loop(self, ctx: HookContext) -> None:
        if not self._trace:
            return
        self._trace.event(
            name="Doom Loop Detected",
            metadata={"iteration": ctx.iteration},
        )

    # --- Helpers ---

    def _get_or_create_iter_span(self, iteration: int) -> Any:
        key = f"iter_{iteration}"
        if key not in self._spans and self._trace:
            self._spans[key] = self._trace.span(
                name=f"Iteration {iteration + 1}",
            )
        return self._spans.get(key, self._trace)


def _truncate(data: Any, max_len: int) -> Any:
    if isinstance(data, str) and len(data) > max_len:
        return data[:max_len] + "..."
    if isinstance(data, list):
        return [_truncate(item, max_len) for item in data]
    return data
