"""Tests for koboi tracing (LangfuseTracingHook)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koboi.hooks.chain import HookContext, HookChain, HookEvent


class TestLangfuseTracingHook:
    def test_hook_is_noop_without_sdk(self):
        """Without langfuse installed or credentials, hook should be a no-op."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="", secret_key="")
        assert hook.available is False

    async def test_execute_returns_ctx_unchanged(self):
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="", secret_key="")
        ctx = HookContext(event=HookEvent.PRE_LLM_CALL, iteration=0, messages=[])
        result = await hook.execute(ctx)
        assert result is ctx

    def test_handles_all_events(self):
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="", secret_key="")
        assert set(hook.handles()) == set(HookEvent)

    def test_from_env(self):
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook.from_env()
        assert isinstance(hook, LangfuseTracingHook)

    def test_flush_no_error_without_client(self):
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="", secret_key="")
        hook.flush()  # should not raise


class TestTracingIntegration:
    async def test_hook_chain_with_langfuse(self):
        """LangfuseTracingHook can be added to a HookChain without error."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        chain = HookChain()
        hook = LangfuseTracingHook(public_key="", secret_key="")
        chain.add(hook)

        ctx = HookContext(event=HookEvent.SESSION_START, iteration=0, messages=[])
        await chain.emit(ctx)

    def test_trace_id_none_without_client(self):
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="", secret_key="")
        assert hook.trace_id is None


class TestLangfuseTracingHookWithMockClient:
    """Tests for LangfuseTracingHook behavior when a mock client is available."""

    @pytest.fixture
    def mock_langfuse_client(self):
        """Create a mock Langfuse client."""
        client = MagicMock()
        mock_trace = MagicMock()
        mock_trace.trace_id = "test-trace-123"
        mock_span = MagicMock()
        mock_generation = MagicMock()

        client.trace.return_value = mock_trace
        mock_trace.span.return_value = mock_span
        mock_trace.generation.return_value = mock_generation
        mock_trace.event.return_value = None

        return client

    def test_trace_method_creates_trace(self, mock_langfuse_client):
        """When langfuse client is available, trace() should create a trace."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="test", secret_key="test")
        hook._client = mock_langfuse_client

        ctx = HookContext(event=HookEvent.SESSION_START)
        hook._dispatch(ctx)

        assert hook._trace is not None
        mock_langfuse_client.trace.assert_called_once()

    def test_trace_id_set_after_session_start(self, mock_langfuse_client):
        """trace_id should be set after SESSION_START event."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="test", secret_key="test")
        hook._client = mock_langfuse_client

        ctx = HookContext(event=HookEvent.SESSION_START)
        hook._dispatch(ctx)

        assert hook.trace_id == "test-trace-123"

    async def test_fail_open_on_exception(self):
        """Hook should not crash when an exception occurs during dispatch."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        mock_client = MagicMock()
        mock_client.trace.side_effect = Exception("Test exception")

        hook = LangfuseTracingHook(public_key="test", secret_key="test")
        hook._client = mock_client

        ctx = HookContext(event=HookEvent.SESSION_START)
        # execute() has try/except, should not raise
        await hook.execute(ctx)

    def test_score_logging(self, mock_langfuse_client):
        """Should be able to log scores to the trace."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="test", secret_key="test")
        hook._client = mock_langfuse_client

        # Initialize trace
        ctx_start = HookContext(event=HookEvent.SESSION_START)
        hook._dispatch(ctx_start)

        # Log a score (if the API supports it)
        if hasattr(hook._trace, "score"):
            hook._trace.score(
                name="relevance",
                value=0.95,
            )

    def test_llm_call_creates_generation(self, mock_langfuse_client):
        """LLM calls should create a generation in the trace."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook
        from koboi.types import AgentResponse, TokenUsage

        hook = LangfuseTracingHook(public_key="test", secret_key="test")
        hook._client = mock_langfuse_client

        # Session start
        ctx_start = HookContext(event=HookEvent.SESSION_START)
        hook._dispatch(ctx_start)

        # Pre LLM call - this creates timing entry
        messages = [{"role": "user", "content": "Hello"}]
        ctx_pre = HookContext(
            event=HookEvent.PRE_LLM_CALL,
            iteration=0,
            messages=messages,
        )
        hook._dispatch(ctx_pre)

        # Verify timing was recorded
        assert "llm_0" in hook._timings

        # Post LLM call - this should end generation
        response = AgentResponse(
            content="Hi there!",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )
        ctx_post = HookContext(
            event=HookEvent.POST_LLM_CALL,
            iteration=0,
            llm_response=response,
        )
        hook._dispatch(ctx_post)

        # Verify timing was cleaned up (generation ended)
        assert "llm_0" not in hook._timings

    def test_tool_use_creates_span(self, mock_langfuse_client):
        """Tool use should create a span in the trace."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="test", secret_key="test")
        hook._client = mock_langfuse_client

        # Session start
        ctx_start = HookContext(event=HookEvent.SESSION_START)
        hook._dispatch(ctx_start)

        # Pre tool use
        ctx_pre = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="test_tool",
            tool_arguments='{"arg": "value"}',
            iteration=0,
        )
        hook._dispatch(ctx_pre)

        # Post tool use
        ctx_post = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="test_tool",
            tool_result="Success",
            iteration=0,
        )
        hook._dispatch(ctx_post)

        # Verify span was created
        assert mock_langfuse_client.trace.return_value.span.called

    def test_doom_loop_creates_event(self, mock_langfuse_client):
        """Doom loop detection should create an event in the trace."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="test", secret_key="test")
        hook._client = mock_langfuse_client

        # Session start
        ctx_start = HookContext(event=HookEvent.SESSION_START)
        hook._dispatch(ctx_start)

        # Doom loop detected
        ctx_doom = HookContext(
            event=HookEvent.DOOM_LOOP_DETECTED,
            iteration=5,
        )
        hook._dispatch(ctx_doom)

        # Verify event was created
        hook._trace.event.assert_called_once()
        call_args = hook._trace.event.call_args
        assert call_args[1]["name"] == "Doom Loop Detected"

    def test_session_end_clears_state(self, mock_langfuse_client):
        """SESSION_END should clear trace, spans, and generations."""
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(public_key="test", secret_key="test")
        hook._client = mock_langfuse_client

        # Session start
        ctx_start = HookContext(event=HookEvent.SESSION_START)
        hook._dispatch(ctx_start)

        # Create some spans
        ctx_llm = HookContext(
            event=HookEvent.PRE_LLM_CALL,
            iteration=0,
            messages=[],
        )
        hook._dispatch(ctx_llm)

        assert len(hook._spans) > 0

        # Session end
        ctx_end = HookContext(event=HookEvent.SESSION_END)
        hook._dispatch(ctx_end)

        assert hook._trace is None
        assert len(hook._spans) == 0
        assert len(hook._generations) == 0
        assert len(hook._timings) == 0
