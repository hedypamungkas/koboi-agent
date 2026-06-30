"""tests/test_client_stream_retry -- Bucket B: stream timeouts are retried.

Regression for the e2e failure where a single stalled upstream LLM call
(job_multi_chain) consumed the whole turn budget and failed the job:
``LLMConnectionError`` (raised by ``HttpTransport`` on httpx stream timeouts)
was not in the retryable set, so ``RetryClient.complete_stream`` re-raised it
immediately. After the fix, pre-stream stalls are retried (bounded), while
mid-stream failures still raise (can't resume a partial stream).
"""

from __future__ import annotations

import pytest

from koboi.client import RetryClient
from koboi.events import TextDeltaEvent
from koboi.llm.base import LLMConnectionError


class _FakeImpl:
    """Minimal LLMClient impl whose complete_stream follows a scripted behavior.

    Each call advances one step: ``conn_err`` (raise pre-yield), ``yield_then_err``
    (yield one event then raise), or ``ok`` (yield one event). The trailing step
    repeats so a short script can describe a retry-then-success sequence.
    """

    def __init__(self, behavior: list[str]):
        self.behavior = behavior
        self.calls = 0

    async def complete_stream(self, messages, tools):
        self.calls += 1
        step = self.behavior[min(self.calls - 1, len(self.behavior) - 1)]
        if step == "conn_err":
            raise LLMConnectionError("simulated pre-stream stall")
        if step == "yield_then_err":
            yield TextDeltaEvent(content="partial")
            raise LLMConnectionError("mid-stream stall")
        yield TextDeltaEvent(content="done")


def _client() -> RetryClient:
    # Build with a dummy provider/key (no network at construction), then swap in
    # the fake impl. Tiny backoff so retries don't slow the test.
    rc = RetryClient(
        api_key="test",
        base_url="http://localhost:8080/v1",
        model="gpt-4o-mini",
        provider="openai",
        retry_backoff_base=0.01,
    )
    return rc


class TestStreamRetry:
    async def test_pre_stream_timeout_is_retried_then_succeeds(self):
        rc = _client()
        fake = _FakeImpl(["conn_err", "conn_err", "ok"])
        rc._impl = fake

        events = [e async for e in rc.complete_stream([], None)]

        assert fake.calls == 3  # 2 retries then success
        assert any(getattr(e, "content", None) == "done" for e in events)

    async def test_mid_stream_timeout_is_not_retried(self):
        rc = _client()
        fake = _FakeImpl(["yield_then_err"])
        rc._impl = fake

        with pytest.raises(LLMConnectionError):
            [e async for e in rc.complete_stream([], None)]

        assert fake.calls == 1  # already yielded -> no retry

    async def test_exhausted_retries_raise(self):
        rc = _client()
        fake = _FakeImpl(["conn_err"])  # always fails
        rc._impl = fake

        with pytest.raises(LLMConnectionError):
            [e async for e in rc.complete_stream([], None)]

        # max_retries=3 -> 4 total attempts (initial + 3 retries).
        assert fake.calls == rc.max_retries + 1
