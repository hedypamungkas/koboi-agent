"""tests/test_pool.py -- ProviderPool + CircuitBreaker + FailoverPolicy unit tests."""

from __future__ import annotations

import pytest

from koboi.llm.base import LLMAuthenticationError, LLMClient
from koboi.llm.pool import (
    CircuitBreaker,
    FailoverPolicy,
    ProviderPool,
    ProviderPoolExhausted,
)
from koboi.types import AgentResponse


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeClient(LLMClient):
    """Scriptable LLMClient for pool tests.

    mode: "ok" always succeeds; "fail" always raises (failover-eligible);
    "fail_once" raises the first call then succeeds; "yield_then_fail" streams
    one event then raises (post-first-byte -> no failover).
    """

    def __init__(self, name: str, mode: str = "ok", model: str = "fake-model"):
        self.name = name
        self._model = model
        self.mode = mode
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, messages, tools=None):
        self.calls += 1
        if self.mode == "fail" or (self.mode == "fail_once" and self.calls == 1):
            raise LLMAuthenticationError(f"{self.name} auth fail")
        return AgentResponse(content=f"{self.name}-answer")

    async def get_embeddings(self, text):
        self.calls += 1
        if self.mode == "fail":
            raise LLMAuthenticationError(f"{self.name} embed fail")
        return [1.0, 2.0, 3.0]

    async def complete_stream(self, messages, tools=None):
        from koboi.events import TextDeltaEvent

        self.calls += 1
        if self.mode == "fail":
            raise LLMAuthenticationError(f"{self.name} stream fail")  # pre-first-byte
        if self.mode == "yield_then_fail":
            yield TextDeltaEvent(content=f"{self.name}-chunk")
            raise LLMAuthenticationError(f"{self.name} mid-stream fail")  # post-first-byte
        yield TextDeltaEvent(content=f"{self.name}-stream")


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_closed_below_threshold(self):
        b = CircuitBreaker(failure_threshold=3, cooldown_s=30)
        c = FakeClient("a")
        assert not b.is_open(c)
        b.record_failure(c)
        b.record_failure(c)
        assert not b.is_open(c)  # 2 < 3

    def test_opens_at_threshold_and_reopens_after_cooldown(self, monkeypatch):
        t = [0.0]
        monkeypatch.setattr("koboi.llm.pool.time.perf_counter", lambda: t[0])
        b = CircuitBreaker(failure_threshold=3, cooldown_s=30)
        c = FakeClient("a")
        for _ in range(3):
            b.record_failure(c)
        assert b.is_open(c)  # tripped
        t[0] = 31.0  # past cooldown
        assert not b.is_open(c)  # half-open: probe allowed, state cleared

    def test_success_resets(self):
        b = CircuitBreaker(failure_threshold=2)
        c = FakeClient("a")
        b.record_failure(c)
        b.record_success(c)
        b.record_failure(c)  # only 1 since reset
        assert not b.is_open(c)


# ---------------------------------------------------------------------------
# FailoverPolicy
# ---------------------------------------------------------------------------


class TestFailoverPolicy:
    def test_first_non_open_non_excluded(self):
        b = CircuitBreaker()
        a, b2, c = FakeClient("a"), FakeClient("b"), FakeClient("c")
        policy = FailoverPolicy()
        assert policy.select([a, b2, c], b, []) is a

    def test_skips_open_and_excluded(self):
        br = CircuitBreaker(failure_threshold=1, cooldown_s=30)
        a, b2, c = FakeClient("a"), FakeClient("b"), FakeClient("c")
        br.record_failure(a)  # opens a (threshold 1)
        assert br.is_open(a)
        policy = FailoverPolicy()
        # a is open, b is excluded -> picks c
        assert policy.select([a, b2, c], br, [b2]) is c

    def test_last_resort_probe_when_all_open(self):
        br = CircuitBreaker(failure_threshold=1, cooldown_s=30)
        a, b2 = FakeClient("a"), FakeClient("b")
        br.record_failure(a)
        br.record_failure(b2)  # both open
        policy = FailoverPolicy()
        assert policy.select([a, b2], br, []) is a  # still returns first non-excluded

    def test_none_when_all_excluded(self):
        a, b2 = FakeClient("a"), FakeClient("b")
        policy = FailoverPolicy()
        assert policy.select([a, b2], CircuitBreaker(), [a, b2]) is None


# ---------------------------------------------------------------------------
# ProviderPool
# ---------------------------------------------------------------------------


class TestProviderPoolComplete:
    async def test_succeeds_on_first(self):
        a, b2 = FakeClient("a"), FakeClient("b")
        pool = ProviderPool([a, b2])
        resp = await pool.complete([{"role": "user", "content": "q"}])
        assert resp.content == "a-answer"
        assert a.calls == 1 and b2.calls == 0

    async def test_fails_over_when_first_raises(self):
        a, b2 = FakeClient("a", mode="fail"), FakeClient("b")
        pool = ProviderPool([a, b2])
        resp = await pool.complete([{"role": "user", "content": "q"}])
        assert resp.content == "b-answer"
        assert a.calls == 1 and b2.calls == 1

    async def test_raises_exhausted_with_chain_when_all_fail(self):
        a = FakeClient("a", mode="fail", model="model-a")
        b2 = FakeClient("b", mode="fail", model="model-b")
        pool = ProviderPool([a, b2])
        with pytest.raises(ProviderPoolExhausted, match="2 member") as exc_info:
            await pool.complete([{"role": "user", "content": "q"}])
        # The chain names BOTH failed members (not just the last); last err chained.
        assert "model-a" in str(exc_info.value) and "model-b" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, LLMAuthenticationError)
        assert a.calls == 1 and b2.calls == 1

    async def test_breaker_trips_after_repeated_failures(self):
        a = FakeClient("a", mode="fail")
        b2 = FakeClient("b")
        pool = ProviderPool([a, b2], breaker=CircuitBreaker(failure_threshold=2, cooldown_s=30))
        # Two failing calls trip a's breaker.
        await pool.complete([{"role": "user", "content": "q"}])
        await pool.complete([{"role": "user", "content": "q"}])
        assert pool.breaker.is_open(a)
        # Third call skips a entirely -> only b is tried.
        a.calls = 0
        await pool.complete([{"role": "user", "content": "q"}])
        assert a.calls == 0
        assert b2.calls >= 3

    async def test_empty_pool_rejected(self):
        with pytest.raises(ValueError):
            ProviderPool([])


class TestProviderPoolStream:
    async def test_failover_before_first_byte(self):
        a = FakeClient("a", mode="fail")  # raises pre-yield
        b2 = FakeClient("b")
        pool = ProviderPool([a, b2])
        chunks = [e.content async for e in pool.complete_stream([{"role": "user", "content": "q"}])]
        assert chunks == ["b-stream"]
        assert a.calls == 1 and b2.calls == 1

    async def test_no_failover_after_first_byte(self):
        a = FakeClient("a", mode="yield_then_fail")  # yields then raises
        b2 = FakeClient("b")
        pool = ProviderPool([a, b2])
        with pytest.raises(LLMAuthenticationError):
            async for _e in pool.complete_stream([{"role": "user", "content": "q"}]):
                pass
        # b must NOT be tried (mid-stream error -> re-raise, no failover)
        assert b2.calls == 0

    async def test_stream_exhaustion_raises_pre_first_byte(self):
        a = FakeClient("a", mode="fail", model="model-a")
        b2 = FakeClient("b", mode="fail", model="model-b")
        pool = ProviderPool([a, b2])
        with pytest.raises(ProviderPoolExhausted, match="2 member"):
            async for _e in pool.complete_stream([{"role": "user", "content": "q"}]):
                pass
        assert a.calls == 1 and b2.calls == 1


class TestProviderPoolEmbeddings:
    async def test_embedding_failover(self):
        a = FakeClient("a", mode="fail")
        b2 = FakeClient("b")
        pool = ProviderPool([a, b2])
        emb = await pool.get_embeddings("text")
        assert emb == [1.0, 2.0, 3.0]
        assert a.calls == 1 and b2.calls == 1

    async def test_embedding_exhaustion_raises_not_none(self):
        a = FakeClient("a", mode="fail", model="model-a")
        b2 = FakeClient("b", mode="fail", model="model-b")
        pool = ProviderPool([a, b2])
        # Must RAISE (not silently return None) on total embedding outage.
        with pytest.raises(ProviderPoolExhausted, match="2 member"):
            await pool.get_embeddings("text")
        assert a.calls == 1 and b2.calls == 1


class TestProviderPoolModel:
    def test_model_is_first_member(self):
        pool = ProviderPool([FakeClient("a", model="gpt-x"), FakeClient("b", model="claude-y")])
        assert pool.model == "gpt-x"

    async def test_last_served_model_tracks_actual_member(self):
        a = FakeClient("a", model="model-a")
        pool = ProviderPool([a])
        assert pool.last_served_model is None  # nothing served yet
        await pool.complete([{"role": "user", "content": "q"}])
        assert pool.last_served_model == "model-a"  # the member that actually answered
