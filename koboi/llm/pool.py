"""koboi/llm/pool.py -- ProviderPool: a multi-provider client with selection policies.

A ``ProviderPool`` holds N ``LLMClient`` s and a ``SelectionPolicy``. It
implements ``LLMClient``, so it is a drop-in wherever a single client goes
(chat, embedding, per-agent). The pool delegates to one client per call; on a
failover-eligible failure it records a breaker strike and tries the next.

Layering: the pool sits ABOVE ``RetryClient``. ``RetryClient`` handles
intra-provider transient retries (``LLMServerError``/``LLMRateLimitError``, plus
``LLMConnectionError`` pre-stream); the pool fails over on the errors
``RetryClient`` does NOT retry -- ``LLMAuthenticationError`` /
``LLMInvalidRequestError`` / ``LLMResponseParseError``, and
``RetryClientError`` (which wraps any non-LLM exception). Stream failover is
only possible BEFORE the first byte (mirroring ``RetryClient``'s ``yielded``
guard); once a stream has yielded, a mid-stream error re-raises (no failover).

W2 ships the ``FailoverPolicy`` + ``CircuitBreaker``. ``round_robin`` / budget
arrive in later waves.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from koboi.llm.base import LLMClient

if TYPE_CHECKING:
    from koboi.events import StreamEvent


# ---------------------------------------------------------------------------
# Circuit breaker -- in-memory, shared per ProviderPool instance
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Tracks per-client failures; opens (skips a client) after a threshold.

    Shared across all callers of one ``ProviderPool``: if one session's calls
    trip a provider, every session using that pool avoids it until cooldown
    expires (desirable -- a downed provider shouldn't be hammered). In-memory
    only (per-process); a persistent spend/health store is a later-wave concern.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._failures: dict[LLMClient, int] = {}
        self._opened_until: dict[LLMClient, float] = {}

    def is_open(self, client: LLMClient) -> bool:
        until = self._opened_until.get(client)
        if until is None:
            return False
        if time.perf_counter() < until:
            return True
        # Cooldown expired -> half-open: allow a probe call (clear state).
        self._opened_until.pop(client, None)
        self._failures[client] = 0
        return False

    def record_failure(self, client: LLMClient) -> None:
        count = self._failures.get(client, 0) + 1
        self._failures[client] = count
        if count >= self.failure_threshold:
            self._opened_until[client] = time.perf_counter() + self.cooldown_s

    def record_success(self, client: LLMClient) -> None:
        self._failures[client] = 0
        self._opened_until.pop(client, None)


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------


class SelectionPolicy(ABC):
    """Pick the next client to try, given breaker state and an exclude set."""

    @abstractmethod
    def select(
        self, clients: list[LLMClient], breaker: CircuitBreaker, exclude: list[LLMClient]
    ) -> LLMClient | None: ...


class FailoverPolicy(SelectionPolicy):
    """First non-open client in declared order, skipping already-tried ones.

    If every client is open or excluded, returns the first non-excluded one (a
    last-resort probe so an all-open pool still attempts a call rather than
    failing without trying); returns ``None`` only if all are excluded.
    """

    def select(self, clients, breaker, exclude):
        for c in clients:
            if c in exclude:
                continue
            if not breaker.is_open(c):
                return c
        for c in clients:  # last resort: first non-excluded (let it raise if it must)
            if c not in exclude:
                return c
        return None


# ---------------------------------------------------------------------------
# ProviderPool
# ---------------------------------------------------------------------------


class ProviderPool(LLMClient):
    """An ``LLMClient`` backed by N providers with failover.

    Construct with the already-built member clients (typically ``RetryClient``
    instances). The pool is provider-type-agnostic, so the same class backs chat
    and embedding pools.
    """

    def __init__(
        self,
        clients: list[LLMClient],
        policy: SelectionPolicy | None = None,
        breaker: CircuitBreaker | None = None,
    ):
        if not clients:
            raise ValueError("ProviderPool requires at least one client")
        self._clients: list[LLMClient] = list(clients)
        self._policy: SelectionPolicy = policy or FailoverPolicy()
        self._breaker: CircuitBreaker = breaker or CircuitBreaker()

    @property
    def clients(self) -> list[LLMClient]:
        return list(self._clients)

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def model(self) -> str:
        """First member's model (telemetry label)."""
        return getattr(self._clients[0], "model", "") or ""

    async def complete(self, messages: list[dict], tools: list[dict] | None = None):
        """Try members in policy order; fail over on any error; raise the last on exhaustion."""
        exclude: list[LLMClient] = []
        last_err: Exception | None = None
        while True:
            client = self._policy.select(self._clients, self._breaker, exclude)
            if client is None:
                break
            try:
                resp = await client.complete(messages, tools)
            except Exception as err:  # RetryClient already exhausted its intra-provider retries
                last_err = err
                self._breaker.record_failure(client)
                exclude.append(client)
                continue
            self._breaker.record_success(client)
            return resp
        if last_err is None:  # unreachable: pool has >=1 client so a failure path set this
            raise RuntimeError("ProviderPool exhausted with no captured error")
        raise last_err

    async def complete_stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator["StreamEvent"]:
        """Fail over only BEFORE the first byte; once yielding, errors re-raise."""
        exclude: list[LLMClient] = []
        last_err: Exception | None = None
        while True:
            client = self._policy.select(self._clients, self._breaker, exclude)
            if client is None:
                break
            yielded = False
            try:
                async for event in client.complete_stream(messages, tools):
                    yielded = True
                    yield event
            except Exception as err:
                last_err = err
                if yielded:
                    raise  # mid-stream: cannot resume / fail over
                self._breaker.record_failure(client)
                exclude.append(client)
                continue
            self._breaker.record_success(client)
            return
        if last_err is not None:
            raise last_err

    async def get_embeddings(self, text: str) -> list[float] | None:
        exclude: list[LLMClient] = []
        last_err: Exception | None = None
        while True:
            client = self._policy.select(self._clients, self._breaker, exclude)
            if client is None:
                break
            try:
                emb = await client.get_embeddings(text)
            except Exception as err:
                last_err = err
                self._breaker.record_failure(client)
                exclude.append(client)
                continue
            self._breaker.record_success(client)
            return emb
        if last_err is not None:
            raise last_err
        return None

    async def close(self) -> None:
        for c in self._clients:
            try:
                await c.close()
            except Exception:  # nosec B110 - best-effort close; one member's failure must not abort the others
                pass
