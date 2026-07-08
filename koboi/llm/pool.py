"""koboi/llm/pool.py -- ProviderPool: a multi-provider client with selection policies.

A ``ProviderPool`` holds N ``LLMClient`` s and a ``SelectionPolicy``. It
implements ``LLMClient``, so it is a drop-in wherever a single client goes
(chat, embedding, per-agent). The pool delegates to one client per call; on a
failover-eligible failure it records a breaker strike and tries the next.

Layering: the pool sits ABOVE ``RetryClient``. ``RetryClient`` handles
intra-provider transient retries and raises an ``LLMError`` (``LLMServerError`` /
``LLMRateLimitError`` after retries, or ``LLMAuthenticationError`` /
``LLMInvalidRequestError`` / ``LLMResponseParseError`` / ``LLMConnectionError``
immediately; non-LLM exceptions are wrapped in ``RetryClientError``, itself an
``LLMError``). The pool therefore narrows its failover catch to ``LLMError`` --
programming bugs (``TypeError``/``KeyError``) propagate immediately so they
don't pollute the breaker. Stream failover is only possible BEFORE the first
byte (mirroring ``RetryClient``'s ``yielded`` guard); once a stream has yielded,
a mid-stream error re-raises (no failover).

Exhaustion (every member failed) raises ``ProviderPoolExhausted`` carrying the
full failure chain (which provider, which error) -- never a silent empty result.
W2 ships the ``FailoverPolicy`` + ``CircuitBreaker``; ``round_robin`` / budget
arrive in later waves.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from koboi.llm.base import LLMClient, LLMError
from koboi.types import AgentResponse

if TYPE_CHECKING:
    from koboi.events import StreamEvent

_logger = logging.getLogger(__name__)


class ProviderPoolExhausted(LLMError):
    """Raised when every member of a ``ProviderPool`` has failed.

    Carries the full failure chain (one entry per tried member) so the operator
    can see WHICH provider was the root cause, not just the last one. The last
    member's exception is chained via ``__cause__``.
    """


def _label(client: LLMClient) -> str:
    """A short, telemetry-friendly identifier for a member (for logs/errors)."""
    return f"{getattr(client, 'provider', '?')}/{getattr(client, 'model', '?')}"


def _exhausted(failures: list[tuple[LLMClient, Exception]]) -> Exception:
    """Build the exhaustion error carrying the full failure chain."""
    if not failures:
        return RuntimeError("ProviderPool exhausted with no captured error (no candidates?)")
    chain = "; ".join(f"{_label(c)}: {type(e).__name__}: {e}" for c, e in failures)
    err = ProviderPoolExhausted(f"ProviderPool exhausted: all {len(failures)} member(s) failed. Chain: {chain}")
    err.__cause__ = failures[-1][1]
    return err


# ---------------------------------------------------------------------------
# Circuit breaker -- in-memory, shared per ProviderPool instance
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Tracks per-client failures; opens (skips a client) after a threshold.

    Shared across all callers of one ``ProviderPool``: if one session's calls
    trip a provider, every session using that pool avoids it until cooldown
    expires (desirable -- a downed provider shouldn't be hammered). In-memory
    only (per-process); a persistent spend/health store is a later-wave concern.

    State transitions are logged (open at threshold, half-open after cooldown)
    so a permanently-dead provider is observable, not silently routed around.
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
        # Cooldown expired -> half-open: clear state and allow a probe call.
        self._opened_until.pop(client, None)
        self._failures[client] = 0
        _logger.info("ProviderPool circuit HALF-OPEN for %s (probing next call)", _label(client))
        return False

    def record_failure(self, client: LLMClient) -> None:
        count = self._failures.get(client, 0) + 1
        self._failures[client] = count
        # Log only the closed->open transition (not every subsequent failure).
        if count >= self.failure_threshold and client not in self._opened_until:
            self._opened_until[client] = time.perf_counter() + self.cooldown_s
            _logger.warning(
                "ProviderPool circuit OPEN for %s after %d failures; cooling down %.0fs",
                _label(client),
                count,
                self.cooldown_s,
            )

    def record_success(self, client: LLMClient) -> None:
        self._failures[client] = 0
        self._opened_until.pop(client, None)


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------


class SelectionPolicy(ABC):
    """Pick the next client to try, given breaker state and an exclude set.

    Contract: return a candidate client, or ``None`` ONLY when every client is in
    ``exclude``. When all non-excluded candidates are breaker-open, return a
    probe candidate (the caller must be prepared for it to fail). ``exclude`` is
    caller-owned and read-only here.
    """

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
        for c in clients:  # last resort: first non-excluded even if breaker-open
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
    and embedding pools. ``last_served_model`` reports which member actually
    answered the most recent successful call (use it for telemetry/cost/eval
    attribution -- ``model`` only labels the first member).
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
        self._last_served_model: str | None = None

    @property
    def clients(self) -> list[LLMClient]:
        return list(self._clients)

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def model(self) -> str:
        """First member's model (a stable label). For which member ACTUALLY
        answered, read ``last_served_model``."""
        return getattr(self._clients[0], "model", "") or ""

    @property
    def last_served_model(self) -> str | None:
        """Model of the member that answered the most recent successful call, or
        ``None`` before any success / after a mid-stream failure."""
        return self._last_served_model

    def _record_success(self, client: LLMClient) -> None:
        self._breaker.record_success(client)
        self._last_served_model = getattr(client, "model", None)

    async def complete(self, messages: list[dict], tools: list[dict] | None = None) -> AgentResponse:
        """Try members in policy order; fail over on ``LLMError``; raise
        ``ProviderPoolExhausted`` (with the chain) when all fail."""
        exclude: list[LLMClient] = []
        failures: list[tuple[LLMClient, Exception]] = []
        while True:
            client = self._policy.select(self._clients, self._breaker, exclude)
            if client is None:
                break
            try:
                resp = await client.complete(messages, tools)
            except LLMError as err:  # RetryClient raised (post-retry or immediately) -> fail over
                failures.append((client, err))
                self._breaker.record_failure(client)
                exclude.append(client)
                continue
            self._record_success(client)
            return resp
        raise _exhausted(failures)

    async def complete_stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Fail over only BEFORE the first byte; once yielding, errors re-raise.
        Pre-first-byte exhaustion raises ``ProviderPoolExhausted``."""
        exclude: list[LLMClient] = []
        failures: list[tuple[LLMClient, Exception]] = []
        while True:
            client = self._policy.select(self._clients, self._breaker, exclude)
            if client is None:
                break
            yielded = False
            try:
                async for event in client.complete_stream(messages, tools):
                    yielded = True
                    yield event
            except LLMError as err:
                if yielded:
                    raise  # mid-stream: cannot resume / fail over
                failures.append((client, err))
                self._breaker.record_failure(client)
                exclude.append(client)
                continue
            self._record_success(client)
            return
        raise _exhausted(failures)

    async def get_embeddings(self, text: str) -> list[float] | None:
        """Fail over on ``LLMError``; raise ``ProviderPoolExhausted`` when all
        members fail. A member returning ``None`` (unsupported) is returned as-is."""
        exclude: list[LLMClient] = []
        failures: list[tuple[LLMClient, Exception]] = []
        while True:
            client = self._policy.select(self._clients, self._breaker, exclude)
            if client is None:
                break
            try:
                emb = await client.get_embeddings(text)
            except LLMError as err:
                failures.append((client, err))
                self._breaker.record_failure(client)
                exclude.append(client)
                continue
            self._record_success(client)
            return emb
        raise _exhausted(failures)

    async def close(self) -> None:
        for c in self._clients:
            try:
                await c.close()
            except Exception as e:  # best-effort close; log + continue (don't abort the others)
                _logger.warning("ProviderPool member close failed for %s: %s", _label(c), e)
