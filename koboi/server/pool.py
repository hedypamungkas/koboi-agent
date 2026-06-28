"""koboi/server/pool -- per-session KoboiAgent registry with serialized runs.

``AgentCore`` is NOT concurrent-safe (``loop.py`` has no lock), so each session
gets its own ``asyncio.Lock``. ``run_stream()`` acquires the lock for the FULL
generator lifetime (i.e. the SSE stream's lifetime) and releases it in
``finally`` -- including on client disconnect, where Starlette cancels the
response task and ``CancelledError`` propagates through the ``async for``.

On-demand LRU eviction: when the cap is reached, the oldest session whose lock
is free is evicted (``await agent.close()``); only if none is free does
``get_or_create`` raise ``PoolFull`` (-> HTTP 429).

M5 seam (doc §16.20): this class's public surface is the de-facto ``SessionStore``
protocol; extracting ``Protocol SessionStore`` is a 1-file annotation change
once a second (Redis) backend exists -- do NOT pre-protocol-ize for one impl.
"""

from __future__ import annotations

import asyncio
import copy
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from koboi.events import StreamEvent

if TYPE_CHECKING:
    from koboi.config import Config
    from koboi.facade import KoboiAgent


class PoolFull(Exception):
    """Raised when the agent cap is reached and no idle session can be evicted."""


class InvalidSessionId(ValueError):
    """Raised when a session_id is not a safe path segment / dict key."""


#: A session_id must be a short, URL/dict-key/filesystem-safe token. It flows
#: from the client ``X-Session-Id`` header into a filesystem path
#: (``./workspace/<id>``) and dict keys, so it MUST reject traversal/abs paths.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def is_safe_session_id(session_id: str) -> bool:
    """True if ``session_id`` is a safe path segment / dict key (no traversal)."""
    return bool(_SESSION_ID_RE.fullmatch(session_id))


class AgentPool:
    """Lazy, per-session KoboiAgent cache with per-session run serialization."""

    def __init__(
        self,
        config: Config,
        *,
        client_factory: Callable[[], Any] | None = None,
        workspace_root: str = "./workspace",
        cap: int = 100,
        extra_tools: tuple = (),
        extra_hooks: tuple = (),
        approval_handler: Any | None = None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory
        self._workspace_root = workspace_root.rstrip("/")
        self._cap = cap
        self._agents: dict[str, KoboiAgent] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_used: dict[str, float] = {}
        # Extensibility (doc §6 Path B): attached to each pooled agent.
        # approval_handler is the M2 seam (None in M1 -> base auto behavior).
        self._extra_tools = tuple(extra_tools)
        self._extra_hooks = tuple(extra_hooks)
        self._approval_handler = approval_handler
        self._closed = False

    def __len__(self) -> int:
        return len(self._agents)

    @property
    def cap(self) -> int:
        return self._cap

    def new_session_id(self) -> str:
        from uuid import uuid4

        return uuid4().hex

    def get(self, session_id: str) -> KoboiAgent | None:
        return self._agents.get(session_id)

    def workdir_for(self, session_id: str) -> str:
        """Per-session sandbox workdir (§16.14).

        Defense-in-depth: ``session_id`` must be a safe path segment (validated
        at the route boundary too); reject anything that could escape the
        workspace root via traversal/absolute paths.
        """
        if not is_safe_session_id(session_id):
            raise InvalidSessionId(f"unsafe session_id: {session_id!r}")
        return f"{self._workspace_root}/{session_id}"

    def _build_agent(self, session_id: str) -> KoboiAgent:
        from koboi.facade import KoboiAgent

        # Deep-copy the resolved base config so per-session stamps don't mutate
        # the shared base. ``Config.raw`` is the live dict -> must copy.
        data = copy.deepcopy(self._config.raw)
        data.setdefault("memory", {})["session_id"] = session_id
        data.setdefault("sandbox", {})["workdir"] = self.workdir_for(session_id)
        agent = KoboiAgent.from_dict(data)
        if self._client_factory is not None:
            # Test seam: replace the facade-built RetryClient with a MockClient.
            # NOTE: the original RetryClient is orphaned (its httpx pool is not
            # explicitly closed). Acceptable for tests; production passes no
            # client_factory so the real client is reused and close()d on evict.
            agent._core.client = self._client_factory()
        # Extensibility (doc §6 Path B): extra tools/hooks + approval handler
        # are attached to each pooled agent. approval_handler is the M2 seam
        # (None in M1 -> base ApprovalHandler's auto behavior).
        for spec in self._extra_tools:
            name, fn, desc, params = spec[0], spec[1], spec[2], spec[3]
            kwargs: dict = {}
            if len(spec) > 4 and spec[4] is not None:
                kwargs["risk_level"] = spec[4]
            agent.add_tool(name, fn, desc, params, **kwargs)
        for hook_spec in self._extra_hooks:
            if callable(hook_spec):
                agent.add_hook(hook_spec)
            else:
                callback = hook_spec[0]
                events = hook_spec[1] if len(hook_spec) > 1 else None
                agent.add_hook(callback, events=events)
        if self._approval_handler is not None:
            agent._core.approval_handler = self._approval_handler
        return agent

    async def get_or_create(self, session_id: str) -> KoboiAgent:
        if not is_safe_session_id(session_id):
            raise InvalidSessionId(f"unsafe session_id: {session_id!r}")
        if session_id in self._agents:
            return self._agents[session_id]
        if len(self._agents) >= self._cap:
            if not await self._evict_one_idle():
                raise PoolFull(f"agent cap reached ({self._cap}); no idle session to evict")
        agent = self._build_agent(session_id)
        self._agents[session_id] = agent
        self._locks[session_id] = asyncio.Lock()
        self._last_used[session_id] = time.monotonic()
        return agent

    async def _evict_one_idle(self) -> bool:
        """Evict the oldest session whose lock is currently free. Returns True if evicted."""
        for session_id in sorted(self._last_used, key=lambda s: self._last_used[s]):
            lock = self._locks.get(session_id)
            if lock is not None and lock.locked():
                continue  # busy -> skip
            await self.evict(session_id)
            return True
        return False

    async def run_stream(self, session_id: str, message: str) -> AsyncIterator[StreamEvent]:
        """Serialize one turn: hold the per-session lock across the whole stream.

        The lock is acquired inside the generator so its lifetime equals the
        stream's lifetime (NOT around StreamingResponse construction -- that
        would release on ``return``, before Starlette iterates the body).
        ``get_or_create`` creates the per-session lock, so it always exists here.
        """
        await self.get_or_create(session_id)  # idempotent; raises PoolFull/InvalidSessionId
        async with self._locks[session_id]:
            self._last_used[session_id] = time.monotonic()
            try:
                async for event in self.get(session_id).run_stream(message):
                    yield event
            finally:
                self._last_used[session_id] = time.monotonic()

    async def get_messages(self, session_id: str) -> list[dict]:
        agent = self._agents.get(session_id)
        if agent is None or agent._core is None or agent._core.memory is None:
            return []
        return agent._core.memory.get_messages()

    async def evict(self, session_id: str) -> bool:
        agent = self._agents.pop(session_id, None)
        self._locks.pop(session_id, None)
        self._last_used.pop(session_id, None)
        if agent is None:
            return False
        try:
            await agent.close()
        except Exception:
            pass
        return True

    async def close_all(self) -> None:
        self._closed = True
        for session_id in list(self._agents.keys()):
            await self.evict(session_id)
