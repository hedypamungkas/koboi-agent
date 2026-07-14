"""koboi/llm/cache.py -- content-addressed, file-backed LLM response cache.

A ``CachedClient`` is an ``LLMClient`` decorator that memoizes chat completions
to a file-backed, content-addressed store (one JSON file per request hash). The
first call is live and stored; subsequent identical calls return the cached
``AgentResponse`` (byte-identical). This is the ``replay_mode: cache`` mechanism
and the runtime half of capture-from-run (capture freezes a run's cache into a
portable sidecar; re-running the captured bundle in cache mode is 100% hits).

Mirrors the ``ProviderPool`` "LLMClient-wraps-LLMClient" precedent. Stdlib +
``koboi.types`` / ``koboi.events`` / ``koboi.llm.base`` only -- bare-install safe.
Embeddings are NEVER cached (delegated straight through); only chat completions.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from koboi.llm.base import LLMClient, LLMError
from koboi.types import AgentResponse, TokenUsage, ToolCall

if TYPE_CHECKING:
    from koboi.events import StreamEvent

_logger = logging.getLogger(__name__)

_CACHE_SCHEMA = "koboi-cache-1"


# --------------------------------------------------------------------------- #
# Cache key + (de)serialization
# --------------------------------------------------------------------------- #
def compute_cache_key(
    model: str | None,
    messages: list[dict],
    tools: list[dict] | None,
    response_format: dict | None,
) -> str:
    """SHA-256 of (model, messages, tools, response_format).

    ``model`` is in the key so a ``model_pin`` drift never collides. tools /
    response_format normalize to ``None`` when falsy. Fails LOUD on
    non-JSON-serializable messages (never a silent key collision).
    """
    key_dict = {
        "model": model or "",
        "messages": messages,
        "tools": tools or None,
        "response_format": response_format or None,
    }
    payload = json.dumps(key_dict, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_response(resp: AgentResponse) -> dict:
    return {
        "content": resp.content,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls
        ],
        "usage": (
            {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "reasoning_tokens": resp.usage.reasoning_tokens,
            }
            if resp.usage
            else None
        ),
        "model": resp.model,
        "base_url": resp.base_url,
    }


def _deserialize_response(d: dict) -> AgentResponse:
    usage = None
    u = d.get("usage")
    if u:
        usage = TokenUsage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            reasoning_tokens=u.get("reasoning_tokens", 0),
        )
    return AgentResponse(
        content=d.get("content"),
        tool_calls=[
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            for tc in (d.get("tool_calls") or [])
        ],
        usage=usage,
        model=d.get("model"),
        base_url=d.get("base_url"),
    )


@dataclass(frozen=True)
class CacheEntry:
    """A single portable cache entry (for sidecar freeze/hydrate)."""

    key: str
    model: str | None
    created_at: str
    payload: dict  # {"schema", "key", "model", "created_at", "response": {...}}


# --------------------------------------------------------------------------- #
# ResponseCache (file-backed, content-addressed, sharded)
# --------------------------------------------------------------------------- #
class ResponseCache:
    """A directory of ``<key[:2]>/<key>.json`` payload files."""

    def __init__(self, cache_dir: str | Path, *, readonly: bool = False) -> None:
        self._dir = Path(cache_dir)
        self._readonly = readonly

    @property
    def dir(self) -> Path:
        return self._dir

    def _path_for(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> AgentResponse | None:
        """Return the cached response, or None on miss / corrupt entry (fail-soft)."""
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _deserialize_response(data.get("response") or {})
        except Exception as exc:  # corrupt entry -- never raise on read
            _logger.warning("cache read failed for %s: %s", key[:8], exc)
            return None

    def put(self, key: str, response: AgentResponse, *, model: str | None = None) -> None:
        if self._readonly:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _CACHE_SCHEMA,
            "key": key,
            "model": model,
            "created_at": _now_iso(),
            "response": _serialize_response(response),
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(path))

    def has(self, key: str) -> bool:
        return self._path_for(key).exists()

    def iter_entries(self) -> Iterator[tuple[str, dict]]:
        """Yield (key, payload) for every stored entry (skips corrupt files)."""
        if not self._dir.exists():
            return
        for shard in sorted(self._dir.iterdir()):
            if not shard.is_dir():
                continue
            for f in sorted(shard.glob("*.json")):
                try:
                    yield f.stem, json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue

    def count(self) -> int:
        return sum(1 for _ in self.iter_entries())

    def load_entries(self, entries: list[tuple[str, dict]]) -> int:
        """Hydrate a sidecar (list of (key, payload)) into this dir. Returns count."""
        n = 0
        for key, payload in entries:
            path = self._path_for(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            n += 1
        return n

    def clear(self) -> int:
        if not self._dir.exists():
            return 0
        n = self.count()
        shutil.rmtree(self._dir)
        return n


# --------------------------------------------------------------------------- #
# CachedClient (LLMClient decorator)
# --------------------------------------------------------------------------- #
class CacheMissPolicy(enum.Enum):
    STORE = "store"  # live call on miss, then store (the user-facing cache mode)
    RAISE = "raise"  # raise on miss (offline-replay / capture-proof mode)


class CacheMissError(LLMError):
    """Raised by ``CachedClient`` on a cache miss when ``on_miss=RAISE``."""


class CachedClient(LLMClient):
    """Wraps an ``LLMClient`` with a file-backed response cache.

    ``complete`` memoizes by ``compute_cache_key``; ``get_embeddings`` delegates
    uncached; ``complete_stream`` replays the cached response as one
    ``TextDeltaEvent`` + ``CompleteEvent`` on a hit (byte-identical content) and
    collects + stores on a miss.
    """

    def __init__(
        self,
        inner: LLMClient,
        cache: ResponseCache,
        *,
        enabled: bool = True,
        on_miss: CacheMissPolicy = CacheMissPolicy.STORE,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._enabled = enabled
        self._on_miss = on_miss
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        # Mirrors ProviderPool's label convention (adapters have no provider attr).
        return getattr(self._inner, "provider", "?")

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AgentResponse:
        if not self._enabled:
            return await self._inner.complete(messages, tools, response_format=response_format)
        key = compute_cache_key(self._inner.model, messages, tools, response_format)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self._on_miss == CacheMissPolicy.RAISE:
            raise CacheMissError(f"cache miss for key {key[:8]} (offline replay mode)")
        # Per-key lock coalesces concurrent identical misses (no double-spend).
        async with self._lock_for(key):
            cached = self._cache.get(key)  # double-check after acquiring
            if cached is not None:
                return cached
            response = await self._inner.complete(messages, tools, response_format=response_format)
            self._cache.put(key, response, model=self._inner.model)
            return response

    async def get_embeddings(self, text: str) -> list[float] | None:
        # Embeddings are NEVER cached (chat-only scope).
        return await self._inner.get_embeddings(text)

    async def complete_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        from koboi.events import CompleteEvent, TextDeltaEvent

        if not self._enabled:
            async for event in self._inner.complete_stream(messages, tools, response_format=response_format):
                yield event
            return
        key = compute_cache_key(self._inner.model, messages, tools, response_format)
        cached = self._cache.get(key)
        if cached is not None:
            if cached.content:
                yield TextDeltaEvent(content=cached.content)
            yield CompleteEvent(response=cached, content=cached.content or "")
            return
        if self._on_miss == CacheMissPolicy.RAISE:
            raise CacheMissError(f"cache miss for key {key[:8]} (offline replay mode)")
        # Stream live, capture the terminal response, then store. (Concurrent
        # identical streams may both stream + store; the store is idempotent.)
        terminal: AgentResponse | None = None
        async for event in self._inner.complete_stream(messages, tools, response_format=response_format):
            if isinstance(event, CompleteEvent) and event.response is not None:
                terminal = event.response
            yield event
        if terminal is not None:
            self._cache.put(key, terminal, model=self._inner.model)

    async def close(self) -> None:
        await self._inner.close()
