"""koboi/workflows/cache_sidecar.py -- portable cache artifact + store backends.

A captured workflow bundle carries its response cache as a *sidecar* so the
bundle re-runs byte-identical + offline (every response is a cache hit). Two
backends share one ``CacheSidecar`` contract:

* ``DirectoryCacheSidecar`` (CLI): a sibling ``<slug>.cache/`` dir using the
  IDENTICAL file format as :class:`~koboi.llm.cache.ResponseCache` (so the
  sidecar dir IS a valid ``cache_dir`` -- freeze = copy, hydrate = point).
* ``SqliteCacheSidecar`` (server): an owner-scoped ``workflows_cache`` table.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from koboi.llm.cache import CacheEntry, ResponseCache


@dataclass(frozen=True)
class CacheSidecarManifest:
    entry_count: int


class CacheSidecar(Protocol):
    """A portable store of (key, payload) cache entries for a captured workflow."""

    def write(self, entries: Iterable[tuple[str, dict]]) -> int: ...
    def read(self) -> Iterator[CacheEntry]: ...
    def __len__(self) -> int: ...
    def clear(self) -> int: ...
    def manifest(self) -> CacheSidecarManifest: ...


class DirectoryCacheSidecar:
    """File-dir sidecar (CLI). Delegates to ``ResponseCache``'s file format."""

    def __init__(self, dir: str | Path) -> None:
        self._dir = Path(dir)
        self._cache = ResponseCache(self._dir)

    @property
    def dir(self) -> Path:
        return self._dir

    def write(self, entries: Iterable[tuple[str, dict]]) -> int:
        return self._cache.load_entries(list(entries))

    def read(self) -> Iterator[CacheEntry]:
        for key, payload in self._cache.iter_entries():
            yield CacheEntry(
                key=key,
                model=payload.get("model"),
                created_at=payload.get("created_at", ""),
                payload=payload,
            )

    def __len__(self) -> int:
        return self._cache.count()

    def clear(self) -> int:
        return self._cache.clear()

    def manifest(self) -> CacheSidecarManifest:
        return CacheSidecarManifest(entry_count=len(self))


class SqliteCacheSidecar:
    """SQLite sidecar (server). Owner + name scoped; idempotent schema."""

    def __init__(self, conn: sqlite3.Connection, owner: str, name: str) -> None:
        self._conn = conn
        self._owner = owner
        self._name = name
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS workflows_cache ("
            "  owner TEXT NOT NULL, name TEXT NOT NULL, cache_key TEXT NOT NULL,"
            "  payload_json TEXT NOT NULL, model TEXT, created_at REAL NOT NULL,"
            "  PRIMARY KEY (owner, name, cache_key)"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_workflows_cache_owner ON workflows_cache(owner, name)")
        self._conn.commit()

    def write(self, entries: Iterable[tuple[str, dict]]) -> int:
        now = time.time()
        n = 0
        for key, payload in entries:
            self._conn.execute(
                "INSERT OR REPLACE INTO workflows_cache "
                "(owner, name, cache_key, payload_json, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self._owner,
                    self._name,
                    key,
                    json.dumps(payload, ensure_ascii=False),
                    payload.get("model"),
                    now,
                ),
            )
            n += 1
        self._conn.commit()
        return n

    def read(self) -> Iterator[CacheEntry]:
        rows = self._conn.execute(
            "SELECT cache_key, payload_json FROM workflows_cache WHERE owner = ? AND name = ?",
            (self._owner, self._name),
        ).fetchall()
        for r in rows:
            try:
                payload = json.loads(r["payload_json"])
                yield CacheEntry(
                    key=r["cache_key"],
                    model=payload.get("model"),
                    created_at=payload.get("created_at", ""),
                    payload=payload,
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    def __len__(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM workflows_cache WHERE owner = ? AND name = ?",
            (self._owner, self._name),
        ).fetchone()
        return row["c"] if row else 0

    def clear(self) -> int:
        cur = self._conn.execute(
            "DELETE FROM workflows_cache WHERE owner = ? AND name = ?",
            (self._owner, self._name),
        )
        self._conn.commit()
        return cur.rowcount

    def manifest(self) -> CacheSidecarManifest:
        return CacheSidecarManifest(entry_count=len(self))
