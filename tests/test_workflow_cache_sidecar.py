"""Tests for koboi.workflows.cache_sidecar (v2 step 3): both backends."""

import sqlite3

import pytest

from koboi.llm.cache import ResponseCache
from koboi.types import AgentResponse
from koboi.workflows.cache_sidecar import (
    CacheSidecarManifest,
    DirectoryCacheSidecar,
    SqliteCacheSidecar,
)

_ENTRIES = [
    (
        "aa" + "1" * 62,
        {
            "schema": "koboi-cache-1",
            "key": "aa" + "1" * 62,
            "model": "m",
            "created_at": "t1",
            "response": {"content": "a"},
        },
    ),
    (
        "bb" + "2" * 62,
        {
            "schema": "koboi-cache-1",
            "key": "bb" + "2" * 62,
            "model": "m",
            "created_at": "t2",
            "response": {"content": "b"},
        },
    ),
]


def _sqlite_conn(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "wf.db"))
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture(params=["directory", "sqlite"])
def sidecar(request, tmp_path):
    if request.param == "directory":
        return DirectoryCacheSidecar(tmp_path / "sidecar")
    return SqliteCacheSidecar(_sqlite_conn(tmp_path), "ownerA", "wf")


class TestCacheSidecarConformance:
    def test_write_then_read_round_trips(self, sidecar):
        n = sidecar.write(_ENTRIES)
        assert n == 2
        entries = list(sidecar.read())
        assert len(entries) == 2
        keys = {e.key for e in entries}
        assert keys == {e[0] for e in _ENTRIES}
        # payload is verbatim
        first = next(e for e in entries if e.key == _ENTRIES[0][0])
        assert first.payload["response"]["content"] == "a"
        assert first.model == "m"

    def test_len(self, sidecar):
        assert len(sidecar) == 0
        sidecar.write(_ENTRIES)
        assert len(sidecar) == 2

    def test_clear(self, sidecar):
        sidecar.write(_ENTRIES)
        assert sidecar.clear() == 2
        assert len(sidecar) == 0

    def test_manifest(self, sidecar):
        sidecar.write(_ENTRIES)
        m = sidecar.manifest()
        assert isinstance(m, CacheSidecarManifest)
        assert m.entry_count == 2

    def test_write_is_upsert_idempotent(self, sidecar):
        sidecar.write(_ENTRIES)
        sidecar.write(_ENTRIES)  # same keys -> upsert, not duplicate
        assert len(sidecar) == 2


class TestDirectorySidecarIsResponseCacheCompatible:
    def test_sidecar_dir_is_a_valid_cache_dir(self, tmp_path):
        # The sidecar dir format == ResponseCache format, so a captured sidecar
        # can be loaded directly as a ResponseCache (the re-run path).
        sidecar = DirectoryCacheSidecar(tmp_path / "sidecar")
        sidecar.write(_ENTRIES)
        cache = ResponseCache(tmp_path / "sidecar")
        key = _ENTRIES[0][0]
        resp = cache.get(key)
        assert isinstance(resp, AgentResponse)
        assert resp.content == "a"


class TestSqliteSidecarIsolation:
    def test_owner_name_isolation(self, tmp_path):
        conn = _sqlite_conn(tmp_path)
        a = SqliteCacheSidecar(conn, "ownerA", "wf")
        b = SqliteCacheSidecar(conn, "ownerB", "wf")
        a.write(_ENTRIES)
        assert len(a) == 2
        assert len(b) == 0  # owner B cannot see owner A's entries
        # same owner, different name
        a2 = SqliteCacheSidecar(conn, "ownerA", "other")
        assert len(a2) == 0

    def test_idempotent_schema(self, tmp_path):
        conn = _sqlite_conn(tmp_path)
        SqliteCacheSidecar(conn, "o", "n")  # creates table
        SqliteCacheSidecar(conn, "o", "n")  # reopen -- no error
        assert True
