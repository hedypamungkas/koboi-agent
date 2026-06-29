"""Unit tests for koboi/server/ownership.py (no FastAPI)."""

from __future__ import annotations

from koboi.server.ownership import OwnershipStore


class TestOwnershipStore:
    def test_set_and_get_owner(self, tmp_path):
        store = OwnershipStore(str(tmp_path / "own.db"))
        store.set_owner("sess1", "alice")
        assert store.get_owner("sess1") == "alice"

    def test_is_owner(self, tmp_path):
        store = OwnershipStore(str(tmp_path / "own.db"))
        store.set_owner("sess1", "alice")
        assert store.is_owner("sess1", "alice") is True
        assert store.is_owner("sess1", "bob") is False
        assert store.is_owner("nonexistent", "alice") is False

    def test_overwrite_owner(self, tmp_path):
        store = OwnershipStore(str(tmp_path / "own.db"))
        store.set_owner("sess1", "alice")
        store.set_owner("sess1", "bob")
        assert store.get_owner("sess1") == "bob"

    def test_delete(self, tmp_path):
        store = OwnershipStore(str(tmp_path / "own.db"))
        store.set_owner("sess1", "alice")
        store.delete("sess1")
        assert store.get_owner("sess1") is None

    def test_get_owner_unknown(self, tmp_path):
        store = OwnershipStore(str(tmp_path / "own.db"))
        assert store.get_owner("nope") is None

    def test_persists_across_connections(self, tmp_path):
        db = str(tmp_path / "own.db")
        s1 = OwnershipStore(db)
        s1.set_owner("sess1", "alice")
        s1.close()
        s2 = OwnershipStore(db)
        assert s2.get_owner("sess1") == "alice"
        s2.close()
