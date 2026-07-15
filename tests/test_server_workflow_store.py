"""Tests for koboi.server.workflow_store (S5): owner-scoped SQLite CRUD."""

import time

from koboi.server.workflow_store import WorkflowStore


class TestWorkflowStore:
    def test_put_get_owner_scoped(self, tmp_path):
        s = WorkflowStore(db_path=str(tmp_path / "wf.db"))
        s.put("alpha", "ownerA", "bundle: alpha", description="d")
        assert s.get("alpha", "ownerA")["bundle_yaml"] == "bundle: alpha"
        assert s.get("alpha", "ownerB") is None  # owner isolation
        assert s.get("missing", "ownerA") is None

    def test_list_by_owner(self, tmp_path):
        s = WorkflowStore(db_path=str(tmp_path / "wf.db"))
        s.put("a", "o1", "x")
        s.put("b", "o1", "y")
        s.put("c", "o2", "z")
        assert [r["name"] for r in s.list_by_owner("o1")] == ["a", "b"]
        assert [r["name"] for r in s.list_by_owner("o2")] == ["c"]

    def test_delete(self, tmp_path):
        s = WorkflowStore(db_path=str(tmp_path / "wf.db"))
        s.put("a", "o1", "x")
        assert s.delete("a", "o1") is True
        assert s.get("a", "o1") is None
        assert s.delete("a", "o1") is False  # already gone
        assert s.delete("a", "o2") is False  # not the owner

    def test_put_upsert_preserves_created_at(self, tmp_path):
        s = WorkflowStore(db_path=str(tmp_path / "wf.db"))
        s.put("a", "o1", "v1")
        first = s.get("a", "o1")
        time.sleep(0.01)
        s.put("a", "o1", "v2", description="updated")
        second = s.get("a", "o1")
        assert second["bundle_yaml"] == "v2"
        assert second["description"] == "updated"
        assert second["created_at"] == first["created_at"]  # preserved on overwrite
        assert second["updated_at"] >= first["updated_at"]

    def test_ping(self, tmp_path):
        assert WorkflowStore(db_path=str(tmp_path / "wf.db")).ping() is True

    def test_self_heals_on_existing_db(self, tmp_path):
        # CREATE TABLE IF NOT EXISTS is idempotent across instances.
        db = str(tmp_path / "wf.db")
        WorkflowStore(db_path=db).put("a", "o1", "x")
        s2 = WorkflowStore(db_path=db)  # reopen
        assert s2.get("a", "o1")["bundle_yaml"] == "x"
