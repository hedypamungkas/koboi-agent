"""Unit tests for koboi/server/jobs.py + AutonomousApprovalHandler (no FastAPI)."""

from __future__ import annotations


from koboi.guardrails.approval import AutonomousApprovalHandler
from koboi.server.jobs import JobRegistry, JobStore
from koboi.trust import TrustDatabase
from koboi.types import RiskLevel


class TestJobStore:
    def test_insert_and_get(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "sess_1", "alice", "hello")
        job = store.get("job_1")
        assert job["status"] == "pending"
        assert job["owner"] == "alice"
        assert job["message"] == "hello"

    def test_update_status(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "sess_1", "alice", "hi")
        store.update_status("job_1", "completed", result_json='{"content":"done"}')
        job = store.get("job_1")
        assert job["status"] == "completed"
        assert "done" in job["result_json"]

    def test_get_unknown_returns_none(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs.db"))
        assert store.get("nonexistent") is None

    def test_list_by_owner(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "s1", "alice", "a")
        store.insert("job_2", "s2", "bob", "b")
        store.insert("job_3", "s3", "alice", "c")
        alice_jobs = store.list_by_owner("alice")
        assert len(alice_jobs) == 2

    def test_find_by_idempotency_key(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "s1", "alice", "a", idempotency_key="key-123")
        found = store.find_by_idempotency_key("key-123")
        assert found["job_id"] == "job_1"
        assert store.find_by_idempotency_key("nonexistent") is None

    def test_list_by_status(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "s1", "alice", "a")
        store.insert("job_2", "s2", "alice", "b")
        store.update_status("job_1", "completed")
        pending = store.list_by_status("pending")
        assert len(pending) == 1
        assert pending[0]["job_id"] == "job_2"

    def test_persists_across_connections(self, tmp_path):
        db = str(tmp_path / "jobs.db")
        s1 = JobStore(db)
        s1.insert("job_1", "s1", "alice", "hello")
        s1.close()
        s2 = JobStore(db)
        assert s2.get("job_1")["message"] == "hello"
        s2.close()


class TestJobRegistry:
    async def test_register_and_get(self):
        reg = JobRegistry()
        record = reg.register("job_1", "sess_1", "alice")
        assert reg.get("job_1") is record
        assert record.status == "pending"

    async def test_append_event_capped(self):
        reg = JobRegistry(max_events=3)
        reg.register("job_1", "s1", "alice")
        for i in range(5):
            reg.append_event("job_1", f"event_{i}")
        record = reg.get("job_1")
        assert len(record.events) == 3
        assert record.events[0] == "event_2"

    async def test_set_terminal(self):
        reg = JobRegistry()
        reg.register("job_1", "s1", "alice")
        reg.set_terminal("job_1", "completed")
        record = reg.get("job_1")
        assert record.status == "completed"
        assert record.terminal.is_set()


class TestAutonomousApprovalHandler:
    def test_safe_tool_allowed(self):
        handler = AutonomousApprovalHandler()
        assert handler.should_approve("calc", "{}", RiskLevel.SAFE) is True

    def test_moderate_tool_allowed(self):
        handler = AutonomousApprovalHandler()
        assert handler.should_approve("search", "{}", RiskLevel.MODERATE) is True

    def test_destructive_without_trust_denied(self):
        handler = AutonomousApprovalHandler()
        assert handler.should_approve("run_shell", "rm -rf /", RiskLevel.DESTRUCTIVE) is False

    def test_destructive_with_trust_allowed(self, tmp_path):
        db = TrustDatabase(str(tmp_path / "trust.db"))
        db.record_decision("run_shell", RiskLevel.DESTRUCTIVE, "allow", always=True)
        handler = AutonomousApprovalHandler(trust_db=db)
        assert handler.should_approve("run_shell", "ls", RiskLevel.DESTRUCTIVE) is True
