"""Unit tests for koboi/server/jobs.py + AutonomousApprovalHandler (no FastAPI)."""

from __future__ import annotations

import asyncio

import pytest

from koboi.guardrails.approval import AutonomousApprovalHandler
from koboi.server.jobs import DuplicateIdempotencyKey, JobRegistry, JobStore
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

    def test_insert_duplicate_idempotency_key_raises(self, tmp_path):
        # M1: a second insert with the same idempotency_key raises and carries
        # the canonical (first) job_id.
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_a", "sess", "alice", "m", idempotency_key="key-1")
        with pytest.raises(DuplicateIdempotencyKey) as ei:
            store.insert("job_b", "sess", "alice", "m", idempotency_key="key-1")
        assert ei.value.existing_job_id == "job_a"

    def test_insert_null_idempotency_key_multiple_ok(self, tmp_path):
        # M1: NULL idempotency_key never conflicts (partial index WHERE NOT NULL).
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_a", "sess", "alice", "m")
        store.insert("job_b", "sess", "alice", "m")  # both NULL -> ok

    def test_insert_distinct_idempotency_keys_ok(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_a", "sess", "alice", "m", idempotency_key="k1")
        store.insert("job_b", "sess", "alice", "m", idempotency_key="k2")

    def test_insert_duplicate_then_connection_reusable(self, tmp_path):
        # M1: after a DuplicateIdempotencyKey (rollback), the shared connection
        # is reusable -- a fresh insert succeeds.
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_a", "sess", "alice", "m", idempotency_key="key-1")
        with pytest.raises(DuplicateIdempotencyKey):
            store.insert("job_b", "sess", "alice", "m", idempotency_key="key-1")
        store.insert("job_c", "sess", "alice", "m", idempotency_key="key-2")
        assert store.get("job_c")["job_id"] == "job_c"

    def test_redact_error_masks_secrets_and_truncates(self):
        # M2: secret-value shapes are masked; long errors are truncated.
        from koboi.server.jobs import _redact_error

        assert "sk-abcdef" not in _redact_error("auth failed for sk-abcdefghijklmnopqrstuvwxyz")
        assert "hunter2" not in _redact_error("config password=hunter2 rejected")
        assert len(_redact_error("x" * 1000)) == 500

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

    def test_run_shell_benign_command_denied_without_trust(self):
        # C2: run_shell is DESTRUCTIVE, so even a benign command (``ls``) is
        # denied in autonomous mode -- denial is risk-based, not pattern-based,
        # so jobs never execute unattended shell without a Trust DB allow-rule.
        handler = AutonomousApprovalHandler()
        assert handler.should_approve("run_shell", "ls -la", RiskLevel.DESTRUCTIVE) is False


class TestGuardrailsJobActive:
    """16.27: verify guardrails + PolicyHook enforce in autonomous job mode.

    Jobs run without human review — the autonomous handler must still deny
    destructive tools without a Trust DB allow-rule, and PolicyHook's hardcoded
    safety (sensitive paths, dangerous commands) must remain enforced.
    """

    async def test_destructive_tool_denied_in_job(self):
        """A destructive tool called by a job is denied; job completes."""

        from koboi.config import Config
        from koboi.events import ErrorEvent, PendingApprovalEvent
        from koboi.facade import KoboiAgent
        from koboi.guardrails.approval import AutonomousApprovalHandler
        from koboi.types import RiskLevel
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        config = Config.from_dict(
            {
                "agent": {"name": "t", "system_prompt": "h", "max_iterations": 3, "mode": "act"},
                "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
                "memory": {"backend": "in_memory"},
                "sandbox": {"backend": "passthrough"},
            },
            validate=True,
        )
        agent = KoboiAgent.from_dict(config.raw)
        agent._core.client = MockClient(
            [
                make_mock_response(tool_calls=[make_mock_tool_call("danger")]),
                make_mock_response(content="done"),
            ]
        )
        agent.add_tool(
            "danger",
            lambda **kw: "should not reach",
            "destructive test",
            {"type": "object", "properties": {}, "required": []},
            risk_level=RiskLevel.DESTRUCTIVE,
        )

        if hasattr(agent._core, "_tool_pipeline"):
            del agent._core._tool_pipeline
        agent._core.approval_handler = AutonomousApprovalHandler(
            trust_db=agent.trust_db,
            audit_trail=agent._core.audit_trail,
        )

        events: list = []

        async def run_agent():
            try:
                async for ev in agent.run_stream("go"):
                    events.append(ev)
            except Exception as exc:
                events.append(ErrorEvent(error=exc))

        await run_agent()

        # Verify the tool result contains "denied".
        tool_results = [e for e in events if type(e).__name__ == "ToolResultEvent"]
        assert any("denied" in getattr(e, "result", "").lower() for e in tool_results)
        # Verify no PendingApprovalEvent was emitted (autonomous — no HITL).
        assert not any(isinstance(e, PendingApprovalEvent) for e in events)


class TestResumeOnStartup:
    """resume_on_startup (#5): running jobs are rehydrated-and-continued (resume=True),
    pending jobs requeued fresh. (Was: running -> failed InterruptedByRestart.)"""

    async def test_running_job_is_resumed_not_failed(self, tmp_path, monkeypatch):
        from koboi.server import jobs
        from koboi.server.jobs import JobRegistry, JobStore, resume_on_startup

        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "s1", "alice", "do thing")
        store.update_status("job_1", "running")

        calls: list[tuple] = []

        async def fake_run_job(
            job_id,
            pool,
            reg,
            st,
            message,
            timeout=1800,
            mode=None,
            max_iterations=None,
            resume=False,
            webhooks=None,
            workflow_ref=None,
            workflow_store=None,
            replay_mode=None,
        ):
            calls.append((job_id, resume))
            return None

        monkeypatch.setattr(jobs, "run_job", fake_run_job)
        count = await resume_on_startup(store, object(), JobRegistry(), timeout=30)
        await asyncio.sleep(0.01)  # let the created task record its call

        assert count == 1
        assert ("job_1", True) in calls  # running -> resume=True (rehydrate-and-continue)
        job = store.get("job_1")
        assert job["status"] != "failed"  # NOT marked failed (old InterruptedByRestart gone)


class TestJobRegistryPerOwner:
    """G5a: active_count_for_owner counts an owner's running jobs."""

    async def test_counts_running_per_owner(self):
        reg = JobRegistry()
        alice = reg.register("job_1", "s1", "alice")
        reg.register("job_2", "s2", "bob")  # stays pending
        alice.status = "running"
        assert reg.active_count_for_owner("alice") == 1
        assert reg.active_count_for_owner("bob") == 0
        reg.get("job_2").status = "running"
        assert reg.active_count_for_owner("bob") == 1
        assert reg.active_count_for_owner("nobody") == 0


class TestJobStoreReap:
    """G5c-a: reap_terminal_older_than deletes old terminal jobs only."""

    def test_reaps_old_terminal_keeps_recent_and_inflight(self, tmp_path):
        import time as _time

        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("old", "s", "a", "m")
        store.update_status("old", "completed")
        store._conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (_time.time() - 200000, "old"))
        store.insert("recent", "s", "a", "m")
        store.update_status("recent", "completed")  # recent terminal → kept
        store.insert("running", "s", "a", "m")
        store.update_status("running", "running")  # in-flight → kept even if old
        store._conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (_time.time() - 200000, "running"))
        store._conn.commit()

        reaped = store.reap_terminal_older_than(_time.time() - 86400)
        assert reaped == ["old"]
        assert store.get("old") is None
        assert store.get("recent") is not None
        assert store.get("running") is not None

    def test_awaiting_human_not_reaped(self, tmp_path):
        """awaiting_human is terminal but deliberately excluded from the reaper (it awaits human action)."""
        import time as _time

        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("handover", "s", "a", "m")
        store.update_status("handover", "awaiting_human")
        store._conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (_time.time() - 200000, "handover"))
        store._conn.commit()

        reaped = store.reap_terminal_older_than(_time.time() - 86400)
        assert "handover" not in reaped
        assert store.get("handover") is not None  # survived — not reaped


class TestJobRegistryQueue:
    """G5c-b: pending-queue admission (run/queue/reject), FIFO, forget."""

    def test_peek_admit_run_queue_reject(self):
        reg = JobRegistry()
        assert reg.peek_admit(2, 2) == "run"  # active 0 < max 2
        reg.register("j1", "s", "a").status = "running"
        assert reg.peek_admit(2, 2) == "run"  # active 1 < max 2
        reg.register("j2", "s", "a").status = "running"
        assert reg.peek_admit(2, 2) == "queue"  # active 2 >= max 2, pending 0 < depth 2
        reg.enqueue_pending("q1")
        reg.enqueue_pending("q2")
        assert reg.peek_admit(2, 2) == "reject"  # pending 2 >= depth 2

    def test_pop_is_fifo_and_remove(self):
        reg = JobRegistry()
        reg.enqueue_pending("a")
        reg.enqueue_pending("b")
        assert reg.pop_pending() == "a"  # FIFO
        assert reg.remove_pending("b") is True
        assert reg.pop_pending() is None
        assert reg.pending_count == 0
        assert reg.remove_pending("nope") is False

    def test_forget_drops_record_and_pending(self):
        reg = JobRegistry()
        reg.register("j1", "s", "a")
        reg.enqueue_pending("j1")
        reg.forget(["j1"])
        assert reg.get("j1") is None
        assert reg.pending_count == 0


class TestJobRegistryReserveAdmit:
    """#50: reserve_admit atomically reserves an admission slot synchronously so
    N concurrent submitters can't all read pre-increment counts during the await
    window between the admission check and register()."""

    def test_five_run_then_sixth_too_many_per_tenant(self):
        reg = JobRegistry()
        owner = "alice"
        # 5 sequential reservations all admit as "run" (global 64, per-tenant 5);
        # each reserved placeholder counts toward the per-tenant cap immediately.
        for _ in range(5):
            decision, rid = reg.reserve_admit(owner, 64, 0, per_tenant_max=5)
            assert decision == "run"
            assert rid.startswith("job_")
        # The 6th is blocked by the per-tenant cap (5 reserved already counted).
        decision6, _ = reg.reserve_admit(owner, 64, 0, per_tenant_max=5)
        assert decision6 == "too_many_jobs_per_tenant"
        # 5 reserved slots still held (reject path self-releases, leaves the 5).
        assert reg.active_count_for_owner(owner) == 5

    def test_per_tenant_disabled_when_none(self):
        reg = JobRegistry()
        # per_tenant_max=None disables the per-tenant check (dev mode).
        for _ in range(10):
            decision, _ = reg.reserve_admit("dev", 64, 0, per_tenant_max=None)
            assert decision == "run"
        assert reg.active_count == 10

    def test_release_reserve_no_slot_leak(self):
        reg = JobRegistry()
        before = reg.active_count_for_owner("alice")
        decision, rid = reg.reserve_admit("alice", 64, 0, per_tenant_max=5)
        assert decision == "run"
        assert reg.active_count_for_owner("alice") == before + 1
        reg.release_reserve(rid)
        assert reg.active_count_for_owner("alice") == before  # slot returned
        # Re-reserve works (the slot was released, not leaked).
        decision2, _ = reg.reserve_admit("alice", 64, 0, per_tenant_max=5)
        assert decision2 == "run"

    def test_release_reserve_idempotent(self):
        reg = JobRegistry()
        _decision, rid = reg.reserve_admit("alice", 64, 0, per_tenant_max=5)
        reg.release_reserve(rid)
        reg.release_reserve(rid)  # no-op, no error
        reg.release_reserve("nonexistent")  # no-op

    def test_global_reject_never_reserves(self):
        reg = JobRegistry()
        # Fill global capacity with 2 running jobs.
        reg.register("j1", "s", "a").status = "running"
        reg.register("j2", "s", "a").status = "running"
        # queue_depth=0 -> reject; no placeholder is created.
        decision, rid = reg.reserve_admit("a", 2, 0, per_tenant_max=5)
        assert decision == "reject"
        reg.release_reserve(rid)  # defensive no-op (nothing was reserved)
        assert rid not in reg._jobs
        assert reg.active_count == 2  # unchanged

    def test_commit_reserve_rekeys_and_sets_pending(self):
        reg = JobRegistry()
        decision, rid = reg.reserve_admit("alice", 64, 0, per_tenant_max=5)
        assert decision == "run"
        record = reg.commit_reserve(rid, real_job_id="jobX", session_id="sess1")
        assert record.job_id == "jobX"
        assert record.session_id == "sess1"
        assert record.status == "pending"
        # Re-keyed under the real job_id; the reserved placeholder id is gone.
        assert reg.get("jobX") is record
        assert reg.get(rid) is None
        # "pending" is NOT counted as active; only running/reserved are.
        assert reg.active_count_for_owner("alice") == 0
        record.status = "running"
        assert reg.active_count_for_owner("alice") == 1

    def test_commit_reserve_same_id_updates_in_place(self):
        reg = JobRegistry()
        decision, rid = reg.reserve_admit("alice", 64, 0, per_tenant_max=5)
        assert decision == "run"
        # Commit with real_job_id == reserved_id (update fields in place).
        record = reg.commit_reserve(rid, real_job_id=rid, session_id="sess1")
        assert record.job_id == rid
        assert record.status == "pending"
        assert reg.get(rid) is record
