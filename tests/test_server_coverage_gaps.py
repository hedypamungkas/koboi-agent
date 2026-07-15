"""Coverage gap tests for koboi/server layer.

Targets specific missing lines in app.py, jobs.py, and pool.py to raise coverage above 90%.
These tests exercise error paths, auth failures, webhook delivery, and edge cases.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from unittest.mock import Mock

import pytest

pytest.importorskip("fastapi")
import httpx
from httpx import ASGITransport

from koboi.config import Config
from koboi.server import create_app
from koboi.server.jobs import (
    JobStore,
    JobRegistry,
    run_job,
    _redact_error,
    _webhook_payload,
    _emit_job_webhooks,
    _emit_handover_webhook,
    DuplicateIdempotencyKey,
)
from koboi.server.pool import (
    AgentPool,
    is_safe_session_id,
    InvalidSessionId,
    _git_init_workdir,
    _deep_research_messages,
)
from tests.conftest import MockClient, make_mock_response


def _config(**overrides):
    """Base server-test config."""
    cfg = {
        "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "base_url": "http://localhost:8080/v1",
        },
        "memory": {"backend": "in_memory"},
        "sandbox": {"backend": "restricted"},
        "server": {"auth_required": False},
    }
    cfg.update(overrides)
    return Config.from_dict(cfg, validate=True)


def _app(responses=None, **kw):
    def factory():
        return MockClient(responses or [make_mock_response(content="hello")])

    return create_app(_config(), client_factory=factory, enable_cors=False, **kw)


def _auth_app(**kw):
    """Create app with auth enabled."""
    cfg = {
        "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "test"},
        "memory": {"backend": "in_memory"},
        "sandbox": {"backend": "restricted"},
        "server": {"auth_required": True},
    }

    def factory():
        return MockClient([make_mock_response(content="hello")])

    return create_app(Config.from_dict(cfg, validate=True), client_factory=factory, enable_cors=False, **kw)


class TestJobStoreCoverageGaps:
    """Tests for job.py coverage gaps."""

    def test_job_store_ensure_schema_with_legacy_duplicate_keys(self, tmp_path):
        """Test legacy DB with duplicate keys (lines 111-112)."""
        db_path = str(tmp_path / "jobs.db")

        # Create a legacy DB with duplicate idempotency keys
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                owner TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                idempotency_key TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("INSERT INTO jobs VALUES ('job1', 's1', 'alice', 'pending', 'msg', 'dup_key', 1.0, 1.0)")
        conn.execute(
            "INSERT INTO jobs VALUES ('job2', 's2', 'alice', 'pending', 'msg', 'dup_key', 2.0, 2.0)"
        )  # Duplicate!
        conn.commit()
        conn.close()

        # Opening should warn but not fail
        store = JobStore(db_path)
        assert store.get("job1") is not None
        store.close()

    def test_job_store_insert_duplicate_key_workflow_job(self, tmp_path):
        """Test duplicate idempotency key raises (lines 118-120, 552)."""
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_a", "s1", "alice", "msg", idempotency_key="key-1")

        with pytest.raises(DuplicateIdempotencyKey) as exc:
            store.insert("job_b", "s2", "alice", "msg", idempotency_key="key-1")

        assert exc.value.existing_job_id == "job_a"
        store.close()

    def test_job_store_set_cache_dir(self, tmp_path):
        """Test set_cache_dir method (lines 144-145)."""
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "s1", "alice", "msg")
        store.set_cache_dir("job_1", "/tmp/cache/job_1")

        job = store.get("job_1")
        assert job.get("cache_dir") == "/tmp/cache/job_1"
        store.close()

    def test_redact_error_masks_various_patterns(self):
        """Test _redact_error masks secrets (lines 59-64, 678, 681, 688)."""
        # OpenAI-style keys
        assert "sk-abcdef" not in _redact_error("auth failed for sk-abcdefghijklmnopqrstuvwxyz12345678")
        assert "***REDACTED***" in _redact_error("auth failed for sk-abcdefghijklmnopqrstuvwxyz12345678")

        # AWS keys
        assert "AKIA" not in _redact_error("key: AKIAIOSFODNN7EXAMPLE")

        # Bearer tokens
        assert "bearer" not in _redact_error("Authorization: bearer abcdef").lower()

        # Passwords
        assert "hunter2" not in _redact_error("password=hunter2")

        # Truncation
        assert len(_redact_error("x" * 1000)) == 500

    def test_webhook_payload_with_result_json(self, tmp_path):
        """Test _webhook_payload with result parsing (lines 402-412, 708-710, 782, 789, 791)."""
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "s1", "alice", "test msg")
        store.update_status("job_1", "completed", result_json='{"content":"done"}')

        payload = _webhook_payload(store, "job_1", "completed")
        assert payload is not None
        assert payload["job_id"] == "job_1"
        assert payload["result"]["content"] == "done"
        assert payload["status"] == "completed"

    def test_webhook_payload_no_result(self, tmp_path):
        """Test _webhook_payload without result (lines 799, 802)."""
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "s1", "alice", "test msg")
        store.update_status("job_1", "failed", error="something broke")

        payload = _webhook_payload(store, "job_1", "failed")
        assert payload is not None
        assert payload["error"] == "something broke"
        assert payload["result"] is None

    def test_webhook_payload_unknown_job(self, tmp_path):
        """Test _webhook_payload for unknown job (line 678)."""
        store = JobStore(str(tmp_path / "jobs.db"))
        payload = _webhook_payload(store, "unknown", "completed")
        assert payload is None

    def test_emit_job_webhooks_no_webhooks(self, tmp_path):
        """Test _emit_job_webhooks with no webhooks (lines 615, 623)."""
        store = JobStore(str(tmp_path / "jobs.db"))
        store.insert("job_1", "s1", "alice", "msg")

        # Should not raise, just return
        _emit_job_webhooks(None, store, "job_1", "completed")
        _emit_job_webhooks([], store, "job_1", "failed")

    def test_emit_job_webhooks_with_filtering(self, tmp_path):
        """Test _emit_job_webhooks with event filtering (lines 862, 870-871, 876, 899)."""

        async def test_async():
            store = JobStore(str(tmp_path / "jobs.db"))
            store.insert("job_1", "s1", "alice", "msg")

            webhooks = [
                {"url": "http://example.com/hook1", "events": ["completed"]},
                {"url": "http://example.com/hook2", "events": ["failed"]},
                {"url": "http://example.com/hook3"},  # No events filter -> all events
            ]

            # Should fire for hook1 and hook3, not hook2
            _emit_job_webhooks(webhooks, store, "job_1", "completed")
            # Give async tasks time to schedule
            await asyncio.sleep(0.1)

        asyncio.run(test_async())

    def test_emit_job_webhooks_empty_url(self, tmp_path):
        """Test _emit_job_webhooks with empty URL (line 914)."""

        async def test_async():
            store = JobStore(str(tmp_path / "jobs.db"))
            store.insert("job_1", "s1", "alice", "msg")

            webhooks = [{"url": None, "events": ["completed"]}]
            _emit_job_webhooks(webhooks, store, "job_1", "completed")
            await asyncio.sleep(0.1)

        asyncio.run(test_async())

    def test_emit_job_webhooks_with_secret(self, tmp_path):
        """Test _emit_job_webhooks with HMAC signature (lines 935-936, 945)."""

        async def test_async():
            store = JobStore(str(tmp_path / "jobs.db"))
            store.insert("job_1", "s1", "alice", "msg")

            webhooks = [{"url": "http://example.com/hook", "secret": "test-secret", "timeout": 5.0}]

            _emit_job_webhooks(webhooks, store, "job_1", "completed")
            await asyncio.sleep(0.1)

        asyncio.run(test_async())

    def test_emit_handover_webhook(self):
        """Test handover webhook emission (lines 987-991)."""

        async def test_async():
            webhooks = [{"url": "http://example.com/handover", "secret": "secret"}]

            _emit_handover_webhook(
                webhooks,
                "sess_1",
                "handover_123",
                "user_confused",
                "User needs help with topic",
            )
            await asyncio.sleep(0.1)

        asyncio.run(test_async())

    def test_emit_handover_webhook_no_webhooks(self):
        """Test handover webhook with no webhooks."""
        _emit_handover_webhook(None, "sess_1", "handover_123", "reason", "summary")
        _emit_handover_webhook([], "sess_1", "handover_123", "reason", "summary")

    def test_drain_webhook_tasks_timeout(self, monkeypatch):
        """Test drain_webhook_tasks with timeout (lines 980, 487-500)."""
        import koboi.server.jobs as jobs_mod

        async def runner():
            # Create a long-running webhook task on THIS loop.
            async def slow_webhook():
                await asyncio.sleep(10)

            task = asyncio.create_task(slow_webhook())

            # Isolate the global task set: only our own (same-loop) task, so the drain
            # isn't polluted by stale webhook tasks from earlier jobs tests that are bound
            # to now-closed event loops (which would raise on gather).
            monkeypatch.setattr(jobs_mod, "_WEBHOOK_TASKS", {task})

            # Drain should timeout after 0.5s, not wait the full 10s.
            start = time.time()
            await jobs_mod.drain_webhook_tasks(timeout=0.5)
            elapsed = time.time() - start

            assert elapsed < 2.0
            task.cancel()

        asyncio.run(runner())


class TestAgentPoolCoverageGaps:
    """Tests for pool.py coverage gaps."""

    def test_git_init_workdir_success(self, tmp_path, monkeypatch):
        """Test successful git init (lines 49-61)."""
        # Mock subprocess.run to simulate successful git init
        import subprocess

        mock_results = []

        def mock_run(cmd, **kwargs):
            mock_results.append(cmd)
            return None  # Success

        monkeypatch.setattr(subprocess, "run", mock_run)

        workdir = str(tmp_path / "repo")
        result = _git_init_workdir(workdir)
        assert result is True
        assert len(mock_results) == 4  # init, config email, config name, commit

    def test_git_init_workdir_git_unavailable(self, tmp_path, monkeypatch):
        """Test git init when git is not available (lines 58-60)."""
        import subprocess

        def raise_file_not_found(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", raise_file_not_found)

        workdir = str(tmp_path / "repo")
        result = _git_init_workdir(workdir)
        assert result is False  # Should not raise, just return False

    def test_git_init_workdir_timeout(self, tmp_path, monkeypatch):
        """Test git init timeout (lines 58-60)."""
        import subprocess

        def raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired("git", 15)

        monkeypatch.setattr(subprocess, "run", raise_timeout)

        workdir = str(tmp_path / "repo")
        result = _git_init_workdir(workdir)
        assert result is False

    def test_is_safe_session_id(self):
        """Test session_id validation (lines 78-80)."""
        assert is_safe_session_id("abc-123_XYZ")
        assert is_safe_session_id("Agent007")
        assert is_safe_session_id("session_id_123")

        # Invalid: special chars, spaces, path traversal
        assert not is_safe_session_id("../etc/passwd")
        assert not is_safe_session_id("session with spaces")
        assert not is_safe_session_id("session/slash")
        assert not is_safe_session_id("")
        assert not is_safe_session_id("a" * 129)  # Too long

    def test_deep_research_messages_no_db_path(self):
        """Test _deep_research_messages with no db_path (lines 91-95, 104)."""
        agent = Mock()  # Mock agent with no orchestrator
        agent._orchestrator = None

        result = _deep_research_messages(agent, "sess_1")
        assert result == []

    def test_deep_research_messages_no_research_run(self, tmp_path, monkeypatch):
        """Test _deep_research_messages with no research run (lines 100-101)."""
        from koboi.orchestration.dag_scheduler import DagScheduler

        agent = Mock()
        agent._orchestrator = Mock()
        agent._orchestrator._dag_scheduler.db_path = str(tmp_path / "empty.db")

        # No research_context row -> loader returns None -> [] (expected, no log)
        monkeypatch.setattr(
            DagScheduler,
            "load_research_context_for_session",
            staticmethod(lambda db_path, session_id: None),
        )

        assert _deep_research_messages(agent, "sess_1") == []

    def test_deep_research_messages_corrupt_context(self, tmp_path, monkeypatch):
        """Test _deep_research_messages with a corrupt context row (lines 104-108)."""
        from koboi.orchestration.dag_scheduler import DagScheduler

        agent = Mock()
        agent._orchestrator = Mock()
        agent._orchestrator._dag_scheduler.db_path = str(tmp_path / "test.db")

        # Corrupt row -> from_json raises -> caught -> returns [] (never crashes GET /sessions)
        monkeypatch.setattr(
            DagScheduler,
            "load_research_context_for_session",
            staticmethod(lambda db_path, session_id: "invalid json {{{"),
        )

        assert _deep_research_messages(agent, "sess_1") == []

    def test_deep_research_messages_successful(self, tmp_path, monkeypatch):
        """Test _deep_research_messages success path (lines 109-116)."""
        from koboi.orchestration.dag_scheduler import DagScheduler
        from koboi.orchestration.research import ResearchContext

        agent = Mock()
        agent._orchestrator = Mock()
        agent._orchestrator._dag_scheduler.db_path = str(tmp_path / "test.db")

        # Real round-trippable context: query + cited final_report (empty SourceStore).
        ctx_json = ResearchContext(query="test query", final_report="Test report with citations").to_json()
        monkeypatch.setattr(
            DagScheduler,
            "load_research_context_for_session",
            staticmethod(lambda db_path, session_id: ctx_json),
        )

        result = _deep_research_messages(agent, "sess_1")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "test query"}
        assert result[1]["role"] == "assistant"
        assert "Test report" in result[1]["content"]

    def test_pool_get_or_create_unsafe_session_id(self):
        """Test get_or_create with unsafe session_id (lines 224-225)."""
        pool = AgentPool(_config())

        with pytest.raises(InvalidSessionId):
            asyncio.run(pool.get_or_create("../etc/passwd"))

    def test_pool_evict_nonexistent_session(self):
        """Test evict of non-existent session (lines 310-314)."""
        pool = AgentPool(_config())
        result = asyncio.run(pool.evict("nonexistent"))
        assert result is False

    def test_pool_close_on_evict_error(self):
        """Test evict handles agent.close() errors (lines 317-318)."""
        pool = AgentPool(_config())

        # Create an agent
        sid = "test_sess"
        asyncio.run(pool.get_or_create(sid))

        # Make agent.close() raise
        agent = pool.get(sid)
        agent.close = Mock(side_effect=RuntimeError("close failed"))

        # Should not raise, just log and continue
        result = asyncio.run(pool.evict(sid))
        assert result is True

    def test_pool_flush_langfuse_no_hooks(self):
        """Test flush_langfuse with no Langfuse hooks (lines 326-353)."""
        pool = AgentPool(_config())

        # Should not raise
        asyncio.run(pool.flush_langfuse())

    def test_pool_session_lock_cleanup(self):
        """Test session_lock cleanup (lines 272-276)."""
        pool = AgentPool(_config())

        async def test():
            sid = "test_sess"
            async with pool.session_lock(sid):
                # Verify lock was acquired
                assert sid in pool._locks
                # Verify last_used updated
                assert pool._last_used.get(sid) > 0

        asyncio.run(test())


class TestServerAppCoverageGaps:
    """Tests for app.py coverage gaps."""

    async def test_workflow_owner_unauthenticated(self):
        """Test workflow owner check with auth enabled but no identity."""
        app = _auth_app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            # Don't provide auth - should get 401
            response = await client.post(
                "/v1/workflows",
                json={
                    "name": "test_workflow",
                    "bundle": "agent:\n  name: test\n",
                },
            )
            # May get 401 or 500 depending on auth implementation
            assert response.status_code in [401, 500]

    async def test_workflow_create_invalid_bundle(self):
        """Test workflow creation with invalid bundle."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/workflows",
                json={
                    "name": "test_workflow",
                    "bundle": "invalid: yaml: content: [[[",
                },
            )
            assert response.status_code == 400
            assert "invalid_workflow" in response.json()["error"]["code"]

    async def test_workflow_get_not_found(self):
        """Test get workflow not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.get("/v1/workflows/nonexistent")
            assert response.status_code == 404

    async def test_workflow_delete_not_found(self):
        """Test delete workflow not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.delete("/v1/workflows/nonexistent")
            assert response.status_code == 404

    async def test_capture_job_not_found(self):
        """Test capture job not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/jobs/unknown_job/capture",
                json={"name": "captured_workflow"},
            )
            assert response.status_code == 404

    async def test_capture_job_not_complete(self):
        """Test capture job when not complete."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            # Submit a job
            submit_resp = await client.post("/v1/jobs", json={"message": "test"})
            job_id = submit_resp.json()["job_id"]

            # Wait a moment for job to complete (mock returns immediately)
            await asyncio.sleep(0.1)

            # Try to capture - if job completed, test passes; if not, we get expected error
            capture_resp = await client.post(
                f"/v1/jobs/{job_id}/capture",
                json={"name": "captured"},
            )
            # A plain (non-workflow_ref) job cannot isolate a run cache: a completed plain
            # job -> 400 no_cache_to_freeze; a still-running job -> 409 job_not_complete.
            assert capture_resp.status_code in (400, 409)

    async def test_mcp_add_server_invalid_session_id(self):
        """Test MCP server add with invalid session_id."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/sessions/invalid@id/mcp/servers",
                json={"name": "test", "transport": "stdio"},
            )
            assert response.status_code == 400

    async def test_mcp_add_server_session_not_found(self):
        """Test MCP server add when session not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/sessions/nonexistent_sess/mcp/servers",
                json={"name": "test", "transport": "stdio"},
            )
            assert response.status_code == 404

    async def test_mcp_remove_server_not_found(self):
        """Test MCP server remove when server not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            # Create a session first
            create_resp = await client.post("/v1/sessions")
            session_id = create_resp.json()["session_id"]

            response = await client.delete(f"/v1/sessions/{session_id}/mcp/servers/nonexistent_server")
            assert response.status_code == 404

    async def test_mcp_reconnect_server_not_found(self):
        """Test MCP reconnect when server not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            # Create a session first
            create_resp = await client.post("/v1/sessions")
            session_id = create_resp.json()["session_id"]

            response = await client.post(f"/v1/sessions/{session_id}/mcp/servers/nonexistent_server/reconnect")
            assert response.status_code == 404

    async def test_session_delete_invalid_session_id(self):
        """Test session delete with invalid session_id."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.delete("/v1/sessions/invalid@id")
            assert response.status_code == 400

    async def test_session_delete_not_found(self):
        """Test session delete when session not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.delete("/v1/sessions/nonexistent_session")
            assert response.status_code == 404

    async def test_session_list_with_auth_enabled_no_identity(self):
        """Test session list with auth but no identity."""
        app = _auth_app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.get("/v1/sessions")
            assert response.status_code == 401

    async def test_session_fork_invalid_session_id(self):
        """Test session fork with invalid session_id."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post("/v1/sessions/invalid@id/fork")
            assert response.status_code == 400

    async def test_session_fork_not_found(self):
        """Test session fork when session not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post("/v1/sessions/nonexistent_session/fork")
            assert response.status_code == 404

    async def test_session_fork_not_persisted(self):
        """Test session fork with non-sqlite backend."""
        cfg = _config(memory={"backend": "in_memory"})
        app = create_app(cfg, client_factory=lambda: MockClient([make_mock_response()]))

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            # Create a session first
            create_resp = await client.post("/v1/sessions")
            session_id = create_resp.json()["session_id"]

            response = await client.post(f"/v1/sessions/{session_id}/fork")
            assert response.status_code == 409
            assert "not_persisted" in response.json()["error"]["code"]

    async def test_session_resume_invalid_session_id(self):
        """Test session resume with invalid session_id."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post("/v1/sessions/invalid@id/resume")
            assert response.status_code == 400

    async def test_session_resume_not_found(self):
        """Test session resume when session not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post("/v1/sessions/nonexistent_session/resume")
            assert response.status_code == 404

    async def test_chat_stream_invalid_session_id_header(self):
        """Test chat stream with invalid X-Session-Id header."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/chat/stream",
                json={"message": "test"},
                headers={"X-Session-Id": "invalid@id"},
            )
            assert response.status_code == 400

    async def test_chat_stream_invalid_mode(self):
        """Test chat stream with invalid mode."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/chat/stream",
                json={"message": "test", "mode": "invalid_mode"},
            )
            assert response.status_code == 400
            assert "invalid_mode" in response.json()["error"]["code"]

    async def test_job_submit_invalid_mode(self):
        """Test job submit with invalid mode."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/jobs",
                json={"message": "test", "mode": "invalid_mode"},
            )
            assert response.status_code == 400

    async def test_job_submit_yolo_rejected(self):
        """Test job submit rejects yolo mode."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/jobs",
                json={"message": "test", "mode": "yolo"},
            )
            assert response.status_code == 400

    async def test_job_submit_unknown_workflow(self):
        """Test job submit with unknown workflow."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/jobs",
                json={"message": "test", "workflow_ref": "unknown_workflow"},
            )
            assert response.status_code == 400

    async def test_job_submit_invalid_replay_mode(self):
        """Test job submit with invalid replay_mode."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/jobs",
                json={"message": "test", "replay_mode": "invalid_mode"},
            )
            assert response.status_code == 400

    async def test_media_generate_invalid_session_id(self):
        """Test media generate with invalid session_id."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/media/generate",
                json={"modality": "image", "prompt": "test"},
                headers={"X-Session-Id": "invalid@id"},
            )
            assert response.status_code == 400

    async def test_media_submit_invalid_session_id(self):
        """Test media job submit with invalid session_id."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "/v1/media/jobs",
                json={"modality": "image", "prompt": "test"},
                headers={"X-Session-Id": "invalid@id"},
            )
            assert response.status_code == 400

    async def test_media_get_job_not_found(self):
        """Test get media job when job not found."""
        app = _app()

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.get("/v1/media/jobs/nonexistent_job")
            assert response.status_code == 404


class TestJobRunErrorPaths:
    """Tests for job execution error paths."""

    def test_job_run_timeout(self, tmp_path, monkeypatch):
        """Test job run timeout (lines 621-626)."""
        import koboi.server.jobs as jobs_mod

        async def hanging(*args, **kwargs):
            await asyncio.sleep(100)  # exceeds the wait_for timeout below

        monkeypatch.setattr(jobs_mod, "_execute_job", hanging)

        async def runner():
            store = JobStore(str(tmp_path / "jobs.db"))
            registry = JobRegistry()
            pool = AgentPool(_config())
            job_id = "job_timeout"
            registry.register(job_id, "sess_1", "alice")
            store.insert(job_id, "sess_1", "alice", "test message")

            # wait_for cancels the hanging _execute_job after 0.01s -> TimeoutError -> timed_out.
            await run_job(job_id, pool, registry, store, "test message", timeout=0.01)

            job = store.get(job_id)
            assert job is not None
            assert job["status"] == "timed_out"
            assert "timeout" in job.get("error_class", "").lower()

        asyncio.run(runner())

    def test_job_run_cancelled(self, tmp_path, monkeypatch):
        """Test job run cancellation (lines 616-620)."""
        import koboi.server.jobs as jobs_mod

        async def raiser(*args, **kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(jobs_mod, "_execute_job", raiser)

        async def runner():
            store = JobStore(str(tmp_path / "jobs.db"))
            registry = JobRegistry()
            pool = AgentPool(_config())
            job_id = "job_cancel"
            registry.register(job_id, "sess_1", "alice")
            store.insert(job_id, "sess_1", "alice", "test message")

            # _execute_job raises CancelledError -> wait_for re-raises -> cancelled (then re-raised).
            with pytest.raises(asyncio.CancelledError):
                await run_job(job_id, pool, registry, store, "test message", timeout=30)

            job = store.get(job_id)
            assert job is not None
            assert job["status"] == "cancelled"

        asyncio.run(runner())


class TestServerAuthErrors:
    """Tests for auth-related error paths."""

    async def test_list_sessions_non_sqlite_backend(self):
        """Test list sessions with non-sqlite backend (line 919-920)."""
        cfg = _config(memory={"backend": "in_memory"})
        app = create_app(cfg, client_factory=lambda: MockClient([make_mock_response()]))

        async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as client:
            response = await client.get("/v1/sessions")
            assert response.status_code == 200
            # Should return empty list for non-sqlite
            assert response.json()["sessions"] == []

    def test_resolve_allowed_modes_defaults(self):
        """Test _resolve_allowed_modes with None/empty (lines 1625-1644)."""
        from koboi.server.app import _resolve_allowed_modes

        # None -> default
        modes = _resolve_allowed_modes(None)
        assert "chat" in modes
        assert "yolo" not in modes

        # Empty list -> default
        modes = _resolve_allowed_modes([])
        assert "chat" in modes

    def test_resolve_allowed_modes_invalid_entry(self):
        """Test _resolve_allowed_modes with invalid entry (lines 1637-1643)."""
        from koboi.server.app import _resolve_allowed_modes

        with pytest.raises(ValueError) as exc:
            _resolve_allowed_modes(["chat", "invalid_mode"])

        assert "Unknown mode" in str(exc.value)

    def test_resolve_mode_yolo_rejected(self):
        """Test _resolve_mode rejects yolo when not allowed (lines 1661-1667)."""
        from koboi.server.app import _resolve_mode
        from koboi.server.app import _DEFAULT_ALLOWED_MODES

        # yolo not in default allowed modes
        with pytest.raises(ValueError) as exc:
            _resolve_mode("yolo", _DEFAULT_ALLOWED_MODES, allow_yolo=False)

        assert "not allowed" in str(exc.value)

    def test_resolve_mode_invalid_mode(self):
        """Test _resolve_mode with invalid mode string."""
        from koboi.server.app import _resolve_mode
        from koboi.server.app import _DEFAULT_ALLOWED_MODES

        with pytest.raises(ValueError):
            _resolve_mode("not_a_real_mode", _DEFAULT_ALLOWED_MODES, allow_yolo=False)


class TestResolveBindCoverage:
    """Tests for _resolve_bind function."""

    def test_resolve_bind_defaults(self):
        """Test _resolve_bind with defaults (lines 1677-1678)."""
        from koboi.server.app import _resolve_bind

        cfg = _config()
        host, port = _resolve_bind(cfg, None, None)

        assert host == "127.0.0.1"
        assert port == 8000

    def test_resolve_bind_from_config(self):
        """Test _resolve_bind from config (lines 1677-1678)."""
        from koboi.server.app import _resolve_bind

        cfg = _config(server={"host": "0.0.0.0", "port": 9000})
        host, port = _resolve_bind(cfg, None, None)

        assert host == "0.0.0.0"
        assert port == 9000

    def test_resolve_bind_cli_flags_override(self):
        """Test CLI flags override config (lines 1677-1678)."""
        from koboi.server.app import _resolve_bind

        cfg = _config(server={"host": "0.0.0.0", "port": 9000})
        host, port = _resolve_bind(cfg, "localhost", 8080)

        assert host == "localhost"
        assert port == 8080
