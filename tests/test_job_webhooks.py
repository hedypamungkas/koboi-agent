"""Tests for koboi/server/jobs.py outbound job webhooks (`jobs.webhooks`).

Unit tests cover delivery (POST fire, event filtering, HMAC signing, payload,
fail-safe on 5xx / network error, no-op) by mocking the HTTP layer; an integration
test drives ``run_job`` and asserts webhooks are scheduled on terminal status.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import koboi.server.jobs as jobs
from koboi.server.jobs import (
    JobStore,
    _deliver_webhooks,
    _emit_job_webhooks,
    _post_webhook,
    run_job,
)


def _row(**over):
    base = {
        "job_id": "job_1",
        "session_id": "sess_1",
        "owner": "alice",
        "status": "completed",
        "result_json": json.dumps({"content": "Hello"}),
        "error": None,
        "error_class": None,
        "retriable": 0,
        "created_at": 1.0,
        "updated_at": 2.0,
    }
    base.update(over)
    return base


class _FakeStore:
    def __init__(self, row):
        self._row = row

    def get(self, job_id):
        return self._row


def _make_fake_httpx(status=200, exc=None):
    """Return a callable standing in for httpx.AsyncClient."""

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            if exc is not None:
                raise exc

            class _Resp:
                pass

            _Resp.status_code = status
            return _Resp()

    return lambda *a, **kw: _Client()


class TestDeliverWebhooks:
    async def test_fires_post_on_completed(self):
        captured = AsyncMock()
        with patch.object(jobs, "_post_webhook", captured):
            await _deliver_webhooks([{"url": "http://x/h", "events": []}], _FakeStore(_row()), "job_1", "completed")
        assert captured.call_count == 1
        url, body, headers, _timeout = captured.call_args.args
        assert url == "http://x/h"
        assert json.loads(body)["status"] == "completed"

    async def test_event_filtering(self):
        captured = AsyncMock()
        with patch.object(jobs, "_post_webhook", captured):
            await _deliver_webhooks(
                [{"url": "http://x/h", "events": ["failed"]}], _FakeStore(_row()), "job_1", "completed"
            )
        assert captured.call_count == 0  # completed not in [failed]

        captured2 = AsyncMock()
        with patch.object(jobs, "_post_webhook", captured2):
            await _deliver_webhooks(
                [{"url": "http://x/h", "events": ["failed"]}], _FakeStore(_row(status="failed")), "job_1", "failed"
            )
        assert captured2.call_count == 1

    async def test_hmac_signature_header(self):
        secret = "s3cr3t"
        captured = AsyncMock()
        with patch.object(jobs, "_post_webhook", captured):
            await _deliver_webhooks(
                [{"url": "http://x/h", "events": [], "secret": secret}], _FakeStore(_row()), "job_1", "completed"
            )
        _url, body, headers, _timeout = captured.call_args.args
        expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert headers["X-Koboi-Signature"] == expected

    async def test_no_secret_no_signature_header(self):
        captured = AsyncMock()
        with patch.object(jobs, "_post_webhook", captured):
            await _deliver_webhooks([{"url": "http://x/h", "events": []}], _FakeStore(_row()), "job_1", "completed")
        _url, _body, headers, _timeout = captured.call_args.args
        assert "X-Koboi-Signature" not in headers

    async def test_payload_content(self):
        captured = AsyncMock()
        with patch.object(jobs, "_post_webhook", captured):
            await _deliver_webhooks([{"url": "http://x/h", "events": []}], _FakeStore(_row()), "job_1", "completed")
        _url, body, _h, _t = captured.call_args.args
        payload = json.loads(body)
        assert payload["job_id"] == "job_1"
        assert payload["status"] == "completed"
        assert payload["event"] == "job.completed"
        assert payload["result"] == {"content": "Hello"}
        assert payload["owner"] == "alice"

    async def test_no_webhooks_is_noop(self):
        captured = AsyncMock()
        with patch.object(jobs, "_post_webhook", captured):
            await _deliver_webhooks([], _FakeStore(_row()), "job_1", "completed")
        assert captured.call_count == 0

    async def test_missing_row_is_noop(self):
        captured = AsyncMock()
        with patch.object(jobs, "_post_webhook", captured):
            await _deliver_webhooks([{"url": "http://x/h", "events": []}], _FakeStore(None), "job_1", "completed")
        assert captured.call_count == 0


class TestPostWebhookFailSafe:
    async def test_5xx_retried_then_logged_no_raise(self):
        with patch.object(jobs.httpx, "AsyncClient", _make_fake_httpx(status=500)):
            await _post_webhook("http://x/h", b"{}", {}, 1.0)  # must not raise

    async def test_connect_error_no_raise(self):
        with patch.object(jobs.httpx, "AsyncClient", _make_fake_httpx(exc=httpx.ConnectError("refused"))):
            await _post_webhook("http://x/h", b"{}", {}, 1.0)

    async def test_success_returns_silently(self):
        with patch.object(jobs.httpx, "AsyncClient", _make_fake_httpx(status=200)):
            await _post_webhook("http://x/h", b"{}", {}, 1.0)


class TestEmitScheduling:
    async def test_fire_and_forget_schedules_task(self):
        delivered = AsyncMock()
        before = set(jobs._WEBHOOK_TASKS)
        with patch.object(jobs, "_deliver_webhooks", delivered):
            _emit_job_webhooks([{"url": "http://x/h", "events": []}], _FakeStore(_row()), "job_1", "completed")
            new_tasks = jobs._WEBHOOK_TASKS - before
            if new_tasks:
                await asyncio.wait_for(asyncio.gather(*new_tasks), timeout=2)
        assert delivered.call_count == 1

    def test_no_webhooks_no_task(self):
        before = len(jobs._WEBHOOK_TASKS)
        _emit_job_webhooks([], _FakeStore(_row()), "job_1", "completed")
        assert len(jobs._WEBHOOK_TASKS) == before


class _FakeRegistry:
    def __init__(self):
        self.terminals = []

    def get(self, job_id):
        return True  # any non-None value so run_job proceeds

    def set_terminal(self, job_id, status):
        self.terminals.append((job_id, status))


class TestRunJobIntegration:
    async def test_completed_emits_webhook(self, tmp_path):
        store = JobStore(db_path=str(tmp_path / "j.db"))
        store.insert("job_1", "sess_1", "alice", "hi", None, None, None)
        registry = _FakeRegistry()
        spy = MagicMock()

        async def _ok(*a, **kw):
            return "done"

        with patch.object(jobs, "_execute_job", _ok), patch.object(jobs, "_emit_job_webhooks", spy):
            await run_job("job_1", None, registry, store, "hi", timeout=10, webhooks=[{"url": "http://x/h"}])
        spy.assert_called_once()
        assert spy.call_args.args[3] == "completed"  # status arg

    async def test_failed_emits_webhook(self, tmp_path):
        store = JobStore(db_path=str(tmp_path / "j.db"))
        store.insert("job_1", "sess_1", "alice", "hi", None, None, None)
        registry = _FakeRegistry()
        spy = MagicMock()

        async def _boom(*a, **kw):
            raise RuntimeError("kaboom")

        with patch.object(jobs, "_execute_job", _boom), patch.object(jobs, "_emit_job_webhooks", spy):
            await run_job("job_1", None, registry, store, "hi", timeout=10, webhooks=[{"url": "http://x/h"}])
        assert spy.call_args.args[3] == "failed"

    async def test_cancelled_emits_webhook(self, tmp_path):
        store = JobStore(db_path=str(tmp_path / "j.db"))
        store.insert("job_1", "sess_1", "alice", "hi", None, None, None)
        registry = _FakeRegistry()
        spy = MagicMock()

        async def _cancelled(*a, **kw):
            raise asyncio.CancelledError()

        with (
            patch.object(jobs, "_execute_job", _cancelled),
            patch.object(jobs, "_emit_job_webhooks", spy),
            pytest.raises(asyncio.CancelledError),
        ):
            await run_job("job_1", None, registry, store, "hi", timeout=10, webhooks=[{"url": "http://x/h"}])
        spy.assert_called_once()
        assert spy.call_args.args[3] == "cancelled"


class TestWebhookTaskLifecycle:
    """The fire-and-forget task's own failure must be surfaced, not swallowed
    (mirrors CommandHook._on_bg_done), and pending tasks must drain on shutdown."""

    async def test_exception_in_deliver_is_logged_not_swallowed(self, caplog):
        # A bug in _deliver_webhooks itself (not a network/5xx failure inside
        # _post_webhook) must not vanish as an unretrieved-task-exception.
        with patch.object(jobs, "_deliver_webhooks", side_effect=ValueError("boom")):
            with caplog.at_level(logging.ERROR):
                jobs._emit_job_webhooks([{"url": "http://x/h"}], _FakeStore(_row()), "job_1", "completed")
                await asyncio.wait_for(asyncio.gather(*jobs._WEBHOOK_TASKS, return_exceptions=True), timeout=2)
        assert any("job webhook delivery failed" in r.message for r in caplog.records)

    async def test_drain_awaits_pending_tasks(self):
        started = asyncio.Event()
        finished = asyncio.Event()

        async def _slow(*a, **kw):
            started.set()
            await asyncio.sleep(0.05)
            finished.set()

        with patch.object(jobs, "_deliver_webhooks", _slow):
            jobs._emit_job_webhooks([{"url": "http://x/h"}], _FakeStore(_row()), "job_1", "completed")
            await started.wait()
            assert not finished.is_set()
            await jobs.drain_webhook_tasks(timeout=2)
        assert finished.is_set()

    async def test_drain_is_noop_when_nothing_pending(self):
        await jobs.drain_webhook_tasks(timeout=1)  # must not raise / hang


class TestConfigThreadingE2E:
    """End-to-end: jobs.webhooks in YAML config actually reaches _emit_job_webhooks
    through create_app -> _register_routes -> _start_job -> run_job. Exercises the
    exact wiring that broke 81 tests during development (routes live in
    _register_routes, not create_app) -- a regression here must fail loudly."""

    async def test_completed_job_fires_configured_webhook(self, tmp_path):
        pytest.importorskip("fastapi")
        import httpx
        from httpx import ASGITransport

        from koboi.config import Config
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response

        cfg = Config.from_dict(
            {
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
                "jobs": {"webhooks": [{"url": "http://webhook.example/h", "events": ["completed"]}]},
            },
            validate=True,
        )
        delivered = []

        async def _fake_post_webhook(url, body, headers, timeout):
            delivered.append((url, json.loads(body)))

        app = create_app(
            cfg,
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
        )
        with patch.object(jobs, "_post_webhook", _fake_post_webhook):
            async with httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app)) as c:
                r = await c.post("/v1/jobs", json={"message": "do"})
                assert r.status_code == 202
                job_id = r.json()["job_id"]
                deadline = asyncio.get_event_loop().time() + 10
                while asyncio.get_event_loop().time() < deadline:
                    body = (await c.get(f"/v1/jobs/{job_id}")).json()
                    if body["status"] == "completed":
                        break
                    await asyncio.sleep(0.1)
                else:
                    pytest.fail("job never completed")
            # fire-and-forget: give the webhook task a moment to run.
            await asyncio.wait_for(asyncio.gather(*jobs._WEBHOOK_TASKS, return_exceptions=True), timeout=5)

        assert len(delivered) == 1
        url, payload = delivered[0]
        assert url == "http://webhook.example/h"
        assert payload["job_id"] == job_id
        assert payload["status"] == "completed"
