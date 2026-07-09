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
