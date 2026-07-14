"""Tests for B5 -- mid-conversation handover webhook.

``_emit_handover_webhook`` (jobs.py) fires on the chat-path HandoverEvent (unlike
``_emit_job_webhooks`` which fires on terminal job status). Reuses the jobs
``_post_webhook`` (2-retry, fail-safe) + HMAC + ``_WEBHOOK_TASKS``. Mirrors
``tests/test_job_webhooks.py``.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from koboi.server.jobs import _emit_handover_webhook


class TestEmitHandoverWebhook:
    async def test_no_webhooks_is_noop(self, monkeypatch):
        called = []
        monkeypatch.setattr("koboi.server.jobs._post_webhook", lambda *a, **kw: called.append(1) or _noop())
        _emit_handover_webhook(None, "s1", "h1", "r", "s")
        _emit_handover_webhook([], "s1", "h1", "r", "s")
        await _drain()
        assert called == []

    async def test_schedules_post_with_correct_payload(self, monkeypatch):
        captured = []
        async def _fake_post(url, body, headers, timeout):
            captured.append((url, json.loads(body), headers))
        monkeypatch.setattr("koboi.server.jobs._post_webhook", _fake_post)
        _emit_handover_webhook(
            [{"url": "https://hook.example/h", "secret": "shh"}],
            session_id="s1", handover_id="h1", reason="low confidence", summary="digest",
        )
        await _drain()
        assert len(captured) == 1
        url, payload, headers = captured[0]
        assert url == "https://hook.example/h"
        assert payload["event"] == "handover.requested"
        assert payload["session_id"] == "s1"
        assert payload["handover_id"] == "h1"
        assert payload["reason"] == "low confidence"
        assert payload["summary"] == "digest"
        # HMAC signature present (verified against the body the receiver would see).
        body_bytes = json.dumps(payload).encode()
        expected_sig = hmac.new(b"shh", body_bytes, hashlib.sha256).hexdigest()
        assert headers["X-Koboi-Signature"] == f"sha256={expected_sig}"

    async def test_no_secret_no_signature(self, monkeypatch):
        captured = []
        async def _fake_post(url, body, headers, timeout):
            captured.append(headers)
        monkeypatch.setattr("koboi.server.jobs._post_webhook", _fake_post)
        _emit_handover_webhook([{"url": "https://hook.example/h"}], "s1", "h1", "r", "")
        await _drain()
        assert "X-Koboi-Signature" not in captured[0]


class TestHandoverWebhookE2E:
    async def test_chat_handover_fires_webhook(self, monkeypatch):
        pytest.importorskip("fastapi")
        import httpx
        from koboi.config import Config
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        captured = []

        async def _fake_post(url, body, headers, timeout):
            captured.append(json.loads(body))

        monkeypatch.setattr("koboi.server.jobs._post_webhook", _fake_post)

        cfg = Config.from_dict(
            {
                "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
                "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "test"},
                "memory": {"backend": "in_memory"},
                "sandbox": {"backend": "restricted"},
                "server": {"auth_required": False},
                "tools": {"builtin": ["transfer_to_human"]},
                "handover": {"webhooks": [{"url": "https://cs.example/h", "secret": "k"}]},
            },
            validate=True,
        )
        handover_resp = make_mock_response(
            content="transferring",
            tool_calls=[make_mock_tool_call("transfer_to_human", {"reason": "complex", "summary": "cust"})],
        )
        app = create_app(cfg, client_factory=lambda: MockClient([handover_resp]), enable_cors=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            async with c.stream(
                "POST", "/v1/chat/stream", json={"message": "help", "mode": "act"}, headers={"X-Session-Id": "s-wh"}
            ) as r:
                await r.aread()
        await _drain()
        assert any(p["event"] == "handover.requested" and p["reason"] == "complex" for p in captured), captured


# ---- helpers ----


async def _noop() -> None:
    return None


async def _drain(timeout: float = 5.0) -> None:
    from koboi.server.jobs import drain_webhook_tasks

    await drain_webhook_tasks(timeout)
