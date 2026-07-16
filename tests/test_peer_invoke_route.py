"""Tests for the A2A inbound receiver route POST /v1/peer/invoke."""

from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport

from koboi.config import Config
from koboi.server.app import create_app
from tests.conftest import MockClient, make_mock_response


def _app(peers_cfg, *, api_keys=None, content="C-answer-42"):
    cfg = Config.from_dict(
        {
            "agent": {"name": "C", "mode": "chat", "system_prompt": "You are C."},
            "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "x"},
            "memory": {"backend": "memory"},
            "peers": peers_cfg,
        }
    )
    return create_app(
        cfg,
        client_factory=lambda: MockClient([make_mock_response(content=content)]),
        api_keys=api_keys,
    )


async def _client(app):
    return httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app))


class TestPeerInvokeRoute:
    async def test_invoke_returns_content(self, app_y):
        async with await _client(app_y) as c:
            r = await c.post("/v1/peer/invoke", json={"message": "hi"}, headers={"Authorization": "Bearer tok-y"})
        assert r.status_code == 200
        body = r.json()
        assert body["content"] == "C-answer-42"
        assert body["peer_id"] == "peer"
        assert body["session_id"].startswith("peer-")

    async def test_401_without_token(self, app_y):
        async with await _client(app_y) as c:
            r = await c.post("/v1/peer/invoke", json={"message": "hi"})
        assert r.status_code == 401

    async def test_401_wrong_token(self, app_y):
        async with await _client(app_y) as c:
            r = await c.post("/v1/peer/invoke", json={"message": "hi"}, headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    async def test_404_when_peers_disabled(self):
        # peers disabled -> need an API key so auth passes; route then 404s.
        app = _app({"enabled": False}, api_keys=["admin"])
        async with await _client(app) as c:
            r = await c.post("/v1/peer/invoke", json={"message": "hi"}, headers={"Authorization": "Bearer admin"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "peers_disabled"

    async def test_ephemeral_session_evicted(self, app_y):
        async with await _client(app_y) as c:
            r = await c.post("/v1/peer/invoke", json={"message": "hi"}, headers={"Authorization": "Bearer tok-y"})
        sid = r.json()["session_id"]
        assert app_y.state.pool.get(sid) is None  # evicted after the call

    async def test_continuity_session_via_header_not_evicted(self, app_y):
        headers = {"Authorization": "Bearer tok-y", "X-Session-Id": "peer-fixed-1"}
        async with await _client(app_y) as c:
            r1 = await c.post("/v1/peer/invoke", json={"message": "first"}, headers=headers)
            r2 = await c.post("/v1/peer/invoke", json={"message": "second"}, headers=headers)
        assert r1.json()["session_id"] == "peer-fixed-1"
        assert r2.json()["session_id"] == "peer-fixed-1"
        assert app_y.state.pool.get("peer-fixed-1") is not None  # continuity sessions stay

    async def test_invalid_mode_400(self, app_y):
        async with await _client(app_y) as c:
            r = await c.post(
                "/v1/peer/invoke",
                json={"message": "hi", "mode": "yolo"},
                headers={"Authorization": "Bearer tok-y"},
            )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_mode"

    async def test_bad_session_id_400(self, app_y):
        async with await _client(app_y) as c:
            r = await c.post(
                "/v1/peer/invoke",
                json={"message": "hi"},
                headers={"Authorization": "Bearer tok-y", "X-Session-Id": "../escape"},
            )
        assert r.status_code == 400

    async def test_act_mode_refused_with_passthrough_sandbox(self, app_y):
        # C3: autonomous peer execution (act) requires a restricted sandbox; the default
        # passthrough sandbox is refused. (chat/plan are exempt -- they mode-block destructive.)
        async with await _client(app_y) as c:
            r = await c.post(
                "/v1/peer/invoke",
                json={"message": "hi", "mode": "act"},
                headers={"Authorization": "Bearer tok-y"},
            )
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "peer_invoke_failed"
        assert "restricted" in r.json()["error"]["message"]

    async def test_message_at_max_length_ok(self, app_y):
        async with await _client(app_y) as c:
            r = await c.post(
                "/v1/peer/invoke",
                json={"message": "x" * 65536},
                headers={"Authorization": "Bearer tok-y"},
            )
        assert r.status_code == 200

    async def test_message_over_max_length_rejected(self, app_y):
        async with await _client(app_y) as c:
            r = await c.post(
                "/v1/peer/invoke",
                json={"message": "x" * 65537},
                headers={"Authorization": "Bearer tok-y"},
            )
        assert r.status_code == 422  # Pydantic max_length=65536

    async def test_malformed_traceparent_ignored(self, app_y):
        # A malformed inbound traceparent is dropped + a fresh root minted (no crash).
        async with await _client(app_y) as c:
            r = await c.post(
                "/v1/peer/invoke",
                json={"message": "hi"},
                headers={"Authorization": "Bearer tok-y", "traceparent": "junk"},
            )
        assert r.status_code == 200
        assert r.json()["content"] == "C-answer-42"


# Fixture defined at module level (pytest discovers it).


@pytest.fixture
def app_y():
    # Peer-only receiver: inbound token, no API keys (auth fix allows this).
    return _app({"enabled": True, "inbound_tokens": ["tok-y"]})
