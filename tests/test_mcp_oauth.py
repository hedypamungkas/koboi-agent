"""tests/test_mcp_oauth.py -- OAuth2 client-credentials + refresh for MCP HTTP (G1)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koboi.mcp.auth import OAuthClientCredentialsAuth, OAuthError, build_mcp_auth
from koboi.mcp.http_client import StreamableHTTPMCPClient


class _FakeResp:
    def __init__(self, status_code=200, payload=None, text="{}"):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = {}

    def json(self):
        return self._payload


# --- build_mcp_auth factory ---


class TestBuildMcpAuth:
    def test_none(self):
        assert build_mcp_auth({"type": "none"}) is None
        assert build_mcp_auth(None) is None

    def test_bearer(self):
        from koboi.llm.auth import BearerAuth

        a = build_mcp_auth({"type": "bearer", "token": "abc"})
        assert isinstance(a, BearerAuth)
        assert a.apply({})["Authorization"] == "Bearer abc"

    def test_bearer_empty_token_is_none(self):
        assert build_mcp_auth({"type": "bearer", "token": ""}) is None

    def test_oauth(self):
        a = build_mcp_auth(
            {"type": "oauth", "token_endpoint": "https://idp/token", "client_id": "c", "client_secret": "s"}
        )
        assert isinstance(a, OAuthClientCredentialsAuth)


# --- OAuth strategy ---


class TestOAuthStrategy:
    def test_client_credentials_fetch_then_apply(self, monkeypatch):
        seen = {}

        def fake_post(url, data=None, timeout=None):
            seen["data"] = data
            return _FakeResp(payload={"access_token": "TOK", "expires_in": 3600})

        monkeypatch.setattr("koboi.mcp.auth.httpx.post", fake_post)
        auth = OAuthClientCredentialsAuth(
            token_endpoint="https://idp/token", client_id="c", client_secret="s", scopes="read write"
        )
        h = auth.apply({})
        assert h["Authorization"] == "Bearer TOK"
        assert seen["data"]["grant_type"] == "client_credentials"
        assert seen["data"]["scope"] == "read write"

    def test_seeded_token_skips_fetch_then_refresh_rotates(self, monkeypatch):
        seen = {}

        def fake_post(url, data=None, timeout=None):
            seen.update(data)
            return _FakeResp(payload={"access_token": "NEW", "expires_in": 3600, "refresh_token": "ROT"})

        monkeypatch.setattr("koboi.mcp.auth.httpx.post", fake_post)
        auth = OAuthClientCredentialsAuth(
            token_endpoint="https://idp/token", client_id="c", refresh_token="RT", access_token="SEED"
        )
        # Seeded token present and not expired -> apply uses it without a fetch.
        assert auth.apply({})["Authorization"] == "Bearer SEED"
        assert seen == {}  # no token endpoint call yet

        auth.refresh(force=True)
        assert seen["grant_type"] == "refresh_token"
        assert seen["refresh_token"] == "RT"
        assert auth.apply({})["Authorization"] == "Bearer NEW"  # rotated

    def test_token_endpoint_error_raises(self, monkeypatch):
        monkeypatch.setattr("koboi.mcp.auth.httpx.post", lambda *a, **k: _FakeResp(status_code=401, text="bad"))
        auth = OAuthClientCredentialsAuth(token_endpoint="https://idp/token", client_id="c")
        with pytest.raises(OAuthError, match="HTTP 401"):
            auth.apply({})

    def test_requires_endpoint_and_client_id(self):
        with pytest.raises(ValueError):
            OAuthClientCredentialsAuth(token_endpoint="", client_id="c")


# --- StreamableHTTPMCPClient integration: 401 -> refresh -> retry once ---


class TestStreamableHTTP401Retry:
    def test_401_triggers_oauth_refresh_and_retry(self, monkeypatch):
        monkeypatch.setattr(
            "koboi.mcp.auth.httpx.post",
            lambda *a, **k: _FakeResp(payload={"access_token": "T2", "expires_in": 3600}),
        )
        r401 = MagicMock(status_code=401, headers={}, text='{"error":"expired"}')
        r401.json.return_value = {"error": "expired"}
        r200 = MagicMock(status_code=200, headers={}, text="{}")
        r200.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "ok"}]},
        }
        mock_client = MagicMock()
        mock_client.post.side_effect = [r401, r200]

        c = StreamableHTTPMCPClient(
            url="https://mcp.example.com/ep",
            auth_config={
                "type": "oauth",
                "token_endpoint": "https://idp/token",
                "client_id": "c",
                "client_secret": "s",
            },
        )
        c._client = mock_client  # pretend connected (skip connect()/SSRF)
        result = c._post_json_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}})
        assert mock_client.post.call_count == 2  # initial 401 + retried 200
        assert result["result"]["content"][0]["text"] == "ok"

    def test_bearer_401_does_not_retry(self, monkeypatch):
        # Bearer auth has no supports_refresh -> a 401 must surface as MCPError, not loop.
        r401 = MagicMock(status_code=401, headers={}, text="bad")
        r401.json.return_value = {"error": "expired"}
        mock_client = MagicMock()
        mock_client.post.return_value = r401

        from koboi.mcp.base import MCPError

        c = StreamableHTTPMCPClient(url="https://mcp.example.com/ep", auth_config={"type": "bearer", "token": "static"})
        c._client = mock_client
        with pytest.raises(MCPError, match="HTTP 401"):
            c._post_json_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}})
        assert mock_client.post.call_count == 1  # no retry

    def test_bearer_header_applied_via_strategy(self):
        mock_client = MagicMock()
        r200 = MagicMock(status_code=200, headers={}, text="{}")
        r200.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
        mock_client.post.return_value = r200

        c = StreamableHTTPMCPClient(url="https://mcp.example.com/ep", auth_config={"type": "bearer", "token": "Z"})
        c._client = mock_client
        c._post_json_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        sent_headers = mock_client.post.call_args.kwargs["headers"]
        assert sent_headers["Authorization"] == "Bearer Z"
