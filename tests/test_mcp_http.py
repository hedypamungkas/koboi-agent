"""Tests for koboi.mcp.http_client -- StreamableHTTPMCPClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koboi.mcp.base import MCPError
from koboi.mcp.http_client import StreamableHTTPMCPClient
from koboi.tools.registry import ToolRegistry
from koboi.types import MCPToolInfo


# --- Helpers ---


def _make_response(status=200, json_data=None, headers=None, text=None, content_type="application/json"):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": content_type, **(headers or {})}
    if json_data is not None:
        resp.json.return_value = json_data
    if text is not None:
        resp.text = text
    return resp


def _init_response(session_id=None):
    """Mock initialize response."""
    headers = {}
    if session_id:
        headers["mcp-session-id"] = session_id
    return _make_response(
        json_data={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-server", "version": "1.0"},
            },
        },
        headers=headers,
    )


def _notification_response():
    """Mock response for notifications (202 Accepted)."""
    return _make_response(status=202)


def _tools_list_response(tools=None):
    """Mock tools/list response."""
    if tools is None:
        tools = [{"name": "echo", "description": "Echo input", "inputSchema": {"type": "object"}}]
    return _make_response(
        json_data={
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"tools": tools},
        }
    )


def _tool_call_response(text="hello world"):
    """Mock tools/call response."""
    return _make_response(
        json_data={
            "jsonrpc": "2.0",
            "id": 3,
            "result": {"content": [{"type": "text", "text": text}]},
        }
    )


def _error_response(code=-32600, message="Invalid request"):
    """Mock JSON-RPC error response."""
    return _make_response(
        json_data={
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": code, "message": message},
        }
    )


# --- Tests ---


class TestStreamableHTTPMCPClientInit:
    def test_defaults(self):
        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        assert client._url == "https://example.com/mcp"
        assert client._session_id is None
        assert client._client is None
        assert client._timeout == 30.0

    def test_strips_trailing_slash(self):
        client = StreamableHTTPMCPClient(url="https://example.com/mcp/")
        assert client._url == "https://example.com/mcp"

    def test_custom_timeout(self):
        client = StreamableHTTPMCPClient(url="https://example.com/mcp", timeout=60.0)
        assert client._timeout == 60.0

    def test_auth_config(self):
        client = StreamableHTTPMCPClient(
            url="https://example.com/mcp",
            auth_config={"type": "bearer", "token": "test-token"},
        )
        assert client._auth_config["type"] == "bearer"

    def test_extra_headers(self):
        client = StreamableHTTPMCPClient(
            url="https://example.com/mcp",
            headers={"X-Custom": "value"},
        )
        assert client._extra_headers == {"X-Custom": "value"}


class TestStreamableHTTPMCPClientSSRF:
    def test_connect_ssrf_blocked(self):
        # M4: a loopback MCP URL is rejected before any connection (no network).
        client = StreamableHTTPMCPClient(url="http://127.0.0.1:1/mcp")
        with pytest.raises(MCPError, match="SSRF-blocked"):
            client.connect()


class TestStdioAllowlist:
    """M4: MCP stdio command allow-list (facade._create_mcp_client)."""

    def test_default_runner_allowed(self):
        from koboi.facade import _create_mcp_client

        client = _create_mcp_client({"transport": "stdio", "command": "npx", "args": ["-y", "x"]}, "stdio", MagicMock())
        assert client is not None

    def test_basename_runner_allowed(self):
        from koboi.facade import _create_mcp_client

        client = _create_mcp_client({"transport": "stdio", "command": "/usr/bin/python3"}, "stdio", MagicMock())
        assert client is not None

    def test_unknown_runner_rejected(self):
        from koboi.facade import _create_mcp_client

        with pytest.raises(ValueError, match="not in allow-list"):
            _create_mcp_client({"transport": "stdio", "command": "/usr/local/bin/evil"}, "stdio", MagicMock())

    def test_config_extension_allows_custom_runner(self):
        from koboi.facade import _create_mcp_client

        class _Cfg:
            def get(self, *path, default=None):
                return ["my-runner"] if path == ("mcp", "allowlist_commands") else default

        client = _create_mcp_client({"transport": "stdio", "command": "my-runner"}, "stdio", MagicMock(), config=_Cfg())
        assert client is not None


class TestStreamableHTTPMCPClientConnect:
    @pytest.fixture(autouse=True)
    def _bypass_ssrf(self):
        # M4: connect() now runs _check_url_ssrf (real DNS). Bypass it here so the
        # connect unit tests don't depend on network/DNS for example.com.
        with patch("koboi.tools.builtin.web._check_url_ssrf", return_value=None):
            yield

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_connect_success(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        info = client.connect()

        assert info["serverInfo"]["name"] == "test-server"
        assert client._client is not None
        assert mock_client.post.call_count == 2

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_connect_captures_session_id(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(session_id="abc-123"),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()

        assert client._session_id == "abc-123"

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_connect_error_response(self, MockClient):
        MockClient.return_value.post.return_value = _error_response()

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        with pytest.raises(MCPError, match="-32600"):
            client.connect()

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_connect_http_error(self, MockClient):
        MockClient.return_value.post.return_value = _make_response(
            status=401,
            text="Unauthorized",
        )

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        with pytest.raises(MCPError, match="HTTP 401"):
            client.connect()

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_connect_cleans_up_on_failure(self, MockClient):
        MockClient.return_value.post.side_effect = Exception("connection failed")

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        with pytest.raises(Exception, match="connection failed"):
            client.connect()

        assert client._client is None


class TestStreamableHTTPMCPClientDiscoverTools:
    @patch("koboi.mcp.http_client.httpx.Client")
    def test_discover_tools(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
            _tools_list_response(),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()
        tools = client.discover_tools()

        assert len(tools) == 1
        assert tools[0].name == "echo"
        assert tools[0].description == "Echo input"

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_discover_multiple_tools(self, MockClient):
        tools_data = [
            {"name": "tool1", "description": "First", "inputSchema": {"type": "object"}},
            {
                "name": "tool2",
                "description": "Second",
                "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        ]
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
            _tools_list_response(tools_data),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()
        tools = client.discover_tools()

        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"


class TestStreamableHTTPMCPClientCallTool:
    @patch("koboi.mcp.http_client.httpx.Client")
    async def test_call_tool(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
            _tool_call_response("result from tool"),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()
        result = await client.call_tool("echo", {"text": "hello"})

        assert result == "result from tool"

    @patch("koboi.mcp.http_client.httpx.Client")
    async def test_call_tool_deduplicates_text(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
            _make_response(
                json_data={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "result": {
                        "content": [
                            {"type": "text", "text": "same"},
                            {"type": "text", "text": "same"},
                            {"type": "text", "text": "different"},
                        ]
                    },
                }
            ),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()
        result = await client.call_tool("echo", {})

        assert result.count("same") == 1
        assert "different" in result

    @patch("koboi.mcp.http_client.httpx.Client")
    async def test_call_tool_empty_content(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
            _make_response(
                json_data={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "result": {"content": []},
                }
            ),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()
        result = await client.call_tool("echo", {})

        # Falls back to str(result)
        assert isinstance(result, str)


class TestStreamableHTTPMCPClientSSE:
    def test_parse_sse_response(self):
        sse_text = (
            'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"ok"}]}}\n\n'
        )
        response = _make_response(text=sse_text, content_type="text/event-stream")

        result = StreamableHTTPMCPClient._parse_sse_response(response)
        assert result["result"]["content"][0]["text"] == "ok"

    def test_parse_sse_skips_non_response_events(self):
        sse_text = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n'
            "\n"
            "event: message\n"
            'data: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n'
            "\n"
        )
        response = _make_response(text=sse_text, content_type="text/event-stream")

        result = StreamableHTTPMCPClient._parse_sse_response(response)
        assert "tools" in result["result"]

    def test_parse_sse_no_response_raises(self):
        sse_text = "event: message\ndata: not-json\n\n"
        response = _make_response(text=sse_text, content_type="text/event-stream")

        with pytest.raises(MCPError, match="No JSON-RPC response"):
            StreamableHTTPMCPClient._parse_sse_response(response)

    @patch("koboi.mcp.http_client.httpx.Client")
    async def test_call_tool_with_sse_response(self, MockClient):
        sse_text = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"sse result"}]}}\n'
            "\n"
        )
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
            _make_response(text=sse_text, content_type="text/event-stream"),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()
        result = await client.call_tool("test", {})

        assert result == "sse result"


class TestStreamableHTTPMCPClientSession:
    @patch("koboi.mcp.http_client.httpx.Client")
    def test_session_id_sent_in_subsequent_requests(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(session_id="sess-abc"),
            _notification_response(),
            _tools_list_response(),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()
        client.discover_tools()

        # Check that session ID was included in tools/list request
        tools_call = mock_client.post.call_args_list[2]  # 3rd call is tools/list
        (tools_call[1].get("headers") or tools_call[0][2] if len(tools_call[0]) > 2 else tools_call[1].get("headers"))
        # The headers are passed as keyword arg
        call_kwargs = mock_client.post.call_args_list[2]
        sent_headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert sent_headers.get("Mcp-Session-Id") == "sess-abc"

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_no_session_id_when_not_returned(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),  # no session id
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()

        assert client._session_id is None


class TestStreamableHTTPMCPClientAuth:
    @patch("koboi.mcp.http_client.httpx.Client")
    def test_bearer_token_in_headers(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(
            url="https://example.com/mcp",
            auth_config={"type": "bearer", "token": "my-token"},
        )
        client.connect()

        # Check initialize request had Authorization header
        init_call = mock_client.post.call_args_list[0]
        sent_headers = init_call.kwargs.get("headers", init_call[1].get("headers", {}))
        assert sent_headers.get("Authorization") == "Bearer my-token"

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_no_auth_header_when_none(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.connect()

        init_call = mock_client.post.call_args_list[0]
        sent_headers = init_call.kwargs.get("headers", init_call[1].get("headers", {}))
        assert "Authorization" not in sent_headers

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_extra_headers_included(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.post.side_effect = [
            _init_response(),
            _notification_response(),
        ]

        client = StreamableHTTPMCPClient(
            url="https://example.com/mcp",
            headers={"X-Custom": "value"},
        )
        client.connect()

        init_call = mock_client.post.call_args_list[0]
        sent_headers = init_call.kwargs.get("headers", init_call[1].get("headers", {}))
        assert sent_headers.get("X-Custom") == "value"


class TestStreamableHTTPMCPClientClose:
    def test_close_with_client(self):
        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        mock_http = MagicMock()
        client._client = mock_http
        client.close()
        mock_http.close.assert_called_once()
        assert client._client is None

    def test_close_without_client(self):
        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client.close()  # should not raise


class TestStreamableHTTPMCPClientErrorHandling:
    @patch("koboi.mcp.http_client.httpx.Client")
    def test_connection_error(self, MockClient):
        import httpx

        MockClient.return_value.post.side_effect = httpx.ConnectError("Connection refused")

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        with pytest.raises(MCPError, match="Connection failed"):
            client.connect()

    @patch("koboi.mcp.http_client.httpx.Client")
    def test_timeout_error(self, MockClient):
        import httpx

        MockClient.return_value.post.side_effect = httpx.TimeoutException("Timed out")

        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        with pytest.raises(MCPError, match="timed out"):
            client.connect()

    def test_not_connected_raises(self):
        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        with pytest.raises(RuntimeError, match="not connected"):
            client._post_json_rpc({"jsonrpc": "2.0", "method": "test"})


class TestRegisterMCPToolsHTTP:
    def test_register_with_http_client(self):
        """Verify register_mcp_tools works with StreamableHTTPMCPClient."""
        client = StreamableHTTPMCPClient(url="https://example.com/mcp")
        client._tools = [
            MCPToolInfo(name="tool1", description="Test tool", input_schema={"type": "object"}),
        ]

        registry = ToolRegistry()
        client.discover_tools = MagicMock(return_value=client._tools)
        client.call_tool = AsyncMock(return_value="ok")

        from koboi.mcp.base import register_mcp_tools

        registered = register_mcp_tools(client, registry)

        assert registered == ["tool1"]
        assert "tool1" in registry._tools
