"""Tests for MCPServer dispatch edge cases and MCPClient transport."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from koboi.mcp.server import MCPServer
from koboi.mcp.client import MCPClient, MCPError, register_mcp_tools


class TestMCPServerDispatch:
    def test_handle_initialize(self):
        server = MCPServer(name="test", version="1.0.0")
        result = server._handle_initialize(1, {})
        assert result["serverInfo"]["name"] == "test"
        assert result["protocolVersion"] == "2024-11-05"

    def test_handle_tools_list_empty(self):
        server = MCPServer(name="test")
        result = server._handle_tools_list(1, {})
        assert result["tools"] == []

    def test_handle_tools_list_with_tools(self):
        server = MCPServer(name="test")

        @server.tool("calc", "Calculate", {"type": "object"})
        def calc():
            return 42

        result = server._handle_tools_list(1, {})
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "calc"

    def test_handle_tools_call_success(self):
        server = MCPServer(name="test")

        @server.tool(
            "add", "Add numbers", {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}
        )
        def add(a, b):
            return a + b

        result = server._handle_tools_call(1, {"name": "add", "arguments": {"a": 2, "b": 3}})
        assert result["content"][0]["text"] == "5"

    def test_handle_tools_call_unknown(self):
        server = MCPServer(name="test")
        result = server._handle_tools_call(1, {"name": "nonexistent", "arguments": {}})
        assert result["isError"] is True

    def test_dispatch_initialized_notification(self):
        server = MCPServer(name="test")
        # Should not raise, no response needed
        server._dispatch({"method": "initialized", "id": None})

    def test_dispatch_unknown_method(self):
        server = MCPServer(name="test")
        responses = []
        server._write_response = lambda msg: responses.append(msg)
        server._dispatch({"method": "unknown", "id": 1})
        assert len(responses) == 1
        assert "error" in responses[0]

    def test_dispatch_handler_exception(self):
        server = MCPServer(name="test")

        @server.tool("bad", "Bad tool", {"type": "object"})
        def bad():
            raise ValueError("oops")

        responses = []
        server._write_response = lambda msg: responses.append(msg)
        server._dispatch({"method": "tools/call", "id": 1, "params": {"name": "bad", "arguments": {}}})
        assert responses[0]["result"]["isError"] is True


class TestMCPError:
    def test_error_creation(self):
        err = MCPError(code=-32601, message="Method not found", data={"method": "foo"})
        assert err.code == -32601
        assert "Method not found" in str(err)
        assert err.data == {"method": "foo"}


class TestMCPClientTransport:
    def test_write_raises_when_not_connected(self):
        client = MCPClient(["echo", "test"])
        with pytest.raises(RuntimeError, match="not connected"):
            client._write({"jsonrpc": "2.0", "method": "test"})

    def test_read_raises_when_not_connected(self):
        client = MCPClient(["echo", "test"])
        with pytest.raises(RuntimeError, match="not connected"):
            client._read()

    def test_close_when_not_connected(self):
        client = MCPClient(["echo", "test"])
        client.close()  # Should not raise

    def test_close_with_process(self):
        client = MCPClient(["echo", "test"])
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        client._process = mock_process
        client.close()
        mock_process.terminate.assert_called_once()

    def test_close_with_timeout_kill(self):
        import subprocess

        client = MCPClient(["echo", "test"])
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
        client._process = mock_process
        client.close()
        mock_process.kill.assert_called_once()

    def test_server_info_property(self):
        client = MCPClient(["echo", "test"])
        assert client.server_info == {}
        client._server_info = {"name": "test"}
        assert client.server_info == {"name": "test"}

    def test_send_request_error_response(self):
        client = MCPClient(["echo", "test"])
        client._process = MagicMock()
        # Mock _write and _read
        client._write = MagicMock()
        client._read = MagicMock(
            return_value={
                "error": {"code": -32600, "message": "Invalid Request"},
            }
        )
        with pytest.raises(MCPError) as exc_info:
            client._send_request("test")
        assert exc_info.value.code == -32600

    def test_send_notification(self):
        client = MCPClient(["echo", "test"])
        client._process = MagicMock()
        client._write = MagicMock()
        client._send_notification("initialized")
        client._write.assert_called_once()

    def test_call_tool_sync(self):
        client = MCPClient(["echo", "test"])
        client._send_request = MagicMock(
            return_value={
                "content": [
                    {"type": "text", "text": "result1"},
                    {"type": "text", "text": "result1"},  # duplicate
                    {"type": "text", "text": "result2"},
                ],
            }
        )
        result = client._call_tool_sync("test_tool", {"arg": "val"})
        assert "result1" in result
        assert "result2" in result
        # Should deduplicate
        assert result.count("result1") == 1

    def test_drain_stderr_no_process(self):
        client = MCPClient(["echo", "test"])
        client._drain_stderr()  # Should not raise

    def test_read_connection_closed(self):
        client = MCPClient(["echo", "test"])
        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = b""
        client._process = mock_process
        with pytest.raises(RuntimeError, match="closed connection"):
            client._read()


class TestRegisterMCPTools:
    def test_registers_tools(self):
        mock_client = MagicMock()
        mock_client.discover_tools.return_value = [
            MagicMock(name="tool1", description="desc1", input_schema={"type": "object"}),
            MagicMock(name="tool2", description="desc2", input_schema={"type": "object"}),
        ]
        registry = MagicMock()
        registered = register_mcp_tools(mock_client, registry)
        assert len(registered) == 2
        assert registry.register.call_count == 2
