"""Tests for koboi/mcp/client.py and koboi/mcp/server.py — expanded coverage."""

from __future__ import annotations

import json
import sys
import subprocess
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from koboi.mcp.client import MCPClient, MCPError, register_mcp_tools
from koboi.mcp.server import MCPServer
from koboi.tools.registry import ToolRegistry
from koboi.types import MCPToolInfo


class TestMCPServer:
    def test_tool_registration(self):
        server = MCPServer(name="test-server", version="1.0")

        @server.tool(
            name="echo",
            description="Echo input",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )
        def echo(text: str) -> str:
            return text

        assert "echo" in server._tools
        assert server._tools["echo"].description == "Echo input"

    def test_handle_initialize(self):
        server = MCPServer(name="test", version="2.0")
        result = server._handle_initialize(1, {})
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "test"
        assert result["serverInfo"]["version"] == "2.0"
        assert "tools" in result["capabilities"]

    def test_handle_tools_list(self):
        server = MCPServer(name="test")

        @server.tool(name="greet", description="Say hi", input_schema={})
        def greet():
            return "hi"

        result = server._handle_tools_list(1, {})
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "greet"

    def test_handle_tools_call(self):
        server = MCPServer(name="test")

        @server.tool(name="add", description="Add", input_schema={})
        def add(a: int, b: int) -> int:
            return a + b

        result = server._handle_tools_call(1, {"name": "add", "arguments": {"a": 2, "b": 3}})
        assert result["content"][0]["text"] == "5"

    def test_handle_tools_call_unknown(self):
        server = MCPServer(name="test")
        result = server._handle_tools_call(1, {"name": "nonexistent", "arguments": {}})
        assert result["isError"] is True

    def test_dispatch_unknown_method(self):
        server = MCPServer(name="test")
        responses = []
        original_write = server._write_response

        def capture_write(msg):
            responses.append(msg)

        server._write_response = capture_write

        server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "unknown/method"})
        assert len(responses) == 1
        assert responses[0]["error"]["code"] == -32601

    def test_dispatch_initialized_notification(self):
        server = MCPServer(name="test")
        # Notification has no id
        server._dispatch({"jsonrpc": "2.0", "method": "initialized"})
        # Should not crash, no response written

    def test_dispatch_tool_exception(self):
        server = MCPServer(name="test")

        @server.tool(name="failing", description="fails", input_schema={})
        def failing():
            raise ValueError("boom")

        responses = []
        server._write_response = lambda msg: responses.append(msg)

        server._dispatch(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "failing", "arguments": {}}}
        )
        assert len(responses) == 1
        assert responses[0]["result"]["isError"] is True


class TestMCPClient:
    def test_client_init(self):
        client = MCPClient(server_command=["echo", "test"])
        assert client.server_command == ["echo", "test"]
        assert client._process is None

    def test_server_info_property(self):
        client = MCPClient(server_command=["echo"])
        client._server_info = {"name": "test"}
        assert client.server_info == {"name": "test"}

    def test_send_request_error_response(self):
        client = MCPClient(server_command=["test"])
        client._process = MagicMock()
        client._process.stdin = MagicMock()
        client._process.stdout = MagicMock()

        error_response = (
            json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "Invalid request"}}).encode()
            + b"\n"
        )
        client._process.stdout.readline.return_value = error_response

        with pytest.raises(MCPError) as exc_info:
            client._send_request("test_method")
        assert exc_info.value.code == -32600

    def test_send_notification(self):
        client = MCPClient(server_command=["test"])
        client._process = MagicMock()
        client._process.stdin = MagicMock()

        client._send_notification("initialized")
        client._process.stdin.write.assert_called_once()

    def test_write_not_connected(self):
        client = MCPClient(server_command=["test"])
        with pytest.raises(RuntimeError, match="not connected"):
            client._write({"test": True})

    def test_read_not_connected(self):
        client = MCPClient(server_command=["test"])
        with pytest.raises(RuntimeError, match="not connected"):
            client._read()

    def test_read_eof(self):
        client = MCPClient(server_command=["test"])
        client._process = MagicMock()
        client._process.stdout = MagicMock()
        client._process.stdout.readline.return_value = b""

        with pytest.raises(RuntimeError, match="closed connection"):
            client._read()

    def test_close_terminates_process(self):
        client = MCPClient(server_command=["test"])
        client._process = MagicMock()
        client._process.poll.return_value = None  # still running
        client._process.wait.return_value = 0

        client.close()
        client._process.terminate.assert_called_once()

    def test_close_no_process(self):
        client = MCPClient(server_command=["test"])
        client.close()  # should not crash


class TestRegisterMCPTools:
    def test_register_tools(self):
        client = MagicMock()
        client.discover_tools.return_value = [
            MCPToolInfo(name="tool1", description="Test tool 1", input_schema={"type": "object"}),
            MCPToolInfo(name="tool2", description="Test tool 2", input_schema={"type": "object"}),
        ]

        registry = ToolRegistry()
        registered = register_mcp_tools(client, registry)
        assert registered == ["tool1", "tool2"]
        assert len(registry._tools) == 2

    @pytest.mark.asyncio
    async def test_registered_tool_handler(self):
        client = MagicMock()
        client.discover_tools.return_value = [
            MCPToolInfo(name="echo", description="Echo", input_schema={"type": "object"}),
        ]
        client.call_tool = AsyncMock(return_value="echoed")

        registry = ToolRegistry()
        register_mcp_tools(client, registry)

        result = await registry.execute("echo", '{"text": "hello"}')
        assert result == "echoed"
