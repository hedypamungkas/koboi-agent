"""Tests for MCP server and client -- covering dispatch, handlers, transport."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, AsyncMock
from io import BytesIO

import pytest

from koboi.mcp.server import MCPServer, _MCPToolDef
from koboi.mcp.client import MCPClient, MCPError, register_mcp_tools


class TestMCPServerDispatch:
    def test_dispatch_initialize(self):
        server = MCPServer("test")
        written = []
        server._write_response = lambda msg: written.append(msg)
        server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert len(written) == 1
        assert "protocolVersion" in written[0]["result"]

    def test_dispatch_initialized_notification(self):
        server = MCPServer("test")
        written = []
        server._write_response = lambda msg: written.append(msg)
        server._dispatch({"jsonrpc": "2.0", "method": "initialized"})
        assert len(written) == 0  # notification, no response

    def test_dispatch_tools_list(self):
        server = MCPServer("test")
        written = []
        server._write_response = lambda msg: written.append(msg)
        server._dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        assert written[0]["result"]["tools"] == []

    def test_dispatch_tools_call_unknown(self):
        server = MCPServer("test")
        written = []
        server._write_response = lambda msg: written.append(msg)
        server._dispatch({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "nope"}})
        assert written[0]["result"]["isError"] is True

    def test_dispatch_unknown_method(self):
        server = MCPServer("test")
        written = []
        server._write_response = lambda msg: written.append(msg)
        server._dispatch({"jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {}})
        assert written[0]["error"]["code"] == -32601

    def test_dispatch_handler_exception(self):
        server = MCPServer("test")
        written = []
        server._write_response = lambda msg: written.append(msg)

        @server.tool("bad", "fails", {})
        def bad_tool():
            raise ValueError("boom")

        server._dispatch({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "bad", "arguments": {}}})
        assert written[0]["result"]["isError"] is True


class TestMCPServerToolRegistration:
    def test_tool_decorator(self):
        server = MCPServer("test")

        @server.tool("greet", "Say hello", {"type": "object", "properties": {"name": {"type": "string"}}})
        def greet(name: str = "world"):
            return f"Hello {name}"

        assert "greet" in server._tools
        assert server._tools["greet"].description == "Say hello"

    def test_tools_list_after_registration(self):
        server = MCPServer("test")

        @server.tool("echo", "Echo input", {"type": "object"})
        def echo(text: str = ""):
            return text

        written = []
        server._write_response = lambda msg: written.append(msg)
        server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert len(written[0]["result"]["tools"]) == 1
        assert written[0]["result"]["tools"][0]["name"] == "echo"

    def test_tools_call_success(self):
        server = MCPServer("test")

        @server.tool("add", "Add numbers", {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}})
        def add(a: int = 0, b: int = 0):
            return a + b

        written = []
        server._write_response = lambda msg: written.append(msg)
        server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add", "arguments": {"a": 2, "b": 3}}})
        assert written[0]["result"]["content"][0]["text"] == "5"


class TestMCPServerHandleInitialize:
    def test_returns_server_info(self):
        server = MCPServer("myserver", "2.0.0")
        result = server._handle_initialize(1, {})
        assert result["serverInfo"]["name"] == "myserver"
        assert result["serverInfo"]["version"] == "2.0.0"
        assert "protocolVersion" in result
        assert "capabilities" in result


class TestMCPServerWriteResponse:
    @patch("koboi.mcp.server.sys.stdout")
    def test_write_response_writes_json(self, mock_stdout):
        server = MCPServer("test")
        server._write_response({"jsonrpc": "2.0", "id": 1, "result": {}})
        mock_stdout.write.assert_called_once()
        written = mock_stdout.write.call_args[0][0]
        assert json.loads(written.strip())["jsonrpc"] == "2.0"


class TestMCPError:
    def test_error_attributes(self):
        err = MCPError(code=-32601, message="not found", data={"extra": 1})
        assert err.code == -32601
        assert err.message == "not found"
        assert err.data == {"extra": 1}
        assert "MCP Error" in str(err)


class TestMCPClientTransport:
    def test_write_raises_when_not_connected(self):
        client = MCPClient(["echo", "test"])
        with pytest.raises(RuntimeError, match="not connected"):
            client._write({"jsonrpc": "2.0", "method": "test"})

    def test_read_raises_when_not_connected(self):
        client = MCPClient(["echo", "test"])
        with pytest.raises(RuntimeError, match="not connected"):
            client._read()

    def test_close_no_process(self):
        client = MCPClient(["echo", "test"])
        client.close()  # should not raise

    def test_server_info_default(self):
        client = MCPClient(["echo", "test"])
        assert client.server_info == {}

    def test_send_request_increments_id(self):
        client = MCPClient(["echo", "test"])
        # Mock the process
        proc = MagicMock()
        stdin = BytesIO()
        stdout = BytesIO()
        proc.stdin = stdin
        proc.stdout = stdout
        proc.poll.return_value = None
        client._process = proc

        # Write a response for the read
        resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}) + "\n"
        stdout.write(resp.encode())
        stdout.seek(0)
        stdin.seek(0)

        result = client._send_request("test", {"key": "value"})
        assert result == {"ok": True}
        assert client._request_id == 1


class TestMCPClientCallToolSync:
    def test_call_tool_sync_deduplicates(self):
        client = MCPClient(["test"])
        proc = MagicMock()
        stdin = BytesIO()
        stdout = BytesIO()
        proc.stdin = stdin
        proc.stdout = stdout
        proc.poll.return_value = None
        client._process = proc

        resp = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "hello"},  # duplicate
                    {"type": "text", "text": "world"},
                ]
            }
        }) + "\n"
        stdout.write(resp.encode())
        stdout.seek(0)
        stdin.seek(0)

        result = client._call_tool_sync("test_tool", {})
        assert result == "hello\nworld"

    def test_call_tool_sync_no_text_falls_back(self):
        client = MCPClient(["test"])
        proc = MagicMock()
        stdin = BytesIO()
        stdout = BytesIO()
        proc.stdin = stdin
        proc.stdout = stdout
        proc.poll.return_value = None
        client._process = proc

        resp = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "result": {"content": [{"type": "image", "data": "base64"}]}
        }) + "\n"
        stdout.write(resp.encode())
        stdout.seek(0)
        stdin.seek(0)

        result = client._call_tool_sync("test_tool", {})
        assert "image" in result


class TestRegisterMCPTools:
    def test_register_mcp_tools(self):
        client = MagicMock()
        t1 = MagicMock()
        t1.name = "tool1"
        t1.description = "Tool 1"
        t1.input_schema = {"type": "object"}
        t2 = MagicMock()
        t2.name = "tool2"
        t2.description = "Tool 2"
        t2.input_schema = {"type": "object"}
        client.discover_tools.return_value = [t1, t2]

        from koboi.tools.registry import ToolRegistry
        registry = ToolRegistry()
        registered = register_mcp_tools(client, registry)
        assert len(registered) == 2
        assert registered[0] == "tool1"
        assert registered[1] == "tool2"


class TestMCPClientClose:
    def test_close_terminate(self):
        client = MCPClient(["test"])
        proc = MagicMock()
        proc.poll.return_value = None
        client._process = proc
        client.close()
        proc.terminate.assert_called_once()

    def test_close_already_exited(self):
        client = MCPClient(["test"])
        proc = MagicMock()
        proc.poll.return_value = 0  # already exited
        client._process = proc
        client.close()
        proc.terminate.assert_not_called()

    def test_close_timeout_kills(self):
        import subprocess
        client = MCPClient(["test"])
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
        client._process = proc
        client.close()
        proc.kill.assert_called_once()
