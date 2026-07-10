"""Tests for koboi.mcp module."""

from __future__ import annotations


from koboi.mcp.server import MCPServer
from koboi.mcp.client import MCPError
from koboi.tools.registry import ToolRegistry


class TestMCPServer:
    def test_tool_registration(self):
        server = MCPServer(name="test-server", version="1.0.0")

        @server.tool(
            name="add",
            description="Add numbers",
            input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        )
        def add(a: int, b: int) -> int:
            return a + b

        assert "add" in server._tools
        assert server._tools["add"].handler(2, 3) == 5

    def test_multiple_tools(self):
        server = MCPServer(name="multi")

        @server.tool(name="t1", description="Tool 1", input_schema={})
        def t1():
            return "one"

        @server.tool(name="t2", description="Tool 2", input_schema={})
        def t2():
            return "two"

        assert len(server._tools) == 2

    def test_handle_initialize(self):
        server = MCPServer(name="test")
        result = server._handle_initialize(None, {})
        assert result["protocolVersion"] == "2025-03-26"
        assert result["serverInfo"]["name"] == "test"

    def test_handle_tools_list(self):
        server = MCPServer(name="test")

        @server.tool(name="hello", description="Says hi", input_schema={"type": "object"})
        def hello():
            return "hi"

        result = server._handle_tools_list(None, {})
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "hello"

    def test_handle_tools_call(self):
        server = MCPServer(name="test")

        @server.tool(name="echo", description="Echo", input_schema={"type": "object"})
        def echo(text: str = ""):
            return text

        result = server._handle_tools_call(None, {"name": "echo", "arguments": {"text": "hello"}})
        assert result["content"][0]["text"] == "hello"

    def test_handle_unknown_tool(self):
        server = MCPServer(name="test")
        result = server._handle_tools_call(None, {"name": "nonexistent", "arguments": {}})
        assert result.get("isError") is True


class TestMCPError:
    def test_error_creation(self):
        err = MCPError(code=-32600, message="Invalid request")
        assert err.code == -32600
        assert "Invalid request" in str(err)


class TestMCPToolRegistration:
    async def test_register_mcp_tools_to_registry(self):
        """Test the register_mcp_tools bridge function using a mock client."""
        from koboi.types import MCPToolInfo

        mock_tools = [
            MCPToolInfo(name="add_todo", description="Add a todo", input_schema={"type": "object"}),
            MCPToolInfo(name="list_todos", description="List todos", input_schema={"type": "object"}),
        ]

        class FakeMCPClient:
            def discover_tools(self):
                return mock_tools

            async def call_tool(self, name, arguments):
                return f"Called {name}"

        registry = ToolRegistry()
        from koboi.mcp.client import register_mcp_tools

        registered = register_mcp_tools(FakeMCPClient(), registry)

        assert len(registered) == 2
        assert "add_todo" in registered
        assert "list_todos" in registered
        result = await registry.execute("add_todo", "{}")
        assert result == "Called add_todo"
