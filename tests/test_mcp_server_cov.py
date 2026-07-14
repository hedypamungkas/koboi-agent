"""koboi/mcp/server.py -- branch coverage for the stdio JSON-RPC MCPServer."""

from __future__ import annotations

import io
import json


from koboi.mcp.server import MCPServer


def _make_server() -> MCPServer:
    s = MCPServer("test-server", version="2.0.0")

    @s.tool("add", description="add", input_schema={"type": "object"})
    def add(a: int, b: int) -> int:
        return a + b

    @s.tool("boom", description="raises", input_schema={"type": "object"})
    def boom() -> None:
        raise RuntimeError("nope")

    @s.resource("file://x", name="x", description="a resource")
    def _x() -> str:
        return "resource-body"

    @s.prompt("greet", description="greeting", arguments=[{"name": "who"}])
    def _greet(args: dict) -> str:
        return f"hi {args.get('who', '?')}"

    return s


def _run_capture(server: MCPServer, stdin_text: str) -> str:
    out = io.StringIO()
    err = io.StringIO()
    import sys

    old_in, old_out, old_err = sys.stdin, sys.stdout, sys.stderr
    sys.stdin = io.StringIO(stdin_text)
    sys.stdout = out
    sys.stderr = err
    try:
        server.run()
    finally:
        sys.stdin, sys.stdout, sys.stderr = old_in, old_out, old_err
    return out.getvalue()


class TestRegistration:
    def test_tool_resource_prompt_registered(self):
        s = _make_server()
        assert "add" in s._tools and "file://x" in s._resources and "greet" in s._prompts

    def test_resource_default_name_is_uri(self):
        s = MCPServer("x")

        @s.resource("file://y", description="d")
        def _y() -> str:
            return ""

        assert s._resources["file://y"].name == "file://y"

    def test_prompt_default_arguments_empty(self):
        s = MCPServer("x")

        @s.prompt("p")
        def _p(args: dict) -> str:
            return ""

        assert s._prompts["p"].arguments == []


class TestRunLoop:
    def test_full_session_via_stdin(self):
        s = _make_server()
        lines = [
            '{"jsonrpc":"2.0","id":1,"method":"initialize"}',
            '{"jsonrpc":"2.0","method":"initialized"}',  # notification, no response
            "",  # blank line skipped
            '{"jsonrpc":"2.0","id":2,"method":"tools/list"}',
            '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":1,"b":2}}}',
            '{"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"file://x"}}',
            '{"jsonrpc":"2.0","id":5,"method":"prompts/get","params":{"name":"greet","arguments":{"who":"A"}}}',
            "not valid json",  # parse error response
            '{"jsonrpc":"2.0","id":6,"method":"unknown/method"}',  # method not found
            '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"boom","arguments":{}}}',  # handler error
        ]
        out = _run_capture(s, "\n".join(lines) + "\n")
        responses = [json.loads(l) for l in out.strip().splitlines() if l.strip()]
        by_id = {r.get("id"): r for r in responses}
        assert by_id[1]["result"]["protocolVersion"] == "2025-03-26"
        assert by_id[2]["result"]["tools"][0]["name"] == "add"
        assert "3" in by_id[3]["result"]["content"][0]["text"]
        assert by_id[4]["result"]["contents"][0]["text"] == "resource-body"
        assert "hi A" in by_id[5]["result"]["messages"][0]["content"]["text"]
        # parse error has id null
        assert any(r.get("id") is None and r.get("error", {}).get("code") == -32700 for r in responses)
        assert by_id[6]["error"]["code"] == -32601
        assert by_id[7]["result"]["isError"] is True


class TestHandlers:
    def test_tools_call_unknown_tool(self):
        s = _make_server()
        res = s._handle_tools_call(1, {"name": "ghost", "arguments": {}})
        assert res["isError"] is True

    def test_resources_read_unknown_uri(self):
        s = _make_server()
        assert s._handle_resources_read(1, {"uri": "nope"}) == {"contents": []}

    def test_prompts_get_unknown_name(self):
        s = _make_server()
        assert s._handle_prompts_get(1, {"name": "ghost"}) == {"description": "", "messages": []}

    def test_resources_list_and_prompts_list(self):
        s = _make_server()
        assert s._handle_resources_list(1, {})["resources"][0]["uri"] == "file://x"
        assert s._handle_prompts_list(1, {})["prompts"][0]["name"] == "greet"

    def test_dispatch_notification_initialized_no_response(self, monkeypatch):
        s = _make_server()
        written = []
        monkeypatch.setattr(s, "_write_response", lambda m: written.append(m))
        s._dispatch({"jsonrpc": "2.0", "method": "initialized"})  # no id
        assert written == []

    def test_dispatch_method_not_found(self, monkeypatch):
        s = _make_server()
        written = []
        monkeypatch.setattr(s, "_write_response", lambda m: written.append(m))
        s._dispatch({"jsonrpc": "2.0", "id": 9, "method": "wat"})
        assert written[0]["error"]["code"] == -32601

    def test_dispatch_handler_exception_is_error_result(self, monkeypatch):
        s = _make_server()
        written = []
        monkeypatch.setattr(s, "_write_response", lambda m: written.append(m))
        s._dispatch({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "boom", "arguments": {}}})
        assert written[0]["result"]["isError"] is True
