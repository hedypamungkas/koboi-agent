"""tests/test_mcp_status.py -- Wave 0 MCP status accessors (G6/G7 foundation)."""

from __future__ import annotations

from unittest.mock import MagicMock

from koboi.facade import KoboiAgent
from koboi.mcp.client import MCPClient
from koboi.mcp.http_client import StreamableHTTPMCPClient
from koboi.types import MCPToolInfo


class TestTransportStatus:
    def test_stdio_is_connected_and_endpoint(self):
        c = MCPClient(["echo", "hi"])
        assert c.is_connected() is False  # no process yet
        assert c.transport == "stdio"
        assert c.endpoint == "echo hi"
        proc = MagicMock()
        proc.poll.return_value = None
        c._process = proc
        assert c.is_connected() is True
        proc.poll.return_value = 0
        assert c.is_connected() is False  # exited

    def test_http_is_connected_and_endpoint(self):
        c = StreamableHTTPMCPClient(url="https://x.example.com/mcp")
        assert c.is_connected() is False
        assert c.transport == "streamable-http"
        assert c.endpoint == "https://x.example.com/mcp"
        c._client = object()
        assert c.is_connected() is True

    def test_tool_names_and_name(self):
        c = MCPClient(["echo"])
        c._tools = [
            MCPToolInfo(name="a", description="", input_schema={}),
            MCPToolInfo(name="b", description="", input_schema={}),
        ]
        assert c.tool_names == ["a", "b"]
        assert c.name == ""  # no server_info yet
        c._server_info = {"serverInfo": {"name": "myserver"}}
        assert c.name == "myserver"


class _FakeMcpClient:
    def __init__(self, endpoint, name="", transport="stdio", connected=True, tools=None):
        self._endpoint = endpoint
        self._name = name
        self._transport = transport
        self._connected = connected
        self.server_info = {"serverInfo": {"name": name}} if name else {}
        self._tools = tools or []

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def name(self):
        return self._name

    @property
    def transport(self):
        return self._transport

    def is_connected(self):
        return self._connected

    @property
    def tool_names(self):
        return list(self._tools)


class _FakeConfig:
    def __init__(self, servers):
        self._servers = servers

    def get(self, *keys, default=None):
        if keys == ("mcp", "servers"):
            return self._servers
        return default


def _agent(clients, servers=None):
    return KoboiAgent(config=_FakeConfig(servers or []), mcp_clients=clients)


class TestMcpStatus:
    def test_live_clients_listed(self):
        a = _agent([_FakeMcpClient("python srv.py", name="todo", tools=["add", "list"])])
        s = a.mcp_status()
        assert len(s) == 1
        e = s[0]
        assert e["name"] == "todo"
        assert e["connected"] is True
        assert e["tool_names"] == ["add", "list"]
        assert e["transport"] == "stdio"

    def test_surfaces_failed_config_server(self):
        a = _agent(
            [_FakeMcpClient("python srv.py", name="todo")],
            servers=[
                {"command": "python", "args": ["srv.py"]},
                {"transport": "streamable-http", "url": "https://dead.example/mcp"},
            ],
        )
        s = a.mcp_status()
        assert len(s) == 2  # 1 live + 1 failed
        failed = [e for e in s if not e["connected"]]
        assert len(failed) == 1
        assert failed[0]["transport"] == "streamable-http"

    def test_matched_config_not_duplicated(self):
        a = _agent(
            [_FakeMcpClient("python srv.py", name="todo")],
            servers=[{"command": "python", "args": ["srv.py"]}],
        )
        assert len(a.mcp_status()) == 1  # live matches config -> no failed dup

    def test_dead_live_client_shows_disconnected(self):
        a = _agent([_FakeMcpClient("python srv.py", name="todo", connected=False)])
        s = a.mcp_status()
        assert s[0]["connected"] is False

    def test_mcp_clients_property_is_copy(self):
        client = _FakeMcpClient("x")
        a = _agent([client])
        assert a.mcp_clients == [client]
        a.mcp_clients.append("intruder")
        assert client in a._mcp_clients  # internal list untouched
