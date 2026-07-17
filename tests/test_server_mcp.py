"""tests/test_server_mcp.py -- G6 per-session MCP server management."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.server import create_app  # noqa: E402
from koboi.server.mcp_registry import SessionMcpRegistry  # noqa: E402
from koboi.tools.registry import ToolRegistry  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Registry unit tests ---


class _FakeClient:
    def __init__(self, name, endpoint="cmd", connected=True, tools=("t1", "t2")):
        self._name = name
        self._endpoint = endpoint
        self._connected = connected
        self._tools = list(tools)

    @property
    def name(self):
        return self._name

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def transport(self):
        return "stdio"

    def is_connected(self):
        return self._connected

    @property
    def tool_names(self):
        return list(self._tools)

    server_info = {}

    def close(self):
        self._connected = False

    def connect(self):
        self._connected = True


class TestSessionMcpRegistry:
    def test_ensure_populated_and_status(self):
        reg = SessionMcpRegistry()
        reg.ensure_populated([_FakeClient("todo-srv", endpoint="python srv.py")])
        s = reg.status()
        assert len(s) == 1
        assert s[0]["id"] == "todo-srv"
        assert s[0]["connected"] is True
        assert s[0]["tool_names"] == ["t1", "t2"]

    def test_register_assigns_unique_id(self):
        reg = SessionMcpRegistry()
        id1 = reg.register(_FakeClient("todo"))
        id2 = reg.register(_FakeClient("todo"))
        assert id1 != id2
        assert reg.get(id1) is not None and reg.get(id2) is not None

    def test_remove_disables_tools_and_closes(self):
        reg = SessionMcpRegistry()
        client = _FakeClient("todo", tools=["add_todo"])
        sid = reg.register(client)
        tools = ToolRegistry()
        tools.register("add_todo", "", {"type": "object"}, fn=lambda **k: "x")
        mcp_clients = [client]
        assert reg.remove(sid, tools, mcp_clients) is True
        assert client.is_connected() is False  # closed
        assert mcp_clients == []  # dropped
        assert "add_todo" not in tools  # disabled
        assert reg.get(sid) is None

    def test_reconnect(self):
        reg = SessionMcpRegistry()
        client = _FakeClient("todo", connected=False)
        sid = reg.register(client)
        assert reg.reconnect(sid) is True
        assert client.is_connected() is True  # connect() called

    def test_remove_unknown_is_false(self):
        assert SessionMcpRegistry().remove("nope", None, []) is False


# --- Endpoint integration tests (real todo_server subprocess) ---


def _app():
    cfg = {
        "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "base_url": "http://localhost:8080/v1",
        },
        "memory": {"backend": "in_memory"},
        "sandbox": {"backend": "restricted"},
        "server": {"auth_required": False},
    }
    from koboi.config import Config

    factory = lambda: MockClient([make_mock_response(content="hello")])  # noqa: E731
    return create_app(Config.from_dict(cfg, validate=True), client_factory=factory, enable_cors=False)


async def test_add_list_reconnect_delete_mcp_server():
    server = os.path.join(REPO, "mcp_servers", "todo_server.py")
    if not os.path.exists(server):
        pytest.skip("todo_server.py not present")
    app = _app()
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
        sid = (await c.post("/v1/sessions")).json()["session_id"]

        # POST: attach the stdio todo_server
        r = await c.post(
            f"/v1/sessions/{sid}/mcp/servers",
            json={"transport": "stdio", "command": sys.executable, "args": [server], "group": "todo"},
        )
        assert r.status_code == 201, r.text
        server_id = r.json()["id"]
        assert r.json()["connected"] is True

        # GET: listed, connected, with the todo tools
        r = await c.get(f"/v1/sessions/{sid}/mcp/servers")
        servers = r.json()["servers"]
        match = [s for s in servers if s["id"] == server_id]
        assert match and match[0]["connected"] is True
        assert "add_todo" in match[0]["tool_names"]

        # reconnect
        r = await c.post(f"/v1/sessions/{sid}/mcp/servers/{server_id}/reconnect")
        assert r.status_code == 200, r.text
        assert r.json()["connected"] is True

        # DELETE: gone
        r = await c.delete(f"/v1/sessions/{sid}/mcp/servers/{server_id}")
        assert r.status_code == 200, r.text
        r = await c.get(f"/v1/sessions/{sid}/mcp/servers")
        assert not any(s["id"] == server_id for s in r.json()["servers"])


async def test_mcp_endpoints_404s():
    app = _app()
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
        sid = (await c.post("/v1/sessions")).json()["session_id"]
        # unknown session
        assert (await c.get("/v1/sessions/no-such/mcp/servers")).status_code == 404
        # unknown server_id on a real session
        assert (await c.delete(f"/v1/sessions/{sid}/mcp/servers/no-such")).status_code == 404
        assert (await c.post(f"/v1/sessions/{sid}/mcp/servers/no-such/reconnect")).status_code == 404


async def test_add_mcp_server_bad_command_400():
    app = _app()
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
        sid = (await c.post("/v1/sessions")).json()["session_id"]
        # runner not in the stdio allow-list -> ValueError -> 400 mcp_connect_failed
        r = await c.post(
            f"/v1/sessions/{sid}/mcp/servers",
            json={"transport": "stdio", "command": "/usr/bin/evil-binary", "args": []},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "mcp_connect_failed"


# --- 29-C: identity-aware register() (no double-index) ---


def test_register_after_ensure_populated_no_duplicate():
    """29-C: ensure_populated + register on the same client must not double-index."""
    reg = SessionMcpRegistry()
    client = _FakeClient("todo")
    reg.ensure_populated([client])  # indexes by slug "todo"
    sid = reg.register(client)  # must reuse existing id, not add a second entry
    assert sid == "todo"
    assert len(reg.status()) == 1  # no duplicate


# --- 29-D: registration failure -> 502 + client closed (not orphaned) ---


async def test_add_mcp_server_register_failure_502(monkeypatch):
    """29-D: connect succeeds but discover_tools fails -> 502 + client.close() called."""
    app = _app()

    class _ConnectsButBadDiscover:
        def __init__(self):
            self.closed = False
            self.logger = None
            self.server_command = []
            self.server_info = {}

        @property
        def name(self):
            return "bad"

        @property
        def endpoint(self):
            return "bad"

        @property
        def transport(self):
            return "stdio"

        @property
        def tool_names(self):
            return []

        def is_connected(self):
            return not self.closed

        def connect(self):
            return {"serverInfo": {"name": "bad"}}

        def discover_tools(self):
            raise RuntimeError("bad tools/list")

        async def call_tool(self, name, arguments):
            return ""

        def close(self):
            self.closed = True

    bad = _ConnectsButBadDiscover()
    monkeypatch.setattr("koboi.facade._create_mcp_client", lambda *a, **k: bad)
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
        sid = (await c.post("/v1/sessions")).json()["session_id"]
        r = await c.post(
            f"/v1/sessions/{sid}/mcp/servers",
            json={"transport": "stdio", "command": "python3", "args": []},
        )
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "mcp_register_failed"
        assert bad.closed is True  # 29-D: client was closed (not orphaned)


# --- 29-F: unexpected exception propagates to 500 (not 400) ---


async def test_add_mcp_server_unexpected_exception_500(monkeypatch):
    """29-F: an exception NOT in the caught family (TypeError) propagates to 500."""
    app = _app()

    def _boom(*a, **k):
        raise TypeError("unexpected server bug")

    monkeypatch.setattr("koboi.facade._create_mcp_client", _boom)
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
        sid = (await c.post("/v1/sessions")).json()["session_id"]
        # 29-F: TypeError is NOT in the caught family -> propagates uncaught (Starlette's
        # ASGI test transport re-raises; in production ServerErrorMiddleware -> 500).
        with pytest.raises(TypeError, match="unexpected server bug"):
            await c.post(
                f"/v1/sessions/{sid}/mcp/servers",
                json={"transport": "stdio", "command": "python3", "args": []},
            )


# --- 29-E: delete_session clears the MCP registry ---


async def test_delete_session_clears_mcp_registry():
    """29-E: deleting a session removes its entry from app.state.mcp_registries."""
    app = _app()
    async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
        sid = (await c.post("/v1/sessions")).json()["session_id"]
        await c.get(f"/v1/sessions/{sid}/mcp/servers")  # creates the registry
        assert sid in app.state.mcp_registries
        await c.delete(f"/v1/sessions/{sid}")
        assert sid not in app.state.mcp_registries  # 29-E: cleaned up
