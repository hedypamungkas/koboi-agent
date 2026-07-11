"""tests/test_mcp_gap_fixes.py -- regression tests for the closed MCP gaps (G3/G8/G10/G11).

Each test pins the FIXED behavior so the gaps cannot silently regress. Additional
gap coverage (G2/G4/G1) is added in later waves.
"""

from __future__ import annotations

import logging

import pytest

from koboi.mcp.base import BaseMCPClient, register_mcp_tools
from koboi.mcp.server import MCPServer
from koboi.tools.registry import ToolRegistry
from koboi.types import MCPToolInfo, RiskLevel


# --- Shared fake client (duck-typed, mirrors tests/test_mcp.py:80-105) ---


class _FakeClient:
    """Minimal client exposing discover_tools + async call_tool."""

    def __init__(self, tools, call_result="ok"):
        self._tools = tools
        self._call_result = call_result

    def discover_tools(self):
        return self._tools

    async def call_tool(self, name, arguments):
        return f"{self._call_result}:{name}"


class _RecordingClient(BaseMCPClient):
    """Records JSON-RPC requests so we can inspect the initialize handshake (G11)."""

    def __init__(self):
        super().__init__(logger=None)
        self.requests: list[tuple[str, dict | None]] = []

    def connect(self):
        return self._do_initialize_handshake()

    def discover_tools(self):
        return []

    async def call_tool(self, name, arguments):
        return ""

    def _send_request_impl(self, method, params=None):
        self.requests.append((method, params))
        if method == "initialize":
            return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "x"}}
        return {}

    def _send_notification_impl(self, method, params=None):
        pass

    def close(self):
        pass


def _tools(*names):
    return [MCPToolInfo(name=n, description=f"tool {n}", input_schema={"type": "object"}) for n in names]


# --- G3: per-MCP-tool risk level ---


class TestG3RiskLevel:
    def test_default_is_safe(self):
        reg = ToolRegistry()
        register_mcp_tools(_FakeClient(_tools("t")), reg)
        assert reg.get_risk_level("t") == RiskLevel.SAFE

    def test_destructive_propagates(self):
        reg = ToolRegistry()
        register_mcp_tools(_FakeClient(_tools("delete_all")), reg, risk_level=RiskLevel.DESTRUCTIVE)
        assert reg.get_risk_level("delete_all") == RiskLevel.DESTRUCTIVE

    def test_set_risk_level_after_register(self):
        reg = ToolRegistry()
        register_mcp_tools(_FakeClient(_tools("send_email")), reg)
        reg.set_risk_level("send_email", RiskLevel.MODERATE)
        assert reg.get_risk_level("send_email") == RiskLevel.MODERATE

    def test_set_risk_level_unknown_is_noop(self):
        reg = ToolRegistry()
        reg.set_risk_level("nope", RiskLevel.DESTRUCTIVE)  # must not raise
        assert reg.get_risk_level("nope") is None


# --- G8: collision guard + opt-in namespacing ---


class TestG8CollisionAndNamespace:
    def test_collision_logs_warning(self, caplog):
        reg = ToolRegistry()
        reg.register("dup", "first", {"type": "object"}, fn=lambda **k: "1", risk_level=RiskLevel.SAFE)
        with caplog.at_level(logging.WARNING, logger="koboi.tools.registry"):
            reg.register("dup", "second", {"type": "object"}, fn=lambda **k: "2", risk_level=RiskLevel.SAFE)
        assert any("overwriting existing tool 'dup'" in r.message for r in caplog.records)

    def test_namespace_prefix_renames_tools(self):
        reg = ToolRegistry()
        registered = register_mcp_tools(_FakeClient(_tools("add", "list")), reg, namespace_prefix="mcp__todo")
        assert registered == ["mcp__todo__add", "mcp__todo__list"]
        assert reg.get_risk_level("mcp__todo__add") == RiskLevel.SAFE
        # bare name is NOT registered -> no shadow
        assert reg.get_risk_level("add") is None

    def test_no_collision_when_namespaced(self, caplog):
        reg = ToolRegistry()
        reg.register("add", "builtin-ish", {"type": "object"}, fn=lambda **k: "b")
        with caplog.at_level(logging.WARNING, logger="koboi.tools.registry"):
            register_mcp_tools(_FakeClient(_tools("add")), reg, namespace_prefix="mcp__s")
        assert not any("overwriting existing tool 'add'" in r.message for r in caplog.records)


# --- G10: mcp.fail_fast ---


class _FakeConfig:
    """Fake Config exposing only the dotted-path .get() _build_mcp / _create_mcp_client read."""

    def __init__(self, servers, fail_fast=False):
        self._servers = servers
        self._fail_fast = fail_fast

    def get(self, *keys, default=None):
        if keys == ("mcp", "servers"):
            return self._servers
        if keys == ("mcp", "fail_fast"):
            return self._fail_fast
        if keys == ("mcp", "namespace"):
            return False
        if keys == ("mcp", "allowlist_commands"):
            return []
        return default


class TestG10FailFast:
    def test_default_swallows_and_returns_empty(self):
        from koboi.facade import _build_mcp

        reg = ToolRegistry()
        # runner not in allow-list -> ValueError, swallowed by default
        cfg = _FakeConfig([{"command": "/usr/bin/evil-binary", "args": []}])
        clients = _build_mcp(cfg, reg, logger=None)
        assert clients == []

    def test_fail_fast_raises(self):
        from koboi.facade import _build_mcp

        reg = ToolRegistry()
        cfg = _FakeConfig([{"command": "/usr/bin/evil-binary", "args": []}], fail_fast=True)
        with pytest.raises(ValueError, match="not in allow-list"):
            _build_mcp(cfg, reg, logger=None)


# --- G11: protocol version alignment (client sends 2025-03-26, tolerates peers) ---


class TestG11ProtocolVersion:
    def test_client_handshake_sends_2025_03_26(self):
        client = _RecordingClient()
        client.connect()
        init = next(p for m, p in client.requests if m == "initialize")
        assert init["protocolVersion"] == "2025-03-26"

    def test_server_advertises_2025_03_26(self):
        srv = MCPServer(name="x")
        assert srv._handle_initialize(0, {})["protocolVersion"] == "2025-03-26"

    def test_client_tolerates_older_peer_version(self):
        # _RecordingClient's fake server replies with 2024-11-05; connect() must not raise.
        client = _RecordingClient()
        info = client.connect()  # would raise if version-gated; it isn't
        assert info["protocolVersion"] == "2024-11-05"


# --- Mode-block nuance: configurable read-only allowlist for chat/plan ---


class TestNuanceModeBlock:
    @staticmethod
    def _ctx(tool_name):
        from koboi.hooks.chain import HookContext, HookEvent

        return HookContext(event=HookEvent.PRE_TOOL_USE, agent=None, iteration=1, tool_name=tool_name)

    async def test_mcp_tool_blocked_in_chat_by_default(self):
        from koboi.hooks.mode_hook import ModeHook
        from koboi.modes import AgentMode, ModeManager

        hook = ModeHook(ModeManager(AgentMode.CHAT))  # no extras
        out = await hook.execute(self._ctx("add_todo"))
        assert out.metadata.get("mode_blocked") is True

    async def test_mcp_tool_allowed_in_chat_when_allowlisted(self):
        from koboi.hooks.mode_hook import ModeHook
        from koboi.modes import AgentMode, ModeManager

        hook = ModeHook(ModeManager(AgentMode.CHAT), extra_read_only=["add_todo"])
        out = await hook.execute(self._ctx("add_todo"))
        assert out.metadata.get("mode_blocked") is not True  # permitted

    async def test_non_allowlisted_still_blocked(self):
        from koboi.hooks.mode_hook import ModeHook
        from koboi.modes import AgentMode, ModeManager

        hook = ModeHook(ModeManager(AgentMode.CHAT), extra_read_only=["add_todo"])
        out = await hook.execute(self._ctx("write_file"))
        assert out.metadata.get("mode_blocked") is True


# --- G4: reconnect / retry ---


class _FlakyConnect:
    def __init__(self, fail_n):
        self.fail_n = fail_n
        self.calls = 0

    def connect(self):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise RuntimeError("boom")


class TestG4ConnectRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        from koboi.facade import _connect_with_retry

        monkeypatch.setattr("time.sleep", lambda *_: None)
        c = _FlakyConnect(fail_n=2)
        _connect_with_retry(c, connect_retries=2, backoff_base=2.0)
        assert c.calls == 3  # 2 failures + 1 success

    def test_exhausts_and_raises(self, monkeypatch):
        from koboi.facade import _connect_with_retry

        monkeypatch.setattr("time.sleep", lambda *_: None)
        c = _FlakyConnect(fail_n=99)
        with pytest.raises(RuntimeError, match="boom"):
            _connect_with_retry(c, connect_retries=2, backoff_base=2.0)
        assert c.calls == 3  # initial + 2 retries


class TestG4EnsureConnectedHttp:
    def test_reconnects_when_client_none(self, monkeypatch):
        from koboi.mcp.http_client import StreamableHTTPMCPClient

        c = StreamableHTTPMCPClient(url="https://example.com/mcp")
        c._client = None
        calls = {"n": 0}
        monkeypatch.setattr(c, "connect", lambda: calls.__setitem__("n", calls["n"] + 1))
        c.ensure_connected()
        assert calls["n"] == 1

    def test_no_reconnect_when_connected(self, monkeypatch):
        from koboi.mcp.http_client import StreamableHTTPMCPClient

        c = StreamableHTTPMCPClient(url="https://example.com/mcp")
        c._client = object()  # truthy -> connected
        calls = {"n": 0}
        monkeypatch.setattr(c, "connect", lambda: calls.__setitem__("n", calls["n"] + 1))
        c.ensure_connected()
        assert calls["n"] == 0


class TestG4EnsureConnectedStdio:
    async def test_respawn_after_kill_and_tool_call_works(self):
        import asyncio
        import os
        import sys

        from koboi.mcp.client import MCPClient

        server = os.path.join(os.getcwd(), "mcp_servers", "todo_server.py")
        if not os.path.exists(server):
            pytest.skip("todo_server.py not present")
        mcp = MCPClient(server_command=[sys.executable, server], connect_timeout=15)
        try:
            mcp.connect()
            mcp.discover_tools()
            proc_before = mcp._process
            mcp._process.kill()
            mcp._process.wait(timeout=5)
            # A tool call after the kill must transparently respawn (ensure_connected).
            res = await asyncio.wait_for(mcp.call_tool("list_todos", {}), timeout=15)
            assert mcp._process is not proc_before  # respawned
            assert mcp._process.poll() is None  # alive
            assert isinstance(res, str)
        finally:
            mcp.close()


# --- G2: resources / prompts primitives ---


class _FakeResourceClient(BaseMCPClient):
    def __init__(self, responses):
        super().__init__(logger=None)
        self._responses = responses

    def connect(self):
        return {}

    def discover_tools(self):
        return []

    async def call_tool(self, name, arguments):
        return ""

    def _send_request_impl(self, method, params=None):
        return self._responses.get(method, {})

    def _send_notification_impl(self, method, params=None):
        pass

    def close(self):
        pass


class TestG2ClientResourcesPrompts:
    def test_list_resources_parses(self):
        from koboi.types import MCPResource

        c = _FakeResourceClient(
            {
                "resources/list": {
                    "resources": [{"uri": "u1", "name": "n", "description": "d", "mimeType": "text/plain"}]
                }
            }
        )
        assert c.list_resources() == [MCPResource(uri="u1", name="n", description="d", mime_type="text/plain")]

    def test_read_resource_joins_text(self):
        c = _FakeResourceClient(
            {"resources/read": {"contents": [{"uri": "u1", "text": "line1"}, {"uri": "u1", "text": "line2"}]}}
        )
        assert c.read_resource("u1") == "line1\nline2"

    def test_list_prompts_parses(self):
        c = _FakeResourceClient(
            {"prompts/list": {"prompts": [{"name": "p", "description": "dd", "arguments": [{"name": "x"}]}]}}
        )
        ps = c.list_prompts()
        assert ps[0].name == "p"
        assert ps[0].arguments == [{"name": "x"}]

    def test_get_prompt_renders(self):
        c = _FakeResourceClient(
            {"prompts/get": {"messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}]}}
        )
        assert c.get_prompt("p", {"x": 1}) == "hi"


class TestG2ServerResourcesPrompts:
    @staticmethod
    def _dispatch_result(server, msg):
        written = []
        server._write_response = lambda m: written.append(m)
        server._dispatch(msg)
        assert written, "no response written"
        return written[0].get("result")

    def test_capabilities_advertise_all_three(self):
        caps = MCPServer("g2")._handle_initialize(0, {})["capabilities"]
        assert {"tools", "resources", "prompts"} <= set(caps)

    def test_resource_list_and_read(self):
        srv = MCPServer("g2")

        @srv.resource(uri="doc://a", name="A", description="doc a")
        def _r():
            return "hello resource"

        r = self._dispatch_result(srv, {"jsonrpc": "2.0", "id": 1, "method": "resources/list", "params": {}})
        assert any(x["uri"] == "doc://a" for x in r["resources"])
        r2 = self._dispatch_result(
            srv, {"jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": {"uri": "doc://a"}}
        )
        assert r2["contents"][0]["text"] == "hello resource"

    def test_prompt_list_and_get(self):
        srv = MCPServer("g2")

        @srv.prompt(name="greet", description="greet", arguments=[{"name": "who"}])
        def _p(args):
            return f"hi {args.get('who', 'world')}"

        r = self._dispatch_result(srv, {"jsonrpc": "2.0", "id": 1, "method": "prompts/list", "params": {}})
        assert any(x["name"] == "greet" for x in r["prompts"])
        r2 = self._dispatch_result(
            srv,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "prompts/get",
                "params": {"name": "greet", "arguments": {"who": "bob"}},
            },
        )
        assert r2["messages"][0]["content"]["text"] == "hi bob"


# --- G5: orchestration MCP wiring ---


class _FakeMcpPairClient:
    def discover_tools(self):
        return _tools("add_todo", "list_todos")

    async def call_tool(self, name, arguments):
        return "ok"


class _FakeCfgNamespace:
    def get(self, *keys, default=None):
        return False if keys == ("mcp", "namespace") else default


class TestG5OrchestrationMcp:
    def test_registrar_registers_shared_tools_with_risk(self):
        from koboi.facade import _mcp_registrar_for_pairs

        reg = ToolRegistry()
        _mcp_registrar_for_pairs(
            [(_FakeMcpPairClient(), {"group": "todo", "risk_level": "moderate"})], _FakeCfgNamespace()
        )(reg)
        assert "add_todo" in reg and "list_todos" in reg
        assert reg.get_risk_level("add_todo") == RiskLevel.MODERATE

    def test_create_configured_agent_registers_mcp_tools(self):
        from unittest.mock import MagicMock

        from koboi.facade import _mcp_registrar_for_pairs
        from koboi.orchestration.factory import AgentFactory
        from koboi.types import AgentDef

        ad = AgentDef(name="a", tools_config={"builtin": ["calculator"]})
        registrar = _mcp_registrar_for_pairs([(_FakeMcpPairClient(), {"group": "todo"})], _FakeCfgNamespace())
        agent = AgentFactory.create_configured_agent(
            ad, MagicMock(), None, None, sandbox=MagicMock(), mcp_registrar=registrar
        )
        # The sub-agent's per-agent registry now contains the shared MCP tool (G5).
        assert "add_todo" in agent.tools
        assert "list_todos" in agent.tools

    def test_agent_without_tools_skips_registrar(self):
        from unittest.mock import MagicMock

        from koboi.orchestration.factory import AgentFactory
        from koboi.types import AgentDef

        ad = AgentDef(name="a", tools_config=None)
        seen = []
        AgentFactory.create_configured_agent(ad, MagicMock(), None, None, mcp_registrar=lambda r: seen.append(r))
        assert seen == []  # no registry -> registrar not called

    def test_share_mcp_false_disables_wiring(self):
        # OrchestrationConfig default share_mcp=True; flipping to False must keep MCP out.
        from koboi.config_models import OrchestrationConfig

        assert OrchestrationConfig().share_mcp is True
        assert OrchestrationConfig(share_mcp=False).share_mcp is False


# --- 24-C: reconnect diagnostics (respawn count + logging + stderr capture) ---


class TestReconnectDiagnostics24C:
    def test_stdio_respawn_increments_count_and_logs(self):
        from unittest.mock import MagicMock

        from koboi.mcp.client import MCPClient

        logger = MagicMock()
        c = MCPClient(["echo"], logger=logger)
        dead = MagicMock()
        dead.poll.return_value = 1  # process exited
        c._process = dead
        c.connect = MagicMock()  # avoid a real subprocess spawn
        c.ensure_connected()
        assert c._respawn_count == 1
        assert logger.log.call_count == 1
        assert "respawn #1" in logger.log.call_args[0][0]

    def test_stdio_stderr_tail_captured_not_discarded(self):
        from unittest.mock import MagicMock

        from koboi.mcp.client import MCPClient

        c = MCPClient(["echo"])
        c._process = MagicMock()
        c._process.stderr = iter([b"crash: bad config\n", b"traceback line\n"])
        c._drain_stderr()
        assert "crash: bad config" in c._stderr_tail
        assert "traceback line" in c._stderr_tail

    def test_http_reconnect_increments_count_and_logs(self):
        from unittest.mock import MagicMock

        from koboi.mcp.http_client import StreamableHTTPMCPClient

        logger = MagicMock()
        c = StreamableHTTPMCPClient(url="https://x.example.com/mcp", logger=logger)
        c._client = None  # session closed -> ensure_connected reconnects
        c.connect = MagicMock()
        c.ensure_connected()
        assert c._respawn_count == 1
        assert logger.log.call_count == 1
        assert "reconnect #1" in logger.log.call_args[0][0]
