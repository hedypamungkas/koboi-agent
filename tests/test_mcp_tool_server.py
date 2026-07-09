"""tests/test_mcp_tool_server.py -- G9 expose-koboi-tools-over-MCP."""

from __future__ import annotations

import asyncio
import threading

from koboi.mcp.tool_server import _make_sync_handler, build_tool_server, select_exposed_tools
from koboi.tools.registry import ToolRegistry
from koboi.types import RiskLevel


def _reg_with_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("safe_tool", "s", {"type": "object"}, fn=lambda **k: "s", risk_level=RiskLevel.SAFE)
    reg.register("mod_tool", "m", {"type": "object"}, fn=lambda **k: "m", risk_level=RiskLevel.MODERATE)
    reg.register("dest_tool", "d", {"type": "object"}, fn=lambda **k: "d", risk_level=RiskLevel.DESTRUCTIVE)
    return reg


class TestSelectExposed:
    def test_safe_only_default(self):
        assert set(select_exposed_tools(_reg_with_tools())) == {"safe_tool"}

    def test_allow_adds_moderate_not_destructive(self):
        out = select_exposed_tools(_reg_with_tools(), allow=["mod_tool", "dest_tool"])
        assert set(out) == {"safe_tool", "mod_tool"}  # DESTRUCTIVE still blocked

    def test_allow_all_exposes_destructive(self):
        out = select_exposed_tools(_reg_with_tools(), allow_all=True)
        assert set(out) == {"safe_tool", "mod_tool", "dest_tool"}


class TestSyncHandlerBridge:
    def test_drives_async_tool(self):
        reg = ToolRegistry()

        async def add(**k):
            return str(k["a"] + k["b"])

        reg.register("add", "", {"type": "object", "properties": {"a": {}, "b": {}}}, fn=add, risk_level=RiskLevel.SAFE)
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        try:
            handler = _make_sync_handler("add", reg, loop)
            assert handler(a=2, b=3) == "5"
        finally:
            loop.call_soon_threadsafe(loop.stop)


_MIN_CONFIG = (
    "agent:\n"
    "  name: mcp-serve-test\n"
    "llm:\n"
    "  provider: openai\n"
    "  model: gpt-4o-mini\n"
    "tools:\n"
    "  builtin: [calculate]\n"
    "memory:\n"
    "  backend: in_memory\n"
)


class TestBuildToolServer:
    def test_exposes_safe_calculate_and_callable(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_MIN_CONFIG)
        server, _registry, loop = build_tool_server(str(cfg))
        try:
            # tools/list exposes the SAFE calculator
            written: list[dict] = []
            server._write_response = lambda m: written.append(m)
            server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
            names = {t["name"] for t in written[0]["result"]["tools"]}
            assert "calculate" in names

            # tools/call drives the real (async) koboi tool through the sync bridge
            written.clear()
            server._dispatch(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "calculate", "arguments": {"expression": "1+1"}},
                }
            )
            text = written[0]["result"]["content"][0]["text"]
            assert "1+1" in text and "2" in text
        finally:
            loop.call_soon_threadsafe(loop.stop)
