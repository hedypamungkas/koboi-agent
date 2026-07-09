"""tests/test_tui_mcp_status.py -- G7 MCP status TUI screen."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")
from koboi.tui.screens.mcp_status import McpStatusScreen, _McpServerRow  # noqa: E402
from koboi.tui.textual_app import KoboiApp  # noqa: E402


class TestMcpServerRow:
    def test_connected_render(self):
        row = _McpServerRow({"name": "todo", "transport": "stdio", "connected": True, "tool_names": ["add", "list"]})
        out = row.render()
        assert "todo" in out and "connected" in out and "add" in out

    def test_dead_render(self):
        row = _McpServerRow({"name": "dead", "transport": "streamable-http", "connected": False, "tool_names": []})
        assert "DISCONNECTED" in row.render()


def _mock_agent(entries):
    from tests.test_textual_tui import _make_mock_agent

    agent = _make_mock_agent()
    agent.mcp_status.return_value = entries
    return agent


class TestMcpStatusScreenOpen:
    @pytest.mark.asyncio
    async def test_opens_via_keybinding(self):
        agent = _mock_agent(
            [{"id": "todo", "name": "todo", "transport": "stdio", "connected": True, "tool_names": ["add_todo"]}]
        )
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            await pilot.press("f2")
            await pilot.pause()
            assert isinstance(app.screen, McpStatusScreen)

    @pytest.mark.asyncio
    async def test_empty_state(self):
        agent = _mock_agent([])
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            await pilot.press("f2")
            await pilot.pause()
            assert isinstance(app.screen, McpStatusScreen)
