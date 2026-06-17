"""Tests for command history and HistorySearchScreen."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koboi.tui.textual_app import KoboiApp
from koboi.tui.widgets.input_box import ChatSubmit, InputBox


def _make_mock_agent():
    mock = MagicMock()
    mock.config.agent_name = "test-agent"
    mock.config.provider = "openai"
    mock.config.model = "gpt-4o-mini"
    mock.config.max_iterations = 10
    mock.config.rag_enabled = False
    mock.core.tools._tools = {}
    mock.core.hooks.list_hooks.return_value = []
    mock.core.input_guardrail = None
    mock.core.output_guardrail = None
    mock.core.rate_limiter = None
    mock.core.approval_handler = None
    mock.core.memory.get_messages.return_value = []
    return mock


class TestCommandHistory:
    @pytest.mark.asyncio
    async def test_history_adds_entries(self):
        agent = _make_mock_agent()
        agent.run_stream = AsyncMock()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("first"))
            app.post_message(ChatSubmit("second"))
            app.post_message(ChatSubmit("third"))
            await pilot.pause()
            assert app._history == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_history_deduplicates_consecutive(self):
        agent = _make_mock_agent()
        agent.run_stream = AsyncMock()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("same"))
            app.post_message(ChatSubmit("same"))
            await pilot.pause()
            assert app._history == ["same"]

    @pytest.mark.asyncio
    async def test_history_caps_at_max(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        app._history_max = 5
        for i in range(10):
            app._add_to_history(f"msg-{i}")
        assert len(app._history) == 5
        assert app._history[0] == "msg-5"
        assert app._history[-1] == "msg-9"

    @pytest.mark.asyncio
    async def test_history_navigation(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            app._history = ["first", "second", "third"]
            input_box.set_history(app._history)

            # Press up to go to last entry
            await pilot.press("up")
            await pilot.pause()
            assert input_box.value == "third"

            # Press up again
            await pilot.press("up")
            await pilot.pause()
            assert input_box.value == "second"

            # Press down
            await pilot.press("down")
            await pilot.pause()
            assert input_box.value == "third"

            # Press down past end clears
            await pilot.press("down")
            await pilot.pause()
            assert input_box.value == ""


from unittest.mock import AsyncMock
