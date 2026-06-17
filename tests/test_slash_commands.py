"""Tests for slash command handling in the Textual TUI."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koboi.tui.textual_app import KoboiApp
from koboi.tui.widgets.chat_log import ChatLog
from koboi.tui.widgets.input_box import ChatSubmit
from koboi.tui.widgets.message_bubble import MessageBubble


def _make_mock_agent():
    mock = MagicMock()
    mock.config.agent_name = "test-agent"
    mock.config.provider = "openai"
    mock.config.model = "gpt-4o-mini"
    mock.config.max_iterations = 10
    mock.config.rag_enabled = False
    mock.core.tools.list_tools.return_value = {}
    mock.core.hooks.list_hooks.return_value = []
    mock.core.input_guardrail = None
    mock.core.output_guardrail = None
    mock.core.rate_limiter = None
    mock.core.approval_handler = None
    mock.core.memory.get_messages.return_value = []
    return mock


class TestSlashCommands:
    @pytest.mark.asyncio
    async def test_help_command(self):
        from koboi.tui.screens.help_overlay import HelpOverlayScreen
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("/help"))
            await pilot.pause()
            assert len(app.screen_stack) > 1
            assert isinstance(app.screen_stack[-1], HelpOverlayScreen)

    @pytest.mark.asyncio
    async def test_info_command(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("/info"))
            await pilot.pause()
            chat = app.query_one("#chat-area", ChatLog)
            bubbles = chat.query(MessageBubble)
            assert len(bubbles) == 1
            assert "test-agent" in bubbles[0]._content

    @pytest.mark.asyncio
    async def test_history_command_empty(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("/history"))
            await pilot.pause()
            chat = app.query_one("#chat-area", ChatLog)
            bubbles = chat.query(MessageBubble)
            assert len(bubbles) == 1
            assert "No messages" in bubbles[0]._content

    @pytest.mark.asyncio
    async def test_tools_command_empty(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("/tools"))
            await pilot.pause()
            chat = app.query_one("#chat-area", ChatLog)
            bubbles = chat.query(MessageBubble)
            assert len(bubbles) == 1
            assert "No tools" in bubbles[0]._content

    @pytest.mark.asyncio
    async def test_reset_command(self):
        agent = _make_mock_agent()
        agent.reset = MagicMock()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_message("user", "hello")
            await pilot.pause()
            app.post_message(ChatSubmit("/reset"))
            await pilot.pause()
            agent.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("/foobar"))
            await pilot.pause()
            chat = app.query_one("#chat-area", ChatLog)
            bubbles = chat.query(MessageBubble)
            assert len(bubbles) == 1
            assert "Unknown command" in bubbles[0]._content

    @pytest.mark.asyncio
    async def test_non_command_sends_to_agent(self):
        agent = _make_mock_agent()

        async def mock_stream(msg):
            from koboi.events import CompleteEvent
            yield CompleteEvent(content="response")

        agent.run_stream = mock_stream

        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("hello"))
            await pilot.pause()
            await pilot.pause()
            status = app.query_one("#status-bar")
            assert status.turn_count == 1

    @pytest.mark.asyncio
    async def test_get_all_commands(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        commands = app._get_all_commands()
        assert "/reset" in commands
        assert "/help" in commands
        assert "/info" in commands
        assert "/history" in commands
        assert "/tools" in commands
