"""Visual regression and integration tests for TUI widgets.

Tier 2 tests: screenshot capture, terminal resize behavior, and
slash command autocomplete integration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from koboi.tui.textual_app import KoboiApp
from koboi.tui.widgets.chat_log import ChatLog
from koboi.tui.widgets.file_suggester import CompositeSuggester
from koboi.tui.widgets.input_box import InputBox
from koboi.tui.widgets.message_bubble import MessageBubble
from koboi.tui.widgets.tool_call import ToolCallWidget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


SAMPLE_DIFF = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
-    print("hi")
+    print("hello")
+    print("world")
     return True
"""


# ---------------------------------------------------------------------------
# Screenshot capture -- verifies widgets render without crashing
# ---------------------------------------------------------------------------


class TestScreenshotCapture:
    @pytest.mark.asyncio
    async def test_screenshot_user_message(self, tmp_path: Path):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_message("user", "Hello world")
            await pilot.pause()
            svg_path = app.save_screenshot("user_msg.svg", path=str(tmp_path))
            assert Path(svg_path).exists()
            assert Path(svg_path).stat().st_size > 100

    @pytest.mark.asyncio
    async def test_screenshot_assistant_message(self, tmp_path: Path):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_message("assistant", "Here is **bold** and `code`")
            await pilot.pause()
            svg_path = app.save_screenshot("assistant_msg.svg", path=str(tmp_path))
            assert Path(svg_path).exists()
            assert Path(svg_path).stat().st_size > 100

    @pytest.mark.asyncio
    async def test_screenshot_system_message(self, tmp_path: Path):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_message("system", "Session started")
            await pilot.pause()
            svg_path = app.save_screenshot("system_msg.svg", path=str(tmp_path))
            assert Path(svg_path).exists()
            assert Path(svg_path).stat().st_size > 100

    @pytest.mark.asyncio
    async def test_screenshot_tool_call_running(self, tmp_path: Path):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("calculator", "tc_1", '{"expr":"1+1"}')
            await pilot.pause()
            svg_path = app.save_screenshot("tool_running.svg", path=str(tmp_path))
            assert Path(svg_path).exists()
            assert Path(svg_path).stat().st_size > 100

    @pytest.mark.asyncio
    async def test_screenshot_tool_call_completed(self, tmp_path: Path):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("calculator", "tc_1", '{"expr":"1+1"}')
            await pilot.pause()
            chat.update_tool_result("tc_1", "2")
            await pilot.pause()
            svg_path = app.save_screenshot("tool_completed.svg", path=str(tmp_path))
            assert Path(svg_path).exists()
            assert Path(svg_path).stat().st_size > 100

    @pytest.mark.asyncio
    async def test_screenshot_diff_view(self, tmp_path: Path):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("editor", "tc_diff", '{"file":"foo.py"}')
            await pilot.pause()
            chat.update_tool_result("tc_diff", SAMPLE_DIFF)
            await pilot.pause()
            svg_path = app.save_screenshot("diff_view.svg", path=str(tmp_path))
            assert Path(svg_path).exists()
            assert Path(svg_path).stat().st_size > 100


# ---------------------------------------------------------------------------
# Resize behavior -- verifies widgets survive terminal resize
# ---------------------------------------------------------------------------


class TestResizeBehavior:
    @pytest.mark.asyncio
    async def test_survive_small_terminal(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert app.query_one("#chat-area").display
            assert app.query_one("#input-box").display
            assert app.query_one("#header-bar").display
            assert app.query_one("#status-bar").display

    @pytest.mark.asyncio
    async def test_survive_large_terminal(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.resize_terminal(200, 50)
            await pilot.pause()
            assert app.query_one("#chat-area").display
            assert app.query_one("#input-box").display
            assert app.query_one("#header-bar").display
            assert app.query_one("#status-bar").display

    @pytest.mark.asyncio
    async def test_messages_survive_resize(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test(size=(120, 40)) as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_message("user", "Hello")
            chat.add_message("assistant", "Hi there")
            await pilot.pause()
            assert len(chat.query(MessageBubble)) == 2
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert len(chat.query(MessageBubble)) == 2

    @pytest.mark.asyncio
    async def test_tool_calls_survive_resize(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test(size=(120, 40)) as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("calculator", "tc_1", '{"expr":"1+1"}')
            await pilot.pause()
            assert len(chat.query(ToolCallWidget)) == 1
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert len(chat.query(ToolCallWidget)) == 1


# ---------------------------------------------------------------------------
# Slash command autocomplete integration
# ---------------------------------------------------------------------------


class TestSlashAutocompleteIntegration:
    @pytest.mark.asyncio
    async def test_suggester_wired_to_input(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test():
            input_box = app.query_one("#input-box", InputBox)
            assert input_box.suggester is not None
            assert isinstance(input_box.suggester, CompositeSuggester)

    @pytest.mark.asyncio
    async def test_slash_suggestion_partial(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test():
            input_box = app.query_one("#input-box", InputBox)
            suggestion = await input_box.suggester.get_suggestion("/h")
            assert suggestion == "/help"

    @pytest.mark.asyncio
    async def test_slash_no_suggestion_plain_text(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test():
            input_box = app.query_one("#input-box", InputBox)
            suggestion = await input_box.suggester.get_suggestion("hello world")
            assert suggestion is None

    @pytest.mark.asyncio
    async def test_slash_suggestion_reset(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test():
            input_box = app.query_one("#input-box", InputBox)
            suggestion = await input_box.suggester.get_suggestion("/re")
            assert suggestion == "/reset"

    @pytest.mark.asyncio
    async def test_slash_exact_match_none(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test():
            input_box = app.query_one("#input-box", InputBox)
            # Exact match returns None per SlashSuggester logic (cmd != lower)
            suggestion = await input_box.suggester.get_suggestion("/help")
            assert suggestion is None
