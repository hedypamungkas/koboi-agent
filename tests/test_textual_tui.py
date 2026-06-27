"""Tests for Textual TUI widgets, bridge, and app."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.events import (
    CompleteEvent,
    ErrorEvent,
    IterationEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from koboi.tui.bridge import (
    StreamBridge,
    StreamComplete,
    StreamDelta,
    StreamError,
    StreamIteration,
    StreamToolCall,
    StreamToolResult,
)
from koboi.tui.textual_app import KoboiApp
from koboi.tui.widgets.chat_log import ChatLog
from koboi.tui.widgets.diff_view import DiffViewWidget, is_diff_content, count_changes
from koboi.tui.widgets.header_bar import HeaderBar
from koboi.tui.widgets.input_box import ChatSubmit, InputBox
from koboi.tui.widgets.message_bubble import MessageBubble
from koboi.tui.widgets.status_bar import StatusBar
from koboi.tui.widgets.thinking_block import ThinkingBlockWidget
from koboi.tui.widgets.tool_call import ToolCallWidget


@pytest.fixture(autouse=True)
def _ensure_event_loop_for_textual_widgets(request):
    """Ensure a current event loop exists for bare Textual widget construction.

    Textual ``Widget`` binds an ``asyncio.Lock`` in ``__init__`` (via ``RLock``),
    which calls ``asyncio.get_event_loop()`` -- so constructing a widget outside
    a running loop raises ``RuntimeError: There is no current event loop``.

    In isolation these sync tests pass (a default loop is lazily created), but
    across the full suite pytest-asyncio closes its loop after earlier async
    tests, leaving ``_set_called=True`` / no loop on MainThread -- so bare widget
    construction in later sync tests then fails. This fixture ensures a loop
    exists for sync tests only; async tests already receive one from
    pytest-asyncio and are left untouched.
    """
    import asyncio
    import inspect

    fn = getattr(request.node, "function", None)
    if fn is not None and inspect.iscoroutinefunction(fn):
        yield
        return
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield


# ============================================================================
# Helpers
# ============================================================================


def _make_mock_agent():
    """Create a fully mocked KoboiAgent."""
    mock = MagicMock()
    mock.config.agent_name = "test-agent"
    mock.config.provider = "openai"
    mock.config.model = "gpt-4o-mini"
    mock.config.max_iterations = 10
    mock.config.rag_enabled = False
    mock.core.tools._tools = {}
    mock.core.tools.list_tools.return_value = {}
    mock.core.hooks.list_hooks.return_value = []
    mock.core.input_guardrail = None
    mock.core.output_guardrail = None
    mock.core.rate_limiter = None
    mock.core.approval_handler = None
    mock.core.memory.get_messages.return_value = []
    return mock


# ============================================================================
# HeaderBar
# ============================================================================


class TestHeaderBar:
    def test_default_values(self):
        bar = HeaderBar()
        assert bar.agent_name == "koboi-agent"
        assert bar.model == ""
        assert bar.mode == "CHAT"

    def test_render_contains_agent_name(self):
        bar = HeaderBar()
        bar.agent_name = "sales-bot"
        rendered = bar.render()
        assert "sales-bot" in rendered

    def test_render_contains_model(self):
        bar = HeaderBar()
        bar.model = "openai/gpt-4o"
        rendered = bar.render()
        assert "openai/gpt-4o" in rendered

    def test_render_contains_mode(self):
        bar = HeaderBar()
        bar.mode = "PLAN"
        rendered = bar.render()
        assert "PLAN" in rendered


# ============================================================================
# StatusBar
# ============================================================================


class TestStatusBar:
    def test_default_values(self):
        bar = StatusBar()
        assert bar.context_pct == 0.0
        assert bar.tokens_used == 0
        assert bar.turn_count == 0
        assert bar.state == "idle"

    def test_progress_bar_full(self):
        assert StatusBar._progress_bar(100.0, width=10) == "█" * 10

    def test_progress_bar_empty(self):
        assert StatusBar._progress_bar(0.0, width=10) == "░" * 10

    def test_progress_bar_partial(self):
        bar = StatusBar._progress_bar(50.0, width=10)
        assert "█" in bar and "░" in bar

    def test_render_shows_tokens(self):
        bar = StatusBar()
        bar.tokens_used = 1500
        bar.max_tokens = 8000
        rendered = bar.render()
        assert "1500/8000" in rendered

    def test_render_shows_turn_count(self):
        bar = StatusBar()
        bar.turn_count = 5
        rendered = bar.render()
        assert "turn 5" in rendered

    def test_render_shows_tool_when_running(self):
        bar = StatusBar()
        bar.state = "running_tool"
        bar.current_tool = "calculator"
        rendered = bar.render()
        assert "calculator" in rendered

    def test_render_hides_tool_when_idle(self):
        bar = StatusBar()
        bar.state = "idle"
        bar.current_tool = "calculator"
        rendered = bar.render()
        assert "calculator" not in rendered

    def test_render_shows_streaming(self):
        bar = StatusBar()
        bar.state = "streaming"
        rendered = bar.render()
        assert "streaming" in rendered


# ============================================================================
# InputBox
# ============================================================================


class TestInputBox:
    def test_input_submitted_message_type(self):
        msg = ChatSubmit("hello")
        assert msg.value == "hello"


# ============================================================================
# StreamBridge
# ============================================================================


class TestStreamBridge:
    @pytest.mark.asyncio
    async def test_text_deltas(self):
        app = MagicMock()
        bridge = StreamBridge(app)

        async def stream():
            yield TextDeltaEvent(content="Hello")
            yield TextDeltaEvent(content=" world")
            yield CompleteEvent(content="Hello world")

        await bridge.process_stream(stream())

        assert app.post_message.call_count == 3
        msg0 = app.post_message.call_args_list[0][0][0]
        assert isinstance(msg0, StreamDelta)
        assert msg0.content == "Hello"
        msg1 = app.post_message.call_args_list[1][0][0]
        assert isinstance(msg1, StreamDelta)
        assert msg1.content == " world"

    @pytest.mark.asyncio
    async def test_tool_call(self):
        app = MagicMock()
        bridge = StreamBridge(app)

        async def stream():
            yield ToolCallEvent(tool_name="calc", tool_call_id="tc_1", arguments='{"x":1}')
            yield CompleteEvent(content="done")

        await bridge.process_stream(stream())

        msg = app.post_message.call_args_list[0][0][0]
        assert isinstance(msg, StreamToolCall)
        assert msg.tool_name == "calc"
        assert msg.tool_call_id == "tc_1"
        assert msg.arguments == '{"x":1}'

    @pytest.mark.asyncio
    async def test_tool_result(self):
        app = MagicMock()
        bridge = StreamBridge(app)

        async def stream():
            yield ToolResultEvent(tool_name="calc", tool_call_id="tc_1", result="42")
            yield CompleteEvent(content="done")

        await bridge.process_stream(stream())

        msg = app.post_message.call_args_list[0][0][0]
        assert isinstance(msg, StreamToolResult)
        assert msg.tool_name == "calc"
        assert msg.result == "42"

    @pytest.mark.asyncio
    async def test_iteration_event(self):
        app = MagicMock()
        bridge = StreamBridge(app)

        async def stream():
            yield IterationEvent(iteration=2, messages_count=10, tokens_estimated=500)
            yield CompleteEvent(content="ok")

        await bridge.process_stream(stream())

        msg = app.post_message.call_args_list[0][0][0]
        assert isinstance(msg, StreamIteration)
        assert msg.iteration == 2
        assert msg.messages_count == 10
        assert msg.tokens_estimated == 500

    @pytest.mark.asyncio
    async def test_error_event(self):
        app = MagicMock()
        bridge = StreamBridge(app)

        async def stream():
            yield ErrorEvent(error=ValueError("bad input"))

        await bridge.process_stream(stream())

        msg = app.post_message.call_args_list[0][0][0]
        assert isinstance(msg, StreamError)
        assert isinstance(msg.error, ValueError)

    @pytest.mark.asyncio
    async def test_complete_event(self):
        app = MagicMock()
        bridge = StreamBridge(app)

        async def stream():
            yield CompleteEvent(content="final answer")

        await bridge.process_stream(stream())

        msg = app.post_message.call_args_list[0][0][0]
        assert isinstance(msg, StreamComplete)
        assert msg.content == "final answer"

    @pytest.mark.asyncio
    async def test_complete_event_empty_content(self):
        app = MagicMock()
        bridge = StreamBridge(app)

        async def stream():
            yield CompleteEvent(content=None)

        await bridge.process_stream(stream())

        msg = app.post_message.call_args_list[0][0][0]
        assert isinstance(msg, StreamComplete)
        assert msg.content == ""

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        app = MagicMock()
        bridge = StreamBridge(app)

        async def stream():
            return
            yield  # make it a generator

        await bridge.process_stream(stream())
        app.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_events(self):
        app = MagicMock()
        bridge = StreamBridge(app)

        async def stream():
            yield IterationEvent(iteration=0, messages_count=5, tokens_estimated=200)
            yield TextDeltaEvent(content="Let me ")
            yield TextDeltaEvent(content="calculate...")
            yield ToolCallEvent(tool_name="calc", tool_call_id="tc_1", arguments="{}")
            yield ToolResultEvent(tool_name="calc", tool_call_id="tc_1", result="42")
            yield TextDeltaEvent(content="The answer is 42.")
            yield CompleteEvent(content="The answer is 42.")

        await bridge.process_stream(stream())

        assert app.post_message.call_count == 7
        types = [type(call[0][0]) for call in app.post_message.call_args_list]
        assert types == [
            StreamIteration,
            StreamDelta,
            StreamDelta,
            StreamToolCall,
            StreamToolResult,
            StreamDelta,
            StreamComplete,
        ]


# ============================================================================
# MessageBubble
# ============================================================================


class TestMessageBubble:
    @pytest.mark.asyncio
    async def test_user_bubble_has_user_class(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test():
            bubble = MessageBubble("user", "hello")
            assert "message-user" in bubble.classes

    @pytest.mark.asyncio
    async def test_assistant_bubble_has_assistant_class(self):
        bubble = MessageBubble("assistant", "hi")
        assert "message-assistant" in bubble.classes

    @pytest.mark.asyncio
    async def test_streamable_bubble_accumulates(self):
        bubble = MessageBubble("assistant", "", is_streamable=True)
        assert bubble._content == ""
        bubble._markdown = MagicMock()
        bubble._last_update = 0
        bubble.update_content("Hello")
        bubble.update_content(" world")
        assert bubble._content == "Hello world"

    def test_set_final_content(self):
        bubble = MessageBubble("assistant", "partial", is_streamable=True)
        bubble._markdown = MagicMock()
        bubble.set_final_content("complete answer")
        assert bubble._content == "complete answer"
        bubble._markdown.update.assert_called_with("complete answer")


# ============================================================================
# ChatLog
# ============================================================================


class TestChatLog:
    @pytest.mark.asyncio
    async def test_add_message(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            bubble = chat.add_message("user", "hello")
            assert isinstance(bubble, MessageBubble)
            await pilot.pause()
            assert len(chat.query(MessageBubble)) == 1

    @pytest.mark.asyncio
    async def test_begin_stream(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            bubble = chat.begin_stream()
            assert isinstance(bubble, MessageBubble)
            assert bubble._is_streamable is True
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_add_system_message(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_system_message("test system msg")
            await pilot.pause()
            assert len(chat.query(MessageBubble)) == 1

    @pytest.mark.asyncio
    async def test_clear_messages(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_message("user", "msg1")
            chat.add_message("assistant", "msg2")
            await pilot.pause()
            assert len(chat.query(MessageBubble)) == 2
            chat.clear_messages()
            await pilot.pause()
            assert len(chat.query(MessageBubble)) == 0

    @pytest.mark.asyncio
    async def test_add_tool_call(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("calculator", "tc_1", '{"x":1}')
            await pilot.pause()
            widgets = chat.query(ToolCallWidget)
            assert len(widgets) == 1


# ============================================================================
# KoboiApp Integration
# ============================================================================


class TestKoboiApp:
    @pytest.mark.asyncio
    async def test_app_compose(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test():
            header = app.query_one("#header-bar", HeaderBar)
            assert header.agent_name == "test-agent"
            assert header.model == "openai/gpt-4o-mini"
            status = app.query_one("#status-bar", StatusBar)
            assert status.turn_count == 0
            assert status.state == "idle"
            app.query_one("#chat-area", ChatLog)
            app.query_one("#input-box", InputBox)

    @pytest.mark.asyncio
    async def test_input_focus_on_mount(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test():
            input_box = app.query_one("#input-box", InputBox)
            assert input_box.has_focus

    @pytest.mark.asyncio
    async def test_submit_message_triggers_stream(self):
        agent = _make_mock_agent()

        async def mock_stream(message):
            yield TextDeltaEvent(content="Hello!")
            yield CompleteEvent(content="Hello!")

        agent.run_stream = mock_stream

        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("Hi"))
            await pilot.pause()
            await pilot.pause()

            status = app.query_one("#status-bar", StatusBar)
            assert status.turn_count == 1

    @pytest.mark.asyncio
    async def test_empty_input_ignored(self):
        agent = _make_mock_agent()
        agent.run_stream = AsyncMock()

        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            await pilot.press("Enter")
            await pilot.pause()

            status = app.query_one("#status-bar", StatusBar)
            assert status.turn_count == 0
            agent.run_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_call_updates_status(self):
        agent = _make_mock_agent()

        async def mock_stream(message):
            yield ToolCallEvent(tool_name="calculator", tool_call_id="tc_1", arguments="{}")
            yield ToolResultEvent(tool_name="calculator", tool_call_id="tc_1", result="42")
            yield CompleteEvent(content="The answer is 42.")

        agent.run_stream = mock_stream

        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("what is 6*7?"))
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()

            status = app.query_one("#status-bar", StatusBar)
            assert status.turn_count == 1

    @pytest.mark.asyncio
    async def test_slash_command_does_not_call_agent(self):
        agent = _make_mock_agent()
        agent.run_stream = AsyncMock()

        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("/help"))
            await pilot.pause()

            agent.run_stream.assert_not_called()
            status = app.query_one("#status-bar", StatusBar)
            assert status.turn_count == 0

    @pytest.mark.asyncio
    async def test_slash_reset_clears_chat(self):
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
            assert len(chat.query(MessageBubble)) == 1  # system message only

    @pytest.mark.asyncio
    async def test_history_tracking(self):
        agent = _make_mock_agent()
        agent.run_stream = AsyncMock()

        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("first"))
            app.post_message(ChatSubmit("second"))
            await pilot.pause()

            assert app._history == ["first", "second"]

    @pytest.mark.asyncio
    async def test_command_palette_opens(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+p")
            await pilot.pause()
            # Should have pushed a screen
            assert len(app._screen_stack) > 1


# ============================================================================
# DiffViewWidget
# ============================================================================


class TestDiffViewWidget:
    def test_is_diff_content_git_header(self):
        text = "diff --git a/foo.py b/foo.py\nindex abc..def 100644\n--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,4 @@\n+new line"
        assert is_diff_content(text) is True

    def test_is_diff_content_hunk_markers(self):
        text = "--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,4 @@\n+added\n context"
        assert is_diff_content(text) is True

    def test_is_diff_content_high_ratio(self):
        lines = ["+line"] * 5 + ["-line"] * 3 + [" context"] * 2
        assert is_diff_content("\n".join(lines)) is True

    def test_is_diff_content_false_positive(self):
        text = "This is normal text with some + and - symbols scattered around."
        assert is_diff_content(text) is False

    def test_is_diff_content_too_short(self):
        assert is_diff_content("+\n-") is False

    def test_count_changes(self):
        text = "+added1\n+added2\n-removed1\n context"
        adds, dels = count_changes(text)
        assert adds == 2
        assert dels == 1

    def test_count_changes_skips_headers(self):
        text = "--- a/foo.py\n+++ b/foo.py\n+added\n-removed"
        adds, dels = count_changes(text)
        assert adds == 1
        assert dels == 1

    def test_parse_diff_styles(self):
        parsed = DiffViewWidget._parse_diff(
            "diff --git a/f\nindex abc\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n+add\n-del\n context"
        )
        styles = [s for _, s in parsed]
        assert "bold yellow" in styles  # diff --git
        assert "bold cyan" in styles  # @@
        assert "green" in styles  # +
        assert "red" in styles  # -
        assert "bold" in styles  # --- / +++


# ============================================================================
# ThinkingBlockWidget
# ============================================================================


class TestThinkingBlockWidget:
    def test_default_collapsed(self):
        widget = ThinkingBlockWidget("I am thinking...")
        assert widget.collapsed is True

    def test_toggle(self):
        widget = ThinkingBlockWidget("thinking text")
        widget.toggle()
        assert widget.collapsed is False
        widget.toggle()
        assert widget.collapsed is True

    @pytest.mark.asyncio
    async def test_mounted_has_header(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            widget = ThinkingBlockWidget("reasoning content here")
            chat.mount(widget)
            await pilot.pause()
            assert widget.is_mounted
            assert widget.collapsed is True
            # Verify children were composed
            assert len(widget.query("Static")) >= 1


# ============================================================================
# ToolCallWidget
# ============================================================================


class TestToolCallWidget:
    def test_initial_state_running(self):
        import time

        widget = ToolCallWidget("calc", "tc_1", '{"x":1}', time.monotonic())
        assert widget._state == "running"
        assert widget.collapsed is True

    def test_set_result_updates_state(self):
        import time

        start = time.monotonic()
        widget = ToolCallWidget("calc", "tc_1", '{"x":1}', start)
        widget.set_result("42", start + 0.3)
        assert widget._state == "completed"
        # Not auto-expanded yet because widget isn't mounted (deferred)
        assert widget._result_pending is True

    def test_set_result_error_state(self):
        import time

        start = time.monotonic()
        widget = ToolCallWidget("calc", "tc_1", "{}", start)
        widget.set_result("Error: division by zero", start + 0.1)
        assert widget._state == "error"

    def test_format_args_json(self):
        result = ToolCallWidget._format_args('{"x": 1, "y": 2}')
        assert '"x": 1' in result

    def test_format_args_empty(self):
        result = ToolCallWidget._format_args("")
        assert result == "(no arguments)"

    def test_format_args_plain(self):
        result = ToolCallWidget._format_args("not json")
        assert result == "not json"

    def test_render_header_running(self):
        import time

        widget = ToolCallWidget("calculator", "tc_1", "{}", time.monotonic())
        header = widget._render_header()
        assert "running" in header
        assert "calculator" in header

    def test_render_header_completed(self):
        import time

        start = time.monotonic()
        widget = ToolCallWidget("calculator", "tc_1", "{}", start)
        widget._end_time = start + 0.5
        widget._result = "42"
        widget._state = "completed"
        header = widget._render_header()
        assert "calculator" in header
        assert "0.5s" in header
        assert "2B" in header


# ============================================================================
# ChatLog Integration (Phase 3)
# ============================================================================


class TestChatLogPhase3:
    @pytest.mark.asyncio
    async def test_add_tool_call_creates_widget(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("calc", "tc_1", '{"x":1}')
            await pilot.pause()
            widgets = chat.query(ToolCallWidget)
            assert len(widgets) == 1
            assert "tc_1" in chat._tool_widgets

    @pytest.mark.asyncio
    async def test_update_tool_result_finds_widget(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("calc", "tc_1", '{"x":1}')
            await pilot.pause()
            chat.update_tool_result("tc_1", "42")
            await pilot.pause()
            assert "tc_1" not in chat._tool_widgets
            widget = chat.query(ToolCallWidget)[0]
            assert widget._state == "completed"

    @pytest.mark.asyncio
    async def test_update_tool_result_missing_id(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.update_tool_result("unknown_id", "result")
            await pilot.pause()
            # Should create a fallback Static
            fallbacks = chat.query(".tool-result")
            assert len(fallbacks) == 1

    @pytest.mark.asyncio
    async def test_collapse_all_tools(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("calc", "tc_1", "{}")
            chat.add_tool_call("search", "tc_2", "{}")
            await pilot.pause()
            chat.expand_all_tools()
            await pilot.pause()
            assert all(not w.collapsed for w in chat.query(ToolCallWidget))
            chat.collapse_all_tools()
            await pilot.pause()
            assert all(w.collapsed for w in chat.query(ToolCallWidget))

    @pytest.mark.asyncio
    async def test_clear_messages_resets_tool_widgets(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("calc", "tc_1", "{}")
            await pilot.pause()
            assert len(chat._tool_widgets) == 1
            chat.clear_messages()
            await pilot.pause()
            assert len(chat._tool_widgets) == 0

    @pytest.mark.asyncio
    async def test_add_iteration_marker(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_iteration_marker(2, 10)
            await pilot.pause()
            markers = chat.query(".iteration-marker")
            assert len(markers) == 1


# ============================================================================
# KoboiApp Integration (Phase 3)
# ============================================================================


class TestKoboiAppPhase3:
    @pytest.mark.asyncio
    async def test_tool_call_passes_full_data(self):
        agent = _make_mock_agent()

        async def mock_stream(message):
            yield ToolCallEvent(tool_name="calc", tool_call_id="tc_1", arguments='{"x":1}')
            yield ToolResultEvent(tool_name="calc", tool_call_id="tc_1", result="42")
            yield CompleteEvent(content="The answer is 42.")

        agent.run_stream = mock_stream

        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("calculate"))
            await pilot.pause()
            await pilot.pause()

            chat = app.query_one("#chat-area", ChatLog)
            widgets = chat.query(ToolCallWidget)
            assert len(widgets) == 1
            assert widgets[0]._tool_name == "calc"
            assert widgets[0]._result == "42"

    @pytest.mark.asyncio
    async def test_iteration_updates_status_and_chat(self):
        agent = _make_mock_agent()

        async def mock_stream(message):
            yield IterationEvent(iteration=1, messages_count=5, tokens_estimated=300)
            yield TextDeltaEvent(content="Hello")
            yield CompleteEvent(content="Hello")

        agent.run_stream = mock_stream

        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.post_message(ChatSubmit("hi"))
            await pilot.pause()
            await pilot.pause()

            status = app.query_one("#status-bar", StatusBar)
            assert status.iteration == 1
            chat = app.query_one("#chat-area", ChatLog)
            markers = chat.query(".iteration-marker")
            assert len(markers) == 1

    @pytest.mark.asyncio
    async def test_tab_toggles_tools_when_not_on_input(self):
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-area", ChatLog)
            chat.add_tool_call("calc", "tc_1", "{}")
            chat.add_tool_call("search", "tc_2", "{}")
            await pilot.pause()

            # All start collapsed
            assert all(w.collapsed for w in chat.query(ToolCallWidget))

            # Focus the chat area so the input box doesn't intercept Tab
            chat.focus()
            await pilot.pause()

            # Call the action directly
            app.action_toggle_all_tools()
            await pilot.pause()
            assert all(not w.collapsed for w in chat.query(ToolCallWidget))

            # Toggle again to collapse
            app.action_toggle_all_tools()
            await pilot.pause()
            assert all(w.collapsed for w in chat.query(ToolCallWidget))
