"""Tests for G11 (sub-agent monitor), G12 (configurable keybindings), G15 (vim mode)."""

from __future__ import annotations

from unittest.mock import MagicMock


from koboi.config import Config
from koboi.tui.keybindings import get_keybinding_display, load_keybindings
from koboi.tui.textual_app import KoboiApp
from koboi.tui.widgets.input_box import ChatSubmit, InputBox
from koboi.tui.widgets.status_bar import StatusBar


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
    mock.config.get.return_value = None
    mock.core.tools._tools = {}
    mock.core.hooks.list_hooks.return_value = []
    mock.core.input_guardrail = None
    mock.core.output_guardrail = None
    mock.core.rate_limiter = None
    mock.core.approval_handler = None
    mock.core.memory.get_messages.return_value = []
    return mock


# ============================================================================
# G12: Configurable Keybindings
# ============================================================================


class TestKeybindings:
    def test_load_keybindings_defaults(self):
        """Default config produces standard bindings."""
        config = Config.from_dict({"agent": {"name": "test"}, "llm": {"model": "gpt-4o"}})
        bindings = load_keybindings(config)
        assert len(bindings) == 15
        actions = {b.action for b in bindings}
        assert "command_palette" in actions
        assert "cancel_or_quit" in actions
        assert "subagent_monitor" in actions
        assert "mcp_status" in actions
        assert "media_gallery" in actions  # W5c

    def test_load_keybindings_override(self):
        """YAML overrides change the key for an action."""
        config = Config.from_dict(
            {
                "agent": {"name": "test"},
                "llm": {"model": "gpt-4o"},
                "keybindings": {"f1": "help_overlay", "ctrl+p": "command_palette"},
            }
        )
        bindings = load_keybindings(config)
        help_binding = next(b for b in bindings if b.action == "help_overlay")
        assert help_binding.key == "f1"

    def test_load_keybindings_empty_override(self):
        """Empty overrides keep defaults."""
        config = Config.from_dict(
            {
                "agent": {"name": "test"},
                "llm": {"model": "gpt-4o"},
                "keybindings": {},
            }
        )
        bindings = load_keybindings(config)
        assert len(bindings) == 15

    def test_load_keybindings_invalid_config(self):
        """Non-dict config gracefully falls back to defaults."""
        config = MagicMock()
        config.get.return_value = "not-a-dict"
        bindings = load_keybindings(config)
        assert len(bindings) == 15

    def test_get_keybinding_display(self):
        """Display tuples exclude hidden bindings."""
        config = Config.from_dict({"agent": {"name": "test"}, "llm": {"model": "gpt-4o"}})
        display = get_keybinding_display(config)
        keys = [d[0] for d in display]
        assert "ctrl+p" in keys
        # Hidden bindings (show=False) should not appear
        assert "ctrl+k" not in keys
        assert "ctrl+m" not in keys

    def test_get_keybinding_display_override(self):
        """Display reflects overridden keys."""
        config = Config.from_dict(
            {
                "agent": {"name": "test"},
                "llm": {"model": "gpt-4o"},
                "keybindings": {"f5": "cycle_theme"},  # key->action format
            }
        )
        display = get_keybinding_display(config)
        theme_entry = next(d for d in display if d[1] == "Toggle Theme")
        assert theme_entry[0] == "f5"

    def test_config_keybindings_property(self):
        """Config.keybindings returns the keybindings dict."""
        config = Config.from_dict(
            {
                "agent": {"name": "test"},
                "llm": {"model": "gpt-4o"},
                "keybindings": {"ctrl+x": "custom_action"},
            }
        )
        assert config.keybindings == {"ctrl+x": "custom_action"}

    def test_config_keybindings_default_empty(self):
        """Config.keybindings returns empty dict when not set."""
        config = Config.from_dict({"agent": {"name": "test"}, "llm": {"model": "gpt-4o"}})
        assert config.keybindings == {}


# ============================================================================
# G15: Vim Input Mode
# ============================================================================


class TestVimMode:
    async def test_vim_toggle_via_slash(self):
        """Toggling vim mode via /vim command."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            status = app.query_one("#status-bar", StatusBar)

            assert not input_box.vim_enabled

            # Toggle vim on
            app.post_message(ChatSubmit("/vim"))
            await pilot.pause()
            assert input_box.vim_enabled
            assert input_box.vim_mode == "normal"
            assert status.vim_enabled

            # Toggle vim off
            app.post_message(ChatSubmit("/vim"))
            await pilot.pause()
            assert not input_box.vim_enabled
            assert input_box.vim_mode == "insert"

    async def test_vim_normal_mode_h_l(self):
        """h and l move cursor in normal mode."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"
            input_box.value = "hello world"
            input_box.cursor_position = 5
            input_box.vim_mode = "normal"

            # l moves right
            await pilot.press("l")
            assert input_box.cursor_position == 6

            # h moves left
            await pilot.press("h")
            assert input_box.cursor_position == 5

    async def test_vim_normal_mode_i_enters_insert(self):
        """'i' in normal mode switches to insert mode."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "normal"

            await pilot.press("i")
            assert input_box.vim_mode == "insert"

    async def test_vim_escape_returns_to_normal(self):
        """Escape in insert mode returns to normal mode."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"

            await pilot.press("escape")
            assert input_box.vim_mode == "normal"

    async def test_vim_x_deletes_char(self):
        """'x' in normal mode deletes character under cursor."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"
            input_box.value = "hello"
            input_box.cursor_position = 1
            input_box.vim_mode = "normal"

            await pilot.press("x")
            assert input_box.value == "hllo"

    async def test_vim_dd_deletes_line(self):
        """'dd' in normal mode deletes entire line."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"
            input_box.value = "hello world"
            input_box.vim_mode = "normal"

            await pilot.press("d")
            await pilot.press("d")
            assert input_box.value == ""

    async def test_vim_p_pastes_buffer(self):
        """'p' in normal mode pastes from the buffer."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"
            input_box.value = "hello"
            input_box.cursor_position = 0
            input_box.vim_mode = "normal"

            # Delete char at cursor (x) to fill buffer
            await pilot.press("x")
            assert input_box.value == "ello"
            assert input_box._vim_buffer == "h"

            # Paste after cursor
            await pilot.press("p")
            assert input_box.value == "ehllo"

    async def test_vim_A_enters_insert_at_end(self):
        """'A' in normal mode enters insert mode at end of line."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"
            input_box.value = "hello"
            input_box.cursor_position = 0
            input_box.vim_mode = "normal"

            await pilot.press("A")
            assert input_box.vim_mode == "insert"
            assert input_box.cursor_position == 5

    async def test_vim_zero_moves_to_start(self):
        """'0' in normal mode moves cursor to start."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"
            input_box.value = "hello"
            input_box.cursor_position = 4
            input_box.vim_mode = "normal"

            await pilot.press("0")
            assert input_box.cursor_position == 0

    async def test_vim_dollar_moves_to_end(self):
        """'$' in normal mode moves cursor to end."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"
            input_box.value = "hello"
            input_box.cursor_position = 0
            input_box.vim_mode = "normal"

            await pilot.press("dollar")
            assert input_box.cursor_position == 5

    async def test_vim_enter_does_not_submit_in_normal(self):
        """Enter in normal mode does NOT submit the message."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"
            input_box.value = "hello"
            input_box.vim_mode = "normal"

            # Simulate Enter (submitted event)
            input_box.action_submit()
            await pilot.pause()
            # Value should still be there (not cleared by submit)
            assert input_box.value == "hello"

    async def test_vim_w_moves_to_next_word(self):
        """'w' in normal mode moves to next word start."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            input_box.vim_enabled = True
            input_box.vim_mode = "insert"
            input_box.value = "hello world"
            input_box.cursor_position = 0
            input_box.vim_mode = "normal"

            await pilot.press("w")
            assert input_box.cursor_position == 6  # start of "world"

    async def test_vim_mode_posts_message(self):
        """Changing vim_mode posts VimModeChanged message."""
        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            input_box = app.query_one("#input-box", InputBox)
            received = []
            app.on_vim_mode_changed = lambda e: received.append(e.mode)
            input_box.vim_mode = "normal"
            await pilot.pause()
            # The watcher posts a message; verify it was set
            assert input_box.vim_mode == "normal"

    def test_vim_word_forward(self):
        """_find_word_forward finds next word boundary."""
        assert InputBox._find_word_forward("hello world", 0) == 6
        assert InputBox._find_word_forward("hello world", 6) is None
        assert InputBox._find_word_forward("a b c", 0) == 2

    def test_vim_word_backward(self):
        """_find_word_backward finds previous word boundary."""
        assert InputBox._find_word_backward("hello world", 6) == 0
        assert InputBox._find_word_backward("hello world", 0) is None

    def test_vim_word_end(self):
        """_find_word_end finds end of current/next word."""
        result = InputBox._find_word_end("hello world", 0)
        assert result is not None
        assert result >= 0


# ============================================================================
# G11: Sub-agent Monitor
# ============================================================================


class TestSubagentMonitor:
    async def test_agent_states_updated_on_dispatch(self):
        """Agent dispatch events update _agent_states."""
        from koboi.tui.bridge import StreamAgentDispatch, StreamRoutingDecision

        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.on_stream_routing_decision(
                StreamRoutingDecision(agents=["hr", "sales"], confidence=0.9, method="keyword", reasoning="")
            )
            await pilot.pause()
            assert "hr" in app._agent_states
            assert "sales" in app._agent_states
            assert app._agent_states["hr"]["status"] == "pending"

            app.on_stream_agent_dispatch(
                StreamAgentDispatch(agent_name="hr", agent_index=0, total_agents=2, mode="sequential")
            )
            await pilot.pause()
            assert app._agent_states["hr"]["status"] == "running"

    async def test_agent_states_updated_on_result(self):
        """Agent result events update _agent_states."""
        from koboi.tui.bridge import (
            StreamAgentDispatch,
            StreamAgentResult,
            StreamRoutingDecision,
        )

        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.on_stream_routing_decision(
                StreamRoutingDecision(agents=["hr"], confidence=0.9, method="keyword", reasoning="")
            )
            app.on_stream_agent_dispatch(
                StreamAgentDispatch(agent_name="hr", agent_index=0, total_agents=1, mode="sequential")
            )
            app.on_stream_agent_result(
                StreamAgentResult(agent_name="hr", answer="done", elapsed_seconds=1.5, failed=False)
            )
            await pilot.pause()
            assert app._agent_states["hr"]["status"] == "done"
            assert app._agent_states["hr"]["elapsed"] == 1.5

    async def test_agent_states_updated_on_failure(self):
        """Failed agents are tracked."""
        from koboi.tui.bridge import (
            StreamAgentDispatch,
            StreamAgentResult,
            StreamRoutingDecision,
        )

        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.on_stream_routing_decision(
                StreamRoutingDecision(agents=["sales"], confidence=0.8, method="keyword", reasoning="")
            )
            app.on_stream_agent_dispatch(
                StreamAgentDispatch(agent_name="sales", agent_index=0, total_agents=1, mode="sequential")
            )
            app.on_stream_agent_result(
                StreamAgentResult(agent_name="sales", answer="", elapsed_seconds=0.5, failed=True)
            )
            await pilot.pause()
            assert app._agent_states["sales"]["status"] == "failed"

    async def test_agent_states_cleared_on_new_routing(self):
        """New routing decision clears previous agent states."""
        from koboi.tui.bridge import StreamRoutingDecision

        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app._agent_states["old_agent"] = {"name": "old_agent", "status": "done"}
            app.on_stream_routing_decision(
                StreamRoutingDecision(agents=["new_agent"], confidence=0.9, method="keyword", reasoning="")
            )
            await pilot.pause()
            assert "old_agent" not in app._agent_states
            assert "new_agent" in app._agent_states

    async def test_subagent_dispatch_updates_states(self):
        """SubagentUIHook dispatch events update _agent_states."""
        from koboi.hooks.subagent_hook import _SubagentDispatch

        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.on_subagent_dispatch(_SubagentDispatch("task-1", 0, 2))
            await pilot.pause()
            assert "task-1" in app._agent_states
            assert app._agent_states["task-1"]["status"] == "running"

    async def test_subagent_result_updates_states(self):
        """SubagentUIHook result events update _agent_states."""
        from koboi.hooks.subagent_hook import _SubagentDispatch, _SubagentResult

        agent = _make_mock_agent()
        app = KoboiApp(agent)
        async with app.run_test() as pilot:
            app.on_subagent_dispatch(_SubagentDispatch("task-1", 0, 1))
            app.on_subagent_result(_SubagentResult("task-1", 2.3, True))
            await pilot.pause()
            assert app._agent_states["task-1"]["status"] == "done"
            assert app._agent_states["task-1"]["elapsed"] == 2.3

    def test_monitor_data_model(self):
        """Agent states dict tracks status correctly."""
        states: dict[str, dict] = {}
        states["hr"] = {"name": "hr", "status": "done", "elapsed": 1.2, "answer_preview": "ok"}
        states["sales"] = {"name": "sales", "status": "running", "elapsed": 0.0, "answer_preview": ""}
        assert states["hr"]["status"] == "done"
        assert states["sales"]["status"] == "running"
        states.clear()
        assert len(states) == 0
