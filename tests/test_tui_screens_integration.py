"""Integration tests for TUI screens and theme cycling.

Tier 1 tests: screen interaction flows (CommandPalette, HistorySearch,
PermissionDialog) and theme cycling via Ctrl+T.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koboi.tui.screens.command_palette import CommandPaletteScreen
from koboi.tui.screens.history_search import HistorySearchScreen
from koboi.tui.screens.permission_dialog import PermissionDialog, PermissionResult
from koboi.tui.textual_app import KoboiApp
from textual.widgets import Input, OptionList


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_command_palette.py and test_textual_tui.py)
# ---------------------------------------------------------------------------


def _make_test_app(screen, on_dismiss=None):
    """Create a minimal app that pushes the given screen."""
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("test")

        def on_mount(self) -> None:
            self.push_screen(screen, on_dismiss)

    return TestApp()


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


# ---------------------------------------------------------------------------
# CommandPaletteScreen integration
# ---------------------------------------------------------------------------


class TestCommandPaletteScreenIntegration:
    @pytest.mark.asyncio
    async def test_select_option_dismisses_value(self):
        screen = CommandPaletteScreen(["/reset", "/info", "/help"])
        dismissed: list = []
        app = _make_test_app(screen, on_dismiss=lambda v: dismissed.append(v))
        async with app.run_test() as pilot:
            option_list = screen.query_one("#palette-list", OptionList)
            option_list.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert dismissed == ["/reset"]

    @pytest.mark.asyncio
    async def test_filter_then_select(self):
        screen = CommandPaletteScreen(["/reset", "/info", "/help", "/tools"])
        dismissed: list = []
        app = _make_test_app(screen, on_dismiss=lambda v: dismissed.append(v))
        async with app.run_test() as pilot:
            search = screen.query_one("#palette-search", Input)
            search.value = "re"
            await pilot.pause()
            await pilot.pause()
            option_list = screen.query_one("#palette-list", OptionList)
            assert len(option_list._options) == 1
            option_list.focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert dismissed == ["/reset"]

    @pytest.mark.asyncio
    async def test_filter_no_match_shows_empty(self):
        screen = CommandPaletteScreen(["/reset", "/info"])
        app = _make_test_app(screen)
        async with app.run_test() as pilot:
            search = screen.query_one("#palette-search", Input)
            search.value = "zzz"
            await pilot.pause()
            option_list = screen.query_one("#palette-list", OptionList)
            assert len(option_list._options) == 0


# ---------------------------------------------------------------------------
# HistorySearchScreen interaction
# ---------------------------------------------------------------------------


class TestHistorySearchScreenInteraction:
    @pytest.mark.asyncio
    async def test_shows_reversed_entries(self):
        screen = HistorySearchScreen(["first", "second", "third"])
        app = _make_test_app(screen)
        async with app.run_test() as pilot:
            option_list = screen.query_one("#history-list", OptionList)
            assert len(option_list._options) == 3
            # Reversed: third, second, first
            assert option_list._options[0].prompt == "third"
            assert option_list._options[2].prompt == "first"

    @pytest.mark.asyncio
    async def test_filter_by_typing(self):
        screen = HistorySearchScreen(["alpha", "beta", "gamma", "alphabet"])
        app = _make_test_app(screen)
        async with app.run_test() as pilot:
            search = screen.query_one("#history-search", Input)
            search.value = "alp"
            await pilot.pause()
            option_list = screen.query_one("#history-list", OptionList)
            # Original reversed: alphabet, gamma, beta, alpha
            # Filter "alp" matches: alphabet, alpha -> 2 entries
            assert len(option_list._options) == 2

    @pytest.mark.asyncio
    async def test_select_entry_dismisses_value(self):
        screen = HistorySearchScreen(["first", "second", "third"])
        dismissed: list = []
        app = _make_test_app(screen, on_dismiss=lambda v: dismissed.append(v))
        async with app.run_test() as pilot:
            option_list = screen.query_one("#history-list", OptionList)
            option_list.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            # Reversed list: "third" is first, so Enter selects it
            assert dismissed == ["third"]

    @pytest.mark.asyncio
    async def test_escape_dismisses_none(self):
        screen = HistorySearchScreen(["entry1"])
        dismissed: list = []
        app = _make_test_app(screen, on_dismiss=lambda v: dismissed.append(v))
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()
            assert dismissed == [None]

    @pytest.mark.asyncio
    async def test_filter_then_clear_shows_all(self):
        screen = HistorySearchScreen(["a", "b", "c"])
        app = _make_test_app(screen)
        async with app.run_test() as pilot:
            search = screen.query_one("#history-search", Input)
            search.value = "a"
            await pilot.pause()
            option_list = screen.query_one("#history-list", OptionList)
            assert len(option_list._options) == 1
            # Clear filter
            search.value = ""
            await pilot.pause()
            assert len(option_list._options) == 3


# ---------------------------------------------------------------------------
# PermissionDialog interaction
# ---------------------------------------------------------------------------


class TestPermissionDialogInteraction:
    @pytest.mark.asyncio
    async def test_press_y_approves_once(self):
        screen = PermissionDialog("shell", '{"cmd":"ls"}', "safe")
        results: list = []
        app = _make_test_app(screen, on_dismiss=lambda v: results.append(v))
        async with app.run_test() as pilot:
            await pilot.press("y")
            await pilot.pause()
            assert len(results) == 1
            assert results[0].approved is True
            assert results[0].always_allow is False

    @pytest.mark.asyncio
    async def test_press_a_always_approves(self):
        screen = PermissionDialog("shell", '{"cmd":"ls"}', "moderate")
        results: list = []
        app = _make_test_app(screen, on_dismiss=lambda v: results.append(v))
        async with app.run_test() as pilot:
            await pilot.press("a")
            await pilot.pause()
            assert len(results) == 1
            assert results[0].approved is True
            assert results[0].always_allow is True

    @pytest.mark.asyncio
    async def test_press_n_denies(self):
        screen = PermissionDialog("shell", '{"cmd":"rm -rf"}', "destructive")
        results: list = []
        app = _make_test_app(screen, on_dismiss=lambda v: results.append(v))
        async with app.run_test() as pilot:
            await pilot.press("n")
            await pilot.pause()
            assert len(results) == 1
            assert results[0].approved is False
            assert results[0].always_allow is False

    @pytest.mark.asyncio
    async def test_escape_denies(self):
        screen = PermissionDialog("shell", "{}", "safe")
        results: list = []
        app = _make_test_app(screen, on_dismiss=lambda v: results.append(v))
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()
            assert len(results) == 1
            assert results[0].approved is False

    @pytest.mark.asyncio
    async def test_d_key_does_not_dismiss(self):
        screen = PermissionDialog("shell", '{"long":"args"}', "safe")
        results: list = []
        app = _make_test_app(screen, on_dismiss=lambda v: results.append(v))
        async with app.run_test() as pilot:
            await pilot.press("d")
            await pilot.pause()
            # 'd' toggles diff display, should NOT dismiss
            assert len(results) == 0
            assert len(app._screen_stack) > 1


# ---------------------------------------------------------------------------
# Theme cycling
# ---------------------------------------------------------------------------


class TestThemeCycling:
    @pytest.mark.asyncio
    async def test_ctrl_t_cycles_dark_to_light(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            assert app.theme == "koboi-dark"
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app.theme == "koboi-light"

    @pytest.mark.asyncio
    async def test_ctrl_t_cycles_back_to_dark(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app.theme == "koboi-light"
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app.theme == "koboi-dark"

    @pytest.mark.asyncio
    async def test_action_cycle_theme_direct(self):
        app = KoboiApp(_make_mock_agent())
        async with app.run_test() as pilot:
            app.action_cycle_theme()
            await pilot.pause()
            assert app.theme == "koboi-light"
            app.action_cycle_theme()
            await pilot.pause()
            assert app.theme == "koboi-dark"
