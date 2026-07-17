"""Tests for CommandPaletteScreen."""

from __future__ import annotations


from koboi.tui.screens.command_palette import CommandPaletteScreen
from textual.widgets import Input, OptionList


class TestCommandPaletteScreen:
    async def test_shows_commands(self):
        screen = CommandPaletteScreen(["/reset", "/info", "/help"])
        app = _make_test_app(screen)
        async with app.run_test():
            option_list = screen.query_one("#palette-list", OptionList)
            assert len(option_list._options) == 3

    async def test_escape_dismisses_with_none(self):
        dismissed = []
        screen = CommandPaletteScreen(["/reset", "/info"])
        app = _make_test_app(screen, on_dismiss=lambda v: dismissed.append(v))
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()
            assert dismissed == [None]

    async def test_filter_on_input(self):
        screen = CommandPaletteScreen(["/reset", "/info", "/help", "/tools"])
        app = _make_test_app(screen)
        async with app.run_test() as pilot:
            search = screen.query_one("#palette-search", Input)
            search.value = "re"
            await pilot.pause()
            option_list = screen.query_one("#palette-list", OptionList)
            # Should filter to only "/reset"
            assert len(option_list._options) == 1


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
