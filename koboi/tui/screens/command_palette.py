"""CommandPaletteScreen -- Ctrl+P command palette overlay."""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList
from textual.containers import Vertical


class CommandPaletteScreen(ModalScreen[Optional[str]]):
    """Modal command palette with search and option list."""

    CSS = """
    CommandPaletteScreen {
        background: rgba(0, 0, 0, 0.5);
    }

    CommandPaletteScreen > Vertical {
        align: center middle;
        width: 100%;
        height: 100%;
    }

    #palette-panel {
        width: 60;
        max-width: 80%;
        height: auto;
        max-height: 50%;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
        margin-bottom: 2;
    }

    #palette-search {
        margin: 0 0 1 0;
    }

    #palette-list {
        height: auto;
        max-height: 24;
    }
    """

    BINDINGS = [("escape", "dismiss_none", "Close")]

    def __init__(self, commands: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._commands = commands

    def compose(self) -> ComposeResult:
        with Vertical():
            with Vertical(id="palette-panel"):
                yield Input(placeholder="Search commands...", id="palette-search")
                yield OptionList(*self._commands, id="palette-list")

    def on_mount(self) -> None:
        self.query_one("#palette-search").focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter the option list as the user types."""
        query = event.value.lower()
        option_list = self.query_one("#palette-list", OptionList)
        option_list.clear_options()
        if query:
            filtered = [cmd for cmd in self._commands if query in cmd.lower()]
        else:
            filtered = self._commands
        if filtered:
            option_list.add_options(filtered)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Dismiss with the selected command."""
        self.dismiss(str(event.option.prompt))
