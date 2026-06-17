"""HistorySearchScreen -- Ctrl+R history search overlay."""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList
from textual.containers import Vertical


class HistorySearchScreen(ModalScreen[Optional[str]]):
    """Modal history search with substring filter."""

    CSS = """
    HistorySearchScreen {
        background: rgba(0, 0, 0, 0.5);
    }

    HistorySearchScreen > Vertical {
        align: center middle;
        width: 100%;
        height: 100%;
    }

    #history-panel {
        width: 70;
        max-width: 80%;
        height: auto;
        max-height: 50%;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }

    #history-search {
        margin: 0 0 1 0;
    }

    #history-list {
        height: auto;
        max-height: 24;
    }
    """

    BINDINGS = [("escape", "dismiss_none", "Close")]

    def __init__(self, history: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        # Show most recent first
        self._history = list(reversed(history))

    def compose(self) -> ComposeResult:
        with Vertical():
            with Vertical(id="history-panel"):
                yield Input(placeholder="Search history...", id="history-search")
                yield OptionList(*self._history[:50], id="history-list")

    def on_mount(self) -> None:
        self.query_one("#history-search").focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter history entries as the user types."""
        query = event.value.lower()
        option_list = self.query_one("#history-list", OptionList)
        option_list.clear_options()
        if query:
            filtered = [entry for entry in self._history if query in entry.lower()]
        else:
            filtered = self._history[:50]
        if filtered:
            option_list.add_options(filtered)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Dismiss with the selected history entry."""
        self.dismiss(str(event.option.prompt))
