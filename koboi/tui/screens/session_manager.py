"""session_manager.py -- Session browser/manager modal."""
from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from koboi.memory_sqlite import SQLiteMemory


class SessionManagerScreen(ModalScreen[Optional[str]]):
    """Modal screen for browsing, resuming, and deleting sessions.

    Returns: selected session_id to resume, or None to dismiss.
    """

    CSS = """
    SessionManagerScreen {
        background: rgba(0, 0, 0, 0.7);
    }
    #session-panel {
        width: 72;
        max-width: 85%;
        height: auto;
        max-height: 70%;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }
    #session-panel Input {
        margin: 0 0 1 0;
    }
    #session-panel OptionList {
        height: auto;
        max-height: 24;
    }
    #session-detail {
        color: $text-muted;
        margin: 1 0 0 0;
        height: auto;
    }
    .hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_none", "Close"),
        Binding("enter", "resume_selected", "Resume"),
        Binding("d", "delete_selected", "Delete"),
    ]

    def __init__(self, db_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._db_path = db_path
        self._sessions: list[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="session-panel"):
            yield Input(placeholder="Filter sessions...", id="session-filter")
            yield OptionList(id="session-list")
            yield Static("", id="session-detail")
            yield Static("Enter=Resume  D=Delete  Esc=Close", classes="hint")

    def on_mount(self) -> None:
        self._refresh_list()
        self.query_one("#session-filter", Input).focus()

    def _refresh_list(self, filter_text: str = "") -> None:
        self._sessions = SQLiteMemory.list_sessions(self._db_path)
        option_list = self.query_one("#session-list", OptionList)
        option_list.clear_options()

        for s in self._sessions:
            title = s.get("title") or s.get("first_message") or s["session_id"][:8]
            if len(title) > 50:
                title = title[:50] + "..."
            count = s.get("message_count", 0)
            label = f"{title}  ({count} msgs)"
            if filter_text and filter_text.lower() not in label.lower():
                continue
            option_list.add_option(Option(label, id=s["session_id"]))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "session-filter":
            self._refresh_list(event.value)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id:
            for s in self._sessions:
                if s["session_id"] == event.option_id:
                    detail = self.query_one("#session-detail", Static)
                    agent = s.get("agent_name") or "?"
                    model = s.get("model") or "?"
                    count = s.get("message_count", 0)
                    detail.update(f"Agent: {agent} | Model: {model} | Messages: {count}")
                    break

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def action_resume_selected(self) -> None:
        option_list = self.query_one("#session-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            self.dismiss(option.id)

    def action_delete_selected(self) -> None:
        option_list = self.query_one("#session-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            session_id = option.id
            if session_id:
                SQLiteMemory.delete_session(self._db_path, session_id)
                self._refresh_list()
