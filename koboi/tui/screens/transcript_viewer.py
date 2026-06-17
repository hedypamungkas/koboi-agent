"""transcript_viewer.py -- Full conversation transcript viewer."""
from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static


class TranscriptViewerScreen(ModalScreen[None]):
    """Full-screen read-only conversation transcript viewer."""

    CSS = """
    TranscriptViewerScreen {
        background: $background;
    }
    #transcript-panel {
        width: 100%;
        height: 100%;
    }
    #transcript-search {
        dock: top;
        height: 3;
        margin: 0 1;
    }
    #transcript-scroll {
        height: 1fr;
    }
    .msg-user {
        background: $surface;
        border-left: solid $accent;
        padding: 0 2;
        margin: 0 0 1 2;
    }
    .msg-assistant {
        background: $surface-darken-1;
        border-left: solid $success;
        padding: 0 2;
        margin: 0 0 1 2;
    }
    .msg-tool {
        color: $text-muted;
        padding: 0 2;
        margin: 0 0 1 4;
    }
    .msg-label {
        text-style: bold;
        margin-bottom: 1;
    }
    .user-label { color: $accent; }
    .assistant-label { color: $success; }
    .tool-label { color: $warning; }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("slash", "focus_search", "Search", show=False),
    ]

    def __init__(self, messages: list[dict[str, Any]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._messages = messages
        self._widgets: list[tuple[str, Static]] = []

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search transcript...", id="transcript-search")
        with VerticalScroll(id="transcript-scroll"):
            for msg in self._messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "") or ""
                if role == "system":
                    continue
                if role == "tool":
                    label = "Tool Result"
                    css_class = "msg-tool"
                    label_class = "tool-label"
                elif role == "user":
                    label = "User"
                    css_class = "msg-user"
                    label_class = "user-label"
                else:
                    label = "Assistant"
                    css_class = "msg-assistant"
                    label_class = "assistant-label"

                label_widget = Static(label, classes=f"msg-label {label_class}")
                content_widget = Static(content, classes=css_class, markup=False)
                self._widgets.append((content.lower(), content_widget))
                yield label_widget
                yield content_widget

    def on_mount(self) -> None:
        self.query_one("#transcript-search", Input).focus()

    def action_focus_search(self) -> None:
        self.query_one("#transcript-search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.lower()
        for text, widget in self._widgets:
            if not query:
                widget.visible = True
            else:
                widget.visible = query in text
