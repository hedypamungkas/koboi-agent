"""yolo_confirm.py -- Non-blocking modal confirmation for YOLO mode activation."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class YoloConfirmDialog(ModalScreen[bool]):
    """Non-blocking overlay for YOLO mode activation confirmation."""

    CSS = """
    YoloConfirmDialog {
        background: rgba(0, 0, 0, 0.6);
    }

    YoloConfirmDialog > Vertical {
        align: center middle;
        width: 100%;
        height: 100%;
    }

    #yolo-panel {
        width: 70;
        max-width: 85%;
        height: auto;
        background: $surface;
        border: tall $warning;
        padding: 1 2;
    }

    #yolo-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    #yolo-warning {
        margin-bottom: 1;
    }

    #yolo-keys {
        color: $text-muted;
        margin-top: 1;
    }

    .yolo-key {
        text-style: bold;
        color: $accent;
    }
    """

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            with Vertical(id="yolo-panel"):
                yield Static("YOLO Mode", id="yolo-title")
                yield Static(
                    "[bold yellow]WARNING:[/bold yellow] YOLO mode bypasses rate limiting, "
                    "approval prompts, and mode restrictions.\n\n"
                    "Only hardcoded safety checks (sensitive paths, dangerous commands) "
                    "remain active.\n\n"
                    "Use with caution.",
                    id="yolo-warning",
                )
                yield Static(
                    "[bold]y[/bold] Activate    [bold]n[/bold] Cancel",
                    id="yolo-keys",
                )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
