"""permission_dialog.py -- Non-blocking modal screen for tool approval."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from koboi.tui.widgets.risk_bar import RiskBar


class PermissionResult:
    """Result returned by the permission dialog."""

    __slots__ = ("approved", "always_allow")

    def __init__(self, approved: bool, always_allow: bool = False):
        self.approved = approved
        self.always_allow = always_allow


class PermissionDialog(ModalScreen[PermissionResult]):
    """Non-blocking overlay for tool approval.

    Shows tool name, risk level bar, argument preview.
    Keys: y=approve once, a=always allow, n=deny, d=full diff.
    """

    CSS = """
    PermissionDialog {
        background: rgba(0, 0, 0, 0.6);
    }

    PermissionDialog > Vertical {
        align: center middle;
        width: 100%;
        height: 100%;
    }

    #perm-panel {
        width: 70;
        max-width: 85%;
        height: auto;
        max-height: 70%;
        background: $surface;
        border: tall $warning;
        padding: 1 2;
    }

    #perm-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    #perm-tool {
        margin-bottom: 1;
    }

    #perm-risk {
        margin-bottom: 1;
    }

    #perm-args-label {
        color: $text-muted;
        margin-top: 1;
    }

    #perm-args {
        height: auto;
        max-height: 12;
        overflow-y: auto;
        background: $surface-darken-1;
        border: solid $primary-background-darken-2;
        padding: 0 1;
        margin-bottom: 1;
    }

    #perm-keys {
        color: $text-muted;
        margin-top: 1;
    }

    .perm-key {
        text-style: bold;
        color: $accent;
    }
    """

    BINDINGS = [
        Binding("y", "approve", "Yes, once"),
        Binding("a", "always", "Always allow"),
        Binding("n", "deny", "No"),
        Binding("d", "show_diff", "Full diff"),
        Binding("escape", "deny", "Cancel"),
    ]

    def __init__(
        self,
        tool_name: str,
        arguments: str,
        risk_level: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._arguments = arguments
        self._risk_level = risk_level
        self._showing_full = False

    def compose(self) -> ComposeResult:
        with Vertical():
            with Vertical(id="perm-panel"):
                yield Static("Permission Required", id="perm-title")
                yield Static(
                    f"[bold]Tool:[/bold]  {self._tool_name}",
                    id="perm-tool",
                )
                _risk_bar = RiskBar(id="perm-risk")
                _risk_bar.risk_level = self._risk_level
                yield _risk_bar
                yield Static("[bold]Preview:[/bold]", id="perm-args-label")
                yield Static(
                    self._format_args_preview(self._arguments),
                    id="perm-args",
                )
                yield Static(
                    "[bold]y[/bold] Yes, once    "
                    "[bold]a[/bold] Always allow    "
                    "[bold]n[/bold] No    "
                    "[bold]d[/bold] Full diff",
                    id="perm-keys",
                )

    def action_approve(self) -> None:
        self.dismiss(PermissionResult(approved=True, always_allow=False))

    def action_always(self) -> None:
        self.dismiss(PermissionResult(approved=True, always_allow=True))

    def action_deny(self) -> None:
        self.dismiss(PermissionResult(approved=False, always_allow=False))

    def action_show_diff(self) -> None:
        """Toggle full argument display (could expand to show full diff)."""
        args_widget = self.query_one("#perm-args", Static)
        self._showing_full = not self._showing_full
        if self._showing_full:
            args_widget.update(self._arguments)
        else:
            args_widget.update(self._format_args_preview(self._arguments))

    @staticmethod
    def _format_args_preview(arguments: str, max_len: int = 300) -> str:
        """Format arguments for preview, truncating if needed."""
        if len(arguments) <= max_len:
            return arguments
        return arguments[:max_len] + "..."
