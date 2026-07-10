"""mcp_status.py -- Modal screen showing MCP server status (G7)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


class _McpServerRow(Static):
    """A single MCP server status row."""

    DEFAULT_CSS = """
    _McpServerRow {
        width: 100%;
        height: auto;
        padding: 0 2;
    }
    _McpServerRow.connected {
        color: $success;
    }
    _McpServerRow.dead {
        color: $error;
    }
    """

    def __init__(self, data: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def render(self) -> str:
        name = self._data.get("name") or self._data.get("id") or "server"
        transport = self._data.get("transport", "?")
        connected = bool(self._data.get("connected"))
        tool_names = self._data.get("tool_names", []) or []
        symbol = "[green]v[/green]" if connected else "[red]x[/red]"
        state = "connected" if connected else "DISCONNECTED"
        line = f"  {symbol}  {name} [dim]({transport}, {len(tool_names)} tool(s))[/dim] [{state}]"
        if tool_names:
            preview = ", ".join(tool_names[:6])
            if len(tool_names) > 6:
                preview += ", ..."
            line += f"\n    [dim]{preview}[/dim]"
        return line


class McpStatusScreen(ModalScreen[None]):
    """Modal overlay showing MCP server status (connected/dead + tools)."""

    CSS = """
    McpStatusScreen {
        background: rgba(0, 0, 0, 0.7);
    }
    #mcp-panel {
        width: 80;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }
    #mcp-panel .heading {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #mcp-panel .summary {
        color: $text-muted;
        margin-bottom: 1;
    }
    #mcp-panel .hint {
        color: $text-muted;
        margin-top: 1;
        text-align: center;
    }
    #mcp-panel .no-servers {
        color: $text-muted;
        text-style: dim;
        padding: 1 0;
    }
    #mcp-list {
        height: auto;
        max-height: 50;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(self, status_entries: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries = status_entries

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="mcp-panel"):
            yield Static("MCP Servers", classes="heading")
            yield Static(self._build_summary(), id="mcp-summary", classes="summary")
            with VerticalScroll(id="mcp-list"):
                yield from self._build_rows()
            yield Static("Press Esc to close", classes="hint")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._refresh)

    def _build_summary(self) -> str:
        total = len(self._entries)
        if not total:
            return "No MCP servers configured"
        connected = sum(1 for e in self._entries if e.get("connected"))
        dead = total - connected
        parts = [f"{total} server(s)", f"[green]{connected} connected[/green]"]
        if dead:
            parts.append(f"[red]{dead} disconnected[/red]")
        return " | ".join(parts)

    def _build_rows(self):
        if not self._entries:
            yield Static("  No MCP servers configured for this agent.", classes="no-servers")
            return
        for idx, entry in enumerate(self._entries):
            row = _McpServerRow(entry, id=f"mcp-row-{idx}")
            row.add_class("connected" if entry.get("connected") else "dead")
            yield row

    def _refresh(self) -> None:
        try:
            self.query_one("#mcp-summary", Static).update(self._build_summary())
        except Exception:  # nosec B110 - best-effort refresh
            pass
        try:
            lst = self.query_one("#mcp-list", VerticalScroll)
            lst.remove_children()
            for child in self._build_rows():
                lst.mount(child)
        except Exception:  # nosec B110 - best-effort refresh
            pass
