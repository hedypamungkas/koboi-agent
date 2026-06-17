"""subagent_monitor.py -- Modal screen showing active and completed sub-agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    pass

_STATUS_SYMBOLS = {
    "pending": "[dim]queued[/dim]",
    "running": "[yellow]> running[/yellow]",
    "done": "[green]v done[/green]",
    "failed": "[red]x failed[/red]",
}


class _AgentRow(Static):
    """A single agent status row in the monitor."""

    DEFAULT_CSS = """
    _AgentRow {
        width: 100%;
        height: auto;
        padding: 0 2;
    }
    _AgentRow.running {
        background: $surface;
        color: $warning;
    }
    _AgentRow.done {
        color: $success;
    }
    _AgentRow.failed {
        color: $error;
    }
    """

    def __init__(self, agent_data: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = agent_data

    def render(self) -> str:
        name = self._data.get("name", "unknown")
        status = self._data.get("status", "pending")
        elapsed = self._data.get("elapsed", 0.0)
        answer_preview = self._data.get("answer_preview", "")
        is_dynamic = self._data.get("is_dynamic", False)
        domain = self._data.get("domain_label")

        symbol = _STATUS_SYMBOLS.get(status, status)
        parts = [f"  {symbol}  {name}"]

        if is_dynamic and domain:
            parts.append(f"({domain})")

        if status in ("done", "failed") and elapsed > 0:
            parts.append(f"[dim]{elapsed:.1f}s[/dim]")

        line = " ".join(parts)

        if status == "done" and answer_preview:
            preview = answer_preview[:120]
            if len(answer_preview) > 120:
                preview += "..."
            line += f"\n    [dim]{preview}[/dim]"

        return line


class SubagentMonitorScreen(ModalScreen[None]):
    """Modal overlay showing all sub-agents and their current status."""

    CSS = """
    SubagentMonitorScreen {
        background: rgba(0, 0, 0, 0.7);
    }
    #monitor-panel {
        width: 80;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }
    #monitor-panel .heading {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #monitor-panel .summary {
        color: $text-muted;
        margin-bottom: 1;
    }
    #monitor-panel .hint {
        color: $text-muted;
        margin-top: 1;
        text-align: center;
    }
    #monitor-panel .no-agents {
        color: $text-muted;
        text-style: dim;
        padding: 1 0;
    }
    #agent-list {
        height: auto;
        max-height: 50;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(self, agent_states: dict[str, dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self._agent_states = agent_states

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="monitor-panel"):
            yield Static("Sub-Agent Monitor", classes="heading")
            yield Static(self._build_summary(), id="monitor-summary", classes="summary")
            with VerticalScroll(id="agent-list"):
                yield from self._build_rows()
            yield Static("Press Esc to close", classes="hint")

    def on_mount(self) -> None:
        """Set up periodic refresh while screen is open."""
        self.set_interval(0.5, self._refresh)

    def _build_summary(self) -> str:
        total = len(self._agent_states)
        running = sum(1 for s in self._agent_states.values() if s.get("status") == "running")
        done = sum(1 for s in self._agent_states.values() if s.get("status") == "done")
        failed = sum(1 for s in self._agent_states.values() if s.get("status") == "failed")
        parts = []
        if total:
            parts.append(f"{total} agent(s)")
        if running:
            parts.append(f"[yellow]{running} running[/yellow]")
        if done:
            parts.append(f"[green]{done} done[/green]")
        if failed:
            parts.append(f"[red]{failed} failed[/red]")
        return " | ".join(parts) if parts else "No agents dispatched"

    def _build_rows(self):
        if not self._agent_states:
            yield Static("  No agents in current orchestration.", classes="no-agents")
            return
        for key in sorted(self._agent_states.keys()):
            data = self._agent_states[key]
            row = _AgentRow(data, id=f"agent-row-{key}")
            status = data.get("status", "pending")
            if status == "running":
                row.add_class("running")
            elif status == "done":
                row.add_class("done")
            elif status == "failed":
                row.add_class("failed")
            yield row

    def _refresh(self) -> None:
        """Re-render the agent list and summary."""
        try:
            summary = self.query_one("#monitor-summary", Static)
            summary.update(self._build_summary())
        except Exception:
            pass

        try:
            agent_list = self.query_one("#agent-list", VerticalScroll)
            agent_list.remove_children()
            for child in self._build_rows():
                agent_list.mount(child)
        except Exception:
            pass
