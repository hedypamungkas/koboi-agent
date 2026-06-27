"""ToolCallWidget -- collapsible tool call with timing, args, result, and diff view."""

from __future__ import annotations

import json

from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static
from textual.widget import Widget

from koboi.tui.widgets.diff_view import DiffViewWidget, SideBySideDiffWidget, is_diff_content, count_changes


class ToolCallWidget(Widget):
    """A collapsible widget representing a single tool call in the chat log.

    Shows a one-line header with tool name, timing, and size.
    Expanding reveals arguments, result, and optional diff view.
    """

    DEFAULT_CSS = """
    ToolCallWidget {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    ToolCallWidget .tool-header {
        width: 100%;
        padding: 0 2;
        background: $surface;
        color: $warning;
    }

    ToolCallWidget .tool-header.running {
        color: $warning;
        text-style: dim;
    }

    ToolCallWidget .tool-header.completed {
        color: $success;
    }

    ToolCallWidget .tool-header.error {
        color: $error;
    }

    ToolCallWidget .tool-body {
        width: 100%;
        padding: 0 2;
        margin: 0 0 0 2;
        max-height: 32;
        overflow-y: auto;
        display: none;
    }

    ToolCallWidget.expanded .tool-body {
        display: block;
    }

    ToolCallWidget .tool-args {
        width: 100%;
        color: $text-muted;
        text-style: dim;
        margin: 0 0 1 0;
        display: none;
    }

    ToolCallWidget.risk-moderate .tool-header {
        color: $warning;
    }

    ToolCallWidget.risk-destructive .tool-header {
        color: $error;
        text-style: bold;
    }
    """

    collapsed: reactive[bool] = reactive(True)

    def __init__(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: str,
        start_time: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._tool_call_id = tool_call_id
        self._arguments = arguments
        self._result: str = ""
        self._start_time = start_time
        self._end_time: float = 0.0
        self._state = "running"
        self._body_populated = False
        self._result_pending = False

    def compose(self):
        yield Static(self._render_header(), classes="tool-header running")
        with Vertical(classes="tool-body"):
            yield Static(self._format_args(self._arguments), classes="tool-args")

    def on_mount(self) -> None:
        """Apply any pending result after the widget is mounted."""
        if self._result_pending:
            self._result_pending = False
            self._apply_result()

    def _render_header(self) -> str:
        triangle = "v" if not self.collapsed else ">"
        arg_summary = self._arg_summary()
        if self._state == "running":
            if arg_summary:
                return f"{triangle} running: {self._tool_name}({arg_summary}) ..."
            return f"{triangle} running: {self._tool_name} ..."
        elapsed = self._end_time - self._start_time
        size = len(self._result)
        if arg_summary:
            parts = [f"{triangle} {self._tool_name}({arg_summary})", f"{elapsed:.1f}s", f"{size}B"]
        else:
            parts = [f"{triangle} {self._tool_name}", f"{elapsed:.1f}s", f"{size}B"]
        if self._state == "error":
            parts.append("ERROR")
        elif is_diff_content(self._result):
            adds, dels = count_changes(self._result)
            if adds or dels:
                parts.append(f"+{adds}/-{dels}")
        return " | ".join(parts)

    @staticmethod
    def _shorten(text: str, max_len: int = 40) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _arg_summary(self) -> str:
        """Extract a short, meaningful summary from the tool arguments JSON."""
        if not self._arguments:
            return ""
        try:
            parsed = json.loads(self._arguments)
        except (json.JSONDecodeError, TypeError):
            return self._shorten(self._arguments, 30)
        if not isinstance(parsed, dict):
            return ""
        # Priority order for common tools
        for key in ("path", "file_path", "directory", "command", "pattern", "query", "key", "title", "query", "url"):
            if key in parsed:
                val = str(parsed[key])
                return self._shorten(val, 45)
        # For delegate_tasks, show count
        if "tasks" in parsed and isinstance(parsed["tasks"], list):
            return f"{len(parsed['tasks'])} tasks"
        # Fallback: first string value
        for val in parsed.values():
            if isinstance(val, str) and val:
                return self._shorten(val, 45)
        return ""

    @staticmethod
    def _format_args(args: str) -> str:
        if not args:
            return "(no arguments)"
        try:
            parsed = json.loads(args)
            return json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            return args[:500] + ("..." if len(args) > 500 else "")

    def set_result(self, result: str, end_time: float) -> None:
        """Populate the tool body with the result and update the header."""
        self._result = result
        self._end_time = end_time
        self._state = "error" if result.startswith("Error:") else "completed"
        # If not mounted yet, defer to on_mount
        if not self.is_mounted:
            self._result_pending = True
            return
        self._apply_result()

    def _apply_result(self) -> None:
        """Actually update the UI with the stored result."""
        try:
            header = self.query_one(".tool-header")
        except Exception:
            return
        header.update(self._render_header())  # type: ignore[attr-defined]  # Textual Static.update; header typed Widget by query_one
        header.remove_class("running")
        header.add_class(self._state)

        if not self._body_populated:
            self._body_populated = True
            try:
                body = self.query_one(".tool-body")
            except Exception:
                return
            if is_diff_content(self._result):
                try:
                    width = self.app.size.width
                except Exception:
                    width = 80
                if width > 120:
                    body.mount(SideBySideDiffWidget(self._result))
                else:
                    body.mount(DiffViewWidget(self._result))
            elif self._result.startswith("Error:"):
                body.mount(Static(self._result, classes="tool-error"))
            else:
                display = self._result[:2000] + ("..." if len(self._result) > 2000 else "")
                body.mount(Static(display))

        # Stay collapsed — header shows tool name, timing, and size.
        # User can click/press to expand for full details.

    def toggle(self) -> None:
        self.collapsed = not self.collapsed

    def watch_collapsed(self, collapsed: bool) -> None:
        self.set_class(not collapsed, "expanded")
        try:
            header = self.query_one(".tool-header")
            header.update(self._render_header())  # type: ignore[attr-defined]  # Textual Static.update; header typed Widget by query_one
        except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
            pass
