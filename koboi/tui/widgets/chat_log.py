"""ChatLog -- scrollable message list replacing the old ChatArea(RichLog)."""

from __future__ import annotations

import time

from textual.containers import VerticalScroll
from textual.widgets import Static

from koboi.tui.widgets.message_bubble import MessageBubble
from koboi.tui.widgets.tool_call import ToolCallWidget


class ChatLog(VerticalScroll):
    """Scrollable chat message container.

    Manages MessageBubble children, auto-scrolls to bottom,
    and supports streaming updates for assistant messages.
    """

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        width: 100%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._streaming_bubble: MessageBubble | None = None
        self._tool_widgets: dict[str, ToolCallWidget] = {}
        self._needs_paragraph_break: bool = False

    def add_message(self, role: str, content: str) -> MessageBubble:
        """Add a complete message and scroll to bottom."""
        bubble = MessageBubble(role, content)
        self.mount(bubble)
        self._scroll_to_end()
        return bubble

    def begin_stream(self) -> MessageBubble:
        """Mount an empty assistant bubble for streaming."""
        bubble = MessageBubble("assistant", "", is_streamable=True)
        self._streaming_bubble = bubble
        self._needs_paragraph_break = False
        self.mount(bubble)
        self._scroll_to_end()
        return bubble

    def append_delta(self, content: str) -> None:
        """Forward a text delta to the current streaming bubble."""
        if self._streaming_bubble is not None:
            if self._needs_paragraph_break:
                self._needs_paragraph_break = False
                # Only insert break if content doesn't already start with one
                if content and not content.startswith("\n"):
                    content = "\n\n" + content
            self._streaming_bubble.update_content(content)
            self._scroll_to_end()

    def finalize_stream(self, content: str) -> None:
        """Replace streaming bubble content with the authoritative final text."""
        if self._streaming_bubble is not None:
            if content:
                self._streaming_bubble.set_final_content(content)
            else:
                self._streaming_bubble.remove()
            self._streaming_bubble = None

    def move_streaming_bubble_to_end(self) -> None:
        """Move the streaming bubble to the bottom of the chat log."""
        if self._streaming_bubble is not None:
            self.move_child(self._streaming_bubble, after=-1)

    def add_tool_call(self, tool_name: str, tool_call_id: str, arguments: str) -> None:
        """Mount a new ToolCallWidget for an in-progress tool call."""
        start_time = time.monotonic()
        widget = ToolCallWidget(tool_name, tool_call_id, arguments, start_time)
        self._tool_widgets[tool_call_id] = widget
        self.mount(widget)
        # Don't move the streaming bubble — it stays at the end so the
        # user sees "Thinking..." / response text below the tool calls.
        # Mark that the next text delta should get a paragraph break
        self._needs_paragraph_break = True
        self._scroll_to_end()

    def update_tool_result(self, tool_call_id: str, result: str) -> None:
        """Update the ToolCallWidget with its result."""
        widget = self._tool_widgets.pop(tool_call_id, None)
        if widget:
            widget.set_result(result, time.monotonic())
        else:
            preview = result[:120] + "..." if len(result) > 120 else result
            self.mount(Static(f"  {tool_call_id} -> {preview}", classes="tool-result"))
        self._scroll_to_end()

    def collapse_all_tools(self) -> None:
        """Collapse all ToolCallWidgets."""
        for widget in self.query(ToolCallWidget):
            widget.collapsed = True

    def expand_all_tools(self) -> None:
        """Expand all ToolCallWidgets."""
        for widget in self.query(ToolCallWidget):
            widget.collapsed = False

    def add_iteration_marker(self, iteration: int, messages_count: int) -> None:
        """Show a subtle iteration divider."""
        marker = Static(
            f"  --- iteration {iteration} ({messages_count} messages) ---",
            classes="iteration-marker",
        )
        self.mount(marker)
        self._scroll_to_end()

    def add_error(self, error: Exception) -> None:
        """Show an error message."""
        bubble = MessageBubble("system", f"Error: {error}")
        self.mount(bubble)
        self._scroll_to_end()

    def add_system_message(self, content: str) -> None:
        """Show a system message (used by slash commands)."""
        bubble = MessageBubble("system", content)
        self.mount(bubble)
        self._scroll_to_end()

    def clear_messages(self) -> None:
        """Remove all messages (for /reset)."""
        self.remove_children()
        self._streaming_bubble = None
        self._tool_widgets.clear()
        self._needs_paragraph_break = False

    def add_routing_decision(self, agents: list[str], method: str, confidence: float) -> None:
        """Show routing info as a compact system line."""
        agents_str = ", ".join(f"[cyan]{a}[/cyan]" for a in agents)
        content = f"  Routed ({method}, {confidence:.0%}): {agents_str}"
        marker = Static(content, classes="orchestration-marker")
        self.mount(marker)
        self._scroll_to_end()

    def add_agent_status(self, agent_name: str, status: str, elapsed: float | None = None) -> None:
        """Show agent status as a compact line."""
        if status == "running":
            content = f"  > {agent_name.upper()} agent working..."
        elif status == "done":
            content = f"  [green]v[/green] {agent_name.upper()} agent completed ({elapsed:.1f}s)"
        else:
            content = f"  [red]x[/red] {agent_name.upper()} agent failed"
        marker = Static(content, classes="orchestration-marker")
        self.mount(marker)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        # Prefer scrolling to the streaming bubble if active (follows live content)
        if self._streaming_bubble is not None:
            self.scroll_to_widget(self._streaming_bubble, animate=False)
            return
        kids = list(self.children)
        if kids:
            self.scroll_to_widget(kids[-1], animate=False)
        else:
            self.scroll_end(animate=False)
