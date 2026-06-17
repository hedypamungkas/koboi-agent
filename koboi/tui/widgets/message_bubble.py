"""MessageBubble -- a single chat message widget with streaming markdown support."""

from __future__ import annotations

import time

from rich.text import Text
from textual.widgets import Markdown, Static
from textual.widget import Widget

from koboi.tui.widgets.thinking_block import ThinkingBlockWidget, THINKING_PATTERNS


class RoleIndicator(Static):
    """Small role label displayed above the message content."""


class MessageBubble(Widget):
    """A single chat message bubble.

    For assistant messages, uses a Markdown widget that supports live updates
    during streaming. For user/system/tool messages, uses Static with Rich markup.
    """

    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 0;
    }

    MessageBubble.message-user {
        align: right top;
        margin-top: 1;
    }

    MessageBubble.message-user > Static.content {
        width: auto;
        max-width: 80%;
        background: $accent-darken-2;
        color: $text;
        padding: 0 1;
    }

    MessageBubble.message-assistant {
        align: left top;
        margin-top: 1;
    }

    MessageBubble.message-assistant > Markdown {
        width: 100%;
        max-width: 100%;
        margin: 0;
    }

    MessageBubble.message-system {
        align: left top;
    }

    MessageBubble.message-system > Static.content {
        width: auto;
        color: $text-muted;
        text-style: italic;
    }

    MessageBubble.message-tool {
        align: left top;
    }

    MessageBubble.message-tool > Static.content {
        width: auto;
        color: $text-muted;
        text-style: dim;
    }

    .role-label {
        width: auto;
        text-style: bold;
        margin: 0;
        height: 1;
    }

    .role-label.user {
        color: $success;
    }

    .role-label.assistant {
        color: $accent;
    }

    .role-label.system {
        color: $text-muted;
    }

    .role-label.tool {
        color: $warning;
    }
    """

    def __init__(
        self,
        role: str,
        content: str,
        *,
        is_streamable: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._role = role
        self._content = content
        self._is_streamable = is_streamable
        self._markdown: Markdown | None = None
        self._last_update: float = 0.0
        self._pending_update: bool = False
        self._update_threshold_chars = 100
        self._update_threshold_secs = 0.2
        self._thinking_extracted = False
        self._has_placeholder = False

        self.add_class(f"message-{role}")

    def compose(self):
        role_names = {"user": "You", "assistant": "Agent", "system": "System", "tool": "Tool"}
        label = role_names.get(self._role, self._role)
        yield RoleIndicator(f"[{label}]", classes=f"role-label {self._role}")

        if self._role == "assistant" and self._is_streamable:
            if not self._content:
                self._has_placeholder = True
                md = Markdown("*Thinking...*")
            else:
                md = Markdown(self._content)
            self._markdown = md
            yield md
        elif self._role == "assistant":
            md = Markdown(self._content)
            self._markdown = md
            yield md
        elif self._role == "user":
            yield Static(Text(self._content), classes="content")
        elif self._role == "tool":
            yield Static(Text(self._content, style="dim"), classes="content")
        else:
            yield Static(Text(self._content, style="italic dim"), classes="content")

    def on_mount(self) -> None:
        """Start thinking animation if placeholder is active."""
        if self._has_placeholder:
            self._animate_placeholder()

    def _animate_placeholder(self) -> None:
        """Cycle the thinking placeholder dots."""
        if not self._has_placeholder or self._markdown is None:
            return
        dots = "." * ((int(time.monotonic() * 2) % 3) + 1)
        self._markdown.update(f"*Thinking{dots}*")
        self.set_timer(0.5, self._animate_placeholder)

    def update_content(self, delta: str) -> None:
        """Append a text delta and throttle-update the Markdown widget."""
        if not self._is_streamable or self._markdown is None:
            return
        if self._has_placeholder:
            if not delta:
                return  # keep "Thinking..." animation alive during empty deltas
            self._has_placeholder = False
            self._content = delta
        else:
            self._content += delta
        self._check_thinking_blocks()
        now = time.monotonic()
        chars_since = len(self._content)
        time_since = now - self._last_update

        if time_since >= self._update_threshold_secs or chars_since >= self._update_threshold_chars:
            self._do_update()
        elif not self._pending_update:
            self._pending_update = True
            delay = max(0.01, self._update_threshold_secs - time_since)
            self.set_timer(delay, self._flush_update)

    def _flush_update(self) -> None:
        if self._pending_update:
            self._pending_update = False
            self._do_update()

    def _do_update(self) -> None:
        if self._markdown is not None:
            self._markdown.update(self._content)
            self._last_update = time.monotonic()

    def set_final_content(self, content: str) -> None:
        """Replace content with the authoritative final text."""
        self._content = content
        self._check_thinking_blocks()
        if self._markdown is not None:
            self._markdown.update(self._content)
        else:
            for child in self.query("Static.content"):
                child.update(Text(self._content))

    def _check_thinking_blocks(self) -> None:
        """Detect and extract thinking blocks from accumulated content."""
        if not self._is_streamable or self._thinking_extracted:
            return
        for pattern in THINKING_PATTERNS:
            match = pattern.search(self._content)
            if match:
                thinking_text = match.group(1).strip()
                self._content = (self._content[: match.start()] + self._content[match.end() :]).strip()
                thinking_widget = ThinkingBlockWidget(thinking_text)
                if self._markdown is not None:
                    self.mount(thinking_widget, before=self._markdown)
                else:
                    self.mount(thinking_widget)
                self._thinking_extracted = True
                break
