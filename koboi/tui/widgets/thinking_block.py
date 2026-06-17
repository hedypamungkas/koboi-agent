"""ThinkingBlockWidget -- collapsible thinking/reasoning block."""
from __future__ import annotations

import re

from textual.reactive import reactive
from textual.widgets import Static
from textual.widget import Widget

THINKING_PATTERNS = [
    re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL),
    re.compile(r"<think>(.*?)</think>", re.DOTALL),
]


class ThinkingBlockWidget(Widget):
    """A collapsible widget that displays LLM thinking/reasoning content."""

    DEFAULT_CSS = """
    ThinkingBlockWidget {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    ThinkingBlockWidget .thinking-header {
        width: 100%;
        padding: 0 2;
        color: $text-muted;
        text-style: italic dim;
    }

    ThinkingBlockWidget .thinking-body {
        width: 100%;
        padding: 0 2 0 4;
        color: $text-muted;
        text-style: dim italic;
        max-height: 16;
        overflow-y: auto;
        display: none;
    }

    ThinkingBlockWidget.expanded .thinking-body {
        display: block;
    }
    """

    collapsed: reactive[bool] = reactive(True)

    def __init__(self, thinking_text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._thinking_text = thinking_text

    def compose(self):
        yield Static("v Thinking...", classes="thinking-header")
        yield Static(self._thinking_text, classes="thinking-body")

    def watch_collapsed(self, collapsed: bool) -> None:
        self.set_class(not collapsed, "expanded")
        try:
            header = self.query_one(".thinking-header")
        except Exception:
            return
        if collapsed:
            header.update("v Thinking...")
        else:
            header.update("^ Thinking")

    def toggle(self) -> None:
        self.collapsed = not self.collapsed
