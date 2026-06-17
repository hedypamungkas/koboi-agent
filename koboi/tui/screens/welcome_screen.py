"""welcome_screen.py -- First-run welcome overlay."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class WelcomeScreen(ModalScreen[None]):
    """First-run welcome overlay with quick-start guide."""

    CSS = """
    WelcomeScreen {
        background: rgba(0, 0, 0, 0.7);
    }
    #welcome-panel {
        width: 70;
        max-width: 85%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: tall $accent;
        padding: 2 3;
    }
    #welcome-panel .title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #welcome-panel .section {
        margin: 1 0;
    }
    #welcome-panel .section-title {
        text-style: bold;
        color: $text;
    }
    #welcome-panel .shortcut {
        color: $accent;
    }
    #welcome-panel .footer {
        text-align: center;
        color: $text-muted;
        margin-top: 2;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("enter", "dismiss", "Close"),
    ]

    def __init__(self, agent_name: str = "", model: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._agent_name = agent_name
        self._model = model

    def compose(self) -> ComposeResult:
        with Vertical(id="welcome-panel"):
            yield Static("KOBOI AGENT", classes="title")
            if self._agent_name:
                yield Static(f"  Agent: {self._agent_name}  |  Model: {self._model}")
            yield Static("")
            yield Static("Getting Started", classes="section-title")
            yield Static("  Type a message to start chatting with the agent")
            yield Static("  Use /mode <name> to switch: Chat / Plan / Act / Auto")
            yield Static("  Type /help or press ? for full reference")
            yield Static("")
            yield Static("Key Shortcuts", classes="section-title")
            yield Static("  [shortcut]Ctrl+P[/shortcut]       Command palette")
            yield Static("  [shortcut]Ctrl+R[/shortcut]       Search history")
            yield Static("  [shortcut]Ctrl+S[/shortcut]       Session manager")
            yield Static("  [shortcut]Ctrl+O[/shortcut]       Transcript viewer")
            yield Static("  [shortcut]Shift+Tab[/shortcut]    Cycle mode")
            yield Static("  [shortcut]?[/shortcut]            Help overlay")
            yield Static("")
            yield Static("Input Tips", classes="section-title")
            yield Static("  Type / to see all commands")
            yield Static("  Use Shift+Enter for multi-line input")
            yield Static("  Use @ to autocomplete file paths")
            yield Static("")
            yield Static("Press Enter or Esc to start", classes="footer")
