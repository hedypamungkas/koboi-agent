"""help_overlay.py -- Rich help overlay modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


_HELP_MODES = """\
Modes:
  CHAT   -- Default. Agent reads and responds, no file changes.
  PLAN   -- Read-only. Agent analyzes and proposes a numbered plan.
  ACT    -- Agent executes with per-action permission prompts.
  AUTO   -- Agent executes with graduated trust (learns from approvals).
"""

_HELP_TIPS = """\
Tips:
  - Type / to see all commands
  - Use Shift+Enter for multi-line input
  - Use @ to autocomplete file paths
  - Tab expands/collapses tool calls (when not in input)
  - { / } jumps between user messages
"""

_HELP_VIM = """\
Vim Mode (/vim to toggle):
  Normal: h/j/k/l  w/b/e  0/$  ^  x/X  dd  yy  p  u
  Insert: i/a/A/I/o  Escape returns to normal
  History: j/k in normal mode, Up/Down in insert mode
"""


class HelpOverlayScreen(ModalScreen[None]):
    """Rich help overlay with commands, shortcuts, and modes."""

    CSS = """
    HelpOverlayScreen {
        background: rgba(0, 0, 0, 0.7);
    }
    #help-panel {
        width: 72;
        max-width: 85%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: tall $accent;
        padding: 2 3;
    }
    #help-panel .heading {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #help-panel .section-title {
        text-style: bold;
        color: $text;
        margin-top: 1;
    }
    #help-panel .hint {
        color: $text-muted;
        margin-top: 2;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(
        self,
        commands: list[str] | None = None,
        bindings: list | None = None,
        current_mode: str = "chat",
        keybinding_display: list[tuple[str, str]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._commands = commands or []
        self._bindings = bindings or []  # type: ignore[assignment]  # Textual BindingsMap accepts a list at runtime
        self._current_mode = current_mode
        self._keybinding_display = keybinding_display

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-panel"):
            yield Static("Keyboard Shortcuts & Commands", classes="heading")
            yield Static("")
            yield Static("Commands", classes="section-title")
            command_descriptions = {
                "/reset": "Clear conversation memory",
                "/info": "Show agent configuration",
                "/history": "Show conversation history",
                "/tools": "List registered tools",
                "/mode": "Switch or show mode (chat/plan/act/auto)",
                "/theme": "Cycle color theme",
                "/sessions": "Open session manager",
                "/fork": "Fork current conversation",
                "/export": "Export conversation (md/json/html)",
                "/skills": "List discovered skills",
                "/compact": "Manually compact context window",
                "/model": "Switch LLM model mid-session",
                "/editor": "Open $EDITOR for long messages",
                "/undo": "Revert last AI commit(s)",
                "/copy": "Copy last response to clipboard",
                "/vim": "Toggle vim input mode",
                "/diagnostics": "Export session diagnostics ZIP",
                "/help": "Show this help",
            }
            for cmd in self._commands:
                desc = command_descriptions.get(cmd, "")
                yield Static(f"  {cmd:<16} {desc}")

            yield Static("")
            yield Static("Keyboard Shortcuts", classes="section-title")
            if self._keybinding_display:
                for key, desc in self._keybinding_display:
                    yield Static(f"  {key:<16} {desc}")
            else:
                shortcut_lines = [
                    "  Ctrl+P          Command palette",
                    "  Ctrl+R          Search history",
                    "  Ctrl+S          Session manager",
                    "  Ctrl+O          Transcript viewer",
                    "  Ctrl+L          Clear chat",
                    "  Ctrl+T          Cycle theme",
                    "  Ctrl+C          Quit",
                    "  Shift+Tab       Cycle mode",
                    "  Tab             Toggle tool calls (when not in input)",
                    "  ?               This help overlay",
                    "  Escape          Focus input / close overlay",
                    "  Up/Down         Navigate command history",
                ]
                for line in shortcut_lines:
                    yield Static(line)

            yield Static("")
            yield Static(_HELP_MODES)
            yield Static(_HELP_TIPS)
            yield Static(_HELP_VIM)
            yield Static("Press Esc to close", classes="hint")
