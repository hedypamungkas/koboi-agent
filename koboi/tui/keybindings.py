"""keybindings.py -- Load and merge configurable keybindings from YAML config.

Default bindings are defined here. Users can override any binding via the
``keybindings:`` section in their agent YAML config.

Example YAML::

    keybindings:
      ctrl+p: command_palette
      f1: help_overlay
      ctrl+m: subagent_monitor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.binding import Binding

if TYPE_CHECKING:
    pass

# Default bindings: action_name -> (key, description, show_in_footer)
_DEFAULT_BINDINGS: dict[str, tuple[str, str, bool]] = {
    "command_palette": ("ctrl+p", "Command Palette", True),
    "history_search": ("ctrl+r", "Search History", True),
    "session_manager": ("ctrl+s", "Sessions", True),
    "transcript_viewer": ("ctrl+o", "Transcript", True),
    "clear_chat": ("ctrl+l", "Clear Chat", True),
    "cycle_theme": ("ctrl+t", "Toggle Theme", True),
    "cancel_or_quit": ("ctrl+c", "Quit", True),
    "focus_input": ("escape", "Focus Input", True),
    "toggle_all_tools": ("tab", "Toggle Tools", False),
    "cycle_mode": ("shift+tab", "Cycle Mode", True),
    "help_overlay": ("question_mark", "Help", True),
    "kill_subagents": ("ctrl+k", "Kill Subagents", False),
    "subagent_monitor": ("ctrl+m", "Sub-Agent Monitor", False),
}


def load_keybindings(config) -> list[Binding]:
    """Build the BINDINGS list from config with defaults as fallback.

    The YAML config uses key->action format::

        keybindings:
          f1: help_overlay
          ctrl+p: command_palette

    Args:
        config: The loaded Config object. Reads ``keybindings:`` section.

    Returns:
        List of ``Binding`` objects for the app.
    """
    try:
        raw_overrides = config.get("keybindings", default={}) or {}
        if not isinstance(raw_overrides, dict):
            raw_overrides = {}
    except Exception:
        raw_overrides = {}

    # Invert key->action to action->key for lookup
    action_to_key: dict[str, str] = {}
    for key, action in raw_overrides.items():
        if isinstance(action, str):
            action_to_key[action] = key

    bindings = []
    for action, (default_key, description, show) in _DEFAULT_BINDINGS.items():
        key = action_to_key.get(action, default_key)
        bindings.append(Binding(key, action, description, show=show))

    return bindings


def get_keybinding_display(config) -> list[tuple[str, str]]:
    """Return (key, description) pairs for display in help overlay.

    Args:
        config: The loaded Config object.

    Returns:
        List of (key_string, description) tuples.
    """
    try:
        raw_overrides = config.get("keybindings", default={}) or {}
        if not isinstance(raw_overrides, dict):
            raw_overrides = {}
    except Exception:
        raw_overrides = {}

    # Invert key->action to action->key
    action_to_key: dict[str, str] = {}
    for key, action in raw_overrides.items():
        if isinstance(action, str):
            action_to_key[action] = key

    result = []
    for action, (default_key, description, show) in _DEFAULT_BINDINGS.items():
        if not show:
            continue
        key = action_to_key.get(action, default_key)
        result.append((key, description))

    return result
