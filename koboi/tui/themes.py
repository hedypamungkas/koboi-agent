"""themes.py -- Koboi Agent TUI theme definitions."""

from __future__ import annotations

from textual.app import App
from textual.theme import Theme

KOBOI_DARK = Theme(
    name="koboi-dark",
    primary="#7c3aed",
    secondary="#3b82f6",
    accent="#7c3aed",
    surface="#1e1b2e",
    background="#0f0d1a",
    warning="#f59e0b",
    success="#10b981",
    error="#ef4444",
    panel="#1e1b2e",
)

KOBOI_LIGHT = Theme(
    name="koboi-light",
    primary="#6d28d9",
    secondary="#2563eb",
    accent="#6d28d9",
    surface="#ffffff",
    background="#f8f7ff",
    warning="#d97706",
    success="#059669",
    error="#dc2626",
    panel="#f3f2ff",
)

THEMES: dict[str, Theme] = {
    "koboi-dark": KOBOI_DARK,
    "koboi-light": KOBOI_LIGHT,
}


def register_themes(app: App, default: str = "koboi-dark") -> None:
    """Register all Koboi themes with the app and set the default."""
    for theme in THEMES.values():
        app.register_theme(theme)
    app.theme = default if default in THEMES else "koboi-dark"
