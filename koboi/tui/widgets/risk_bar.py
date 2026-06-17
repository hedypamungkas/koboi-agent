"""risk_bar.py -- Visual risk level indicator widget."""
from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class RiskBar(Static):
    """Visual risk level indicator.

    Renders as a progress-bar-style display:
        ████░░░░░░ MODERATE
    """

    risk_level: reactive[str] = reactive("safe")
    bar_width: reactive[int] = reactive(10)

    RISK_COLORS = {
        "safe": "green",
        "moderate": "yellow",
        "destructive": "red",
    }

    RISK_FILL = {
        "safe": 3,
        "moderate": 6,
        "destructive": 10,
    }

    def render(self) -> str:
        level = self.risk_level.lower()
        fill = self.RISK_FILL.get(level, 0)
        width = self.bar_width
        filled = min(fill, width)
        empty = width - filled
        color = self.RISK_COLORS.get(level, "white")

        bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
        label = f"[{color} bold]{level.upper()}[/{color} bold]"
        return f"{bar} {label}"
