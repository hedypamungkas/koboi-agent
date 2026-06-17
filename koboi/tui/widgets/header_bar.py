"""HeaderBar -- top bar showing agent name, mode, model."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

_MODE_COLORS = {
    "CHAT": "green",
    "PLAN": "blue",
    "ACT": "yellow",
    "AUTO": "cyan",
}

_MODE_BADGES = {
    "CHAT": "CHAT",
    "PLAN": "PLAN | READ-ONLY",
    "ACT": "ACT",
    "AUTO": "AUTO | graduated trust",
}


class HeaderBar(Static):
    """A horizontal header bar displaying agent metadata."""

    agent_name: reactive[str] = reactive("koboi-agent")
    model: reactive[str] = reactive("")
    mode: reactive[str] = reactive("CHAT")

    def __init__(
        self,
        agent_name: str = "koboi-agent",
        model: str = "",
        mode: str = "CHAT",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.model = model
        self.mode = mode

    def render(self) -> str:
        mode_upper = self.mode.upper()
        color = _MODE_COLORS.get(mode_upper, "white")
        badge = _MODE_BADGES.get(mode_upper, mode_upper)

        parts = [f"  {self.agent_name}"]
        parts.append(f"  [{color}]{badge}[/{color}]")
        if self.model:
            parts.append(f"  {self.model}")
        return " | ".join(parts)
