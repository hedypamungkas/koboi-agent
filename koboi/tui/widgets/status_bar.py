"""StatusBar -- bottom status line showing context, tokens, turns, mode."""
from textual.reactive import reactive
from textual.widgets import Static

_MODE_COLORS = {
    "chat": "green",
    "plan": "blue",
    "act": "yellow",
    "auto": "cyan",
}


class StatusBar(Static):
    """Status bar showing session metrics and mode state."""

    context_pct: reactive[float] = reactive(0.0)
    tokens_used: reactive[int] = reactive(0)
    max_tokens: reactive[int] = reactive(8000)
    turn_count: reactive[int] = reactive(0)
    current_tool: reactive[str] = reactive("")
    state: reactive[str] = reactive("idle")
    iteration: reactive[int] = reactive(0)
    mode: reactive[str] = reactive("chat")
    skill_count: reactive[int] = reactive(0)
    orchestration_agents: reactive[int] = reactive(0)
    orchestration_completed: reactive[int] = reactive(0)
    orchestration_current: reactive[str] = reactive("")
    task_summary: reactive[str] = reactive("")
    vim_enabled: reactive[bool] = reactive(False)
    vim_mode: reactive[str] = reactive("insert")

    def render(self) -> str:
        ctx_bar = self._progress_bar(self.context_pct)
        token_str = f"{self.tokens_used}/{self.max_tokens} tokens"
        turn_str = f"turn {self.turn_count}"

        parts = [f"  ctx: {ctx_bar} {self.context_pct:.0f}%", token_str, turn_str]
        if self.iteration > 0:
            parts.append(f"iter {self.iteration}")

        if self.state == "running_tool" and self.current_tool:
            parts.append(f"tool: {self.current_tool}")
        elif self.state == "streaming":
            parts.append("streaming...")
        elif self.state == "waiting_approval":
            parts.append("waiting for approval")
        elif self.state == "orchestrating":
            if self.orchestration_current:
                parts.append(f"orchestrating: {self.orchestration_current} ({self.orchestration_completed}/{self.orchestration_agents})")
            else:
                parts.append(f"orchestrating ({self.orchestration_completed}/{self.orchestration_agents})")

        if self.task_summary:
            parts.append(f"tasks: {self.task_summary}")

        # Mode indicator
        mode_lower = self.mode.lower()
        color = _MODE_COLORS.get(mode_lower, "white")
        parts.append(f"[{color}]{mode_lower.upper()}[/{color}]")

        # Vim mode indicator
        if self.vim_enabled:
            vim_color = "green" if self.vim_mode == "insert" else "yellow"
            parts.append(f"[{vim_color}]-- {self.vim_mode.upper()} --[/{vim_color}]")

        if self.skill_count > 0:
            parts.append(f"skills: {self.skill_count}")

        return " | ".join(parts)

    @staticmethod
    def _progress_bar(pct: float, width: int = 10) -> str:
        filled = int(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)
