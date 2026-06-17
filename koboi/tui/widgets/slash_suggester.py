"""SlashSuggester -- inline autocomplete for slash commands."""
from __future__ import annotations

from textual.suggester import Suggester


class SlashSuggester(Suggester):
    """Suggests slash command names when input starts with '/'."""

    def __init__(self, command_names: list[str], **kwargs) -> None:
        super().__init__(use_cache=False, case_sensitive=True, **kwargs)
        self._commands = sorted(command_names)

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        lower = value.lower()
        for cmd in self._commands:
            if cmd.startswith(lower) and cmd != lower:
                return cmd
        return None
