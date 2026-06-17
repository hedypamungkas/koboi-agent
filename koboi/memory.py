"""koboi/memory.py -- Conversation memory with pluggable backends."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for memory backends. Both sync and async implementations satisfy this."""

    def add_user_message(self, content: str | list) -> None: ...
    def add_assistant_message(self, content: str | None, tool_calls: list[dict] | None = None) -> None: ...
    def add_tool_result(self, tool_call_id: str, content: str) -> None: ...
    def add_context_message(self, content: str, label: str = "") -> None: ...
    def get_messages(self) -> list[dict]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...


class ConversationMemory:
    def __init__(self, logger: AgentLogger | None = None, system_prompt: str | None = None):
        self._system_prompt = system_prompt
        self._messages: list[dict] = []
        self._logger = logger

    def _snapshot(self, trigger: str) -> None:
        if self._logger:
            self._logger.log_memory_snapshot(self.get_messages(), trigger)

    def add_user_message(self, content: str | list) -> None:
        self._messages.append({"role": "user", "content": content})
        self._snapshot("add_user_message")

    def add_assistant_message(self, content: str | None, tool_calls: list[dict] | None = None) -> None:
        msg: dict = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
            msg["content"] = content or ""
        self._messages.append(msg)
        self._snapshot("add_assistant_message")

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        self._snapshot("add_tool_result")

    def add_context_message(self, content: str, label: str = "") -> None:
        self._messages.append({"role": "system", "content": content})
        self._snapshot(f"add_context_message ({label})")

    def get_messages(self) -> list[dict]:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._messages)
        return messages

    def replace_messages(self, messages: list[dict]) -> None:
        """Replace all messages with a new list (used by context compaction)."""
        self._messages.clear()
        self._messages.extend(messages)
        self._snapshot("replace_messages")

    def clear(self) -> None:
        self._messages.clear()
        self._snapshot("clear")

    def __len__(self) -> int:
        return len(self._messages)
