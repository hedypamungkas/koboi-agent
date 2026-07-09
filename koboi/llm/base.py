"""koboi/llm/base.py -- Abstract base class and error hierarchy for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from koboi.types import AgentResponse

if TYPE_CHECKING:
    from koboi.events import StreamEvent


class LLMError(Exception):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMAuthenticationError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMServerError(LLMError):
    pass


class LLMInvalidRequestError(LLMError):
    pass


class LLMResponseParseError(LLMError):
    pass


class LLMClient(ABC):
    @property
    def model(self) -> str:
        """Model identifier. Override in subclasses or return empty string."""
        return ""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AgentResponse: ...

    @abstractmethod
    async def get_embeddings(self, text: str) -> list[float] | None: ...

    async def complete_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Default streaming: fall back to non-streaming, yield as single chunk.

        ``response_format`` is a provider-agnostic JSON Schema dict; when set,
        the provider should enforce (or strongly encourage) a JSON response
        conforming to it. OpenAI uses native ``response_format``; Anthropic
        emulates it via a forced ``tool_use`` (see AnthropicAdapter).
        """
        from koboi.events import CompleteEvent, TextDeltaEvent

        response = await self.complete(messages, tools, response_format=response_format)
        if response.content:
            yield TextDeltaEvent(content=response.content)
        yield CompleteEvent(response=response)

    async def close(self) -> None:  # noqa: B027 - optional override hook; default no-op for stateless clients
        """Release underlying resources (HTTP transport, etc). Default no-op."""
