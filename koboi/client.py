"""koboi/client.py -- Async LLM client facade with retry logic."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from koboi.llm.base import LLMClient, LLMConnectionError, LLMError, LLMRateLimitError, LLMServerError
from koboi.llm.factory import create_client
from koboi.types import AgentResponse

if TYPE_CHECKING:
    from koboi.events import StreamEvent
    from koboi.logger import AgentLogger

PLACEHOLDER_KEYS = {"", "your-api-key-here", "sk-your-api-key", "sk-xxx"}
_UNRESOLVED_PATTERN = re.compile(r"^\$\{")


class RetryClientError(LLMError):
    pass


ClientError = RetryClientError


_RETRYABLE_ERRORS = (LLMServerError, LLMRateLimitError)

# Streaming adds ``LLMConnectionError`` (raised by ``HttpTransport`` on
# httpx timeouts/connect failures) to the retryable set. A stalled upstream
# (no bytes streamed yet) is a transient failure we can safely retry; once a
# single event has been yielded we can no longer retry (can't resume a partial
# stream), so the ``yielded`` guard in ``complete_stream`` still raises then.
_STREAM_RETRYABLE_ERRORS = _RETRYABLE_ERRORS + (LLMConnectionError,)
_MAX_CLIENT_RETRIES = 3


class RetryClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        logger: AgentLogger | None = None,
        provider: str = "openai",
        timeout: float = 120.0,
        max_tokens: int | None = None,
        auth_token: str = "",
        auth_type: str = "api_key",
        max_retries: int = _MAX_CLIENT_RETRIES,
        retry_backoff_base: float = 2.0,
        temperature: float | None = None,
        extra_params: dict | None = None,
    ):
        from koboi.llm.registry import ProviderRegistry

        self.provider = provider
        self.auth_type = auth_type

        resolved = ProviderRegistry.resolve_env(
            provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            auth_token=auth_token,
        )
        self.api_key = resolved["api_key"]
        self.base_url = resolved["base_url"]
        self._model = resolved["model"]
        self.logger = logger

        if provider == "cloudflare" and not self.base_url:
            account_id = resolved.get("account_id", "")
            if account_id:
                self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

        self._raw_auth_token = resolved.get("auth_token", "")

        if self.auth_type == "oauth_token":
            self.api_key = ""

        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.temperature = temperature
        self._validate_config()

        self._impl = create_client(
            provider=provider,
            model=self._model,
            api_key=self.api_key,
            base_url=self.base_url,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
            auth_token=self._raw_auth_token,
            temperature=temperature,
            extra_params=extra_params,
        )

    @property
    def model(self) -> str:
        return self._model

    def _is_placeholder(self, value: str) -> bool:
        return not value or value in PLACEHOLDER_KEYS or bool(_UNRESOLVED_PATTERN.match(value))

    def _validate_config(self) -> None:
        if self.auth_type == "oauth_token":
            if self._is_placeholder(self._raw_auth_token):
                raise RetryClientError(
                    "OAuth token not configured. Set ANTHROPIC_AUTH_TOKEN in .env or auth_token in config YAML."
                )
            return
        if self._is_placeholder(self.api_key):
            from koboi.llm.registry import ProviderRegistry

            desc = ProviderRegistry.get(self.provider)
            env_key = desc.env_key_api if desc else "OPENAI_API_KEY"
            raise RetryClientError(f"API key not configured. Set {env_key} in .env or config YAML.")

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AgentResponse:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._impl.complete(messages, tools)
            except _RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_backoff_base**attempt
                    if isinstance(e, LLMRateLimitError) and e.retry_after:
                        wait = e.retry_after
                    await asyncio.sleep(wait)
                    continue
                raise
            except LLMError:
                raise
            except Exception as e:
                raise RetryClientError(f"Unexpected error: {e}") from e
        raise last_error

    async def complete_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            yielded = False
            try:
                async for event in self._impl.complete_stream(messages, tools):
                    yielded = True
                    yield event
                return  # stream completed successfully
            except _STREAM_RETRYABLE_ERRORS as e:
                last_error = e
                if yielded:
                    raise  # can't retry mid-stream
                if attempt < self.max_retries:
                    wait = self.retry_backoff_base**attempt
                    if isinstance(e, LLMRateLimitError) and e.retry_after:
                        wait = e.retry_after
                    await asyncio.sleep(wait)
                    continue
                raise
            except LLMError:
                raise
            except Exception as e:
                if yielded:
                    raise
                raise RetryClientError(f"Unexpected error: {e}") from e
        raise last_error

    async def get_embeddings(self, text: str) -> list[float] | None:
        return await self._impl.get_embeddings(text)

    async def close(self) -> None:
        await self._impl.close()


# A "client" is anything implementing the LLMClient interface. RetryClient is the
# default (single-provider, retrying) impl; ProviderPool (koboi/llm/pool.py) is a
# multi-provider impl. Widened from RetryClient so a pool flows through the
# orchestration/server/subagent paths unchanged.
Client = LLMClient
