"""koboi/llm/openai_adapter.py -- OpenAI-compatible provider adapter (async)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from koboi.events import CompleteEvent, TextDeltaEvent, ToolCallEvent
from koboi.llm.base import LLMClient, LLMResponseParseError
from koboi.llm.http_transport import HttpTransport
from koboi.types import AgentResponse, ToolCall, TokenUsage

if TYPE_CHECKING:
    from koboi.logger import AgentLogger

_logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMClient):
    def __init__(
        self,
        model: str,
        transport: HttpTransport,
        embedding_model: str = "text-embedding-3-small",
        logger: AgentLogger | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_params: dict | None = None,
    ):
        self._model = model
        self._transport = transport
        self._embedding_model = embedding_model
        self._logger = logger
        self._temperature = temperature
        # None = unset -> omitted from the body (don't force-cap at a default).
        self._max_tokens = max_tokens
        # Forward-as-is generation params (top_p/stop/seed/response_format/
        # reasoning_effort/thinking/max_completion_tokens/...), merged verbatim
        # into the request body by _apply_generation_params.
        self._extra_params = extra_params

    def _apply_generation_params(self, body: dict) -> None:
        """Inject max_tokens + forward-as-is extra params into the request body.

        Drops ``max_tokens`` when ``max_completion_tokens`` is forwarded --
        OpenAI o-series rejects the two together.
        """
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        if self._extra_params:
            body.update(self._extra_params)
            if "max_completion_tokens" in self._extra_params:
                body.pop("max_tokens", None)

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AgentResponse:
        body: dict = {"model": self._model, "messages": messages}
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        if self._logger:
            self._logger.log_llm_request(messages, tools)

        self._apply_generation_params(body)
        data = await self._transport.post("/chat/completions", body)

        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError) as e:
            raise LLMResponseParseError(f"Unexpected OpenAI response structure: {e}") from e

        content = msg.get("content")
        tool_calls = self._parse_tool_calls(msg.get("tool_calls"))
        usage = self._parse_usage(data.get("usage"))

        result = AgentResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=self._model,
            base_url=getattr(self._transport, "base_url", None),
        )

        if self._logger:
            self._logger.log_llm_response(result)

        return result

    async def complete_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[TextDeltaEvent | ToolCallEvent | CompleteEvent]:
        body: dict = {"model": self._model, "messages": messages, "stream": True}
        # Request usage in the final stream chunk — without this, OpenAI-compatible
        # gateways omit usage from streamed responses (token accounting goes null).
        body["stream_options"] = {"include_usage": True}
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        if self._logger:
            self._logger.log_llm_request(messages, tools)

        self._apply_generation_params(body)
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        usage = None

        async for raw_line in self._transport.post_stream("/chat/completions", body):
            line = raw_line.decode("utf-8", errors="replace").strip()

            if not line.startswith("data: "):
                continue

            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # Usage from final chunk (if present)
            usage_raw = chunk.get("usage")
            if usage_raw:
                usage = self._parse_usage(usage_raw)

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            # Text delta
            delta_content = delta.get("content")
            if delta_content:
                content_parts.append(delta_content)
                yield TextDeltaEvent(content=delta_content)

            # Tool call deltas
            delta_tool_calls = delta.get("tool_calls")
            if delta_tool_calls:
                for tc_delta in delta_tool_calls:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_delta.get("id", ""),
                            "name": "",
                            "arguments": "",
                        }
                    acc = tool_calls_acc[idx]
                    if tc_delta.get("id"):
                        acc["id"] = tc_delta["id"]
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        acc["name"] = fn["name"]
                    if fn.get("arguments"):
                        acc["arguments"] += fn["arguments"]

        # Assemble final response
        full_content = "".join(content_parts) or None
        parsed_tool_calls = []
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            parsed_tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
            )
            yield ToolCallEvent(
                tool_name=tc["name"],
                tool_call_id=tc["id"],
                arguments=tc["arguments"],
            )

        final = AgentResponse(
            content=full_content,
            tool_calls=parsed_tool_calls,
            usage=usage,
            model=self._model,
            base_url=getattr(self._transport, "base_url", None),
        )
        if self._logger:
            self._logger.log_llm_response(final)
        yield CompleteEvent(response=final)

    async def get_embeddings(self, text: str) -> list[float] | None:
        try:
            data = await self._transport.post(
                "/embeddings",
                {
                    "model": self._embedding_model,
                    "input": text,
                },
            )
            return data["data"][0]["embedding"]
        except Exception as e:
            if "405" in str(e):
                _logger.warning(
                    "Embedding endpoint not available (server returned 405). "
                    "SemanticRetriever will fall back to keyword retrieval. Error: %s",
                    e,
                )
            else:
                _logger.warning("Embedding request failed: %s", e)
            return None

    @staticmethod
    def _parse_tool_calls(raw: list | None) -> list[ToolCall]:
        if not raw:
            return []
        calls = []
        for tc in raw:
            func = tc.get("function", {})
            calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", "{}"),
                )
            )
        return calls

    @staticmethod
    def _parse_usage(raw: dict | None) -> TokenUsage | None:
        if not raw:
            return None
        # Reasoning tokens live under completion_tokens_details.reasoning_tokens
        # (OpenAI-compatible spec). Some gateways also surface a top-level alias.
        details = raw.get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens") or raw.get("reasoning_tokens") or 0
        return TokenUsage(
            prompt_tokens=raw.get("prompt_tokens", 0) or 0,
            completion_tokens=raw.get("completion_tokens", 0) or 0,
            reasoning_tokens=int(reasoning),
        )

    async def close(self) -> None:
        await self._transport.close()
