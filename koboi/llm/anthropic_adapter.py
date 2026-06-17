"""koboi/llm/anthropic_adapter.py -- Anthropic Messages API provider adapter (async)."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from koboi.events import CompleteEvent, TextDeltaEvent, ToolCallEvent
from koboi.llm.base import LLMClient
from koboi.llm.http_transport import HttpTransport
from koboi.types import AgentResponse, ToolCall, TokenUsage

if TYPE_CHECKING:
    from koboi.logger import AgentLogger

_DEFAULT_MAX_TOKENS = 4096


class AnthropicAdapter(LLMClient):
    def __init__(
        self,
        model: str,
        transport: HttpTransport,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        logger: AgentLogger | None = None,
        temperature: float | None = None,
    ):
        self._model = model
        self._transport = transport
        self._max_tokens = max_tokens
        self._logger = logger
        self._temperature = temperature

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AgentResponse:
        system_content, remaining = self._extract_system(messages)
        translated = self._translate_messages(remaining)

        body: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": translated,
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if system_content:
            body["system"] = system_content
        if tools:
            body["tools"] = self._translate_tools(tools)
            body["tool_choice"] = {"type": "auto"}

        if self._logger:
            self._logger.log_llm_request(messages, tools)

        data = await self._transport.post("/messages", body)
        result = self._parse_response(data)

        if self._logger:
            self._logger.log_llm_response(result)

        return result

    async def complete_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[TextDeltaEvent | ToolCallEvent | CompleteEvent]:
        system_content, remaining = self._extract_system(messages)
        translated = self._translate_messages(remaining)

        body: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": translated,
            "stream": True,
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if system_content:
            body["system"] = system_content
        if tools:
            body["tools"] = self._translate_tools(tools)
            body["tool_choice"] = {"type": "auto"}

        if self._logger:
            self._logger.log_llm_request(messages, tools)

        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        usage_input = 0
        usage_output = 0

        async for raw_line in self._transport.post_stream("/messages", body):
            line = raw_line.decode("utf-8", errors="replace").strip()

            if line.startswith("event: "):
                continue
            if not line.startswith("data: "):
                continue

            payload = line[len("data: "):]
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            event_type = chunk.get("type", "")

            if event_type == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        content_parts.append(text)
                        yield TextDeltaEvent(content=text)
                elif delta.get("type") == "input_json_delta":
                    idx = chunk.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    partial = delta.get("partial_json", "")
                    tool_calls_acc[idx]["arguments"] += partial

            elif event_type == "content_block_start":
                block = chunk.get("content_block", {})
                if block.get("type") == "tool_use":
                    idx = chunk.get("index", 0)
                    tool_calls_acc[idx] = {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": "",
                    }

            elif event_type == "message_delta":
                delta = chunk.get("delta", {})
                if delta.get("stop_reason"):
                    pass
                usage_chunk = chunk.get("usage", {})
                usage_output += usage_chunk.get("output_tokens", 0)

            elif event_type == "message_start":
                msg = chunk.get("message", {})
                usage_raw = msg.get("usage", {})
                usage_input = usage_raw.get("input_tokens", 0)

            elif event_type == "message_stop":
                break

        full_content = "".join(content_parts) or None
        parsed_tool_calls = []
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            parsed_tool_calls.append(ToolCall(
                id=tc["id"], name=tc["name"], arguments=tc["arguments"],
            ))
            yield ToolCallEvent(
                tool_name=tc["name"],
                tool_call_id=tc["id"],
                arguments=tc["arguments"],
            )

        final = AgentResponse(
            content=full_content,
            tool_calls=parsed_tool_calls,
            usage=TokenUsage(prompt_tokens=usage_input, completion_tokens=usage_output),
        )
        if self._logger:
            self._logger.log_llm_response(final)
        yield CompleteEvent(response=final)

    async def get_embeddings(self, text: str) -> list[float] | None:
        return None

    @staticmethod
    def _extract_system(messages: list[dict]) -> tuple[str, list[dict]]:
        parts: list[str] = []
        remaining: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    parts.append(content)
            else:
                remaining.append(msg)
        return "\n\n".join(parts), remaining

    def _translate_messages(self, messages: list[dict]) -> list[dict]:
        result: list[dict] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            if role == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = self._translate_user_content(content)
                result.append({"role": "user", "content": content})
            elif role == "assistant":
                result.append(self._translate_assistant(msg))
            elif role == "tool":
                tool_results, consumed = self._collect_tool_results(messages, i)
                result.append({"role": "user", "content": tool_results})
                i += consumed
                continue

            i += 1

        return self._ensure_alternating_roles(result)

    @staticmethod
    def _translate_assistant(msg: dict) -> dict:
        content: list = []
        text = msg.get("content")
        if text:
            content.append({"type": "text", "text": text})

        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                args = {}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "input": args,
            })

        if not content:
            content.append({"type": "text", "text": ""})
        return {"role": "assistant", "content": content}

    @staticmethod
    def _translate_user_content(blocks: list) -> list:
        """Convert OpenAI-format content blocks to Anthropic format.

        Translates image_url blocks to Anthropic's image source format.
        Text blocks pass through unchanged.
        """
        result = []
        for block in blocks:
            if not isinstance(block, dict):
                result.append({"type": "text", "text": str(block)})
                continue
            btype = block.get("type", "text")
            if btype == "text":
                result.append(block)
            elif btype == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Parse data:image/png;base64,<data>
                    try:
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                    except (ValueError, IndexError):
                        media_type = "image/png"
                        data = url
                    result.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    })
                else:
                    result.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
            else:
                result.append(block)
        return result

    @staticmethod
    def _collect_tool_results(messages: list[dict], start: int) -> tuple[list[dict], int]:
        blocks: list[dict] = []
        consumed = 0
        for j in range(start, len(messages)):
            if messages[j].get("role") != "tool":
                break
            blocks.append({
                "type": "tool_result",
                "tool_use_id": messages[j].get("tool_call_id", ""),
                "content": messages[j].get("content", ""),
            })
            consumed += 1
        return blocks, consumed

    @staticmethod
    def _ensure_alternating_roles(messages: list[dict]) -> list[dict]:
        if not messages:
            return messages

        merged: list[dict] = [messages[0]]
        for msg in messages[1:]:
            prev = merged[-1]
            if msg["role"] == prev["role"]:
                prev_content = prev["content"]
                curr_content = msg["content"]

                if isinstance(prev_content, list) and isinstance(curr_content, list):
                    prev_content.extend(curr_content)
                elif isinstance(prev_content, str) and isinstance(curr_content, str):
                    prev["content"] = prev_content + "\n" + curr_content
                else:
                    prev_parts = prev_content if isinstance(prev_content, list) else [{"type": "text", "text": prev_content}]
                    curr_parts = curr_content if isinstance(curr_content, list) else [{"type": "text", "text": curr_content}]
                    prev["content"] = prev_parts + curr_parts
            else:
                merged.append(msg)

        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "."})

        return merged

    @staticmethod
    def _translate_tools(tools: list[dict]) -> list[dict]:
        result = []
        for tool in tools:
            func = tool.get("function", {})
            result.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    @staticmethod
    def _parse_response(data: dict) -> AgentResponse:
        content_blocks = data.get("content", [])

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=json.dumps(block.get("input", {})),
                ))

        content = "\n".join(text_parts) if text_parts else None

        usage_raw = data.get("usage", {})
        usage = None
        if usage_raw:
            usage = TokenUsage(
                prompt_tokens=usage_raw.get("input_tokens", 0),
                completion_tokens=usage_raw.get("output_tokens", 0),
            )

        return AgentResponse(content=content, tool_calls=tool_calls, usage=usage)

    async def close(self) -> None:
        await self._transport.close()
