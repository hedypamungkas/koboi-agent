"""Tests for koboi.llm.factory module."""

from __future__ import annotations

import asyncio
import json

import pytest

from koboi.llm.anthropic_adapter import AnthropicAdapter
from koboi.llm.base import LLMClient, LLMInvalidRequestError
from koboi.llm.factory import create_client
from koboi.llm.openai_adapter import OpenAIAdapter


class TestCreateClient:
    def test_openai_provider_returns_openai_adapter(self):
        client = create_client(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert isinstance(client, OpenAIAdapter)
        assert isinstance(client, LLMClient)

    def test_empty_provider_defaults_to_openai(self):
        client = create_client(
            provider="",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert isinstance(client, OpenAIAdapter)

    def test_openai_compatible_alias(self):
        client = create_client(
            provider="openai_compatible",
            model="local-model",
            api_key="key",
            base_url="http://localhost:8080/v1",
        )
        assert isinstance(client, OpenAIAdapter)

    def test_anthropic_provider_returns_anthropic_adapter(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
        )
        assert isinstance(client, AnthropicAdapter)
        assert isinstance(client, LLMClient)

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMInvalidRequestError, match="Unknown provider"):
            create_client(provider="gemini", model="x", api_key="k")

    def test_anthropic_default_base_url(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
        )
        assert "anthropic.com" in client._transport._base_url

    def test_anthropic_custom_base_url(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            base_url="https://custom-proxy.example.com/v1",
        )
        assert client._transport._base_url == "https://custom-proxy.example.com/v1"

    def test_anthropic_max_tokens_forwarded(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            max_tokens=8192,
        )
        assert client._max_tokens == 8192

    def test_anthropic_auth_token_forwarded(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            auth_token="oauth-token-here",
        )
        transport = client._transport
        headers = transport._auth.apply({})
        assert "x-api-key" in headers
        assert "Authorization" in headers

    def test_anthropic_x_api_key_only(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-key-only",
        )
        headers = client._transport._auth.apply({})
        assert headers.get("x-api-key") == "sk-ant-key-only"
        assert "anthropic-version" in headers

    def test_anthropic_version_header_always_present(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
        )
        headers = client._transport._auth.apply({})
        assert headers["anthropic-version"] == "2023-06-01"

    def test_timeout_forwarded(self):
        client = create_client(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            timeout=60.0,
        )
        assert client._transport._timeout == 60.0


class _SpyTransport:
    """Fake HTTP transport: records the request body, returns canned data.

    Stands in for HttpTransport so we can assert exactly which keys reach the
    provider request body (the real egress point) without network or API keys.
    """

    base_url = "https://example.test/v1"

    def __init__(self, canned: dict):
        self.canned = canned
        self.body: dict | None = None

    async def post(self, path: str, body: dict) -> dict:
        self.body = body
        return self.canned

    async def post_stream(self, path: str, body: dict):
        self.body = json.loads(json.dumps(body))
        yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'.encode()
        yield b"data: [DONE]"

    async def close(self) -> None:
        pass


_OPENAI_CANNED = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
_ANTHROPIC_CANNED = {"content": [{"type": "text", "text": "ok"}], "usage": {}}


class TestLLMParamForwarding:
    """Body-level proof that configured LLM params reach the provider request."""

    def test_openai_max_tokens_reaches_body(self):
        spy = _SpyTransport(_OPENAI_CANNED)
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=spy, max_tokens=8192)
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}]))
        assert spy.body["max_tokens"] == 8192

    def test_openai_max_tokens_omitted_when_unset(self):
        spy = _SpyTransport(_OPENAI_CANNED)
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=spy)
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}]))
        assert "max_tokens" not in spy.body

    def test_openai_extra_params_reach_body(self):
        spy = _SpyTransport(_OPENAI_CANNED)
        adapter = OpenAIAdapter(
            model="gpt-4o-mini",
            transport=spy,
            extra_params={
                "top_p": 0.1,
                "stop": ["\n"],
                "seed": 42,
                "response_format": {"type": "json_object"},
                "reasoning_effort": "high",
            },
        )
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}]))
        assert spy.body["top_p"] == 0.1
        assert spy.body["stop"] == ["\n"]
        assert spy.body["seed"] == 42
        assert spy.body["response_format"] == {"type": "json_object"}
        assert spy.body["reasoning_effort"] == "high"

    def test_openai_max_completion_tokens_suppresses_max_tokens(self):
        # OpenAI o-series rejects max_tokens alongside max_completion_tokens.
        spy = _SpyTransport(_OPENAI_CANNED)
        adapter = OpenAIAdapter(
            model="o3-mini",
            transport=spy,
            max_tokens=8000,
            extra_params={"max_completion_tokens": 6000, "reasoning_effort": "high"},
        )
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}]))
        assert spy.body["max_completion_tokens"] == 6000
        assert "max_tokens" not in spy.body

    def test_anthropic_extra_params_reach_body(self):
        spy = _SpyTransport(_ANTHROPIC_CANNED)
        adapter = AnthropicAdapter(
            model="claude-sonnet-4-20250514",
            transport=spy,
            extra_params={
                "top_p": 0.1,
                "top_k": 40,
                "thinking": {"type": "enabled", "budget_tokens": 2000},
            },
        )
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}]))
        assert spy.body["top_p"] == 0.1
        assert spy.body["top_k"] == 40
        assert spy.body["thinking"] == {"type": "enabled", "budget_tokens": 2000}

    def test_anthropic_max_tokens_falls_back_to_default(self):
        # Anthropic's API requires max_tokens; unset -> adapter uses 4096.
        spy = _SpyTransport(_ANTHROPIC_CANNED)
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=spy)
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}]))
        assert spy.body["max_tokens"] == 4096

    def test_build_client_applies_llm_overrides(self):
        # Per-agent overrides (orchestration llm_config) merge over the top-level
        # llm: block via _build_client(llm_overrides=...).
        from koboi.config import Config
        from koboi.facade import _build_client

        config = Config.from_dict(
            {"agent": {"name": "t"}, "llm": {"provider": "openai", "model": "m", "api_key": "k", "temperature": 0.9}}
        )
        base = _build_client(config, logger=None)
        assert base._impl._temperature == 0.9
        assert base._impl._max_tokens is None

        overridden = _build_client(config, logger=None, llm_overrides={"temperature": 0.1, "max_tokens": 1234})
        assert overridden._impl._temperature == 0.1
        assert overridden._impl._max_tokens == 1234

    def test_cloudflare_forwards_params_reach_body(self):
        # Cloudflare inherits the OpenAI adapter via _create_openai (registry.py);
        # prove its body actually carries max_tokens + extra params (not just by
        # construction).
        cf = create_client(
            provider="cloudflare",
            model="@cf/meta/llama-3.1-70b-instruct",
            api_key="k",
            max_tokens=512,
            extra_params={"top_p": 0.1},
        )
        captured: dict = {}

        async def spy(path: str, body: dict) -> dict:
            captured["body"] = json.loads(json.dumps(body))
            return _OPENAI_CANNED

        cf._transport.post = spy  # create_client returns the OpenAIAdapter directly
        asyncio.run(cf.complete([{"role": "user", "content": "hi"}]))
        assert captured["body"]["max_tokens"] == 512
        assert captured["body"]["top_p"] == 0.1

    def test_openai_params_reach_streamed_body(self):
        # The SSE server is streaming-only; prove complete_stream forwards
        # max_tokens + extra params into the streamed request body.
        spy = _SpyTransport(_OPENAI_CANNED)
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=spy, max_tokens=8192, extra_params={"top_p": 0.1})

        async def drain() -> None:
            async for _ev in adapter.complete_stream([{"role": "user", "content": "hi"}]):
                pass

        asyncio.run(drain())
        assert spy.body["max_tokens"] == 8192
        assert spy.body["top_p"] == 0.1

    def test_build_client_provider_switch_uses_override_key(self):
        # An agent that switches provider must use its OWN key, not inherit the
        # parent's wrong-provider key (which would cause an opaque 401).
        from koboi.config import Config
        from koboi.facade import _build_client
        from koboi.llm.anthropic_adapter import AnthropicAdapter

        config = Config.from_dict(
            {"agent": {"name": "t"}, "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-openai-X"}}
        )
        client = _build_client(
            config,
            logger=None,
            llm_overrides={"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "sk-ant-Y"},
        )
        assert isinstance(client._impl, AnthropicAdapter)
        headers = client._impl._transport._auth.apply({})
        assert headers.get("x-api-key") == "sk-ant-Y"  # NOT the inherited "sk-openai-X"

    def test_build_client_provider_switch_without_key_does_not_inherit_parent(self, monkeypatch):
        # Switching provider with no key must NOT silently reuse the parent key;
        # it raises a clear "key not configured for <new provider>" instead.
        from koboi.client import RetryClientError
        from koboi.config import Config
        from koboi.facade import _build_client

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        config = Config.from_dict(
            {"agent": {"name": "t"}, "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-openai-X"}}
        )
        with pytest.raises(RetryClientError, match="ANTHROPIC_API_KEY"):
            _build_client(
                config, logger=None, llm_overrides={"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
            )


class TestOrchestrationClientLifecycle:
    def test_close_closes_dedicated_per_agent_clients(self):
        # KoboiAgent.close() must close each agent's dedicated client (built when
        # llm_config has overrides), not just the shared orchestrator client.
        from unittest.mock import AsyncMock, MagicMock

        from koboi.facade import KoboiAgent

        shared = MagicMock()
        shared.close = AsyncMock()
        dedicated = MagicMock()
        dedicated.close = AsyncMock()

        agent_shared = MagicMock()
        agent_shared.client = shared
        agent_shared.memory.close = MagicMock()
        agent_dedicated = MagicMock()
        agent_dedicated.client = dedicated
        agent_dedicated.memory.close = MagicMock()

        orch = MagicMock()
        orch._agents_map = {"a": agent_shared, "b": agent_dedicated}
        orch.client = shared

        agent = KoboiAgent(orchestrator=orch)
        asyncio.run(agent.close())

        dedicated.close.assert_awaited_once()  # per-agent client closed
        shared.close.assert_awaited_once()  # shared closed exactly once (not double-closed)
