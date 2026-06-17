"""Tests for koboi.llm.auth module."""

from __future__ import annotations

from koboi.llm.auth import (
    APIKeyHeaderAuth,
    BearerAuth,
    CompositeAuth,
    StaticHeaderAuth,
)


class TestBearerAuth:
    def test_adds_authorization_header(self):
        auth = BearerAuth("my-token")
        headers = auth.apply({})
        assert headers["Authorization"] == "Bearer my-token"

    def test_preserves_existing_headers(self):
        auth = BearerAuth("tok")
        headers = auth.apply({"Content-Type": "application/json"})
        assert headers["Authorization"] == "Bearer tok"
        assert headers["Content-Type"] == "application/json"

    def test_overrides_existing_auth(self):
        auth = BearerAuth("new")
        headers = auth.apply({"Authorization": "old"})
        assert headers["Authorization"] == "Bearer new"


class TestAPIKeyHeaderAuth:
    def test_default_header_name(self):
        auth = APIKeyHeaderAuth("sk-ant-key")
        headers = auth.apply({})
        assert headers["x-api-key"] == "sk-ant-key"

    def test_custom_header_name(self):
        auth = APIKeyHeaderAuth("value", header_name="X-Custom-Auth")
        headers = auth.apply({})
        assert headers["X-Custom-Auth"] == "value"


class TestStaticHeaderAuth:
    def test_adds_static_header(self):
        auth = StaticHeaderAuth("anthropic-version", "2023-06-01")
        headers = auth.apply({})
        assert headers["anthropic-version"] == "2023-06-01"


class TestCompositeAuth:
    def test_applies_multiple_strategies(self):
        auth = CompositeAuth(
            [
                APIKeyHeaderAuth("sk-ant-key"),
                StaticHeaderAuth("anthropic-version", "2023-06-01"),
            ]
        )
        headers = auth.apply({"Content-Type": "application/json"})
        assert headers["x-api-key"] == "sk-ant-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["Content-Type"] == "application/json"

    def test_empty_strategies(self):
        auth = CompositeAuth([])
        headers = auth.apply({"Accept": "json"})
        assert headers == {"Accept": "json"}

    def test_later_strategy_overrides(self):
        auth = CompositeAuth(
            [
                APIKeyHeaderAuth("key1"),
                APIKeyHeaderAuth("key2"),
            ]
        )
        headers = auth.apply({})
        assert headers["x-api-key"] == "key2"
