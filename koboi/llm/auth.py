"""koboi/llm/auth.py -- Authentication strategy pattern for LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class AuthStrategy(ABC):
    @abstractmethod
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        ...


class BearerAuth(AuthStrategy):
    def __init__(self, token: str):
        self._token = token

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers["Authorization"] = f"Bearer {self._token}"
        return headers


class APIKeyHeaderAuth(AuthStrategy):
    def __init__(self, api_key: str, header_name: str = "x-api-key"):
        self._api_key = api_key
        self._header_name = header_name

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers[self._header_name] = self._api_key
        return headers


class StaticHeaderAuth(AuthStrategy):
    def __init__(self, name: str, value: str):
        self._name = name
        self._value = value

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers[self._name] = self._value
        return headers


class CompositeAuth(AuthStrategy):
    def __init__(self, strategies: list[AuthStrategy]):
        self._strategies = strategies

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        for strategy in self._strategies:
            headers = strategy.apply(headers)
        return headers
