"""koboi/eval/t/mock.py -- Mock LLM clients for deterministic, API-key-free eval tests."""

from __future__ import annotations

import json
from collections.abc import Callable

from koboi.llm.base import LLMClient
from koboi.types import AgentResponse, TokenUsage, ToolCall


class ScriptedClient(LLMClient):
    """Deterministic LLM client that serves scripted ``AgentResponse`` objects in order.

    Mirrors ``tests/conftest.py:MockClient`` so ``.eval.py`` tests run without an
    API key. Once the script is exhausted it returns a terminal empty response so
    the agent loop exits cleanly instead of hanging.
    """

    def __init__(self, responses: list[AgentResponse], *, model: str = "scripted"):
        self._responses = list(responses)
        self._index = 0
        self._model = model
        self.call_count = 0

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AgentResponse:
        self.call_count += 1
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return response
        return AgentResponse(content="No more scripted responses", tool_calls=[])

    async def get_embeddings(self, text: str) -> list[float] | None:
        return None


def scripted_response(
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> AgentResponse:
    """Build a scripted :class:`~koboi.types.AgentResponse` for ``MOCK_RESPONSES``."""
    return AgentResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def scripted_tool_call(name: str, arguments: dict | None = None) -> ToolCall:
    """Build a scripted :class:`~koboi.types.ToolCall` (arguments JSON-encoded)."""
    return ToolCall(id=f"tc_{name}", name=name, arguments=json.dumps(arguments or {}))


class DispatchingClient(LLMClient):
    """Content-matching mock client for orchestration evals (W6.1 -- NOT sequential).

    Unlike ``ScriptedClient`` (serves responses in order), ``DispatchingClient`` dispatches on the
    CONTENT of ``messages`` via a callable. This fits deep_research's dynamic LLM call sequence
    (plan -> N nodes -> coverage -> maybe re-plan -> synthesis) where the exact call count/order
    isn't known upfront. Used for ``--mock`` orchestration evals (CI-safe, no API key).
    """

    def __init__(self, dispatch: Callable[[list[dict]], AgentResponse], *, model: str = "dispatch-mock") -> None:
        self._dispatch = dispatch
        self._model = model
        self.call_count = 0

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AgentResponse:
        self.call_count += 1
        return self._dispatch(messages)

    async def get_embeddings(self, text: str) -> list[float] | None:
        return None


def deep_research_dispatch(
    *,
    node_answer: str = "Found: the topic is X and Y. Source: https://example.com/topic",
    coverage_score: float = 0.95,
    synthesis: str = "## Report\nThe topic is X [1] and Y [1].",
) -> Callable[[list[dict]], AgentResponse]:
    """Build a content-dispatching callable for mock deep_research evals (W6.1).

    Detects the LLM call phase (planner / coverage / synthesis / node) by message content +
    returns the right ``AgentResponse``. Mirrors the ``_FakeClient`` in
    ``tests/orchestration/test_deep_research.py``. Use with ``DispatchingClient`` + ``MOCK_CLIENT``.
    """

    def _dispatch(messages: list[dict]) -> AgentResponse:
        text = " ".join(m.get("content", "") for m in messages)
        if "research planner" in text:
            return scripted_response(
                content=json.dumps(
                    {
                        "needs_workflow": True,
                        "reason": "research",
                        "steps": [
                            {
                                "id": "research_topic",
                                "instruction": "Investigate the topic",
                                "depends_on": [],
                                "search_queries": ["topic overview"],
                            }
                        ],
                    }
                )
            )
        if "evaluating how thoroughly" in text:
            return scripted_response(
                content=json.dumps(
                    {
                        "overall_score": coverage_score,
                        "coverage": {"Investigate the topic": coverage_score},
                        "follow_up_queries": [],
                    }
                )
            )
        if "synthesizing a cited research report" in text:
            return scripted_response(content=synthesis)
        return scripted_response(content=node_answer)

    return _dispatch
