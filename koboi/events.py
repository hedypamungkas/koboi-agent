"""koboi/events.py -- Sealed StreamEvent union type for streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from koboi.types import AgentResponse


@dataclass
class TextDeltaEvent:
    content: str


@dataclass
class ToolCallEvent:
    tool_name: str
    tool_call_id: str
    arguments: str


@dataclass
class ToolResultEvent:
    tool_name: str
    tool_call_id: str
    result: str


@dataclass
class IterationEvent:
    iteration: int
    messages_count: int = 0
    tokens_estimated: int = 0


@dataclass
class CompleteEvent:
    response: AgentResponse | None = None
    content: str = ""
    elapsed_seconds: float = 0.0
    iterations_used: int = 0
    tools_used: list[str] = field(default_factory=list)


@dataclass
class ErrorEvent:
    error: Exception


@dataclass
class RoutingDecisionEvent:
    """Emitted when the router selects agents for a query."""

    agents: list[str]
    confidence: float
    method: str
    reasoning: str
    domain_label: str | None = None


@dataclass
class AgentDispatchEvent:
    """Emitted when a sub-agent is about to execute."""

    agent_name: str
    agent_index: int
    total_agents: int
    mode: str


@dataclass
class AgentResultEvent:
    """Emitted when a sub-agent finishes execution."""

    agent_name: str
    answer: str
    elapsed_seconds: float
    tokens_used: int
    is_dynamic: bool = False
    domain_label: str | None = None
    failed: bool = False


@dataclass
class OrchestrationCompleteEvent:
    """Emitted when the orchestrator finishes."""

    final_answer: str
    elapsed_seconds: float
    agent_results: list
    execution_mode: str
    routing_agents: list[str]
    routing_confidence: float


StreamEvent = Union[
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    IterationEvent,
    CompleteEvent,
    ErrorEvent,
    RoutingDecisionEvent,
    AgentDispatchEvent,
    AgentResultEvent,
    OrchestrationCompleteEvent,
]


_EVENT_TYPE_MAP: dict[type, str] = {
    TextDeltaEvent: "text_delta",
    ToolCallEvent: "tool_call",
    ToolResultEvent: "tool_result",
    IterationEvent: "iteration",
    CompleteEvent: "complete",
    ErrorEvent: "error",
    RoutingDecisionEvent: "routing_decision",
    AgentDispatchEvent: "agent_dispatch",
    AgentResultEvent: "agent_result",
    OrchestrationCompleteEvent: "orchestration_complete",
}

# Fields that need rounding to 2 decimal places
_ROUND_FIELDS = {"elapsed_seconds"}


def event_to_dict(event: StreamEvent) -> dict:
    """Convert a StreamEvent to a JSON-serializable dict."""
    from dataclasses import asdict

    event_type = _EVENT_TYPE_MAP.get(type(event))
    if event_type is None:
        return {"type": "unknown", "data": str(event)}

    # Special cases with non-serializable fields
    if isinstance(event, CompleteEvent):
        usage = None
        if event.response and event.response.usage:
            usage = {
                "prompt_tokens": event.response.usage.prompt_tokens,
                "completion_tokens": event.response.usage.completion_tokens,
                "total_tokens": event.response.usage.total_tokens,
            }
        return {
            "type": event_type,
            "content": event.content,
            "elapsed_seconds": round(event.elapsed_seconds, 2),
            "iterations_used": event.iterations_used,
            "tools_used": event.tools_used,
            "token_usage": usage,
        }
    if isinstance(event, ErrorEvent):
        return {"type": event_type, "error": str(event.error)}

    # Generic: asdict handles all remaining dataclass fields
    d = asdict(event)
    d["type"] = event_type
    for field_name in _ROUND_FIELDS:
        if field_name in d:
            d[field_name] = round(d[field_name], 2)
    return d
