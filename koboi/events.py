"""koboi/events.py -- Sealed StreamEvent union type for streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    trace_id: str = ""
    # RunResult.metadata parity (rag_results, guardrail_outcomes) so streamed runs
    # are eval/observable. Empty for adapter-emitted CompleteEvents.
    metadata: dict = field(default_factory=dict)


@dataclass
class ErrorEvent:
    error: Exception
    code: str = "internal_error"
    retriable: bool = False


@dataclass
class PendingApprovalEvent:
    """Emitted when a tool call is awaiting human approval (HITL)."""

    approval_id: str
    tool_name: str
    arguments: str
    risk_level: str
    tool_call_id: str = ""
    reason: str = ""
    timeout_seconds: float = 120.0


@dataclass
class HandoverEvent:
    """Emitted when the agent yields the conversation to a human operator (B1).

    The agent's run ends cleanly (``AgentHandoverError`` propagated out, releasing
    ``pool.session_lock``); a human operator takes over via ``POST /transfer`` +
    a new ``/chat/stream`` on the same session. ``summary`` carries the warm-handoff
    digest so the operator sees a case card, not a raw transcript.
    """

    handover_id: str
    reason: str
    summary: str = ""
    tool_call_id: str = ""


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
    # W2: deep-research stamps research_sources / coverage / depth here; merged into
    # OrchestratorResult.metadata by run(). Other modes leave it empty.
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchEvent:
    """W2: a research node ran a web search."""

    query: str
    results_count: int


@dataclass
class FetchEvent:
    """W2: a research node fetched a URL."""

    url: str
    status: int
    chars: int


@dataclass
class SourceEvent:
    """W2: a finding was added to the research SourceStore (assigned a citation id)."""

    citation_id: int
    node_id: str
    preview: str


@dataclass
class MediaGeneratedEvent:
    """W3: media generation event."""

    modality: str
    prompt: str


@dataclass
class CoverageEvent:
    """W2: a coverage-evaluation round completed."""

    depth: int
    score: float
    gaps: list[str]


StreamEvent = (
    TextDeltaEvent
    | ToolCallEvent
    | ToolResultEvent
    | IterationEvent
    | CompleteEvent
    | ErrorEvent
    | PendingApprovalEvent
    | HandoverEvent
    | RoutingDecisionEvent
    | AgentDispatchEvent
    | AgentResultEvent
    | OrchestrationCompleteEvent
    | SearchEvent
    | FetchEvent
    | SourceEvent
    | CoverageEvent
    | MediaGeneratedEvent
)


_EVENT_TYPE_MAP: dict[type, str] = {
    TextDeltaEvent: "text_delta",
    ToolCallEvent: "tool_call",
    ToolResultEvent: "tool_result",
    IterationEvent: "iteration",
    CompleteEvent: "complete",
    ErrorEvent: "error",
    PendingApprovalEvent: "pending_approval",
    HandoverEvent: "handover",
    RoutingDecisionEvent: "routing_decision",
    AgentDispatchEvent: "agent_dispatch",
    AgentResultEvent: "agent_result",
    OrchestrationCompleteEvent: "orchestration_complete",
    SearchEvent: "search",
    FetchEvent: "fetch",
    SourceEvent: "source",
    CoverageEvent: "coverage",
    MediaGeneratedEvent: "media_generated",
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
        resp = event.response
        usage = None
        if resp and resp.usage:
            u = resp.usage
            usage = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "reasoning_tokens": getattr(u, "reasoning_tokens", 0),
                "total_tokens": u.total_tokens,
                # E2E/telemetry-friendly aliases (the names consumers ask for):
                "token_input": u.prompt_tokens,
                "token_output": u.completion_tokens,
                "token_reasoning": getattr(u, "reasoning_tokens", 0),
            }
        return {
            "type": event_type,
            "content": event.content,
            "elapsed_seconds": round(event.elapsed_seconds, 2),
            "iterations_used": event.iterations_used,
            "tools_used": event.tools_used,
            "token_usage": usage,
            "model_name": (resp.model if resp and resp.model else None),
            "url_provider": (resp.base_url if resp and resp.base_url else None),
            "trace_id": event.trace_id or None,
            "metadata": event.metadata,
        }
    if isinstance(event, ErrorEvent):
        return {
            "type": event_type,
            "error": str(event.error),
            "code": event.code,
            "retriable": event.retriable,
        }

    # Generic: asdict handles all remaining dataclass fields
    d = asdict(event)
    d["type"] = event_type
    for field_name in _ROUND_FIELDS:
        if field_name in d:
            d[field_name] = round(d[field_name], 2)
    return d
