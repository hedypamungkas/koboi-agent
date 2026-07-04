"""Tests for events.py -- event_to_dict and orchestration events."""

from __future__ import annotations

from koboi.events import (
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    IterationEvent,
    CompleteEvent,
    ErrorEvent,
    PendingApprovalEvent,
    RoutingDecisionEvent,
    AgentDispatchEvent,
    AgentResultEvent,
    OrchestrationCompleteEvent,
    event_to_dict,
)
from koboi.types import AgentResponse, TokenUsage


class TestEventToDict:
    def test_text_delta(self):
        event = TextDeltaEvent(content="hello")
        d = event_to_dict(event)
        assert d["type"] == "text_delta"
        assert d["content"] == "hello"

    def test_tool_call(self):
        event = ToolCallEvent(tool_name="calc", tool_call_id="tc1", arguments='{"x": 1}')
        d = event_to_dict(event)
        assert d["type"] == "tool_call"
        assert d["tool_name"] == "calc"

    def test_tool_result(self):
        event = ToolResultEvent(tool_name="calc", tool_call_id="tc1", result="42")
        d = event_to_dict(event)
        assert d["type"] == "tool_result"
        assert d["result"] == "42"

    def test_iteration(self):
        event = IterationEvent(iteration=3, messages_count=10, tokens_estimated=500)
        d = event_to_dict(event)
        assert d["type"] == "iteration"
        assert d["iteration"] == 3

    def test_complete_with_usage(self):
        resp = AgentResponse(
            content="answer",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        )
        event = CompleteEvent(response=resp, content="answer", elapsed_seconds=1.5)
        d = event_to_dict(event)
        assert d["type"] == "complete"
        assert d["token_usage"]["prompt_tokens"] == 100

    def test_complete_without_response(self):
        event = CompleteEvent(content="done")
        d = event_to_dict(event)
        assert d["type"] == "complete"
        assert d["token_usage"] is None

    def test_error(self):
        event = ErrorEvent(error=ValueError("bad value"))
        d = event_to_dict(event)
        assert d["type"] == "error"
        assert "bad value" in d["error"]

    def test_routing_decision(self):
        event = RoutingDecisionEvent(
            agents=["a1", "a2"],
            confidence=0.9,
            method="llm",
            reasoning="test",
            domain_label="finance",
        )
        d = event_to_dict(event)
        assert d["type"] == "routing_decision"
        assert d["agents"] == ["a1", "a2"]
        assert d["domain_label"] == "finance"

    def test_agent_dispatch(self):
        event = AgentDispatchEvent(
            agent_name="a1",
            agent_index=0,
            total_agents=2,
            mode="parallel",
        )
        d = event_to_dict(event)
        assert d["type"] == "agent_dispatch"
        assert d["agent_name"] == "a1"

    def test_agent_result(self):
        event = AgentResultEvent(
            agent_name="a1",
            answer="ok",
            elapsed_seconds=1.0,
            tokens_used=100,
            is_dynamic=True,
            domain_label="test",
            failed=False,
        )
        d = event_to_dict(event)
        assert d["type"] == "agent_result"
        assert d["agent_name"] == "a1"

    def test_orchestration_complete(self):
        event = OrchestrationCompleteEvent(
            final_answer="done",
            elapsed_seconds=2.0,
            agent_results=[],
            execution_mode="parallel",
            routing_agents=["a1"],
            routing_confidence=0.9,
        )
        d = event_to_dict(event)
        assert d["type"] == "orchestration_complete"
        assert d["final_answer"] == "done"

    def test_unknown_event(self):
        d = event_to_dict("not an event")
        assert d["type"] == "unknown"

    def test_pending_approval(self):
        event = PendingApprovalEvent(
            approval_id="ap_123",
            tool_name="run_shell",
            tool_call_id="call_1",
            arguments='{"cmd": "ls"}',
            risk_level="destructive",
            reason="risk-based",
            timeout_seconds=120,
        )
        d = event_to_dict(event)
        assert d["type"] == "pending_approval"
        assert d["approval_id"] == "ap_123"
        assert d["tool_name"] == "run_shell"
        assert d["tool_call_id"] == "call_1"
        assert d["risk_level"] == "destructive"
        assert d["timeout_seconds"] == 120

    def test_pending_approval_defaults(self):
        event = PendingApprovalEvent(
            approval_id="ap_1",
            tool_name="t",
            tool_call_id="c",
            arguments="{}",
            risk_level="moderate",
        )
        d = event_to_dict(event)
        assert d["reason"] == ""
        assert d["timeout_seconds"] == 120.0
