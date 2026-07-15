"""Tier A: offline integrated Deep Research + media e2e.

W3 per-node tests are active. W4 post-synthesis tests are skipped until the
generate_research_media call is re-applied to main's orchestrator.
"""

from __future__ import annotations

import json

import pytest

from koboi.events import MediaGeneratedEvent
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.router import KeywordRouter
from koboi.types import AgentResponse, ToolCall


class _FakeClient:
    def __init__(self, node_answer="Found: X [1].", coverage_score=0.95, emit_image_call=False):
        self.node_answer = node_answer
        self.coverage_score = coverage_score
        self.emit_image_call = emit_image_call
        self.model = "fake-model"
        self.provider = "fake"

    async def complete(self, messages, tools=None, response_format=None):
        text = " ".join(m.get("content", "") for m in messages)
        if "research planner" in text:
            return AgentResponse(content=json.dumps({"needs_workflow": True, "reason": "r", "steps": [{"id": "n", "instruction": "Investigate", "depends_on": []}]}), tool_calls=[])
        if "evaluating how thoroughly" in text:
            return AgentResponse(content=json.dumps({"overall_score": self.coverage_score, "coverage": {}, "follow_up_queries": []}), tool_calls=[])
        if "synthesizing" in text:
            return AgentResponse(content="## Report\nX [1].", tool_calls=[])
        if self.emit_image_call and not any(m.get("role") == "tool" for m in messages):
            return AgentResponse(content="", tool_calls=[ToolCall(id="tc", name="generate_image", arguments=json.dumps({"prompt": "diagram"}))])
        return AgentResponse(content=self.node_answer, tool_calls=[])

    async def complete_stream(self, messages, tools=None):
        yield "## Report\nX [1]."


def _orch(client, research, media_conf, tmp_path):
    return Orchestrator(client=client, router=KeywordRouter(), research=research, media_conf=media_conf,
                        dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")))


class TestDeepResearchMediaIntegration:
    async def test_node_media_event(self, tmp_path):
        """W3: node emitting generate_image -> MediaGeneratedEvent."""
        client = _FakeClient(emit_image_call=True)
        orch = _orch(client, {"max_depth": 1, "coverage_threshold": 0.7, "capabilities": ["web", "image"]},
                     {"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path / "art")}}, tmp_path)
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        media_events = [e for e in events if isinstance(e, MediaGeneratedEvent)]
        assert media_events, "expected MediaGeneratedEvent"
        assert media_events[0].modality == "image"

    async def test_post_synthesis_media_briefing(self, tmp_path):
        """W4: skipped until re-applied to main's orchestrator."""
        pytest.skip("W4 post-synthesis media briefing not yet re-applied to main's orchestrator")

    async def test_multi_image_briefing(self, tmp_path):
        pytest.skip("W4 post-synthesis not yet re-applied")

    async def test_no_media_when_disabled(self, tmp_path):
        pytest.skip("W4 post-synthesis not yet re-applied")
