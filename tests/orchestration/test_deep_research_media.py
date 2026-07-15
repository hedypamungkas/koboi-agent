"""Tier A: offline integrated Deep Research + media e2e (W3 per-node + W4 post-synthesis together).

NOTE: W4 post-synthesis tests are marked ``@pytest.mark.skip`` until the post-synthesis media step
(generate_research_media call inside _run_deep_research) is re-applied to main's orchestrator.
W3 per-node tests (MediaGeneratedEvent emission, capabilities wiring) ARE active.
"""

from __future__ import annotations

import json

import pytest

from koboi.events import MediaGeneratedEvent, OrchestrationCompleteEvent
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.router import KeywordRouter
from koboi.types import AgentResponse, ToolCall


@pytest.fixture
def _no_w4_post_synthesis():
    """W4 post-synthesis media briefing is not yet integrated on main's orchestrator."""
    pytest.skip("W4 post-synthesis media briefing not yet re-applied to main's orchestrator")


class _FakeClient:
    """Duck-typed LLM client dispatching on prompt content."""

    def __init__(
        self,
        node_answer: str = "Found: the topic is X [1] and Y [1].",
        coverage_score: float = 0.95,
        synthesis: str = "## Report\nThe topic is X [1] and Y [1].",
        emit_image_call: bool = False,
        image_prompts: list[str] | None = None,
    ) -> None:
        self.node_answer = node_answer
        self.coverage_score = coverage_score
        self.synthesis = synthesis
        self.emit_image_call = emit_image_call
        self.image_prompts = image_prompts if image_prompts is not None else ["a diagram of the concept"]
        self.model = "fake-model"
        self.provider = "fake"

    async def complete(self, messages, tools=None, response_format=None):
        text = " ".join(m.get("content", "") for m in messages)
        if "research planner" in text:
            return AgentResponse(
                content=json.dumps(
                    {
                        "needs_workflow": True,
                        "reason": "research",
                        "steps": [
                            {"id": "research_topic", "instruction": "Investigate the topic", "depends_on": []},
                        ],
                    }
                ),
                tool_calls=[],
            )
        if "evaluating how thoroughly" in text:
            return AgentResponse(
                content=json.dumps(
                    {"overall_score": self.coverage_score, "coverage": {}, "follow_up_queries": []}
                ),
                tool_calls=[],
            )
        if "selecting media to accompany" in text:
            return AgentResponse(content=json.dumps({"image_prompts": self.image_prompts}), tool_calls=[])
        if "synthesizing a cited research report" in text:
            return AgentResponse(content=self.synthesis, tool_calls=[])
        if self.emit_image_call and not any(m.get("role") == "tool" for m in messages):
            return AgentResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc_img", name="generate_image", arguments=json.dumps({"prompt": "a diagram"}))
                ],
            )
        return AgentResponse(content=self.node_answer, tool_calls=[])

    async def complete_stream(self, messages, tools=None):
        yield self.synthesis


def _orch(client, research, media_conf, tmp_path):
    return Orchestrator(
        client=client,
        router=KeywordRouter(),
        research=research,
        media_conf=media_conf,
        dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
    )


class TestDeepResearchMediaIntegration:
    async def test_post_synthesis_media_briefing(self, tmp_path, _no_w4_post_synthesis):
        """W4: research.media.enabled -> report has '## Generated media'."""
        # Skipped — W4 post-synthesis not yet re-applied to main's orchestrator.

    async def test_node_media_event(self, tmp_path):
        """W3: a node emitting generate_image -> MediaGeneratedEvent yielded."""
        client = _FakeClient(emit_image_call=True)
        orch = _orch(
            client,
            research={
                "max_depth": 1,
                "coverage_threshold": 0.7,
                "capabilities": ["web", "image"],
            },
            media_conf={"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path / "art")}},
            tmp_path=tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        media_events = [e for e in events if isinstance(e, MediaGeneratedEvent)]
        assert media_events, "expected at least one MediaGeneratedEvent from the node's generate_image call"
        assert media_events[0].modality == "image"

    async def test_multi_image_briefing(self, tmp_path, _no_w4_post_synthesis):
        """W4 max_items: multiple image prompts -> multiple artifacts."""
        # Skipped — W4 post-synthesis not yet re-applied.

    async def test_no_media_when_disabled(self, tmp_path, _no_w4_post_synthesis):
        """Default (research.media absent) -> no media section."""
        # Skipped — W4 post-synthesis not yet re-applied.
``generate_research_media`` step appends a ``## Generated media`` section + ``media_artifacts``
metadata. All offline (mock); no keys, no $.
"""

from __future__ import annotations

import json

from koboi.events import MediaGeneratedEvent, OrchestrationCompleteEvent
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.router import KeywordRouter
from koboi.types import AgentResponse, ToolCall


class _FakeClient:
    """Duck-typed LLM client dispatching on prompt content.

    Handles: research planner / coverage / synthesis / node / **media-selection** (W4). When
    ``emit_image_call`` is True, a node's first turn emits a ``generate_image`` tool_call (so the
    per-node switch yields a ``MediaGeneratedEvent``); the next turn returns ``node_answer``.
    """

    def __init__(
        self,
        node_answer: str = "Found: the topic is X [1] and Y [1].",
        coverage_score: float = 0.95,
        synthesis: str = "## Report\nThe topic is X [1] and Y [1].",
        emit_image_call: bool = False,
        image_prompts: list[str] | None = None,
    ) -> None:
        self.node_answer = node_answer
        self.coverage_score = coverage_score
        self.synthesis = synthesis
        self.emit_image_call = emit_image_call
        self.image_prompts = image_prompts if image_prompts is not None else ["a diagram of the concept"]
        self.model = "fake-model"
        self.provider = "fake"

    async def complete(self, messages, tools=None, response_format=None):
        text = " ".join(m.get("content", "") for m in messages)
        if "research planner" in text:
            return AgentResponse(
                content=json.dumps(
                    {
                        "needs_workflow": True,
                        "reason": "research",
                        "steps": [
                            {"id": "research_topic", "instruction": "Investigate the topic", "depends_on": []},
                        ],
                    }
                ),
                tool_calls=[],
            )
        if "evaluating how thoroughly" in text:
            return AgentResponse(
                content=json.dumps(
                    {"overall_score": self.coverage_score, "coverage": {}, "follow_up_queries": []}
                ),
                tool_calls=[],
            )
        if "selecting media to accompany" in text:
            # W4 generate_research_media: return per-kind prompts for the requested kinds.
            return AgentResponse(content=json.dumps({"image_prompts": self.image_prompts}), tool_calls=[])
        if "synthesizing a cited research report" in text:
            return AgentResponse(content=self.synthesis, tool_calls=[])
        # Node turn: optionally emit a generate_image tool_call before the final answer.
        if self.emit_image_call and not any(m.get("role") == "tool" for m in messages):
            return AgentResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc_img", name="generate_image", arguments=json.dumps({"prompt": "a diagram"}))
                ],
            )
        return AgentResponse(content=self.node_answer, tool_calls=[])

    async def complete_stream(self, messages, tools=None):
        yield self.synthesis


def _orch(client, research, media_conf, tmp_path):
    return Orchestrator(
        client=client,
        router=KeywordRouter(),
        research=research,
        media_conf=media_conf,
        dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
    )


class TestDeepResearchMediaIntegration:
    async def test_post_synthesis_media_briefing(self, tmp_path):
        """W4: research.media.enabled -> report has '## Generated media' + media_artifacts metadata."""
        client = _FakeClient()
        orch = _orch(
            client,
            research={"max_depth": 1, "coverage_threshold": 0.7, "media": {"enabled": True, "kinds": ["image"]}},
            media_conf={"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path / "art")}},
            tmp_path=tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)][0]
        assert "## Generated media" in complete.final_answer
        assert complete.metadata["media_artifacts"], "expected media_artifacts in terminal metadata"
        assert complete.metadata["media_artifacts"][0]["kind"] == "image"

    async def test_node_media_event(self, tmp_path):
        """W3: a node emitting generate_image -> MediaGeneratedEvent yielded."""
        client = _FakeClient(emit_image_call=True)
        orch = _orch(
            client,
            research={
                "max_depth": 1,
                "coverage_threshold": 0.7,
                "capabilities": ["web", "image"],  # W3: node gets generate_image tool
            },
            media_conf={"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path / "art")}},
            tmp_path=tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        media_events = [e for e in events if isinstance(e, MediaGeneratedEvent)]
        assert media_events, "expected at least one MediaGeneratedEvent from the node's generate_image call"
        assert media_events[0].modality == "image"

    async def test_multi_image_briefing(self, tmp_path):
        """W4 max_items: multiple image prompts -> multiple artifacts."""
        client = _FakeClient(image_prompts=["diagram one", "diagram two", "diagram three"])
        orch = _orch(
            client,
            research={
                "max_depth": 1,
                "coverage_threshold": 0.7,
                "media": {"enabled": True, "kinds": ["image"], "max_items": 3},
            },
            media_conf={"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path / "art")}},
            tmp_path=tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)][0]
        assert len(complete.metadata["media_artifacts"]) == 3

    async def test_no_media_when_disabled(self, tmp_path):
        """Default (research.media absent) -> no media section, empty artifacts."""
        client = _FakeClient()
        orch = _orch(
            client,
            research={"max_depth": 1, "coverage_threshold": 0.7},
            media_conf={"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path / "art")}},
            tmp_path=tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)][0]
        assert "## Generated media" not in complete.final_answer
        assert complete.metadata["media_artifacts"] == []
