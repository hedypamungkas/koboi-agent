"""Tier A: offline integrated Deep Research + media e2e.

W3 per-node tests + W4 post-synthesis tests (Wave2 #6 un-skipped after the
``generate_research_media`` call was wired into ``_synthesize_research``).
"""

from __future__ import annotations

import json
from decimal import Decimal

from koboi.events import MediaGeneratedEvent, OrchestrationCompleteEvent, TextDeltaEvent
from koboi.media.types import MediaResult, MediaUnit
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.router import KeywordRouter
from koboi.types import AgentResponse, ToolCall


class _FakeClient:
    """Canned-response LLM.

    Branches (matched on prompt substring):
    - "research planner"   -> needs_workflow=True + 1 step
    - "evaluating how"     -> coverage judge
    - "synthesizing"       -> cited report body
    - "selecting media"    -> W4 media-selection JSON (image_prompts etc.)
    - otherwise            -> node answer (optionally with a generate_image tool_call)
    """

    def __init__(
        self,
        node_answer="Found: X [1].",
        coverage_score=0.95,
        emit_image_call=False,
        media_selection: dict | None = None,
    ):
        self.node_answer = node_answer
        self.coverage_score = coverage_score
        self.emit_image_call = emit_image_call
        self.media_selection = media_selection
        self.model = "fake-model"
        self.provider = "fake"

    async def complete(self, messages, tools=None, response_format=None):
        text = " ".join(m.get("content", "") for m in messages)
        if "research planner" in text:
            return AgentResponse(
                content=json.dumps(
                    {
                        "needs_workflow": True,
                        "reason": "r",
                        "steps": [{"id": "n", "instruction": "Investigate", "depends_on": []}],
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
        if "synthesizing" in text:
            return AgentResponse(content="## Report\nX [1].", tool_calls=[])
        if "selecting media" in text:
            payload = self.media_selection if self.media_selection is not None else {"image_prompts": ["diagram of X"]}
            return AgentResponse(content=json.dumps(payload), tool_calls=[])
        if self.emit_image_call and not any(m.get("role") == "tool" for m in messages):
            return AgentResponse(
                content="",
                tool_calls=[ToolCall(id="tc", name="generate_image", arguments=json.dumps({"prompt": "diagram"}))],
            )
        return AgentResponse(content=self.node_answer, tool_calls=[])

    async def complete_stream(self, messages, tools=None):
        yield "## Report\nX [1]."


class _MockMediaBackend:
    """Records generate_* calls; returns a canned ok MediaResult per kind."""

    def __init__(self):
        self.calls: dict[str, int] = {}

    async def _gen(self, kind: str, req) -> MediaResult:
        self.calls[kind] = self.calls.get(kind, 0) + 1
        return MediaResult(
            request_id=f"req-{kind}",
            modality=kind,
            status="ok",
            local_path=f"/tmp/research_{kind}.bin",
            content_type="application/octet-stream",
            cost_usd=Decimal("0.01"),
            billing_unit=MediaUnit.IMAGE,
            model=f"mock-{kind}",
        )

    async def generate_image(self, req):
        return await self._gen("image", req)

    async def generate_speech(self, req):
        return await self._gen("speech", req)

    async def generate_video(self, req):
        return await self._gen("video", req)

    async def generate_music(self, req):
        return await self._gen("music", req)


def _orch(client, research, media_conf, tmp_path, media_backend=None):
    return Orchestrator(
        client=client,
        router=KeywordRouter(),
        research=research,
        media_conf=media_conf,
        media_backend=media_backend,
        dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
    )


class TestDeepResearchMediaIntegration:
    async def test_node_media_event(self, tmp_path):
        """W3: node emitting generate_image -> MediaGeneratedEvent."""
        client = _FakeClient(emit_image_call=True)
        orch = _orch(
            client,
            {"max_depth": 1, "coverage_threshold": 0.7, "capabilities": ["web", "image"]},
            {"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path / "art")}},
            tmp_path,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        media_events = [e for e in events if isinstance(e, MediaGeneratedEvent)]
        assert media_events, "expected MediaGeneratedEvent"
        assert media_events[0].modality == "image"

    async def test_post_synthesis_media_briefing(self, tmp_path):
        """W4 (Wave2 #6): post-synthesis auto multimedia briefing fires when configured.

        capabilities=["image"] + media.enabled + a real media_backend -> the synthesized
        report gains a ``## Generated media`` section AND the backend's generate_image is
        invoked exactly once for the single selected prompt.
        """
        backend = _MockMediaBackend()
        client = _FakeClient()  # default media_selection returns one image_prompts entry
        orch = _orch(
            client,
            {"max_depth": 1, "coverage_threshold": 0.7, "capabilities": ["image"]},
            {"enabled": True},
            tmp_path,
            media_backend=backend,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        # The synthesized report + media section land in TextDeltaEvent content.
        text_chunks = [e.content for e in events if isinstance(e, TextDeltaEvent)]
        combined = "".join(text_chunks)
        assert "## Generated media" in combined, f"media section missing from report: {combined!r}"
        assert backend.calls.get("image") == 1, f"expected 1 image generate call, got {backend.calls}"
        # Completion still surfaces -- the run terminates cleanly.
        assert any(isinstance(e, OrchestrationCompleteEvent) for e in events)

    async def test_multi_image_briefing(self, tmp_path):
        """W4: max_items caps the number of generations per kind."""
        backend = _MockMediaBackend()
        client = _FakeClient(media_selection={"image_prompts": ["p1", "p2", "p3"]})
        orch = _orch(
            client,
            {
                "max_depth": 1,
                "coverage_threshold": 0.7,
                "capabilities": ["image"],
                "media": {"max_items": 2},
            },
            {"enabled": True},
            tmp_path,
            media_backend=backend,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        text_chunks = [e.content for e in events if isinstance(e, TextDeltaEvent)]
        assert "## Generated media" in "".join(text_chunks)
        # max_items=2 -> exactly two generate_image calls even though three prompts were selected.
        assert backend.calls.get("image") == 2

    async def test_no_media_when_disabled(self, tmp_path):
        """W4: media section is absent when ``media.enabled`` is false even if capabilities include image."""
        backend = _MockMediaBackend()
        client = _FakeClient()
        orch = _orch(
            client,
            {"max_depth": 1, "coverage_threshold": 0.7, "capabilities": ["image"]},
            {"enabled": False},  # media disabled
            tmp_path,
            media_backend=backend,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        text_chunks = [e.content for e in events if isinstance(e, TextDeltaEvent)]
        assert "## Generated media" not in "".join(text_chunks)
        assert backend.calls == {}, f"backend should not have been called, got {backend.calls}"

    async def test_no_media_when_backend_missing(self, tmp_path):
        """W4: graceful skip when ``media_backend`` is None (the default mock-conf-only path)."""
        client = _FakeClient()
        orch = _orch(
            client,
            {"max_depth": 1, "coverage_threshold": 0.7, "capabilities": ["image"]},
            {"enabled": True},
            tmp_path,
            media_backend=None,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        text_chunks = [e.content for e in events if isinstance(e, TextDeltaEvent)]
        assert "## Generated media" not in "".join(text_chunks)

    async def test_no_media_when_capabilities_have_no_media_tokens(self, tmp_path):
        """W4: graceful skip when capabilities only contain non-media tokens (e.g. 'web')."""
        backend = _MockMediaBackend()
        client = _FakeClient()
        orch = _orch(
            client,
            {"max_depth": 1, "coverage_threshold": 0.7, "capabilities": ["web"]},  # no media tokens
            {"enabled": True},
            tmp_path,
            media_backend=backend,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        text_chunks = [e.content for e in events if isinstance(e, TextDeltaEvent)]
        assert "## Generated media" not in "".join(text_chunks)
        assert backend.calls == {}, f"backend should not have been called, got {backend.calls}"

    async def test_max_items_non_coercible_soft_fails(self, tmp_path):
        """max_items with non-integer value (e.g. 'two') soft-fails instead of crashing."""
        backend = _MockMediaBackend()
        client = _FakeClient()
        orch = _orch(
            client,
            {
                "max_depth": 1,
                "coverage_threshold": 0.7,
                "capabilities": ["image"],
                "media": {"max_items": "two"},  # non-coercible string
            },
            {"enabled": True},
            tmp_path,
            media_backend=backend,
        )
        # Should not raise - returns report unchanged
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        text_chunks = [e.content for e in events if isinstance(e, TextDeltaEvent)]
        combined = "".join(text_chunks)
        # Report should be present but media section absent (soft failure)
        assert "## Report" in combined, f"base report missing: {combined!r}"
        assert "## Generated media" not in combined, f"media section should be absent after soft fail"
        assert backend.calls == {}, f"backend should not have been called, got {backend.calls}"
        # Run should complete cleanly
        assert any(isinstance(e, OrchestrationCompleteEvent) for e in events)

    async def test_max_items_negative_clamps_to_zero(self, tmp_path):
        """Negative max_items clamps to 0 (no generation) instead of slicing wrong."""
        backend = _MockMediaBackend()
        client = _FakeClient(media_selection={"image_prompts": ["p1", "p2"]})
        orch = _orch(
            client,
            {
                "max_depth": 1,
                "coverage_threshold": 0.7,
                "capabilities": ["image"],
                "media": {"max_items": -1},  # negative - clamps to 0
            },
            {"enabled": True},
            tmp_path,
            media_backend=backend,
        )
        events = [e async for e in orch._run_deep_research("Tell me about X")]
        text_chunks = [e.content for e in events if isinstance(e, TextDeltaEvent)]
        combined = "".join(text_chunks)
        # When max_items is 0, no generation occurs and media section is absent (soft-fail)
        assert "## Generated media" not in combined, f"media section should be absent when max_items=0: {combined!r}"
        # No calls should be made (get returns None if key missing, treat as 0)
        assert backend.calls.get("image") in (0, None), f"expected 0 calls, got {backend.calls.get('image')}"
        assert any(isinstance(e, OrchestrationCompleteEvent) for e in events)
