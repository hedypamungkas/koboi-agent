"""W4 tests: post-synthesis media briefing (generate_research_media + helpers)."""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

from koboi.config_models import ResearchConfig
from koboi.media.types import MediaResult, MediaUnit
from koboi.orchestration.research import (
    RESEARCH_MEDIA_SCHEMA,
    build_media_selection_prompt,
    generate_research_media,
)


class _MockClient:
    """Returns a canned JSON string as ``resp.content``; optionally raises."""

    def __init__(self, content: str = "", raise_: bool = False) -> None:
        self._content = content
        self._raise = raise_

    async def complete(self, messages, response_format=None):
        if self._raise:
            raise RuntimeError("LLM down")
        return SimpleNamespace(content=self._content)


class _MockBackend:
    """Records calls; optionally raises for selected kinds."""

    def __init__(self, raise_kinds: tuple[str, ...] = ()) -> None:
        self.raise_kinds = raise_kinds
        self.calls: dict[str, int] = {}

    async def _gen(self, kind: str, req) -> MediaResult:
        self.calls[kind] = self.calls.get(kind, 0) + 1
        if kind in self.raise_kinds:
            raise RuntimeError(f"{kind} boom")
        return MediaResult(
            request_id="x",
            modality=kind,
            status="ok",
            local_path=f"/tmp/{kind}.bin",
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


def _media_json(**fields) -> str:
    return json.dumps(fields)


class TestBuildMediaSelectionPrompt:
    def test_embeds_report_and_kinds(self):
        prompt = build_media_selection_prompt("solid-state battery report", ["image", "speech"])
        assert "solid-state battery report" in prompt
        assert "image" in prompt and "speech" in prompt

    def test_truncates_long_report(self):
        prompt = build_media_selection_prompt("x" * 10000, ["image"])
        assert len(prompt) < 12000  # 4000-char report cap + prompt boilerplate


class TestGenerateResearchMedia:
    async def test_generates_image_artifact(self):
        client = _MockClient(_media_json(image_prompts=["a diagram of the concept"]))
        backend = _MockBackend()
        section, artifacts = await generate_research_media(client, "report", ["image"], 1, backend)
        assert section.startswith("\n\n## Generated media")
        assert len(artifacts) == 1
        assert artifacts[0]["kind"] == "image"
        assert artifacts[0]["prompt"] == "a diagram of the concept"
        assert artifacts[0]["local_path"] == "/tmp/image.bin"
        assert backend.calls["image"] == 1

    async def test_respects_max_items(self):
        client = _MockClient(_media_json(image_prompts=["p1", "p2", "p3"]))
        backend = _MockBackend()
        _section, artifacts = await generate_research_media(client, "report", ["image"], 2, backend)
        assert len(artifacts) == 2
        assert backend.calls["image"] == 2

    async def test_multiple_kinds(self):
        client = _MockClient(_media_json(image_prompts=["img"], speech_texts=["summary"]))
        backend = _MockBackend()
        _section, artifacts = await generate_research_media(client, "report", ["image", "speech"], 1, backend)
        kinds = sorted(a["kind"] for a in artifacts)
        assert kinds == ["image", "speech"]

    async def test_client_failure_returns_empty(self):
        client = _MockClient(raise_=True)
        _section, artifacts = await generate_research_media(client, "report", ["image"], 1, _MockBackend())
        assert (_section, artifacts) == ("", [])

    async def test_one_kind_failure_skipped_others_proceed(self):
        client = _MockClient(_media_json(image_prompts=["img"], speech_texts=["summary"]))
        backend = _MockBackend(raise_kinds=("image",))  # image raises, speech ok
        _section, artifacts = await generate_research_media(client, "report", ["image", "speech"], 1, backend)
        assert len(artifacts) == 1
        assert artifacts[0]["kind"] == "speech"

    async def test_no_kinds_returns_empty(self):
        _section, artifacts = await generate_research_media(_MockClient(), "report", [], 1, _MockBackend())
        assert (_section, artifacts) == ("", [])

    async def test_no_backend_returns_empty(self):
        _section, artifacts = await generate_research_media(_MockClient(), "report", ["image"], 1, None)
        assert (_section, artifacts) == ("", [])

    async def test_no_prompts_returns_empty(self):
        # LLM returned no prompts for the requested kind.
        client = _MockClient(_media_json(image_prompts=[]))
        _section, artifacts = await generate_research_media(client, "report", ["image"], 1, _MockBackend())
        assert (_section, artifacts) == ("", [])


class TestSchema:
    def test_schema_has_all_kind_fields(self):
        assert set(RESEARCH_MEDIA_SCHEMA["properties"]) == {
            "image_prompts",
            "speech_texts",
            "video_prompts",
            "music_prompts",
        }


class TestResearchConfigMedia:
    def test_accepts_media_field(self):
        cfg = ResearchConfig(media={"enabled": True, "kinds": ["image"], "max_items": 1})
        assert cfg.media["enabled"] is True
        assert cfg.media["kinds"] == ["image"]

    def test_default_media_empty(self):
        assert ResearchConfig().media == {}
