"""W3 tests: deep-research media integration (capability helpers, factory dep, event, switch)."""

from __future__ import annotations

from koboi.events import MediaGeneratedEvent, event_to_dict
from koboi.orchestration.factory import AgentFactory
from koboi.orchestration.orchestrator import _MEDIA_MODALITY
from koboi.orchestration.research import (
    RESEARCH_NODE_PREAMBLE,
    media_tools_for_capabilities,
    preamble_with_media,
)


class TestResearchMediaHelpers:
    def test_media_tools_for_capabilities_maps_known_tokens(self):
        assert media_tools_for_capabilities(["web", "image", "video"]) == ["generate_image", "generate_video"]

    def test_media_tools_for_capabilities_ignores_unknown(self):
        assert media_tools_for_capabilities(["web"]) == []
        assert media_tools_for_capabilities(["bogus"]) == []

    def test_media_tools_for_capabilities_all(self):
        assert media_tools_for_capabilities(["image", "video", "music", "speech"]) == [
            "generate_image",
            "generate_video",
            "generate_music",
            "generate_speech",
        ]

    def test_preamble_with_media_adds_clause_for_media_cap(self):
        text = preamble_with_media(["image"])
        assert text.startswith(RESEARCH_NODE_PREAMBLE)
        assert "generate_*" in text  # the media clause mentions the generate_* tools
        assert len(text) > len(RESEARCH_NODE_PREAMBLE)

    def test_preamble_without_media_cap_is_base(self):
        assert preamble_with_media(["web"]) == RESEARCH_NODE_PREAMBLE
        assert preamble_with_media([]) == RESEARCH_NODE_PREAMBLE


class TestFactoryMediaProviderDep:
    def test_media_provider_dep_set_when_passed(self):
        backend = object()  # any backend; the dep store is opaque
        registry = AgentFactory._build_tools_from_config({"builtin": ["generate_image"]}, media_provider=backend)
        assert registry is not None
        assert registry.get_dep("media_provider") is backend
        assert "generate_image" in registry

    def test_no_media_provider_dep_when_absent(self):
        registry = AgentFactory._build_tools_from_config({"builtin": ["web_search"]})
        assert registry is not None
        assert registry.get_dep("media_provider") is None


class TestMediaGeneratedEvent:
    def test_event_to_dict_serializes(self):
        event = MediaGeneratedEvent(modality="image", prompt="a cat")
        data = event_to_dict(event)
        assert data["modality"] == "image"
        assert data["prompt"] == "a cat"

    def test_event_type_registered(self):
        from koboi.events import _EVENT_TYPE_MAP

        assert _EVENT_TYPE_MAP[MediaGeneratedEvent] == "media_generated"


class TestSwitchMapping:
    """The per-node tool-call switch emits MediaGeneratedEvent via _MEDIA_MODALITY. Verify the map
    drives the correct modality for each media tool name (the switch itself is a one-liner)."""

    def test_all_media_tools_mapped(self):
        assert _MEDIA_MODALITY == {
            "generate_image": "image",
            "generate_video": "video",
            "generate_music": "music",
            "generate_speech": "speech",
        }

    def test_event_from_tool_call(self):
        # Simulates the switch: tc.name in _MEDIA_MODALITY -> MediaGeneratedEvent(modality=...)
        for name, modality in _MEDIA_MODALITY.items():
            event = MediaGeneratedEvent(modality=_MEDIA_MODALITY[name], prompt="x")
            assert event.modality == modality
