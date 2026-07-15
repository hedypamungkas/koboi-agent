"""Tests for koboi.media.model_profile (ModelProfile registry + validate_request)."""

from __future__ import annotations

from koboi.media.model_profile import (
    get_model_profile,
    load_profiles_from_config,
    validate_request,
)
from koboi.media.types import MediaRequest, MediaUnit


class TestGetModelProfile:
    def test_finds_builtin_veo3(self):
        p = get_model_profile("veo3-1-full-text-to-video")
        assert p is not None
        assert p.durations == [4, 6, 8]
        assert p.modality == "video"

    def test_finds_builtin_kling(self):
        p = get_model_profile("kling-v3-4k-text-to-video")
        assert p is not None
        assert p.durations is None  # any duration

    def test_finds_builtin_tts(self):
        p = get_model_profile("tts-xai-v1")
        assert p is not None
        assert p.param_overrides == {"voice": None}

    def test_unknown_returns_none(self):
        assert get_model_profile("nonexistent-model") is None
        assert get_model_profile(None) is None


class TestValidateRequest:
    def test_duration_corrected_to_nearest(self):
        """5s → 8s for veo3 (durations [4,6,8]); 5 is closest to 4 or 6."""
        req = MediaRequest(modality="video", prompt="x", model="veo3-1-full-text-to-video", duration_seconds=5)
        profile = get_model_profile("veo3-1-full-text-to-video")
        req = validate_request(req, profile)
        assert req.duration_seconds in (4.0, 6.0)  # nearest to 5 from [4,6,8]

    def test_duration_corrected_wan(self):
        """5s → 6s for wan (durations [6] only)."""
        req = MediaRequest(modality="video", prompt="x", model="wan-2-1-pro-image-to-video", duration_seconds=5)
        profile = get_model_profile("wan-2-1-pro-image-to-video")
        req = validate_request(req, profile)
        assert req.duration_seconds == 6.0

    def test_duration_already_supported_passes(self):
        """8s is supported for veo3 → unchanged."""
        req = MediaRequest(modality="video", prompt="x", model="veo3-1-full-text-to-video", duration_seconds=8)
        profile = get_model_profile("veo3-1-full-text-to-video")
        req = validate_request(req, profile)
        assert req.duration_seconds == 8.0

    def test_kling_any_duration_passes(self):
        """Kling has no duration constraint → any value passes."""
        req = MediaRequest(modality="video", prompt="x", model="kling-v3-4k-text-to-video", duration_seconds=5)
        profile = get_model_profile("kling-v3-4k-text-to-video")
        req = validate_request(req, profile)
        assert req.duration_seconds == 5.0  # unchanged

    def test_voice_override_tts(self):
        """TTS profiles override voice to None (omit → gateway default)."""
        req = MediaRequest(modality="speech", prompt="hello", model="tts-xai-v1", voice="alloy")
        profile = get_model_profile("tts-xai-v1")
        req = validate_request(req, profile)
        assert req.voice is None  # overridden to None

    def test_no_profile_passes_through(self):
        """No profile → request unchanged (backward compat)."""
        req = MediaRequest(modality="video", prompt="x", model="unknown-model", duration_seconds=5, voice="alloy")
        req = validate_request(req, None)
        assert req.duration_seconds == 5.0
        assert req.voice == "alloy"


class TestLoadProfilesFromConfig:
    def test_loads_custom_profile(self):
        load_profiles_from_config(
            [
                {"name": "my-custom-model", "modality": "image", "timeout": 600.0, "billing_unit": "image"},
            ]
        )
        p = get_model_profile("my-custom-model")
        assert p is not None
        assert p.timeout == 600.0
        assert p.billing_unit == MediaUnit.IMAGE

    def test_bad_entry_does_not_crash(self):
        load_profiles_from_config([{"name": "", "modality": "x"}])  # empty name → skipped
        load_profiles_from_config([{"modality": "image"}])  # no name → skipped
        # Should not raise.
