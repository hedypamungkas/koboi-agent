"""W2 tests: speech synthesis (TTS) -- sync, mirrors the image pattern (no async polling)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from koboi.media.backend import MediaBackend, build_media
from koboi.media.budget import CountingSpeechProvider
from koboi.media.providers.mock import MockSpeechProvider
from koboi.media.providers.surplus import SurplusSpeechProvider
from koboi.media.store import MediaStore
from koboi.media.types import MediaBudget, MediaRequest, MediaResult, MediaUnit
from koboi.tools.builtin import register_all
from koboi.tools.registry import ToolRegistry
from koboi.types import RiskLevel


class TestMockSpeechProvider:
    async def test_returns_audio_bytes(self):
        res = await MockSpeechProvider().synthesize_speech(MediaRequest(modality="speech", prompt="hello world"))
        assert res.status == "ok"
        assert res.data == b"mock-speech-audio"
        assert res.content_type == "audio/mpeg"
        assert res.billing_unit == MediaUnit.CHAR
        assert res.billing_quantity == 11.0  # len("hello world")


class TestSurplusSpeechProvider:
    async def test_synthesize_default_mp3(self):
        provider = SurplusSpeechProvider(api_key="k", model="venice-kokoro-tts")
        provider._transport.post_bytes = AsyncMock(return_value=b"mp3bytes")
        res = await provider.synthesize_speech(
            MediaRequest(modality="speech", prompt="hi", voice="alloy", response_format="mp3")
        )
        assert res.status == "ok"
        assert res.data == b"mp3bytes"
        assert res.content_type == "audio/mpeg"
        assert res.billing_unit == MediaUnit.CHAR
        assert res.billing_quantity == 2.0
        provider._transport.post_bytes.assert_awaited()

    async def test_content_type_wav(self):
        provider = SurplusSpeechProvider(api_key="k")
        provider._transport.post_bytes = AsyncMock(return_value=b"wavbytes")
        res = await provider.synthesize_speech(MediaRequest(modality="speech", prompt="hi", response_format="wav"))
        assert res.content_type == "audio/wav"

    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("SURPLUS_API_KEY", raising=False)
        provider = SurplusSpeechProvider(api_key="")
        with pytest.raises(ValueError):
            await provider.synthesize_speech(MediaRequest(modality="speech", prompt="hi"))

    def test_unsupported_auth_mode(self):
        with pytest.raises(NotImplementedError):
            SurplusSpeechProvider(api_key="k", auth_mode="x402")

    async def test_close_calls_transport(self):
        provider = SurplusSpeechProvider(api_key="k")
        provider._transport.close = AsyncMock()
        await provider.close()
        provider._transport.close.assert_awaited()


class TestMediaBudgetSpeech:
    def test_remaining_speech_chars_cap(self):
        assert MediaBudget(max_speech_chars=100, used_speech_chars=100).remaining("speech") is False

    def test_record_speech_accrues_chars(self):
        budget = MediaBudget()
        budget.record(MediaResult(request_id="r", modality="speech", status="ok", billing_quantity=42.0))
        assert budget.used_speech_chars == 42


class TestCountingSpeechProvider:
    async def test_rejects_when_exhausted(self):
        budget = MediaBudget(max_speech_chars=0)
        provider = CountingSpeechProvider(MockSpeechProvider(), budget)
        res = await provider.synthesize_speech(MediaRequest(modality="speech", prompt="hi"))
        assert res.status == "rejected"

    async def test_delegates_and_records(self):
        budget = MediaBudget(max_speech_chars=1000)
        provider = CountingSpeechProvider(MockSpeechProvider(), budget)
        res = await provider.synthesize_speech(MediaRequest(modality="speech", prompt="hello"))
        assert res.status == "ok"
        assert budget.used_speech_chars == 5


class TestBuildMediaSpeech:
    def test_builds_speech_when_configured(self, tmp_path):
        backend = build_media({"enabled": True, "speech": {"provider": "mock"}, "storage": {"dir": str(tmp_path)}})
        assert backend is not None
        assert backend.speech is not None

    def test_budget_wraps_speech(self, tmp_path):
        backend = build_media(
            {
                "enabled": True,
                "speech": {"provider": "mock"},
                "budget": {"max_cost_usd": 1.0},
                "storage": {"dir": str(tmp_path)},
            }
        )
        assert isinstance(backend.speech, CountingSpeechProvider)


class TestMediaBackendSpeech:
    async def test_generate_speech_materializes(self, tmp_path):
        backend = MediaBackend(speech=MockSpeechProvider(), store=MediaStore(backend="local", dir=str(tmp_path)))
        res = await backend.generate_speech(MediaRequest(modality="speech", prompt="hi"))
        assert res.status == "ok"
        assert res.local_path is not None
        assert Path(res.local_path).exists()

    async def test_generate_speech_without_slot_fails(self):
        backend = MediaBackend(speech=None)
        res = await backend.generate_speech(MediaRequest(modality="speech", prompt="hi"))
        assert res.status == "failed"
        assert "not configured" in (res.rejection_reason or "")


class TestGenerateSpeechTool:
    @staticmethod
    def _registry(tmp_path) -> ToolRegistry:
        registry = ToolRegistry()
        register_all(registry)
        backend = build_media({"enabled": True, "speech": {"provider": "mock"}, "storage": {"dir": str(tmp_path)}})
        registry.set_dep("media_provider", backend)
        return registry

    async def test_generate_speech(self, tmp_path):
        out = await self._registry(tmp_path).execute("generate_speech", json.dumps({"prompt": "welcome"}))
        assert out.startswith("Speech saved:")
        assert str(tmp_path) in out

    async def test_not_configured(self):
        registry = ToolRegistry()
        register_all(registry)
        out = await registry.execute("generate_speech", json.dumps({"prompt": "hi"}))
        assert "media not configured" in out

    def test_flags(self):
        registry = ToolRegistry()
        register_all(registry)
        td = registry.get_definition("generate_speech")
        assert td is not None
        assert td.risk_level == RiskLevel.MODERATE
        assert td.idempotent is False
        assert td.timeout == 120.0
