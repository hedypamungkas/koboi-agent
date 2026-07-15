"""W5a tests: programmatic API + create_all_configured parity + STT (transcription)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from koboi.facade import KoboiAgent
from koboi.llm.http_transport import HttpTransport
from koboi.llm.auth import BearerAuth
from koboi.media.backend import MediaBackend, build_media
from koboi.media.providers.mock import MockImageProvider, MockTranscriptionProvider
from koboi.media.providers.surplus import SurplusTranscriptionProvider
from koboi.media.types import MediaRequest, MediaResult
from koboi.orchestration.factory import AgentFactory
from koboi.tools.builtin import register_all
from koboi.tools.registry import ToolRegistry


# ── MediaBackend.generate dispatch ──


class _RecordingBackend(MediaBackend):
    """MediaBackend that records which generate_* ran."""

    def __init__(self) -> None:
        super().__init__(image=MockImageProvider())
        self.called: list[str] = []

    async def generate_image(self, req):  # type: ignore[override]
        self.called.append("image")
        return await super().generate_image(req)

    async def generate_video(self, req):  # type: ignore[override]
        self.called.append("video")
        return MediaResult(request_id="x", modality="video", status="ok")

    async def generate_music(self, req):  # type: ignore[override]
        self.called.append("music")
        return MediaResult(request_id="x", modality="music", status="ok")

    async def generate_speech(self, req):  # type: ignore[override]
        self.called.append("speech")
        return MediaResult(request_id="x", modality="speech", status="ok")


class TestMediaBackendGenerate:
    async def test_dispatches_by_modality(self):
        backend = _RecordingBackend()
        await backend.generate(MediaRequest(modality="image", prompt="x"))
        await backend.generate(MediaRequest(modality="video", prompt="x"))
        await backend.generate(MediaRequest(modality="music", prompt="x"))
        await backend.generate(MediaRequest(modality="speech", prompt="x"))
        assert backend.called == ["image", "video", "music", "speech"]

    async def test_unknown_modality_not_configured(self):
        backend = MediaBackend()
        res = await backend.generate(MediaRequest(modality="hologram", prompt="x"))
        assert res.status == "failed"


# ── STT providers ──


class TestMockTranscription:
    async def test_returns_stub_text(self):
        text = await MockTranscriptionProvider().transcribe(b"\x00\x01\x02" * 10)
        assert "30 bytes" in text  # len(audio)


class TestSurplusTranscription:
    async def test_transcribe_via_post_form(self):
        provider = SurplusTranscriptionProvider(api_key="k", model="whisper-large-v3")
        provider._transport.post_form = AsyncMock(return_value={"text": "hello world"})
        text = await provider.transcribe(b"audio-bytes", language_code="en")
        assert text == "hello world"
        provider._transport.post_form.assert_awaited()

    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("SURPLUS_API_KEY", raising=False)
        provider = SurplusTranscriptionProvider(api_key="")
        import pytest

        with pytest.raises(ValueError):
            await provider.transcribe(b"x")

    async def test_close_calls_transport(self):
        provider = SurplusTranscriptionProvider(api_key="k")
        provider._transport.close = AsyncMock()
        await provider.close()
        provider._transport.close.assert_awaited()


class TestPostFormTransport:
    async def test_post_form_returns_json(self):
        transport = HttpTransport("https://api.example.com/v1", BearerAuth("k"))
        transport._client.request = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(status_code=200, json=lambda: {"text": "ok"}, headers={})
        )
        out = await transport.post_form(
            "/audio/transcriptions", files={"file": ("a", b"x", "oct")}, data={"model": "m"}
        )
        assert out == {"text": "ok"}


# ── transcribe_audio tool ──


def _registry_with_transcription(tmp_path) -> ToolRegistry:
    registry = ToolRegistry()
    register_all(registry)
    backend = build_media({"enabled": True, "transcription": {"provider": "mock"}, "storage": {"dir": str(tmp_path)}})
    registry.set_dep("media_provider", backend)
    return registry


class TestTranscribeAudioTool:
    async def test_transcribes_local_file(self, tmp_path):
        import json

        audio = tmp_path / "clip.bin"
        audio.write_bytes(b"\x00" * 16)
        out = await _registry_with_transcription(tmp_path).execute(
            "transcribe_audio", json.dumps({"file_path": str(audio)})
        )
        assert "16 bytes" in out

    async def test_not_configured(self):
        import json

        registry = ToolRegistry()
        register_all(registry)
        out = await registry.execute("transcribe_audio", json.dumps({"file_path": "/nope"}))
        assert "media not configured" in out

    async def test_no_source_error(self, tmp_path):
        import json

        out = await _registry_with_transcription(tmp_path).execute("transcribe_audio", json.dumps({}))
        assert "provide" in out.lower()

    def test_flags(self):
        registry = ToolRegistry()
        register_all(registry)
        td = registry.get_definition("transcribe_audio")
        assert td is not None
        assert td.idempotent is True  # read-only analysis, safe to re-fire on resume


# ── KoboiAgent programmatic API ──


class _FakeBackend:
    def __init__(self) -> None:
        self.gen: list[str] = []

    async def generate(self, req):
        self.gen.append(req.modality)
        return MediaResult(request_id="x", modality=req.modality, status="ok")

    async def transcribe(self, audio, **opts):
        return f"transcribed {len(audio)} bytes"


def _koboi_with_media(backend) -> KoboiAgent:
    registry = ToolRegistry()
    registry.set_dep("media_provider", backend)
    return KoboiAgent(core=SimpleNamespace(tools=registry))  # type: ignore[arg-type]


class TestKoboiAgentMedia:
    async def test_media_generate_uses_dep_store(self):
        backend = _FakeBackend()
        agent = _koboi_with_media(backend)
        res = await agent.media_generate(MediaRequest(modality="image", prompt="x"))
        assert res.status == "ok"
        assert backend.gen == ["image"]

    async def test_media_transcribe(self):
        backend = _FakeBackend()
        agent = _koboi_with_media(backend)
        text = await agent.media_transcribe(b"abc")
        assert text == "transcribed 3 bytes"

    async def test_media_generate_not_configured(self):
        agent = _koboi_with_media(None)  # no media dep set
        # rebuild registry without the dep
        registry = ToolRegistry()
        agent = KoboiAgent(core=SimpleNamespace(tools=registry))  # type: ignore[arg-type]
        res = await agent.media_generate(MediaRequest(modality="image", prompt="x"))
        assert res.status == "failed"

    async def test_media_transcribe_not_configured_raises(self):
        import pytest

        registry = ToolRegistry()
        agent = KoboiAgent(core=SimpleNamespace(tools=registry))  # type: ignore[arg-type]
        with pytest.raises(RuntimeError):
            await agent.media_transcribe(b"x")


# ── create_all_configured parity ──


class TestCreateAllConfiguredMedia:
    async def test_threads_media_provider(self):
        from koboi.types import AgentDef
        from tests.conftest import MockClient

        backend = object()
        agents = AgentFactory.create_all_configured(
            [AgentDef(name="a", tools_config={"builtin": ["generate_image"]})],
            MockClient(),
            media_provider=backend,
        )
        assert agents["a"].tools.get_dep("media_provider") is backend


# ── build_media transcription ──


class TestBuildMediaTranscription:
    def test_transcription_slot_populated(self, tmp_path):
        backend = build_media(
            {"enabled": True, "transcription": {"provider": "mock"}, "storage": {"dir": str(tmp_path)}}
        )
        assert backend is not None
        assert isinstance(backend.transcription, MockTranscriptionProvider)

    def test_no_transcription_when_absent(self, tmp_path):
        backend = build_media({"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path)}})
        assert backend is not None
        assert backend.transcription is None
