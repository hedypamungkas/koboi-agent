"""Tests for MEDIA coverage gaps -- Surplus provider edge cases, budget exhaustion, tool error paths, backend materialization failures."""

from __future__ import annotations

import base64
import json
from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest

from koboi.media.async_job import MediaJob
from koboi.media.backend import MediaBackend, build_media
from koboi.media.budget import (
    CountingMusicProvider,
    CountingSpeechProvider,
    CountingVideoProvider,
    MediaBudget,
)
from koboi.media.providers.surplus import (
    SurplusImageProvider,
    SurplusMusicProvider,
    SurplusSpeechProvider,
    SurplusTranscriptionProvider,
    SurplusVideoProvider,
)
from koboi.media.store import MediaStore
from koboi.media.types import MediaRequest, MediaResult, MediaUnit
from koboi.tools.builtin import register_all
from koboi.tools.registry import ToolRegistry


# ============================================================================
# Surplus provider tests -- request body building, response parsing, auth modes
# ============================================================================


class TestSurplusImageProviderRequestBuilding:
    async def test_size_quality_in_request_body(self):
        provider = SurplusImageProvider(api_key="test_key")
        provider._transport.post = AsyncMock(
            return_value={"data": [{"b64_json": base64.b64encode(b"x").decode()}], "usage": {}}
        )
        result = await provider.generate_image(MediaRequest(prompt="test", size="1024x1024", quality="high"))
        # Verify the request was successful
        assert result.status == "ok"
        # Verify the mock was called
        provider._transport.post.assert_called_once()
        # The first positional arg is the endpoint, second is the body
        args = provider._transport.post.call_args[0]
        assert args[0] == "/images/generations"
        body = args[1]
        assert body["size"] == "1024x1024"
        assert body["quality"] == "high"

    async def test_quality_parameter(self):
        provider = SurplusImageProvider(api_key="test_key")
        provider._transport.post = AsyncMock(
            return_value={"data": [{"b64_json": base64.b64encode(b"x").decode()}], "usage": {}}
        )
        result = await provider.generate_image(MediaRequest(prompt="test", quality="low"))
        assert result.status == "ok"
        args = provider._transport.post.call_args[0]
        body = args[1]
        assert body["quality"] == "low"

    async def test_idempotency_key_as_metadata(self):
        provider = SurplusImageProvider(api_key="test_key")
        provider._transport.post = AsyncMock(
            return_value={"data": [{"b64_json": base64.b64encode(b"x").decode()}], "usage": {}}
        )
        result = await provider.generate_image(MediaRequest(prompt="test", idempotency_key="test-key-123"))
        assert result.status == "ok"
        args = provider._transport.post.call_args[0]
        body = args[1]
        assert body["metadata"] == {"idempotency_key": "test-key-123"}


class TestSurplusImageProviderResponseParsing:
    async def test_b64_decode_error_returns_failed(self):
        provider = SurplusImageProvider(api_key="test_key")
        provider._transport.post = AsyncMock(return_value={"data": [{"b64_json": "invalid-base64!!"}], "usage": {}})
        result = await provider.generate_image(MediaRequest(prompt="test"))
        assert result.status == "failed"
        assert "failed to decode b64_json" in result.rejection_reason

    async def test_b64_type_error_returns_failed(self):
        provider = SurplusImageProvider(api_key="test_key")
        provider._transport.post = AsyncMock(return_value={"data": [{"b64_json": 12345}], "usage": {}})
        result = await provider.generate_image(MediaRequest(prompt="test"))
        assert result.status == "failed"
        assert "failed to decode b64_json" in result.rejection_reason

    async def test_missing_b64_and_url_returns_failed(self):
        provider = SurplusImageProvider(api_key="test_key")
        provider._transport.post = AsyncMock(return_value={"data": [{}], "usage": {}})
        result = await provider.generate_image(MediaRequest(prompt="test"))
        assert result.status == "failed"
        assert "neither b64_json nor url" in result.rejection_reason


class TestSurplusVideoProviderRequestBuilding:
    async def test_video_body_all_parameters(self):
        provider = SurplusVideoProvider(api_key="test_key")
        provider._transport.post = AsyncMock(
            return_value={"id": "job_123", "status": "queued", "poll_url": "http://poll", "cancel_url": "http://cancel"}
        )
        req = MediaRequest(
            modality="video",
            prompt="test video",
            aspect_ratio="16:9",
            duration_seconds=10.0,
            audio=True,
            input_images=["http://image.jpg"],
            end_image_url="http://end.jpg",
            webhook_url="http://webhook",
            idempotency_key="video-key",
        )
        await provider.submit_video(req)
        provider._transport.post.assert_called_once()
        args = provider._transport.post.call_args[0]
        body = args[1]
        assert body["aspect_ratio"] == "16:9"
        assert body["duration_seconds"] == 10.0
        assert body["audio"] is True
        assert body["image_url"] == "http://image.jpg"
        assert body["end_image_url"] == "http://end.jpg"
        assert body["webhook_url"] == "http://webhook"
        assert body["metadata"] == {"idempotency_key": "video-key"}

    async def test_audio_false_in_body(self):
        provider = SurplusVideoProvider(api_key="test_key")
        provider._transport.post = AsyncMock(
            return_value={"id": "job_123", "status": "queued", "poll_url": "http://poll", "cancel_url": "http://cancel"}
        )
        req = MediaRequest(modality="video", prompt="test", audio=False)
        await provider.submit_video(req)
        args = provider._transport.post.call_args[0]
        body = args[1]
        assert body["audio"] is False

    async def test_input_image_url_first_element(self):
        provider = SurplusVideoProvider(api_key="test_key")
        provider._transport.post = AsyncMock(
            return_value={"id": "job_123", "status": "queued", "poll_url": "http://poll", "cancel_url": "http://cancel"}
        )
        req = MediaRequest(modality="video", prompt="test", input_images=["url1", "url2"])
        await provider.submit_video(req)
        args = provider._transport.post.call_args[0]
        body = args[1]
        assert body["image_url"] == "url1"


class TestSurplusMusicProviderRequestBuilding:
    async def test_music_body_all_parameters(self):
        provider = SurplusMusicProvider(api_key="test_key")
        provider._transport.post = AsyncMock(
            return_value={"id": "job_456", "status": "queued", "poll_url": "http://poll", "cancel_url": "http://cancel"}
        )
        req = MediaRequest(
            modality="music",
            prompt="test music",
            duration_seconds=30.0,
            lyrics_prompt="lyrics here",
            force_instrumental=True,
            voice="male",
            language_code="en",
            webhook_url="http://webhook",
            idempotency_key="music-key",
        )
        await provider.submit_music(req)
        args = provider._transport.post.call_args[0]
        body = args[1]
        assert body["duration_seconds"] == 30.0
        assert body["lyrics_prompt"] == "lyrics here"
        assert body["force_instrumental"] is True
        assert body["voice"] == "male"
        assert body["language_code"] == "en"
        assert body["webhook_url"] == "http://webhook"
        assert body["metadata"] == {"idempotency_key": "music-key"}

    async def test_force_instrumental_false(self):
        provider = SurplusMusicProvider(api_key="test_key")
        provider._transport.post = AsyncMock(
            return_value={"id": "job_456", "status": "queued", "poll_url": "http://poll", "cancel_url": "http://cancel"}
        )
        req = MediaRequest(modality="music", prompt="test", force_instrumental=False)
        await provider.submit_music(req)
        args = provider._transport.post.call_args[0]
        body = args[1]
        assert body["force_instrumental"] is False


class TestSurplusDownloadUrlExtraction:
    async def test_download_url_from_results_array(self):
        from koboi.media.providers.surplus import _download_url_from_raw

        data = {"results": [{"url": "http://result.jpg"}]}
        url = _download_url_from_raw(data)
        assert url == "http://result.jpg"

    async def test_download_url_from_artifacts_array(self):
        from koboi.media.providers.surplus import _download_url_from_raw

        data = {"artifacts": [{"download_url": "http://artifact.mp4"}]}
        url = _download_url_from_raw(data)
        assert url == "http://artifact.mp4"

    async def test_download_url_empty_string_when_no_url(self):
        from koboi.media.providers.surplus import _download_url_from_raw

        data = {"other_field": "value"}
        url = _download_url_from_raw(data)
        assert url == ""

    async def test_download_url_from_non_list_results(self):
        from koboi.media.providers.surplus import _download_url_from_raw

        data = {"results": "not-a-list"}
        url = _download_url_from_raw(data)
        assert url == ""

    async def test_download_url_from_empty_list(self):
        from koboi.media.providers.surplus import _download_url_from_raw

        data = {"results": []}
        url = _download_url_from_raw(data)
        assert url == ""

    async def test_download_url_from_list_with_dict_no_url(self):
        from koboi.media.providers.surplus import _download_url_from_raw

        data = {"results": [{"other": "value"}]}
        url = _download_url_from_raw(data)
        assert url == ""


class TestSurplusProviderAuthModes:
    def test_video_provider_unsupported_auth_mode(self):
        with pytest.raises(NotImplementedError, match="surplus auth_mode 'x402' not implemented"):
            SurplusVideoProvider(api_key="test", auth_mode="x402")

    def test_music_provider_unsupported_auth_mode(self):
        with pytest.raises(NotImplementedError, match="surplus auth_mode 'mpp' not implemented"):
            SurplusMusicProvider(api_key="test", auth_mode="mpp")

    def test_speech_provider_unsupported_auth_mode(self):
        with pytest.raises(NotImplementedError, match="surplus auth_mode 'oauth2' not implemented"):
            SurplusSpeechProvider(api_key="test", auth_mode="oauth2")

    def test_transcription_provider_unsupported_auth_mode(self):
        with pytest.raises(NotImplementedError, match="surplus auth_mode 'bearer2' not implemented"):
            SurplusTranscriptionProvider(api_key="test", auth_mode="bearer2")


class TestSurplusSpeechProvider:
    async def test_speech_voice_with_vv_prefix_sent(self):
        provider = SurplusSpeechProvider(api_key="test_key")
        provider._transport.post_bytes = AsyncMock(return_value=b"mp3-data")
        await provider.synthesize_speech(MediaRequest(prompt="hello", voice="vv_kokoro"))
        args = provider._transport.post_bytes.call_args[0]
        body = args[1]
        assert body["voice"] == "vv_kokoro"

    async def test_speech_non_vv_voice_not_sent(self):
        provider = SurplusSpeechProvider(api_key="test_key")
        provider._transport.post_bytes = AsyncMock(return_value=b"mp3-data")
        await provider.synthesize_speech(MediaRequest(prompt="hello", voice="alloy"))
        args = provider._transport.post_bytes.call_args[0]
        body = args[1]
        assert "voice" not in body

    async def test_speech_speed_sent(self):
        provider = SurplusSpeechProvider(api_key="test_key")
        provider._transport.post_bytes = AsyncMock(return_value=b"mp3-data")
        await provider.synthesize_speech(MediaRequest(prompt="hello", speed=1.5))
        args = provider._transport.post_bytes.call_args[0]
        body = args[1]
        assert body["speed"] == 1.5

    async def test_speech_language_sent(self):
        provider = SurplusSpeechProvider(api_key="test_key")
        provider._transport.post_bytes = AsyncMock(return_value=b"mp3-data")
        await provider.synthesize_speech(MediaRequest(prompt="hello", language_code="en"))
        args = provider._transport.post_bytes.call_args[0]
        body = args[1]
        assert body["language"] == "en"


class TestSurplusTranscriptionProvider:
    async def test_transcribe_with_language(self):
        provider = SurplusTranscriptionProvider(api_key="test_key")
        provider._transport.post_form = AsyncMock(return_value={"text": "hello world"})
        await provider.transcribe(b"audio-data", language_code="en")
        data = provider._transport.post_form.call_args[1]["data"]
        assert data["language"] == "en"

    async def test_transcribe_with_prompt(self):
        provider = SurplusTranscriptionProvider(api_key="test_key")
        provider._transport.post_form = AsyncMock(return_value={"text": "hello world"})
        await provider.transcribe(b"audio-data", prompt="context hint")
        data = provider._transport.post_form.call_args[1]["data"]
        assert data["prompt"] == "context hint"

    async def test_transcribe_with_custom_model(self):
        provider = SurplusTranscriptionProvider(api_key="test_key")
        provider._transport.post_form = AsyncMock(return_value={"text": "hello world"})
        await provider.transcribe(b"audio-data", model="whisper-medium")
        data = provider._transport.post_form.call_args[1]["data"]
        assert data["model"] == "whisper-medium"

    async def test_transcribe_returns_text_field(self):
        provider = SurplusTranscriptionProvider(api_key="test_key")
        provider._transport.post_form = AsyncMock(return_value={"text": "transcribed text"})
        result = await provider.transcribe(b"audio-data")
        assert result == "transcribed text"

    async def test_transcribe_missing_text_returns_empty_string(self):
        provider = SurplusTranscriptionProvider(api_key="test_key")
        provider._transport.post_form = AsyncMock(return_value={"other": "value"})
        result = await provider.transcribe(b"audio-data")
        assert result == ""


# ============================================================================
# Media tool tests -- error paths, model_config, URL fetch failures
# ============================================================================


class TestMediaToolsModelConfig:
    async def test_generate_image_with_model_from_tool_config(self, tmp_path):
        # Test that model from tool_config is passed through (line 58)
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.generate_image = AsyncMock(return_value=MediaResult(request_id="x", modality="image", status="ok"))
        registry.set_dep("media_provider", backend)

        # Call the tool function directly with _tool_config
        from koboi.tools.builtin.media import generate_image

        result = await generate_image(
            prompt="cat", _deps={"media_provider": backend}, _tool_config={"image_model": "venice-z-image-turbo"}
        )
        assert "Image saved" in result
        # Verify the model was passed in the request
        backend.generate_image.assert_called_once()
        req = backend.generate_image.call_args[0][0]
        assert req.model == "venice-z-image-turbo"


class TestMediaToolsExceptionHandling:
    async def test_generate_image_exception_returns_error_string(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.generate_image = AsyncMock(side_effect=RuntimeError("provider crashed"))
        registry.set_dep("media_provider", backend)

        result = await registry.execute("generate_image", json.dumps({"prompt": "cat"}))
        assert result.startswith("Error: image generation failed")

    async def test_generate_video_exception_returns_error_string(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.generate_video = AsyncMock(side_effect=ValueError("video error"))
        registry.set_dep("media_provider", backend)

        result = await registry.execute("generate_video", json.dumps({"prompt": "test video"}))
        assert result.startswith("Error: video generation failed")

    async def test_generate_music_backend_not_configured(self):
        registry = ToolRegistry()
        register_all(registry)

        result = await registry.execute("generate_music", json.dumps({"prompt": "test music"}))
        assert "media not configured" in result

    async def test_generate_music_exception_returns_error_string(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.generate_music = AsyncMock(side_effect=Exception("music crashed"))
        registry.set_dep("media_provider", backend)

        result = await registry.execute("generate_music", json.dumps({"prompt": "test music"}))
        assert result.startswith("Error: music generation failed")

    async def test_generate_speech_exception_returns_error_string(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.synthesize_speech = AsyncMock(side_effect=RuntimeError("TTS error"))
        registry.set_dep("media_provider", backend)

        result = await registry.execute("generate_speech", json.dumps({"prompt": "hello"}))
        assert result.startswith("Error: speech generation failed")


class TestTranscribeAudioTool:
    async def test_transcribe_from_url_fetch_exception(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock, patch

        backend = Mock()
        registry.set_dep("media_provider", backend)

        with patch("koboi.tools.builtin.media.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = httpx.HTTPError("network error")
            result = await registry.execute("transcribe_audio", json.dumps({"url": "http://audio.mp3"}))
            assert "failed to fetch audio" in result

    async def test_transcribe_from_file_read_exception(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        registry.set_dep("media_provider", backend)

        result = await registry.execute("transcribe_audio", json.dumps({"file_path": "/nonexistent/audio.mp3"}))
        assert "failed to read" in result

    async def test_transcribe_neither_file_nor_url(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        registry.set_dep("media_provider", backend)

        result = await registry.execute("transcribe_audio", json.dumps({}))
        assert "provide 'file_path' or 'url'" in result

    async def test_transcribe_backend_exception_returns_error(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.transcribe = AsyncMock(side_effect=RuntimeError("STT crashed"))
        registry.set_dep("media_provider", backend)

        # Create a real temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"audio")
            temp_path = f.name

        try:
            result = await registry.execute("transcribe_audio", json.dumps({"file_path": temp_path}))
            assert "Error: transcription failed" in result
        finally:
            import os

            os.unlink(temp_path)

    async def test_transcribe_empty_text_returns_fallback(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.transcribe = AsyncMock(return_value="")
        registry.set_dep("media_provider", backend)

        # Create a real temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"audio")
            temp_path = f.name

        try:
            result = await registry.execute("transcribe_audio", json.dumps({"file_path": temp_path}))
            assert result == "(no speech transcribed)"
        finally:
            import os

            os.unlink(temp_path)


class TestMediaJobTools:
    async def test_submit_media_job_not_configured(self):
        registry = ToolRegistry()
        register_all(registry)

        result = await registry.execute("submit_media_job", json.dumps({"modality": "video", "prompt": "test"}))
        assert "media not configured" in result

    async def test_submit_media_job_exception_returns_error(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.submit_media_job = AsyncMock(side_effect=ValueError("submit failed"))
        registry.set_dep("media_provider", backend)

        result = await registry.execute("submit_media_job", json.dumps({"modality": "video", "prompt": "test"}))
        assert "Error: submit failed" in result

    async def test_check_media_job_not_configured(self):
        registry = ToolRegistry()
        register_all(registry)

        result = await registry.execute("check_media_job", json.dumps({"job_id": "job_123"}))
        assert "media not configured" in result

    async def test_check_media_job_exception_returns_error(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.check_media_job = AsyncMock(side_effect=RuntimeError("poll crashed"))
        registry.set_dep("media_provider", backend)

        result = await registry.execute("check_media_job", json.dumps({"job_id": "job_123"}))
        assert "Error: check failed" in result

    async def test_check_media_job_none_returns_message(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        backend.check_media_job = AsyncMock(return_value=None)
        registry.set_dep("media_provider", backend)

        result = await registry.execute("check_media_job", json.dumps({"job_id": "job_123"}))
        assert "No media job with id=job_123" in result

    async def test_check_media_job_succeeded_with_path(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        job = MediaJob(
            job_id="job_123",
            kind="video",
            status="succeeded",
            result=MediaResult(request_id="x", modality="video", status="ok", local_path="/tmp/video.mp4"),
        )
        backend.check_media_job = AsyncMock(return_value=job)
        registry.set_dep("media_provider", backend)

        result = await registry.execute("check_media_job", json.dumps({"job_id": "job_123"}))
        assert "succeeded -> /tmp/video.mp4" in result

    async def test_check_media_job_status_string(self):
        registry = ToolRegistry()
        register_all(registry)
        from unittest.mock import Mock

        backend = Mock()
        job = MediaJob(job_id="job_123", kind="video", status="running")
        backend.check_media_job = AsyncMock(return_value=job)
        registry.set_dep("media_provider", backend)

        result = await registry.execute("check_media_job", json.dumps({"job_id": "job_123"}))
        assert "running" in result


# ============================================================================
# Budget tests -- exhaustion paths, delegated methods
# ============================================================================


class TestCountingVideoProviderDelegatedMethods:
    async def test_submit_video_delegates(self):
        from koboi.media.providers.mock import MockVideoProvider

        inner = MockVideoProvider()
        inner.submit_video = AsyncMock(return_value=MediaJob(job_id="job_1", kind="video", status="queued"))
        budget = MediaBudget()
        provider = CountingVideoProvider(inner, budget)

        req = MediaRequest(modality="video", prompt="test")
        job = await provider.submit_video(req)
        assert job.job_id == "job_1"
        inner.submit_video.assert_called_once_with(req)

    async def test_poll_video_delegates(self):
        from koboi.media.providers.mock import MockVideoProvider

        inner = MockVideoProvider()
        job = MediaJob(job_id="job_1", kind="video", status="running")
        inner.poll_video = AsyncMock(return_value=job)
        budget = MediaBudget()
        provider = CountingVideoProvider(inner, budget)

        result = await provider.poll_video(job)
        assert result.status == "running"
        inner.poll_video.assert_called_once_with(job)

    async def test_fetch_video_artifact_delegates(self):
        from koboi.media.providers.mock import MockVideoProvider

        inner = MockVideoProvider()
        job = MediaJob(job_id="job_1", kind="video", status="succeeded")
        inner.fetch_video_artifact = AsyncMock(return_value=b"video-data")
        budget = MediaBudget()
        provider = CountingVideoProvider(inner, budget)

        data = await provider.fetch_video_artifact(job)
        assert data == b"video-data"
        inner.fetch_video_artifact.assert_called_once_with(job)

    async def test_cancel_video_delegates(self):
        from koboi.media.providers.mock import MockVideoProvider

        inner = MockVideoProvider()
        job = MediaJob(job_id="job_1", kind="video", status="running")
        inner.cancel_video = AsyncMock()
        budget = MediaBudget()
        provider = CountingVideoProvider(inner, budget)

        await provider.cancel_video(job)
        inner.cancel_video.assert_called_once_with(job)


class TestCountingMusicProviderDelegatedMethods:
    async def test_submit_music_delegates(self):
        from koboi.media.providers.mock import MockMusicProvider

        inner = MockMusicProvider()
        inner.submit_music = AsyncMock(return_value=MediaJob(job_id="job_1", kind="music", status="queued"))
        budget = MediaBudget()
        provider = CountingMusicProvider(inner, budget)

        req = MediaRequest(modality="music", prompt="test")
        job = await provider.submit_music(req)
        assert job.job_id == "job_1"
        inner.submit_music.assert_called_once_with(req)

    async def test_poll_music_delegates(self):
        from koboi.media.providers.mock import MockMusicProvider

        inner = MockMusicProvider()
        job = MediaJob(job_id="job_1", kind="music", status="running")
        inner.poll_music = AsyncMock(return_value=job)
        budget = MediaBudget()
        provider = CountingMusicProvider(inner, budget)

        result = await provider.poll_music(job)
        assert result.status == "running"
        inner.poll_music.assert_called_once_with(job)

    async def test_fetch_music_artifact_delegates(self):
        from koboi.media.providers.mock import MockMusicProvider

        inner = MockMusicProvider()
        job = MediaJob(job_id="job_1", kind="music", status="succeeded")
        inner.fetch_music_artifact = AsyncMock(return_value=b"music-data")
        budget = MediaBudget()
        provider = CountingMusicProvider(inner, budget)

        data = await provider.fetch_music_artifact(job)
        assert data == b"music-data"
        inner.fetch_music_artifact.assert_called_once_with(job)

    async def test_cancel_music_delegates(self):
        from koboi.media.providers.mock import MockMusicProvider

        inner = MockMusicProvider()
        job = MediaJob(job_id="job_1", kind="music", status="running")
        inner.cancel_music = AsyncMock()
        budget = MediaBudget()
        provider = CountingMusicProvider(inner, budget)

        await provider.cancel_music(job)
        inner.cancel_music.assert_called_once_with(job)


class TestCountingSpeechProviderBudgetExhaustion:
    async def test_exhausted_budget_returns_rejected(self):
        from koboi.media.providers.mock import MockSpeechProvider

        inner = MockSpeechProvider()
        budget = MediaBudget(max_speech_chars=10)
        budget.used_speech_chars = 10  # Exhausted
        provider = CountingSpeechProvider(inner, budget)

        req = MediaRequest(modality="speech", prompt="test")
        result = await provider.synthesize_speech(req)
        assert result.status == "rejected"
        assert "budget exhausted" in result.rejection_reason

    async def test_budget_records_speech_chars(self):
        from koboi.media.providers.mock import MockSpeechProvider

        inner = MockSpeechProvider()
        inner.synthesize_speech = AsyncMock(
            return_value=MediaResult(
                request_id="x",
                modality="speech",
                status="ok",
                billing_unit=MediaUnit.CHAR,
                billing_quantity=50.0,
            )
        )
        budget = MediaBudget()
        provider = CountingSpeechProvider(inner, budget)

        req = MediaRequest(modality="speech", prompt="test text for speech synthesis")
        await provider.synthesize_speech(req)
        assert budget.used_speech_chars == 50

    async def test_close_delegates(self):
        from koboi.media.providers.mock import MockSpeechProvider

        inner = MockSpeechProvider()
        inner.close = AsyncMock()
        budget = MediaBudget()
        provider = CountingSpeechProvider(inner, budget)

        await provider.close()
        inner.close.assert_called_once()


# ============================================================================
# MediaBackend tests -- configuration errors, materialization failures, NotImplementedError fallback
# ============================================================================


class TestMediaBackendConfiguration:
    async def test_music_not_configured(self):
        backend = MediaBackend(music=None, store=None)
        result = await backend.generate_music(MediaRequest(modality="music", prompt="test"))
        assert result.status == "failed"
        assert "not configured" in result.rejection_reason

    async def test_transcribe_raises_runtime_error(self):
        backend = MediaBackend(transcription=None, store=None)
        with pytest.raises(RuntimeError, match="transcription not configured"):
            await backend.transcribe(b"audio-data")


class TestMediaBackendMaterialization:
    async def test_materialization_failure_logs_warning(self, tmp_path, caplog):

        store = MediaStore(backend="local", dir=str(tmp_path))
        store.save = AsyncMock(side_effect=RuntimeError("disk full"))

        from koboi.media.providers.mock import MockImageProvider

        backend = MediaBackend(image=MockImageProvider(), store=store)
        result = await backend.generate_image(MediaRequest(prompt="test"))

        # Result should still be ok, but materialization failed
        assert result.status == "ok"
        assert any("media artifact materialization failed" in record.message for record in caplog.records)

    async def test_store_none_skips_materialization(self):
        from koboi.media.providers.mock import MockImageProvider

        backend = MediaBackend(image=MockImageProvider(), store=None)
        result = await backend.generate_image(MediaRequest(prompt="test"))
        assert result.status == "ok"
        assert result.local_path is None


class TestMediaBackendAsyncJobs:
    async def test_check_media_job_unknown_kind_no_provider(self, tmp_path):
        backend = MediaBackend(store=None)
        job = MediaJob(job_id="job_1", kind="unknown", status="running")
        backend._jobs["job_1"] = job

        result = await backend.check_media_job("job_1")
        assert result.status == "running"  # Unchanged, no provider to poll

    async def test_check_media_job_provider_none_returns_job(self, tmp_path):
        backend = MediaBackend(video=None, store=None)
        job = MediaJob(job_id="job_1", kind="video", status="running")
        backend._jobs["job_1"] = job

        result = await backend.check_media_job("job_1")
        assert result == job

    async def test_check_media_job_polls_video_provider(self, tmp_path):
        from koboi.media.providers.mock import MockVideoProvider

        video = MockVideoProvider()
        job = MediaJob(job_id="job_1", kind="video", status="running")
        video.poll_video = AsyncMock(return_value=MediaJob(job_id="job_1", kind="video", status="succeeded"))
        video.fetch_video_artifact = AsyncMock(return_value=b"video-data")
        backend = MediaBackend(video=video, store=None)
        backend._jobs["job_1"] = job

        result = await backend.check_media_job("job_1")
        assert result.status == "succeeded"
        video.poll_video.assert_called_once()
        video.fetch_video_artifact.assert_called_once()

    async def test_check_media_job_fetch_failure_marks_failed(self, tmp_path):
        from koboi.media.providers.mock import MockVideoProvider

        video = MockVideoProvider()
        job = MediaJob(job_id="job_1", kind="video", status="running")
        video.poll_video = AsyncMock(return_value=MediaJob(job_id="job_1", kind="video", status="succeeded"))
        video.fetch_video_artifact = AsyncMock(side_effect=RuntimeError("fetch failed"))
        backend = MediaBackend(video=video, store=None)
        backend._jobs["job_1"] = job

        result = await backend.check_media_job("job_1")
        # When fetch fails, job.status is set to failed
        assert result.status == "failed"
        # job.result may be None if the poll didn't set it, the rejection_reason is only set if job.result is not None
        # This tests lines 165-169 in backend.py
        assert result.job_id == "job_1"


class TestMediaBackendBuildMediaStorageFallback:
    def test_r2_not_implemented_falls_back_to_local(self, tmp_path):
        from unittest.mock import patch

        # Mock boto3 import to fail
        with patch.dict("sys.modules", {"boto3": None}):
            backend = build_media(
                {
                    "enabled": True,
                    "image": {"provider": "mock"},
                    "storage": {"backend": "r2", "bucket": "test", "endpoint_url": "http://r2", "dir": str(tmp_path)},
                }
            )

            assert backend is not None
            assert backend.store is not None
            assert backend.store._backend == "local"

    def test_s3_not_implemented_falls_back_to_local(self, tmp_path):
        from unittest.mock import patch

        # Mock boto3 import to fail
        with patch.dict("sys.modules", {"boto3": None}):
            backend = build_media(
                {
                    "enabled": True,
                    "image": {"provider": "mock"},
                    "storage": {"backend": "s3", "bucket": "test", "dir": str(tmp_path)},
                }
            )

            assert backend is not None
            assert backend.store is not None
            assert backend.store._backend == "local"


class TestMediaBackendCloseErrors:
    async def test_close_logs_provider_errors(self, tmp_path):
        from koboi.media.providers.mock import MockImageProvider

        image = MockImageProvider()
        image.close = AsyncMock(side_effect=RuntimeError("close failed"))
        backend = MediaBackend(image=image, store=None)

        # Should not raise, just log
        await backend.close()
        image.close.assert_awaited()

    async def test_close_store_exception_propagates(self, tmp_path):
        # Note: The store.close() is NOT wrapped in try/except like provider.close()
        # This is intentional since store.close() should be safe
        from koboi.media.providers.mock import MockImageProvider

        store = MediaStore(backend="local", dir=str(tmp_path))
        store.close = AsyncMock(side_effect=RuntimeError("store close failed"))
        backend = MediaBackend(image=MockImageProvider(), store=store)

        # Exception should propagate (lines 191 not wrapped)
        with pytest.raises(RuntimeError, match="store close failed"):
            await backend.close()


# ============================================================================
# MediaBudget tests -- remaining() method edge cases
# ============================================================================


class TestMediaBudgetRemaining:
    def test_remaining_image_max_images(self):
        budget = MediaBudget(max_images=5)
        budget.used_images = 5
        assert not budget.remaining("image")

    def test_remaining_video_max_seconds(self):
        budget = MediaBudget(max_video_seconds=60.0)
        budget.used_video_seconds = 60.0
        assert not budget.remaining("video")

    def test_remaining_music_max_seconds(self):
        budget = MediaBudget(max_music_seconds=120.0)
        budget.used_music_seconds = 120.0
        assert not budget.remaining("music")

    def test_remaining_speech_max_chars(self):
        budget = MediaBudget(max_speech_chars=1000)
        budget.used_speech_chars = 1000
        assert not budget.remaining("speech")

    def test_remaining_with_est_cost(self):
        budget = MediaBudget(max_cost_usd=10.0)
        budget.used_cost_usd = Decimal("8.0")
        assert not budget.remaining("image", est_cost=2.1)
        assert budget.remaining("image", est_cost=1.9)
