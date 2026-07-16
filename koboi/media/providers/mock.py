"""koboi/media/providers/mock.py -- offline deterministic image provider (default).

No network. Returns a fixed tiny PNG for any prompt so the full tool pipeline
(risk/approval/audit/journal) can be exercised offline and tests stay deterministic.
Mirrors ``koboi.websearch.providers.mock`` (offline default).
"""

from __future__ import annotations

import base64
import uuid
from decimal import Decimal

from koboi.media.async_job import MediaJob
from koboi.media.base import (
    BaseImageProvider,
    BaseMusicProvider,
    BaseSpeechProvider,
    BaseTranscriptionProvider,
    BaseVideoProvider,
)
from koboi.media.registry import (
    register_image_provider,
    register_music_provider,
    register_speech_provider,
    register_transcription_provider,
    register_video_provider,
)
from koboi.media.types import MediaRequest, MediaResult, MediaUnit

# Minimal valid 1x1 transparent PNG (well-known constant).
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@register_image_provider("mock", description="Offline deterministic placeholder image (default; no network)")
class MockImageProvider(BaseImageProvider):
    """Offline placeholder. Returns ``_PNG_1X1`` regardless of prompt."""

    def __init__(self, model: str = "mock-image", width: int = 1, height: int = 1) -> None:
        self._model = model
        self._width = width
        self._height = height

    async def generate_image(self, req: MediaRequest) -> MediaResult:
        n = max(1, int(req.n or 1))
        return MediaResult(
            request_id=req.idempotency_key or uuid.uuid4().hex,
            modality="image",
            status="ok",
            data=_PNG_1X1,
            content_type="image/png",
            width=self._width,
            height=self._height,
            cost_usd=Decimal("0"),
            billing_unit=MediaUnit.IMAGE,
            billing_quantity=float(n),
            model=req.model or self._model,
        )


@register_video_provider("mock", description="Offline deterministic placeholder video (default; no network)")
class MockVideoProvider(BaseVideoProvider):
    """Offline placeholder. ``submit_video`` returns an already-succeeded job."""

    _PLACEHOLDER = b"mock-video-artifact"

    def __init__(
        self,
        model: str = "mock-video",
        duration_seconds: float = 2.0,
        width: int = 320,
        height: int = 240,
    ) -> None:
        self._model = model
        self._duration = duration_seconds
        self._width = width
        self._height = height

    def _result(self, req: MediaRequest) -> MediaResult:
        duration = float(req.duration_seconds or self._duration)
        return MediaResult(
            request_id=req.idempotency_key or "mock",
            modality="video",
            status="ok",
            content_type="video/mp4",
            duration_seconds=duration,
            width=self._width,
            height=self._height,
            cost_usd=Decimal("0"),
            billing_unit=MediaUnit.JOB,
            billing_quantity=1,
            model=req.model or self._model,
        )

    async def submit_video(self, req: MediaRequest) -> MediaJob:
        return MediaJob(job_id="mock-video-job", kind="video", status="succeeded", result=self._result(req))

    async def poll_video(self, job: MediaJob) -> MediaJob:
        return job

    async def fetch_video_artifact(self, job: MediaJob) -> bytes:
        return self._PLACEHOLDER

    async def cancel_video(self, job: MediaJob) -> None:
        return None


@register_music_provider("mock", description="Offline deterministic placeholder music (default; no network)")
class MockMusicProvider(BaseMusicProvider):
    """Offline placeholder. ``submit_music`` returns an already-succeeded job."""

    _PLACEHOLDER = b"mock-music-artifact"

    def __init__(self, model: str = "mock-music", duration_seconds: float = 4.0) -> None:
        self._model = model
        self._duration = duration_seconds

    def _result(self, req: MediaRequest) -> MediaResult:
        duration = float(req.duration_seconds or self._duration)
        return MediaResult(
            request_id=req.idempotency_key or "mock",
            modality="music",
            status="ok",
            content_type="audio/mpeg",
            duration_seconds=duration,
            cost_usd=Decimal("0"),
            billing_unit=MediaUnit.JOB,
            billing_quantity=1,
            model=req.model or self._model,
        )

    async def submit_music(self, req: MediaRequest) -> MediaJob:
        return MediaJob(job_id="mock-music-job", kind="music", status="succeeded", result=self._result(req))

    async def poll_music(self, job: MediaJob) -> MediaJob:
        return job

    async def fetch_music_artifact(self, job: MediaJob) -> bytes:
        return self._PLACEHOLDER

    async def cancel_music(self, job: MediaJob) -> None:
        return None


@register_speech_provider("mock", description="Offline deterministic placeholder speech (default; no network)")
class MockSpeechProvider(BaseSpeechProvider):
    """Offline placeholder. Returns a fixed audio byte string."""

    _PLACEHOLDER = b"mock-speech-audio"

    def __init__(self, model: str = "mock-tts") -> None:
        self._model = model

    async def synthesize_speech(self, req: MediaRequest) -> MediaResult:
        return MediaResult(
            request_id=req.idempotency_key or "mock",
            modality="speech",
            status="ok",
            data=self._PLACEHOLDER,
            content_type="audio/mpeg",
            cost_usd=Decimal("0"),
            billing_unit=MediaUnit.CHAR,
            billing_quantity=float(len(req.prompt or "")),
            model=req.model or self._model,
        )


@register_transcription_provider(
    "mock", description="Offline deterministic transcription placeholder (default; no network)"
)
class MockTranscriptionProvider(BaseTranscriptionProvider):
    """Offline placeholder. Returns a deterministic stub string."""

    def __init__(self, model: str = "mock-stt") -> None:
        self._model = model

    async def transcribe(
        self,
        audio: bytes,
        *,
        language_code: str | None = None,
        prompt: str | None = None,
        model: str | None = None,
    ) -> str:
        return f"[mock transcription of {len(audio)} bytes]"
