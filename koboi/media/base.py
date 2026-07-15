"""koboi/media/base.py -- generation-provider ABCs (image modality, W0).

One ABC per capability. W0 ships ``BaseImageProvider`` only; ``BaseVideoProvider``
and ``BaseAudioProvider`` arrive in W1/W2. A provider class may implement several
ABCs -- the surplus gateway serves image/video/audio behind one provider class,
mirroring how ``koboi.websearch.providers.firecrawl`` registers both a search and a
fetch provider.

PROVENANCE + SAFETY CONTRACT (every provider must honor):
  * Never return a ``url`` without a populated ``url_expires_at`` -- gateway artifact
    links expire (Surplus: 15 min direct S3 / 3 h proxy). Materialize bytes via the
    ``MediaStore`` before any reference is persisted.
  * A content-filter rejection is a ``status="rejected"`` ``MediaResult`` carrying
    ``safety_blocked=True`` + ``rejection_reason`` -- NOT a raised exception.
  * Never echo untrusted prompt text verbatim into result metadata (prompt injection
    into downstream artifact pipelines).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from koboi.media.async_job import MediaJob, run_async_job
from koboi.media.types import MediaRequest, MediaResult


class BaseImageProvider(ABC):
    """text/image -> image. Synchronous on the surplus gateway."""

    @abstractmethod
    async def generate_image(self, req: MediaRequest) -> MediaResult:
        """Generate one or more images (``req.n``) from ``req.prompt``."""

    async def close(self) -> None:
        """Release HTTP transports. Default no-op; HTTP-backed providers override."""
        return None


class BaseVideoProvider(ABC):
    """text/image -> video (async job: submit -> poll -> fetch).

    Providers implement the three transport calls (``submit_video``/``poll_video``/
    ``fetch_video_artifact``); the bounded poll loop lives in ``async_job.run_async_job`` and is
    invoked by the concrete ``generate_video`` facade. ``poll_interval``/``max_wait`` are class-level
    defaults (subclasses override via instance attributes for config-driven tuning).
    """

    poll_interval: float = 5.0
    max_wait: float = 1500.0  # under Surplus's 30-min job expiry

    @abstractmethod
    async def submit_video(self, req: MediaRequest) -> MediaJob:
        """Submit a generation job; return the initial ``MediaJob`` (status typically ``queued``)."""

    @abstractmethod
    async def poll_video(self, job: MediaJob) -> MediaJob:
        """Refresh ``job.status`` (+ metadata on success). Called on the poll interval."""

    @abstractmethod
    async def fetch_video_artifact(self, job: MediaJob) -> bytes:
        """Download the terminal artifact bytes (called once on success)."""

    async def cancel_video(self, job: MediaJob) -> None:
        """Cancel an in-flight job. Optional; default no-op."""
        return None

    async def generate_video(self, req: MediaRequest) -> MediaResult:
        """Blocking facade: submit -> poll until terminal -> fetch artifact."""
        return await run_async_job(
            self.submit_video,
            self.poll_video,
            self.fetch_video_artifact,
            req,
            "video",
            poll_interval=self.poll_interval,
            max_wait=self.max_wait,
        )

    async def close(self) -> None:
        """Release HTTP transports. Default no-op; HTTP-backed providers override."""
        return None


class BaseMusicProvider(ABC):
    """text -> music/SFX (async job: submit -> poll -> fetch). Same shape as video."""

    poll_interval: float = 3.0
    max_wait: float = 480.0

    @abstractmethod
    async def submit_music(self, req: MediaRequest) -> MediaJob: ...

    @abstractmethod
    async def poll_music(self, job: MediaJob) -> MediaJob: ...

    @abstractmethod
    async def fetch_music_artifact(self, job: MediaJob) -> bytes: ...

    async def cancel_music(self, job: MediaJob) -> None:
        """Cancel an in-flight job. Optional; default no-op."""
        return None

    async def generate_music(self, req: MediaRequest) -> MediaResult:
        """Blocking facade: submit -> poll until terminal -> fetch artifact."""
        return await run_async_job(
            self.submit_music,
            self.poll_music,
            self.fetch_music_artifact,
            req,
            "music",
            poll_interval=self.poll_interval,
            max_wait=self.max_wait,
        )

    async def close(self) -> None:
        """Release HTTP transports. Default no-op; HTTP-backed providers override."""
        return None


class BaseSpeechProvider(ABC):
    """text -> speech (TTS). Synchronous; mirrors ``BaseImageProvider``."""

    @abstractmethod
    async def synthesize_speech(self, req: MediaRequest) -> MediaResult:
        """Synthesize audio bytes from ``req.prompt`` (the text to speak)."""

    async def close(self) -> None:
        """Release HTTP transports. Default no-op; HTTP-backed providers override."""
        return None


class BaseTranscriptionProvider(ABC):
    """audio bytes -> text (STT). Synchronous; returns transcribed text (not a MediaResult).

    Distinct from the generation ABCs: STT is audio->text analysis, so it returns ``str`` and
    does not flow through ``MediaResult``/materialization.
    """

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        language_code: str | None = None,
        prompt: str | None = None,
        model: str | None = None,
    ) -> str:
        """Transcribe ``audio`` bytes to text."""

    async def close(self) -> None:
        """Release HTTP transports. Default no-op; HTTP-backed providers override."""
        return None
