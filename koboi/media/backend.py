"""koboi/media/backend.py -- MediaBackend facade + ``build_media`` (W0 image + W1 video/music).

The single ``media_provider`` dep is a ``MediaBackend`` that dispatches per modality. This is the
object behind the future ``agent.media.generate(req)`` programmatic API (W5) and the
``generate_image``/``generate_video``/``generate_music`` tools' ``deps=["media_provider"]`` (W0/W1).

``build_media`` composes the runtime from the ``media:`` YAML section:

  * an image provider (always built; mock default),
  * video / music providers only when ``media.video`` / ``media.music`` are present,
  * an optional ``Counting*Provider`` wrapper around each when ``media.budget`` caps are set,
  * a ``MediaStore`` (``media.storage``) that materializes short-lived gateway URIs to disk.

Returns ``None`` when ``media.enabled`` is false (opt-in); the tools then report
"media not configured".
"""

from __future__ import annotations

import logging
import uuid

from koboi.media.async_job import MediaJob, _TERMINAL
from koboi.media.base import (
    BaseImageProvider,
    BaseMusicProvider,
    BaseSpeechProvider,
    BaseTranscriptionProvider,
    BaseVideoProvider,
)
from koboi.media.budget import (
    CountingImageProvider,
    CountingMusicProvider,
    CountingSpeechProvider,
    CountingVideoProvider,
)
from koboi.media.registry import (
    build_image_provider,
    build_music_provider,
    build_speech_provider,
    build_transcription_provider,
    build_video_provider,
    load_custom_components,
)
from koboi.media.store import MediaStore
from koboi.media.types import MediaBudget, MediaRequest, MediaResult

_logger = logging.getLogger(__name__)


class MediaBackend:
    """Per-modality dispatcher + artifact materialization. The ``media_provider`` dep."""

    def __init__(
        self,
        image: BaseImageProvider | None = None,
        video: BaseVideoProvider | None = None,
        music: BaseMusicProvider | None = None,
        speech: BaseSpeechProvider | None = None,
        transcription: BaseTranscriptionProvider | None = None,
        store: MediaStore | None = None,
    ) -> None:
        self.image = image
        self.video = video
        self.music = music
        self.speech = speech
        self.transcription = transcription
        self.store = store
        self._jobs: dict[str, MediaJob] = {}

    async def generate_image(self, req: MediaRequest) -> MediaResult:
        """Generate an image, then materialize the artifact before returning."""
        if self.image is None:
            return _not_configured(req, "image")
        return await self._materialize(await self.image.generate_image(req))

    async def generate_video(self, req: MediaRequest) -> MediaResult:
        """Generate a video (blocking submit->poll->fetch), then materialize."""
        if self.video is None:
            return _not_configured(req, "video")
        return await self._materialize(await self.video.generate_video(req))

    async def generate_music(self, req: MediaRequest) -> MediaResult:
        """Generate music (blocking submit->poll->fetch), then materialize."""
        if self.music is None:
            return _not_configured(req, "music")
        return await self._materialize(await self.music.generate_music(req))

    async def generate_speech(self, req: MediaRequest) -> MediaResult:
        """Synthesize speech, then materialize the audio artifact."""
        if self.speech is None:
            return _not_configured(req, "speech")
        return await self._materialize(await self.speech.synthesize_speech(req))

    async def generate(self, req: MediaRequest) -> MediaResult:
        """Modality-agnostic dispatch (W5 programmatic API entry). Routes on ``req.modality``."""
        modality = (req.modality or "image").lower()
        if modality == "image":
            return await self.generate_image(req)
        if modality == "video":
            return await self.generate_video(req)
        if modality == "music":
            return await self.generate_music(req)
        if modality == "speech":
            return await self.generate_speech(req)
        return _not_configured(req, modality)

    async def transcribe(
        self,
        audio: bytes,
        *,
        language_code: str | None = None,
        prompt: str | None = None,
        model: str | None = None,
    ) -> str:
        """Transcribe audio bytes to text (STT). Raises if transcription is not configured."""
        if self.transcription is None:
            raise RuntimeError("transcription not configured (enable media.transcription)")
        return await self.transcription.transcribe(audio, language_code=language_code, prompt=prompt, model=model)

    async def submit_media_job(self, req: MediaRequest) -> MediaJob:
        """Non-blocking submit (W5c #4). Video/music -> provider submit (async job, returns pending);
        image/speech -> sync ``generate`` (fast), wrapped as a succeeded job so the surface is uniform.
        The returned ``MediaJob.job_id`` is polled via ``check_media_job``.
        """
        modality = (req.modality or "image").lower()
        if modality == "video" and self.video is not None:
            job = await self.video.submit_video(req)
        elif modality == "music" and self.music is not None:
            job = await self.music.submit_music(req)
        else:
            # image/speech are synchronous + fast: run now + wrap as a completed job.
            result = await self.generate(req)
            job = MediaJob(
                job_id=result.request_id or uuid.uuid4().hex,
                kind=modality,
                status="succeeded" if result.status == "ok" else "failed",
                result=result,
            )
        self._jobs[job.job_id] = job
        return job

    async def check_media_job(self, job_id: str) -> MediaJob | None:
        """Poll a submitted job once (W5c #4). Returns ``None`` if unknown; on success fetches +
        materializes the artifact (sets ``job.result.local_path``). Call repeatedly until terminal.
        """
        job = self._jobs.get(job_id)
        if job is None or job.status in _TERMINAL:
            return job
        provider, poll_name, fetch_name = {
            "video": (self.video, "poll_video", "fetch_video_artifact"),
            "music": (self.music, "poll_music", "fetch_music_artifact"),
        }.get(job.kind, (None, None, None))
        if provider is None:
            return job
        job = await getattr(provider, poll_name)(job)
        if job.status == "succeeded":
            try:
                data = await getattr(provider, fetch_name)(job)
                if job.result is None:
                    job.result = MediaResult(request_id=job_id, modality=job.kind)
                job.result.data = data
                job.result.status = "ok"
                job.result = await self._materialize(job.result)
            except Exception as e:  # noqa: BLE001 - best-effort finalize; surface as failed
                job.status = "failed"
                if job.result is not None:
                    job.result.status = "failed"
                    job.result.rejection_reason = f"artifact fetch failed: {e}"
        self._jobs[job_id] = job
        return job

    async def _materialize(self, result: MediaResult) -> MediaResult:
        """Persist the artifact to durable storage (gateway URIs expire) before returning."""
        if self.store is not None and result.status == "ok" and (result.data is not None or result.url):
            try:
                result.local_path = await self.store.save(result)
            except Exception as e:  # noqa: BLE001 - best-effort; the tool surfaces the raw result
                _logger.warning("media artifact materialization failed: %s", e)
        return result

    async def close(self) -> None:
        """Close every configured provider (HTTP transport) and the store."""
        for provider in (self.image, self.video, self.music, self.speech, self.transcription):
            if provider is not None:
                try:
                    await provider.close()
                except Exception as e:  # noqa: BLE001 - best-effort teardown
                    _logger.debug("media provider close failed: %s", e)
        if self.store is not None:
            await self.store.close()


def _not_configured(req: MediaRequest, modality: str) -> MediaResult:
    return MediaResult(
        request_id=req.idempotency_key or "",
        modality=modality,
        status="failed",
        rejection_reason=f"{modality} generation not configured (enable media.{modality})",
    )


def build_media(media_conf: dict | None) -> MediaBackend | None:
    """Compose a ``MediaBackend`` from the ``media:`` config. ``None`` when disabled.

    ``media_conf`` is the raw ``media:`` dict read via ``config.get("media", ...)``.
    """
    if not media_conf or not media_conf.get("enabled"):
        return None

    custom = media_conf.get("custom_modules") or []
    if custom:
        load_custom_components(custom)

    # ModelProfile: load config-declared profiles (extends the built-in registry).
    profiles_conf = media_conf.get("profiles") or []
    if profiles_conf:
        from koboi.media.model_profile import load_profiles_from_config

        load_profiles_from_config(profiles_conf)

    budget_conf = media_conf.get("budget") or {}
    budget: MediaBudget | None = None
    if budget_conf:
        budget = MediaBudget(
            max_cost_usd=float(budget_conf.get("max_cost_usd", 5.0)),
            max_images=int(budget_conf.get("max_images", 20)),
            max_video_seconds=float(budget_conf.get("max_video_seconds", 60.0)),
            max_music_seconds=float(budget_conf.get("max_music_seconds", 120.0)),
        )

    image = build_image_provider(media_conf)
    if budget is not None:
        image = CountingImageProvider(image, budget)

    video = build_video_provider(media_conf) if media_conf.get("video") else None
    if video is not None and budget is not None:
        video = CountingVideoProvider(video, budget)

    music = build_music_provider(media_conf) if media_conf.get("music") else None
    if music is not None and budget is not None:
        music = CountingMusicProvider(music, budget)

    speech = build_speech_provider(media_conf) if media_conf.get("speech") else None
    if speech is not None and budget is not None:
        speech = CountingSpeechProvider(speech, budget)

    # STT returns str (not MediaResult), so no counting proxy; relies on the USD ceiling + tool risk.
    transcription = build_transcription_provider(media_conf) if media_conf.get("transcription") else None

    storage_conf = media_conf.get("storage") or {}
    store_dir = storage_conf.get("dir", "./media_artifacts")
    store_kwargs: dict = {"backend": storage_conf.get("backend", "local"), "dir": store_dir}
    if store_kwargs["backend"] in ("r2", "s3"):
        store_kwargs.update(
            {
                "bucket": storage_conf.get("bucket"),
                "endpoint_url": storage_conf.get("endpoint_url"),
                "region": storage_conf.get("region"),
                "access_key_id": storage_conf.get("access_key_id"),
                "secret_access_key": storage_conf.get("secret_access_key"),
                "key_prefix": storage_conf.get("key_prefix"),
            }
        )
    try:
        store = MediaStore(**store_kwargs)
    except NotImplementedError as e:
        # Non-local backends (or a missing [media-cloud] extra) degrade to local storage.
        _logger.warning("%s; falling back to local storage at %s", e, store_dir)
        store = MediaStore(backend="local", dir=store_dir)

    return MediaBackend(image=image, video=video, music=music, speech=speech, transcription=transcription, store=store)
