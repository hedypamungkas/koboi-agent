"""koboi/media/budget.py -- budget-metering wrapper for image providers (W0).

Wraps a real image provider to charge each call against a shared ``MediaBudget``.
When the budget is exhausted, returns a ``status="rejected"`` result instead of
calling the (billed) inner provider -- fail-soft, never raises. Mirrors
``koboi.websearch.providers.counting``. Not registered (a wrapper, not a selectable
provider); ``MediaBackend`` constructs it around the configured provider + budget.
"""

from __future__ import annotations

from koboi.media.base import BaseImageProvider, BaseMusicProvider, BaseSpeechProvider, BaseVideoProvider
from koboi.media.types import MediaBudget, MediaRequest, MediaResult


class CountingImageProvider(BaseImageProvider):
    """Delegating image provider that charges each call against the budget."""

    def __init__(self, inner: BaseImageProvider, budget: MediaBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def generate_image(self, req: MediaRequest) -> MediaResult:
        # Pre-check the USD ceiling + image-count cap. True cost is unknown until the
        # provider returns, so the post-call ``record()`` accrues the actual spend; this
        # gate just prevents new calls once a hard cap is already hit.
        if not self._budget.remaining("image"):
            return MediaResult(
                request_id=req.idempotency_key or "",
                modality="image",
                status="rejected",
                rejection_reason="media budget exhausted",
            )
        result = await self._inner.generate_image(req)
        self._budget.record(result)
        return result

    async def close(self) -> None:
        await self._inner.close()


class CountingVideoProvider(BaseVideoProvider):
    """Delegating video provider that charges each generation against the budget.

    Pre-checks the budget before submitting the (expensive, minutes-long) job; records actual
    spend + duration on completion. Eventual enforcement -- a job may exceed the cap before the
    next call is blocked.
    """

    def __init__(self, inner: BaseVideoProvider, budget: MediaBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def submit_video(self, req: MediaRequest):  # type: ignore[override]
        return await self._inner.submit_video(req)

    async def poll_video(self, job):  # type: ignore[override]
        return await self._inner.poll_video(job)

    async def fetch_video_artifact(self, job) -> bytes:  # type: ignore[override]
        return await self._inner.fetch_video_artifact(job)

    async def cancel_video(self, job) -> None:  # type: ignore[override]
        await self._inner.cancel_video(job)

    async def generate_video(self, req: MediaRequest) -> MediaResult:  # type: ignore[override]
        if not self._budget.remaining("video"):
            return MediaResult(
                request_id=req.idempotency_key or "",
                modality="video",
                status="rejected",
                rejection_reason="media budget exhausted",
            )
        result = await self._inner.generate_video(req)
        self._budget.record(result)
        return result

    async def close(self) -> None:
        await self._inner.close()


class CountingMusicProvider(BaseMusicProvider):
    """Delegating music provider that charges each generation against the budget."""

    def __init__(self, inner: BaseMusicProvider, budget: MediaBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def submit_music(self, req: MediaRequest):  # type: ignore[override]
        return await self._inner.submit_music(req)

    async def poll_music(self, job):  # type: ignore[override]
        return await self._inner.poll_music(job)

    async def fetch_music_artifact(self, job) -> bytes:  # type: ignore[override]
        return await self._inner.fetch_music_artifact(job)

    async def cancel_music(self, job) -> None:  # type: ignore[override]
        await self._inner.cancel_music(job)

    async def generate_music(self, req: MediaRequest) -> MediaResult:  # type: ignore[override]
        if not self._budget.remaining("music"):
            return MediaResult(
                request_id=req.idempotency_key or "",
                modality="music",
                status="rejected",
                rejection_reason="media budget exhausted",
            )
        result = await self._inner.generate_music(req)
        self._budget.record(result)
        return result

    async def close(self) -> None:
        await self._inner.close()


class CountingSpeechProvider(BaseSpeechProvider):
    """Delegating speech provider that charges each synthesis against the budget."""

    def __init__(self, inner: BaseSpeechProvider, budget: MediaBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def synthesize_speech(self, req: MediaRequest) -> MediaResult:  # type: ignore[override]
        if not self._budget.remaining("speech"):
            return MediaResult(
                request_id=req.idempotency_key or "",
                modality="speech",
                status="rejected",
                rejection_reason="media budget exhausted",
            )
        result = await self._inner.synthesize_speech(req)
        self._budget.record(result)
        return result

    async def close(self) -> None:
        await self._inner.close()
