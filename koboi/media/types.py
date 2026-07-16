"""koboi/media/types.py -- normalized generation request/result + budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class MediaUnit(str, Enum):
    """Billing unit a generation was charged in (normalized across providers).

    Different modalities bill in different units -- image is per-image or
    per-megapixel (or per-token for gpt-5-image), video is per-job or per-second,
    TTS is per-char, STT is per-minute. ``MediaResult.cost_usd`` unifies the
    *reporting*; ``billing_unit``/``billing_quantity`` keep the original unit so a
    budget guard never compares tokens against dollars.
    """

    IMAGE = "image"  # per image (most image models)
    MEGAPIXEL = "megapixel"  # per megapixel (resolution-driven image)
    TOKEN = "token"  # per-million-tokens billing unit (not a credential)  # nosec B105
    JOB = "job"  # per job (video, music) -- W1
    SECOND = "second"  # per second of output (Happy Horse video, billed audio) -- W1
    CHAR = "char"  # per character (TTS) -- W2
    MINUTE = "minute"  # per minute (STT) -- W2


@dataclass
class MediaRequest:
    """A single generation request. W0: image. W1: video + music. W2: speech + transcription."""

    modality: str = "image"  # "image" | "video" | "music" | "speech" | "transcription"
    prompt: str = ""
    model: str | None = None
    # image params
    n: int = 1
    size: str | None = None  # e.g. "1024x1024"
    quality: str | None = None  # "low" | "medium" | "high"
    response_format: str | None = None  # "b64_json" | "url" (sync image); "mp3" | "wav" (speech)
    speed: float | None = None  # speech synthesis speed multiplier
    # video params
    aspect_ratio: str | None = None  # "16:9", "9:16", ...
    duration_seconds: float | None = None  # video / music output length
    audio: bool | None = None  # include an audio track (video)
    input_images: list[str] | None = None  # i2v / image-edit (URLs or data URIs)
    end_image_url: str | None = None  # video transition end frame
    # music params
    lyrics_prompt: str | None = None
    force_instrumental: bool | None = None
    voice: str | None = None  # music / speech
    language_code: str | None = None
    webhook_url: str | None = None  # async-job terminal callback
    # universal
    idempotency_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaResult:
    """Normalized generation outcome. ``cost_usd`` is the unified reporting field."""

    request_id: str
    modality: str
    status: str = "ok"  # "ok" | "rejected" | "failed"
    data: bytes | None = None
    url: str | None = None
    url_expires_at: float | None = None  # unix epoch; gateway URIs are NOT durable
    local_path: str | None = None  # materialized artifact path (set by MediaBackend)
    content_type: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    cost_usd: Decimal | None = None
    billing_unit: MediaUnit | None = None
    billing_quantity: float | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)
    safety_blocked: bool = False
    rejection_reason: str | None = None
    model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaBudget:
    """Hard caps for generation spend. W0 enforces the USD ceiling + image count.

    Mirrors the bounded-counter pattern of ``orchestration.research.ResearchBudget``
    and the fail-soft ``websearch.providers.counting`` proxies: exhaustion returns a
    ``status="rejected"`` result, never raises. Per-modality caps for video/audio
    arrive in W1/W2.
    """

    max_cost_usd: float = 5.0
    max_images: int = 20
    max_video_seconds: float = 60.0
    max_music_seconds: float = 120.0
    max_speech_chars: int = 50000
    used_cost_usd: Decimal = Decimal("0")
    used_images: int = 0
    used_video_seconds: float = 0.0
    used_music_seconds: float = 0.0
    used_speech_chars: int = 0

    def remaining(self, modality: str = "image", est_cost: float = 0.0) -> bool:
        if float(self.used_cost_usd) + est_cost >= self.max_cost_usd:
            return False
        if modality == "image" and self.used_images >= self.max_images:
            return False
        if modality == "video" and self.used_video_seconds >= self.max_video_seconds:
            return False
        if modality == "music" and self.used_music_seconds >= self.max_music_seconds:
            return False
        if modality == "speech" and self.used_speech_chars >= self.max_speech_chars:
            return False
        return True

    def record(self, result: MediaResult) -> None:
        """Accrue spend from a completed result (no-op for rejected/failed)."""
        if result.status != "ok":
            return
        if result.cost_usd is not None:
            self.used_cost_usd += result.cost_usd
        if result.modality == "image":
            qty = result.billing_quantity if result.billing_quantity is not None else 1
            self.used_images += max(1, int(qty))
        elif result.modality == "video":
            self.used_video_seconds += float(result.duration_seconds or result.billing_quantity or 0)
        elif result.modality == "music":
            self.used_music_seconds += float(result.duration_seconds or result.billing_quantity or 0)
        elif result.modality == "speech":
            self.used_speech_chars += int(result.billing_quantity or 0)
