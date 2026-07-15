"""koboi/media/providers/surplus.py -- Surplus Intelligence image provider.

Surplus (``https://api.surplusintelligence.ai/v1``) is an OpenAI-compatible inference
marketplace. Image generation hits ``POST /v1/images/generations`` (synchronous) with
the standard OpenAI request shape ``{model, prompt, n, size, response_format}`` and
response ``{data: [{b64_json | url}], usage: {...}}``. Auth is Bearer (``inf_xxx``);
x402/MPP arrive in W5 as additional ``AuthStrategy`` subclasses.

Reuses ``koboi.llm.http_transport.HttpTransport`` + ``koboi.llm.auth.BearerAuth`` so
retry/backoff and status->LLMError mapping are shared with the LLM transport. The
gateway bills image gen per-image / per-megapixel / per-token depending on model;
``cost_usd``/``billing_unit`` are derived from the response ``usage`` block when present
(the exact usage field names are gateway-specific and best-effort).
"""

from __future__ import annotations

import base64
import logging
import os
import uuid
from collections.abc import Callable
from decimal import Decimal

from koboi.llm.auth import BearerAuth
from koboi.llm.http_transport import HttpTransport
from koboi.media.async_job import MediaJob
from koboi.media.base import (
    BaseImageProvider,
    BaseMusicProvider,
    BaseSpeechProvider,
    BaseTranscriptionProvider,
    BaseVideoProvider,
)
from koboi.media.model_profile import get_model_profile, validate_request
from koboi.media.registry import (
    register_image_provider,
    register_music_provider,
    register_speech_provider,
    register_transcription_provider,
    register_video_provider,
)
from koboi.media.types import MediaRequest, MediaResult, MediaUnit

_logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.surplusintelligence.ai/v1"
_DEFAULT_MODEL = "venice-z-image-turbo"

# Models metered per-million-tokens (gpt-5-image family) vs the per-image default.
_TOKEN_METERED_MODEL_PREFIXES = ("gpt-5-image", "gpt-5.4-image")


@register_image_provider("surplus", description="Surplus Intelligence gateway (OpenAI-compatible image gen)")
class SurplusImageProvider(BaseImageProvider):
    """Image generation via the Surplus gateway (OpenAI-compatible ``/images/generations``)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        auth_mode: str = "bearer",
        timeout: float = 300.0,
    ) -> None:
        if auth_mode != "bearer":
            raise NotImplementedError(
                f"surplus auth_mode '{auth_mode}' not implemented in W0 (use 'bearer'); x402/MPP arrive in W5"
            )
        self._api_key = api_key or os.getenv("SURPLUS_API_KEY", "")
        self._model = model
        self._transport = HttpTransport(base_url or _DEFAULT_BASE_URL, BearerAuth(self._api_key), timeout=timeout)

    async def generate_image(self, req: MediaRequest) -> MediaResult:
        if not self._api_key:
            raise ValueError("surplus provider requires an api_key (media.image.surplus.api_key or SURPLUS_API_KEY)")
        model = req.model or self._model
        body: dict = {"model": model, "prompt": req.prompt, "n": max(1, int(req.n or 1))}
        if req.size:
            body["size"] = req.size
        if req.quality:
            body["quality"] = req.quality
        body["response_format"] = req.response_format or "b64_json"
        if req.idempotency_key:
            body["metadata"] = {"idempotency_key": req.idempotency_key}

        data = await self._transport.post("/images/generations", body)
        return _parse_image_response(data, model, req)

    async def close(self) -> None:
        await self._transport.close()


def _parse_image_response(data: dict, model: str, req: MediaRequest) -> MediaResult:
    """Map the OpenAI-shape image response onto a normalized ``MediaResult``."""
    request_id = req.idempotency_key or uuid.uuid4().hex
    items = data.get("data") or []
    if not items or not isinstance(items[0], dict):
        return MediaResult(
            request_id=request_id,
            modality="image",
            status="failed",
            rejection_reason="gateway returned no image data",
            model=model,
            raw=data,
        )

    first = items[0]
    b64 = first.get("b64_json")
    url = first.get("url")

    result = MediaResult(
        request_id=request_id,
        modality="image",
        status="ok",
        content_type="image/png",
        model=model,
        raw=data,
    )
    if b64:
        try:
            result.data = base64.b64decode(b64)
        except (ValueError, TypeError) as e:
            return MediaResult(
                request_id=request_id,
                modality="image",
                status="failed",
                rejection_reason=f"failed to decode b64_json: {e}",
                model=model,
                raw=data,
            )
    elif url:
        # Gateway URLs are short-lived; MediaBackend materializes via MediaStore.
        result.url = str(url)
    else:
        return MediaResult(
            request_id=request_id,
            modality="image",
            status="failed",
            rejection_reason="gateway returned neither b64_json nor url",
            model=model,
            raw=data,
        )

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    result.raw_usage = usage
    result.cost_usd, result.billing_unit, result.billing_quantity = _normalize_cost(usage, model, req.n or 1)
    return result


def _normalize_cost(usage: dict, model: str, n: int) -> tuple[Decimal | None, MediaUnit, float]:
    """Derive (cost_usd, billing_unit, billing_quantity) from the gateway usage block."""
    token_metered = isinstance(model, str) and model.startswith(_TOKEN_METERED_MODEL_PREFIXES)
    if token_metered:
        unit = MediaUnit.TOKEN
        qty = float(usage.get("output_tokens") or usage.get("total_tokens") or 0)
    else:
        unit = MediaUnit.IMAGE
        qty = float(usage.get("images") or n)
    return _extract_cost_usd(usage), unit, qty


def _extract_cost_usd(usage: dict) -> Decimal | None:
    """Prefer an explicit USD field; fall back to microdollar (USDC) fields."""
    for key in ("cost_usd", "usd_cost"):
        if usage.get(key) is not None:
            return _to_decimal(usage[key])
    for key in ("buyer_cost_usd", "estimated_cost_usd", "cost_usdc"):
        if usage.get(key) is not None:
            d = _to_decimal(usage[key])
            return None if d is None else d / Decimal("1000000")
    return None


def _to_decimal(value) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (ValueError, TypeError, ArithmeticError):
        return None


# ── Async generation (video + music) -- W1 ──────────────────────────────────
# Surplus async jobs share one envelope: POST /v1/{video|music}/generations -> 202
# {id, status, poll_url, cancel_url, job_token, estimated_cost_usd}; poll
# GET /v1/{kind}/generations/:id through queued -> submitted -> running ->
# succeeded|failed|canceled|expired; fetch the artifact via the download URL.

_PER_SECOND_VIDEO_MARKERS = ("happyhorse", "happy-horse", "minimax")


def _is_per_second_video(model: str | None) -> bool:
    """Happy Horse / MiniMax video bills per-second; others per-job."""
    m = (model or "").lower()
    return any(marker in m for marker in _PER_SECOND_VIDEO_MARKERS)


def _parse_job_response(data: dict, kind: str) -> MediaJob:
    """Parse a Surplus 202 submit response into a ``MediaJob``."""
    job_id = data.get("id") or data.get("job_id") or ""
    return MediaJob(
        job_id=str(job_id),
        kind=kind,
        status=str(data.get("status") or "queued"),
        poll_url=data.get("poll_url"),
        cancel_url=data.get("cancel_url"),
        job_token=data.get("job_token"),
        estimated_cost_usd=_to_decimal(data.get("estimated_cost_usd") or data.get("estimated_cost_usdc")),
        max_cost_usd=_to_decimal(data.get("max_cost_usd") or data.get("max_cost_usdc")),
        raw=data if isinstance(data, dict) else {},
    )


def _download_url_from_raw(data: dict) -> str:
    """Extract the artifact URL from a poll response.

    Surplus returns the artifact URL in various fields depending on the modality:
    ``download_url`` (some), ``url`` (some), or inside ``results``/``artifacts`` arrays.
    """
    url = data.get("download_url") or data.get("url")
    if not url:
        for field in ("results", "artifacts"):
            items = data.get(field)
            if isinstance(items, list) and items and isinstance(items[0], dict):
                url = items[0].get("url") or items[0].get("download_url")
                if url:
                    break
    return str(url or "")


def _download_url(job: MediaJob) -> str:
    return _download_url_from_raw(job.raw or {})


def _build_video_body(req: MediaRequest, model: str) -> dict:
    body: dict = {"model": req.model or model, "prompt": req.prompt}
    if req.aspect_ratio:
        body["aspect_ratio"] = req.aspect_ratio
    if req.duration_seconds is not None:
        body["duration_seconds"] = req.duration_seconds
    if req.audio is not None:
        body["audio"] = req.audio
    if req.input_images:
        body["image_url"] = req.input_images[0]
    if req.end_image_url:
        body["end_image_url"] = req.end_image_url
    if req.webhook_url:
        body["webhook_url"] = req.webhook_url
    if req.idempotency_key:
        body["metadata"] = {"idempotency_key": req.idempotency_key}
    return body


def _build_music_body(req: MediaRequest, model: str) -> dict:
    body: dict = {"model": req.model or model, "prompt": req.prompt}
    if req.duration_seconds is not None:
        body["duration_seconds"] = req.duration_seconds
    if req.lyrics_prompt:
        body["lyrics_prompt"] = req.lyrics_prompt
    if req.force_instrumental is not None:
        body["force_instrumental"] = req.force_instrumental
    if req.voice:
        body["voice"] = req.voice
    if req.language_code:
        body["language_code"] = req.language_code
    if req.webhook_url:
        body["webhook_url"] = req.webhook_url
    if req.idempotency_key:
        body["metadata"] = {"idempotency_key": req.idempotency_key}
    return body


def _build_video_result(data: dict, model: str) -> MediaResult:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    duration = data.get("duration_seconds")
    per_second = _is_per_second_video(model)
    unit = MediaUnit.SECOND if per_second else MediaUnit.JOB
    qty = float(duration) if (per_second and duration is not None) else 1.0
    return MediaResult(
        request_id="",
        modality="video",
        status="ok",
        content_type="video/mp4",
        url=_download_url_from_raw(data) or None,
        duration_seconds=float(duration) if duration is not None else None,
        width=int(data["width"]) if data.get("width") is not None else None,
        height=int(data["height"]) if data.get("height") is not None else None,
        model=model,
        raw_usage=usage,
        cost_usd=_extract_cost_usd(usage) or _extract_cost_usd(data),
        billing_unit=unit,
        billing_quantity=qty,
        raw=data if isinstance(data, dict) else {},
    )


def _build_music_result(data: dict, model: str) -> MediaResult:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    duration = data.get("duration_seconds")
    return MediaResult(
        request_id="",
        modality="music",
        status="ok",
        content_type="audio/mpeg",
        url=_download_url_from_raw(data) or None,
        duration_seconds=float(duration) if duration is not None else None,
        model=model,
        raw_usage=usage,
        cost_usd=_extract_cost_usd(usage) or _extract_cost_usd(data),
        billing_unit=MediaUnit.JOB,
        billing_quantity=1,
        raw=data if isinstance(data, dict) else {},
    )


async def _submit_async(transport: HttpTransport, endpoint: str, body: dict, kind: str, api_key: str) -> MediaJob:
    if not api_key:
        raise ValueError(
            f"surplus {kind} provider requires an api_key (media.{kind}.surplus.api_key or SURPLUS_API_KEY)"
        )
    data = await transport.post(endpoint, body)
    return _parse_job_response(data, kind)


async def _poll_async(
    transport: HttpTransport, endpoint: str, job: MediaJob, build_result: Callable[[dict], MediaResult]
) -> MediaJob:
    data = await transport.get(f"{endpoint}/{job.job_id}")
    if isinstance(data, dict):
        job.raw = data
        job.status = str(data.get("status") or job.status)
    if job.status == "succeeded":
        _logger.info("poll succeeded; raw keys: %s", list(data.keys()) if isinstance(data, dict) else "non-dict")
        job.result = build_result(data if isinstance(data, dict) else {})
    return job


async def _fetch_async(transport: HttpTransport, job: MediaJob) -> bytes:
    url = (job.result.url if job.result is not None else None) or _download_url(job)
    if not url:
        _logger.warning(
            "fetch: no download URL found. raw keys: %s, raw preview: %s",
            list(job.raw.keys()) if isinstance(job.raw, dict) else "none",
            str(job.raw)[:500] if job.raw else "empty",
        )
        raise ValueError(f"no artifact download_url on succeeded {job.kind} job {job.job_id}")
    data = await transport.get_bytes(url)
    _logger.info("fetch: %d bytes from %s", len(data), url[:100])
    return data


async def _cancel_async(transport: HttpTransport, endpoint: str, job: MediaJob) -> None:
    await transport.delete(f"{endpoint}/{job.job_id}")


@register_video_provider("surplus", description="Surplus Intelligence gateway (OpenAI-compatible video gen, async)")
class SurplusVideoProvider(BaseVideoProvider):
    """Video generation via the Surplus gateway (async job: ``/v1/video/generations``)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = _DEFAULT_BASE_URL,
        model: str = "kling-v3",
        auth_mode: str = "bearer",
        timeout: float = 120.0,
        poll_interval: float = 5.0,
        max_wait: float = 1500.0,
    ) -> None:
        if auth_mode != "bearer":
            raise NotImplementedError(
                f"surplus auth_mode '{auth_mode}' not implemented in W1 (use 'bearer'); x402/MPP arrive in W5"
            )
        self._api_key = api_key or os.getenv("SURPLUS_API_KEY", "")
        self._model = model
        self._transport = HttpTransport(base_url or _DEFAULT_BASE_URL, BearerAuth(self._api_key), timeout=timeout)
        self.poll_interval = poll_interval
        self.max_wait = max_wait

    async def submit_video(self, req: MediaRequest) -> MediaJob:
        req = validate_request(req, get_model_profile(req.model or self._model))
        return await _submit_async(
            self._transport, "/video/generations", _build_video_body(req, self._model), "video", self._api_key
        )

    async def poll_video(self, job: MediaJob) -> MediaJob:
        return await _poll_async(
            self._transport, "/video/generations", job, lambda d: _build_video_result(d, self._model)
        )

    async def fetch_video_artifact(self, job: MediaJob) -> bytes:
        return await _fetch_async(self._transport, job)

    async def cancel_video(self, job: MediaJob) -> None:
        return await _cancel_async(self._transport, "/video/generations", job)

    async def close(self) -> None:
        await self._transport.close()


@register_music_provider("surplus", description="Surplus Intelligence gateway (OpenAI-compatible music gen, async)")
class SurplusMusicProvider(BaseMusicProvider):
    """Music generation via the Surplus gateway (async job: ``/v1/music/generations``)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = _DEFAULT_BASE_URL,
        model: str = "venice-minimax-music-v26",
        auth_mode: str = "bearer",
        timeout: float = 120.0,
        poll_interval: float = 3.0,
        max_wait: float = 480.0,
    ) -> None:
        if auth_mode != "bearer":
            raise NotImplementedError(
                f"surplus auth_mode '{auth_mode}' not implemented in W1 (use 'bearer'); x402/MPP arrive in W5"
            )
        self._api_key = api_key or os.getenv("SURPLUS_API_KEY", "")
        self._model = model
        self._transport = HttpTransport(base_url or _DEFAULT_BASE_URL, BearerAuth(self._api_key), timeout=timeout)
        self.poll_interval = poll_interval
        self.max_wait = max_wait

    async def submit_music(self, req: MediaRequest) -> MediaJob:
        req = validate_request(req, get_model_profile(req.model or self._model))
        return await _submit_async(
            self._transport, "/music/generations", _build_music_body(req, self._model), "music", self._api_key
        )

    async def poll_music(self, job: MediaJob) -> MediaJob:
        return await _poll_async(
            self._transport, "/music/generations", job, lambda d: _build_music_result(d, self._model)
        )

    async def fetch_music_artifact(self, job: MediaJob) -> bytes:
        return await _fetch_async(self._transport, job)

    async def cancel_music(self, job: MediaJob) -> None:
        return await _cancel_async(self._transport, "/music/generations", job)

    async def close(self) -> None:
        await self._transport.close()


# ── Speech-to-text (STT) -- W5a (sync, multipart) ────────────────────────────


@register_transcription_provider("surplus", description="Surplus Intelligence gateway (OpenAI-compatible STT, sync)")
class SurplusTranscriptionProvider(BaseTranscriptionProvider):
    """Speech-to-text via the Surplus gateway (sync ``/v1/audio/transcriptions``, multipart)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = _DEFAULT_BASE_URL,
        model: str = "whisper-large-v3",
        auth_mode: str = "bearer",
        timeout: float = 120.0,
    ) -> None:
        if auth_mode != "bearer":
            raise NotImplementedError(
                f"surplus auth_mode '{auth_mode}' not implemented in W5a (use 'bearer'); x402/MPP arrive in W5c"
            )
        self._api_key = api_key or os.getenv("SURPLUS_API_KEY", "")
        self._model = model
        self._transport = HttpTransport(base_url or _DEFAULT_BASE_URL, BearerAuth(self._api_key), timeout=timeout)

    async def transcribe(
        self,
        audio: bytes,
        *,
        language_code: str | None = None,
        prompt: str | None = None,
        model: str | None = None,
    ) -> str:
        if not self._api_key:
            raise ValueError(
                "surplus transcription provider requires an api_key "
                "(media.transcription.surplus.api_key or SURPLUS_API_KEY)"
            )
        files = {"file": ("audio.mp3", audio, "audio/mpeg")}
        data: dict = {"model": model or self._model}
        if language_code:
            data["language"] = language_code
        if prompt:
            data["prompt"] = prompt
        result = await self._transport.post_form("/audio/transcriptions", files=files, data=data)
        return str(result.get("text", ""))

    async def close(self) -> None:
        await self._transport.close()


# ── Speech synthesis (TTS) -- W2 (sync, binary response) ─────────────────────

_AUDIO_CONTENT_TYPE: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
}


@register_speech_provider("surplus", description="Surplus Intelligence gateway (OpenAI-compatible TTS, sync)")
class SurplusSpeechProvider(BaseSpeechProvider):
    """Speech synthesis via the Surplus gateway (sync ``/v1/audio/speech``)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = _DEFAULT_BASE_URL,
        model: str = "venice-kokoro-tts",
        auth_mode: str = "bearer",
        timeout: float = 120.0,
    ) -> None:
        if auth_mode != "bearer":
            raise NotImplementedError(
                f"surplus auth_mode '{auth_mode}' not implemented in W2 (use 'bearer'); x402/MPP arrive in W5"
            )
        self._api_key = api_key or os.getenv("SURPLUS_API_KEY", "")
        self._model = model
        self._transport = HttpTransport(base_url or _DEFAULT_BASE_URL, BearerAuth(self._api_key), timeout=timeout)

    async def synthesize_speech(self, req: MediaRequest) -> MediaResult:
        if not self._api_key:
            raise ValueError(
                "surplus speech provider requires an api_key (media.speech.surplus.api_key or SURPLUS_API_KEY)"
            )
        fmt = req.response_format or "mp3"
        body: dict = {
            "model": req.model or self._model,
            "input": req.prompt,
            "response_format": fmt,
        }
        # Surplus/Venice TTS voices use vv_<id> handles; "alloy" (OpenAI) is rejected. Only send
        # voice when it's a valid Surplus handle; else let the gateway pick its default.
        if req.voice and req.voice.startswith("vv_"):
            body["voice"] = req.voice
        if req.speed is not None:
            body["speed"] = req.speed
        if req.language_code:
            body["language"] = req.language_code
        data = await self._transport.post_bytes("/audio/speech", body)
        # /audio/speech returns a binary audio body with no usage block, so cost_usd is unknown;
        # the per-character cap (max_speech_chars) is the binding budget limit for speech.
        return MediaResult(
            request_id=req.idempotency_key or uuid.uuid4().hex,
            modality="speech",
            status="ok",
            data=data,
            content_type=_AUDIO_CONTENT_TYPE.get(fmt, "audio/mpeg"),
            billing_unit=MediaUnit.CHAR,
            billing_quantity=float(len(req.prompt or "")),
            model=req.model or self._model,
        )

    async def close(self) -> None:
        await self._transport.close()
