"""koboi/media/model_profile.py -- per-model capability registry (the platform's model knowledge).

Each generation model on a gateway (Surplus, Replicate, etc.) has model-specific constraints:
supported durations, voice formats, billing units, response field names, timeouts. ``ModelProfile``
captures these declaratively so providers can **validate/auto-correct before the billed API call**
(saves $ on rejected calls) + **adaptively parse responses** (handles field-name variation).

This is gateway-agnostic: a profile says "veo3 needs 8s duration" regardless of which gateway serves
the model. Providers consult ``get_model_profile(req.model)`` — if a profile exists, it enriches the
request/response handling; if not, the provider falls back to its defaults (backward compatible).

Built-in profiles are registered at import for all models validated in the live smoke campaign.
Users can add custom profiles via ``media.profiles`` YAML config or ``register_model_profile()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from koboi.media.types import MediaRequest, MediaUnit

_logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Per-model capability metadata. Consulted by providers before submit + after parse."""

    name: str
    modality: str  # "image" | "video" | "speech" | "transcription"
    durations: list[int] | None = None  # supported durations (seconds); None = any
    sizes: list[str] | None = None  # image: ["1024x1024"]
    aspect_ratios: list[str] | None = None  # video: ["16:9", "9:16"]
    voices: list[str] | None = None  # TTS: ["vv_alloy"]; None = omit voice (gateway default)
    timeout: float = 120.0  # per-model HTTP timeout override
    billing_unit: MediaUnit | None = None  # per-image/token/job/second/char
    artifact_field: str = "results"  # gateway response field containing the artifact array
    artifact_url_key: str = "url"  # key inside each artifact item for the download URL
    param_overrides: dict[str, Any] = field(default_factory=dict)  # e.g. {"voice": None} = omit


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MODEL_PROFILES: dict[str, ModelProfile] = {}


def register_model_profile(profile: ModelProfile) -> ModelProfile:
    """Register a model profile (idempotent — last write wins)."""
    _MODEL_PROFILES[profile.name] = profile
    return profile


def get_model_profile(name: str | None) -> ModelProfile | None:
    """Look up a profile by model name. Returns ``None`` if unknown (backward compat)."""
    if not name:
        return None
    return _MODEL_PROFILES.get(name)


def validate_request(req: MediaRequest, profile: ModelProfile | None) -> MediaRequest:
    """Auto-correct the request using the profile (duration clamp, param overrides).

    Returns the (possibly modified) ``req``. If ``profile`` is ``None``, returns ``req`` unchanged.
    """
    if profile is None:
        return req

    # Duration: clamp to nearest supported value (e.g. 5s → 8s for veo3, 5s → 6s for wan).
    if profile.durations and req.duration_seconds is not None:
        requested = int(req.duration_seconds)
        if requested not in profile.durations:
            corrected = min(profile.durations, key=lambda d: abs(d - requested))
            _logger.info(
                "ModelProfile: duration %ds not supported for '%s' (supported: %s) → corrected to %ds",
                requested,
                profile.name,
                profile.durations,
                corrected,
            )
            req.duration_seconds = float(corrected)

    # Param overrides: apply model-specific adjustments (e.g. {"voice": None} = omit voice).
    for key, value in profile.param_overrides.items():
        if key == "voice" and value is None:
            req.voice = None  # omit voice → gateway picks default
        elif hasattr(req, key):
            setattr(req, key, value)

    return req


def load_profiles_from_config(profiles_config: list[dict]) -> None:
    """Load custom profiles from YAML ``media.profiles`` config (list of dicts)."""
    for entry in profiles_config:
        try:
            name = entry.get("name", "")
            modality = entry.get("modality", "image")
            if not name:
                continue
            billing_str = entry.get("billing_unit")
            billing = None
            if billing_str:
                try:
                    billing = MediaUnit(billing_str)
                except ValueError:
                    pass
            register_model_profile(
                ModelProfile(
                    name=name,
                    modality=modality,
                    durations=entry.get("durations"),
                    sizes=entry.get("sizes"),
                    aspect_ratios=entry.get("aspect_ratios"),
                    voices=entry.get("voices"),
                    timeout=float(entry.get("timeout", 120.0)),
                    billing_unit=billing,
                    artifact_field=entry.get("artifact_field", "results"),
                    artifact_url_key=entry.get("artifact_url_key", "url"),
                    param_overrides=entry.get("param_overrides", {}),
                )
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning("Failed to load model profile from config: %s — %s", entry.get("name"), e)


# ---------------------------------------------------------------------------
# Built-in profiles (from the live smoke campaign — gateway-verified data)
# ---------------------------------------------------------------------------

# --- Image ---
register_model_profile(
    ModelProfile(name="venice-z-image-turbo", modality="image", timeout=60.0, billing_unit=MediaUnit.IMAGE)
)
register_model_profile(
    ModelProfile(name="venice-nano-banana-pro", modality="image", timeout=60.0, billing_unit=MediaUnit.IMAGE)
)
register_model_profile(
    ModelProfile(name="venice-gpt-image-2", modality="image", timeout=300.0, billing_unit=MediaUnit.IMAGE)
)
register_model_profile(ModelProfile(name="gpt-5-image", modality="image", timeout=300.0, billing_unit=MediaUnit.TOKEN))
register_model_profile(
    ModelProfile(name="gpt-5.4-image-2", modality="image", timeout=300.0, billing_unit=MediaUnit.TOKEN)
)

# --- Video text-to-video ---
register_model_profile(ModelProfile(name="kling-v3-4k-text-to-video", modality="video", billing_unit=MediaUnit.JOB))
register_model_profile(
    ModelProfile(name="seedance-1-5-pro-text-to-video", modality="video", billing_unit=MediaUnit.JOB)
)
register_model_profile(
    ModelProfile(name="veo3-1-full-text-to-video", modality="video", durations=[4, 6, 8], billing_unit=MediaUnit.JOB)
)
register_model_profile(
    ModelProfile(name="veo3-1-fast-text-to-video", modality="video", durations=[4, 6, 8], billing_unit=MediaUnit.JOB)
)

# --- Video image-to-video ---
register_model_profile(ModelProfile(name="kling-v3-pro-image-to-video", modality="video", billing_unit=MediaUnit.JOB))
register_model_profile(
    ModelProfile(name="happyhorse-1-1-image-to-video", modality="video", billing_unit=MediaUnit.SECOND)
)
register_model_profile(
    ModelProfile(name="seedance-2-0-fast-image-to-video", modality="video", billing_unit=MediaUnit.JOB)
)
register_model_profile(
    ModelProfile(name="wan-2-1-pro-image-to-video", modality="video", durations=[6], billing_unit=MediaUnit.JOB)
)
register_model_profile(
    ModelProfile(
        name="gemini-omni-flash-image-to-video", modality="video", durations=[4, 6, 8, 10], billing_unit=MediaUnit.JOB
    )
)
register_model_profile(
    ModelProfile(name="veo3-full-image-to-video", modality="video", durations=[8], billing_unit=MediaUnit.JOB)
)
register_model_profile(
    ModelProfile(name="veo3-fast-image-to-video", modality="video", durations=[8], billing_unit=MediaUnit.JOB)
)

# --- TTS (all: omit voice → gateway picks default; Surplus rejects OpenAI "alloy") ---
for _tts in ("tts-xai-v1", "tts-gemini-3-1-flash", "venice-elevenlabs-tts-turbo-v2-5"):
    register_model_profile(
        ModelProfile(
            name=_tts,
            modality="speech",
            billing_unit=MediaUnit.CHAR,
            param_overrides={"voice": None},
        )
    )

# --- STT (multipart needs format hint) ---
register_model_profile(
    ModelProfile(
        name="venice-whisper-large-v3",
        modality="transcription",
        billing_unit=MediaUnit.MINUTE,
        param_overrides={"filename": "audio.mp3", "content_type": "audio/mpeg"},
    )
)
