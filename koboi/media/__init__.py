"""koboi/media -- multimodal generation abstraction (image/video/music/speech + STT).

Provider-agnostic generation registries mirroring ``koboi.websearch``: decorator-based registries
(``image``/``video``/``music``/``speech``/``transcription``) + ``@register_*_provider`` +
``build_*_provider``. Built-in providers: ``mock`` (offline default) and ``surplus`` (Surplus
Intelligence gateway). The ``generate_image``/``generate_video``/``generate_music``/``generate_speech``/
``transcribe_audio`` tools (``koboi.tools.builtin.media``) delegate to a ``MediaBackend`` injected via
the tool registry's dep store (``media_provider``).
"""

from __future__ import annotations

from koboi.media.async_job import MediaJob, run_async_job
from koboi.media.backend import MediaBackend, build_media
from koboi.media.model_profile import (
    ModelProfile,
    get_model_profile,
    load_profiles_from_config,
    register_model_profile,
    validate_request,
)
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
    ProviderEntry,
    ProviderRegistry,
    build_image_provider,
    build_music_provider,
    build_speech_provider,
    build_transcription_provider,
    build_video_provider,
    image_provider_registry,
    load_custom_components,
    music_provider_registry,
    register_image_provider,
    register_music_provider,
    register_speech_provider,
    register_transcription_provider,
    register_video_provider,
    speech_provider_registry,
    transcription_provider_registry,
    video_provider_registry,
)
from koboi.media.store import MediaStore
from koboi.media.types import MediaBudget, MediaRequest, MediaResult, MediaUnit

# Register built-in providers (idempotent; decorators fire on import of each module).
from koboi.media.providers import mock as _mock  # noqa: F401
from koboi.media.providers import surplus as _surplus  # noqa: F401

__all__ = [
    # Types
    "MediaRequest",
    "MediaResult",
    "MediaBudget",
    "MediaUnit",
    "MediaJob",
    "run_async_job",
    # ABCs
    "BaseImageProvider",
    "BaseVideoProvider",
    "BaseMusicProvider",
    "BaseSpeechProvider",
    "BaseTranscriptionProvider",
    # Registry
    "ProviderRegistry",
    "ProviderEntry",
    "image_provider_registry",
    "video_provider_registry",
    "music_provider_registry",
    "speech_provider_registry",
    "transcription_provider_registry",
    "register_image_provider",
    "register_video_provider",
    "register_music_provider",
    "register_speech_provider",
    "register_transcription_provider",
    "build_image_provider",
    "build_video_provider",
    "build_music_provider",
    "build_speech_provider",
    "build_transcription_provider",
    "load_custom_components",
    # Backend + budget + store
    "MediaBackend",
    "build_media",
    "CountingImageProvider",
    "CountingVideoProvider",
    "CountingMusicProvider",
    "CountingSpeechProvider",
    "MediaStore",
    # Model profiles
    "ModelProfile",
    "get_model_profile",
    "register_model_profile",
    "validate_request",
    "load_profiles_from_config",
]
