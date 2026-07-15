"""koboi/media/registry.py -- decorator-based generation-provider registry.

Mirrors ``koboi.websearch.registry`` (which mirrors ``koboi.rag.registry``): a
``ProviderRegistry`` + a decorator that registers a provider class, a builder that
composes the provider from the ``media:`` YAML section, and a custom-module loader
for YAML-driven extensibility.

W0 ships the image registry only (``image_provider_registry`` /
``@register_image_provider`` / ``build_image_provider``). Video and audio registries
are additive in W1/W2 -- copy this pattern per modality.

Two provider-specific behaviors (identical to websearch):
  1. Nested per-provider config -- a provider's kwargs come from
     ``media.image.<provider_name>.<key>`` with shared top-level knobs (e.g.
     ``media.image.model``) as fallback.
  2. Secret redaction -- providers carry credentials (``api_key``/``token``), so any
     config that reaches a log line passes through ``_redact``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

from koboi.media.base import (
    BaseImageProvider,
    BaseMusicProvider,
    BaseSpeechProvider,
    BaseTranscriptionProvider,
    BaseVideoProvider,
)

_logger = logging.getLogger(__name__)

# Keys whose values must never reach logs / cache-key material (provider credentials).
_SECRET_KEYS = frozenset({"api_key", "token", "secret", "password", "x_payment_signature"})


# ---------------------------------------------------------------------------
# Generic provider registry
# ---------------------------------------------------------------------------


class ProviderEntry:
    """Metadata for a registered media provider."""

    __slots__ = ("cls", "parameters", "description", "config_aliases", "inject")

    def __init__(
        self,
        cls: type,
        parameters: dict[str, dict[str, Any]],
        description: str = "",
        config_aliases: dict[str, str] | None = None,
        inject: list[str] | None = None,
    ):
        self.cls = cls
        self.parameters = parameters
        self.description = description
        self.config_aliases = config_aliases or {}
        self.inject = inject or []


class ProviderRegistry:
    """Generic registry for media providers (image now; video/audio in W1/W2)."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, ProviderEntry] = {}

    def register(
        self,
        name: str,
        cls: type,
        *,
        description: str = "",
        config_aliases: dict[str, str] | None = None,
        inject: list[str] | None = None,
    ) -> None:
        params = _extract_parameters(cls)
        if config_aliases:
            valid_params = set(params.keys())
            for yaml_key, param_name in config_aliases.items():
                if param_name not in valid_params:
                    raise ValueError(
                        f"config_aliases maps '{yaml_key}' to '{param_name}', "
                        f"but {cls.__name__}.__init__ has no such parameter. "
                        f"Available: {valid_params}"
                    )
        self._entries[name] = ProviderEntry(
            cls=cls,
            parameters=params,
            description=description,
            config_aliases=config_aliases,
            inject=inject,
        )

    def get(self, name: str) -> ProviderEntry | None:
        return self._entries.get(name)

    def list_available(self) -> list[str]:
        return sorted(self._entries.keys())

    def clear(self) -> None:
        self._entries.clear()


def _extract_parameters(cls: type) -> dict[str, dict[str, Any]]:
    """Return ``param_name -> {"default": ..., "annotation": ...}`` for ``cls.__init__``."""
    sig = inspect.signature(cls.__init__)  # type: ignore[misc]  # __init__ params drive config->kwargs resolution
    params: dict[str, dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        entry: dict[str, Any] = {}
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        if param.annotation is not inspect.Parameter.empty:
            entry["annotation"] = param.annotation
        params[name] = entry
    return params


# ---------------------------------------------------------------------------
# Module-level registry (image -- W0)
# ---------------------------------------------------------------------------

image_provider_registry = ProviderRegistry("image")
video_provider_registry = ProviderRegistry("video")
music_provider_registry = ProviderRegistry("music")
speech_provider_registry = ProviderRegistry("speech")
transcription_provider_registry = ProviderRegistry("transcription")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def register_image_provider(
    name: str,
    description: str = "",
    *,
    config_aliases: dict[str, str] | None = None,
    inject: list[str] | None = None,
) -> Callable[[type], type]:
    """Register an image provider class.

    Usage::

        @register_image_provider("surplus", description="Surplus Intelligence gateway")
        class SurplusImageProvider(BaseImageProvider):
            ...
    """

    def decorator(cls: type) -> type:
        image_provider_registry.register(
            name, cls, description=description, config_aliases=config_aliases, inject=inject
        )
        return cls

    return decorator


def register_video_provider(
    name: str,
    description: str = "",
    *,
    config_aliases: dict[str, str] | None = None,
    inject: list[str] | None = None,
) -> Callable[[type], type]:
    """Register a video provider class (async job). See ``register_image_provider``."""

    def decorator(cls: type) -> type:
        video_provider_registry.register(
            name, cls, description=description, config_aliases=config_aliases, inject=inject
        )
        return cls

    return decorator


def register_music_provider(
    name: str,
    description: str = "",
    *,
    config_aliases: dict[str, str] | None = None,
    inject: list[str] | None = None,
) -> Callable[[type], type]:
    """Register a music provider class (async job). See ``register_image_provider``."""

    def decorator(cls: type) -> type:
        music_provider_registry.register(
            name, cls, description=description, config_aliases=config_aliases, inject=inject
        )
        return cls

    return decorator


def register_speech_provider(
    name: str,
    description: str = "",
    *,
    config_aliases: dict[str, str] | None = None,
    inject: list[str] | None = None,
) -> Callable[[type], type]:
    """Register a speech (TTS) provider class. See ``register_image_provider``."""

    def decorator(cls: type) -> type:
        speech_provider_registry.register(
            name, cls, description=description, config_aliases=config_aliases, inject=inject
        )
        return cls

    return decorator


def register_transcription_provider(
    name: str,
    description: str = "",
    *,
    config_aliases: dict[str, str] | None = None,
    inject: list[str] | None = None,
) -> Callable[[type], type]:
    """Register a transcription (STT) provider class. See ``register_image_provider``."""

    def decorator(cls: type) -> type:
        transcription_provider_registry.register(
            name, cls, description=description, config_aliases=config_aliases, inject=inject
        )
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _redact(conf: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``conf`` with secret values masked (for safe logging)."""
    redacted: dict[str, Any] = {}
    for k, v in conf.items():
        redacted[k] = "***" if k in _SECRET_KEYS and v else v
    return redacted


def _merged_provider_conf(parent_conf: dict[str, Any], provider_name: str) -> dict[str, Any]:
    """Build the kwargs source for a provider: shared top-level knobs + per-provider sub-dict.

    Shared knobs are the non-dict keys on ``parent_conf`` (excluding the ``provider``
    selector). The per-provider sub-dict (``parent_conf[provider_name]``) overrides them.
    """
    merged: dict[str, Any] = {}
    for k, v in parent_conf.items():
        if k == "provider" or isinstance(v, dict):
            continue
        merged[k] = v
    merged.update(parent_conf.get(provider_name, {}) or {})
    return merged


def _resolve_kwargs(entry: ProviderEntry, conf: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``__init__`` kwargs from ``conf`` using entry metadata.

    Iterates the provider's introspected params (not the config keys), so only declared
    params are ever forwarded -- unknown YAML keys are silently ignored. ``config_aliases``
    remaps a YAML key to a param name.
    """
    config_aliases = entry.config_aliases
    kwargs: dict[str, Any] = {}
    for param_name in entry.parameters:
        yaml_key = param_name
        for yk, pn in config_aliases.items():
            if pn == param_name:
                yaml_key = yk
                break
        if yaml_key in conf:
            kwargs[param_name] = conf[yaml_key]
        elif "default" in entry.parameters[param_name]:
            kwargs[param_name] = entry.parameters[param_name]["default"]
    return kwargs


def _build_provider(
    registry: ProviderRegistry,
    kind: str,
    parent_conf: dict[str, Any],
    fallback_name: str,
) -> BaseImageProvider | BaseVideoProvider | BaseMusicProvider | BaseSpeechProvider | BaseTranscriptionProvider:
    """Resolve ``parent_conf['provider']`` -> instance, with ``fallback_name`` on miss.

    ``parent_conf`` is the stage section (``media.image``). The provider's kwargs come
    from ``_merged_provider_conf``.
    """
    provider_name = parent_conf.get("provider", fallback_name)
    entry = registry.get(provider_name)
    if entry is None:
        _logger.warning(
            "Unknown %s provider '%s', falling back to '%s'. Available: %s. Config: %s",
            kind,
            provider_name,
            fallback_name,
            registry.list_available(),
            _redact(parent_conf),
        )
        entry = registry.get(fallback_name)
        if entry is None:
            raise ValueError(f"No {kind} providers registered")

    conf = _merged_provider_conf(parent_conf, provider_name)
    kwargs = _resolve_kwargs(entry, conf)
    return entry.cls(**kwargs)  # type: ignore[no-any-return]


def build_image_provider(media_conf: dict[str, Any] | None) -> BaseImageProvider:
    """Build an image provider from the ``media:`` config (``media.image.*``).

    Defaults to ``mock`` when unset (offline-safe). Unknown provider -> ``mock``.
    """
    return _build_provider(  # type: ignore[return-value]
        image_provider_registry,
        "image",
        (media_conf or {}).get("image", {}) or {},
        fallback_name="mock",
    )


def build_video_provider(media_conf: dict[str, Any] | None) -> BaseVideoProvider:
    """Build a video provider from the ``media:`` config (``media.video.*``).

    Defaults to ``mock`` when unset (offline-safe). Unknown provider -> ``mock``.
    """
    return _build_provider(  # type: ignore[return-value]
        video_provider_registry,
        "video",
        (media_conf or {}).get("video", {}) or {},
        fallback_name="mock",
    )


def build_music_provider(media_conf: dict[str, Any] | None) -> BaseMusicProvider:
    """Build a music provider from the ``media:`` config (``media.music.*``).

    Defaults to ``mock`` when unset (offline-safe). Unknown provider -> ``mock``.
    """
    return _build_provider(  # type: ignore[return-value]
        music_provider_registry,
        "music",
        (media_conf or {}).get("music", {}) or {},
        fallback_name="mock",
    )


def build_speech_provider(media_conf: dict[str, Any] | None) -> BaseSpeechProvider:
    """Build a speech (TTS) provider from the ``media:`` config (``media.speech.*``).

    Defaults to ``mock`` when unset (offline-safe). Unknown provider -> ``mock``.
    """
    return _build_provider(  # type: ignore[return-value]
        speech_provider_registry,
        "speech",
        (media_conf or {}).get("speech", {}) or {},
        fallback_name="mock",
    )


def build_transcription_provider(media_conf: dict[str, Any] | None) -> BaseTranscriptionProvider:
    """Build a transcription (STT) provider from the ``media:`` config (``media.transcription.*``).

    Defaults to ``mock`` when unset (offline-safe). Unknown provider -> ``mock``.
    """
    return _build_provider(  # type: ignore[return-value]
        transcription_provider_registry,
        "transcription",
        (media_conf or {}).get("transcription", {}) or {},
        fallback_name="mock",
    )


# ---------------------------------------------------------------------------
# Custom module loading (YAML-driven extensibility)
# ---------------------------------------------------------------------------


def load_custom_components(custom_modules: list[str]) -> None:
    """Import modules so ``@register_image_provider`` decorators fire on import.

    YAML config example::

        media:
          custom_modules:
            - mycorp.media_providers.comfyui
    """
    for module_path in custom_modules:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            _logger.warning("Failed to import custom media module '%s': %s", module_path, e)
