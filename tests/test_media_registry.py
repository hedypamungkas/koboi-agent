"""Tests for koboi.media.registry (decorator, builders, redaction, custom modules)."""

from __future__ import annotations

import logging

from koboi.media.base import BaseImageProvider
from koboi.media.registry import (
    _redact,
    build_image_provider,
    image_provider_registry,
    load_custom_components,
    register_image_provider,
)
from koboi.media.types import MediaRequest, MediaResult


@register_image_provider("test-custom-image", description="test fixture provider")
class _CustomImage(BaseImageProvider):
    def __init__(self, label: str = "default") -> None:
        self.label = label

    async def generate_image(self, req: MediaRequest) -> MediaResult:
        return MediaResult(request_id="x", modality="image", status="ok")


class TestRegistryBuiltins:
    def test_mock_and_surplus_registered(self):
        available = image_provider_registry.list_available()
        assert "mock" in available
        assert "surplus" in available

    def test_build_defaults_to_mock(self):
        provider = build_image_provider({})
        assert type(provider).__name__ == "MockImageProvider"

    def test_build_unknown_falls_back_to_mock(self, caplog):
        with caplog.at_level(logging.WARNING, logger="koboi.media.registry"):
            provider = build_image_provider({"image": {"provider": "does-not-exist"}})
        assert type(provider).__name__ == "MockImageProvider"
        assert any("does-not-exist" in r.message for r in caplog.records)


class TestRegistryCustom:
    def test_build_custom_resolves_kwargs(self):
        provider = build_image_provider(
            {"image": {"provider": "test-custom-image", "test-custom-image": {"label": "zzz"}}}
        )
        assert isinstance(provider, _CustomImage)
        assert provider.label == "zzz"

    def test_build_custom_uses_default_when_subdict_absent(self):
        provider = build_image_provider({"image": {"provider": "test-custom-image"}})
        assert isinstance(provider, _CustomImage)
        assert provider.label == "default"


class TestRedaction:
    def test_redact_masks_secrets(self):
        out = _redact({"api_key": "secret", "x_payment_signature": "sig", "model": "x"})
        assert out["api_key"] == "***"
        assert out["x_payment_signature"] == "***"
        assert out["model"] == "x"

    def test_redact_keeps_falsy_secrets_as_is(self):
        out = _redact({"api_key": "", "model": "x"})
        assert out["api_key"] == ""


class TestLoadCustomComponents:
    def test_warns_on_bad_path(self, caplog):
        with caplog.at_level(logging.WARNING, logger="koboi.media.registry"):
            load_custom_components(["nonexistent.pkg.module"])  # must not raise
        assert any("nonexistent.pkg.module" in r.message for r in caplog.records)

    def test_imports_real_module_idempotently(self):
        load_custom_components(["koboi.media.providers.mock"])  # no error
