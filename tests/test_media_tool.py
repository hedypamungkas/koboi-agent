"""Tests for the generate_image tool + facade _build_tools media wiring."""

from __future__ import annotations

import json
from pathlib import Path

from koboi.media.backend import build_media
from koboi.tools.registry import ToolRegistry
from koboi.tools.builtin import register_all
from koboi.types import RiskLevel


def _registry_with_media(tmp_path) -> ToolRegistry:
    registry = ToolRegistry()
    register_all(registry)
    backend = build_media({"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path)}})
    registry.set_dep("media_provider", backend)
    return registry


class TestGenerateImageTool:
    async def test_generates_and_reports_saved_path(self, tmp_path):
        registry = _registry_with_media(tmp_path)
        out = await registry.execute("generate_image", json.dumps({"prompt": "a cat"}))
        assert out.startswith("Image saved:")
        assert str(tmp_path) in out

    async def test_not_configured_when_dep_absent(self):
        registry = ToolRegistry()
        register_all(registry)
        out = await registry.execute("generate_image", json.dumps({"prompt": "a cat"}))
        assert "media not configured" in out

    def test_tool_definition_flags(self):
        registry = ToolRegistry()
        register_all(registry)
        td = registry.get_definition("generate_image")
        assert td is not None
        assert td.risk_level == RiskLevel.MODERATE
        assert td.idempotent is False
        assert td.group == "media"


class TestFacadeBuildToolsMediaWiring:
    @staticmethod
    def _config(media: dict | None) -> "object":
        from koboi.config import Config

        return Config(
            {
                "agent": {"name": "t", "max_iterations": 3, "system_prompt": "x"},
                "llm": {"model": "m", "api_key": "k", "base_url": "http://x/v1"},
                "tools": {"builtin": ["generate_image"]},
                "media": media or {},
            }
        )

    def test_wires_media_when_enabled(self, tmp_path):
        from koboi.facade import _build_tools

        cfg = self._config({"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path)}})
        registry = _build_tools(cfg)
        assert registry.get_dep("media_provider") is not None

    def test_no_media_dep_when_disabled(self):
        from koboi.facade import _build_tools

        cfg = self._config({"enabled": False})
        registry = _build_tools(cfg)
        assert registry.get_dep("media_provider") is None
