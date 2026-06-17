"""Tests for koboi/hooks/registry.py -- Hook registry and build_hook_chain."""

from __future__ import annotations

import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from koboi.hooks.registry import (
    HookEntry,
    _REGISTRY,
    build_hook_chain,
    register_hook,
    list_entries,
    _build_notif_events,
)
from koboi.hooks.chain import HookEvent, HookChain
from koboi.config import Config
from koboi.logger import AgentLogger


def _write_config(tmp_path, config_data):
    p = tmp_path / "test_config.yaml"
    with open(p, "w") as f:
        yaml.dump(config_data, f)
    return p


def _base_config():
    return {
        "agent": {"name": "test-agent", "max_iterations": 5, "system_prompt": "You are helpful."},
        "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
    }


class TestHookEntry:
    def test_creation(self):
        entry = HookEntry(
            name="TestHook",
            config_key="test.key",
            should_add=lambda config, **kw: True,
            factory=lambda config, **kw: MagicMock(),
        )
        assert entry.name == "TestHook"
        assert entry.config_key == "test.key"

    def test_frozen(self):
        entry = HookEntry(
            name="TestHook",
            config_key="test.key",
            should_add=lambda config, **kw: True,
            factory=lambda config, **kw: MagicMock(),
        )
        with pytest.raises(AttributeError):
            entry.name = "changed"


class TestRegistry:
    def test_registry_has_expected_entries(self):
        names = [e.name for e in _REGISTRY]
        assert "AuditHook" in names
        assert "ModeHook" in names
        assert "TelemetryHook" in names
        assert "CarryoverHook" in names
        assert "DoomLoopHook" in names
        assert "TaskHook" in names
        assert "NotificationHook" in names
        assert "LangfuseTracingHook" in names

    def test_list_entries_returns_copy(self):
        entries = list_entries()
        assert len(entries) >= 8
        # Modifying the copy shouldn't affect the original
        entries.clear()
        assert len(list_entries()) >= 8


class TestBuildHookChain:
    def test_basic_chain_has_logging_hook(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger)
        assert isinstance(chain, HookChain)
        assert len(chain._hooks) >= 1
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "LoggingHook" in hook_names

    def test_chain_with_telemetry(self, tmp_path):
        cfg = _base_config()
        cfg["harness"] = {"telemetry": True}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "TelemetryHook" in hook_names

    def test_chain_with_carryover(self, tmp_path):
        cfg = _base_config()
        cfg["harness"] = {"carryover": True}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "CarryoverHook" in hook_names

    def test_chain_with_doom_loop(self, tmp_path):
        cfg = _base_config()
        cfg["harness"] = {"doom_loop": {"consecutive_identical_threshold": 3}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "DoomLoopHook" in hook_names

    def test_chain_with_tasks(self, tmp_path):
        cfg = _base_config()
        cfg["harness"] = {"tasks": {"reminder_interval": 5}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "TaskHook" in hook_names

    def test_chain_with_audit_trail(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        audit_trail = MagicMock()
        chain = build_hook_chain(config, logger, audit_trail=audit_trail)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "AuditHook" in hook_names

    def test_chain_without_audit_trail(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger, audit_trail=None)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "AuditHook" not in hook_names

    def test_chain_with_mode_manager(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        mode_manager = MagicMock()
        chain = build_hook_chain(config, logger, mode_manager=mode_manager)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "ModeHook" in hook_names

    def test_chain_without_mode_manager(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger, mode_manager=None)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "ModeHook" not in hook_names

    def test_chain_with_notifications(self, tmp_path):
        cfg = _base_config()
        cfg["harness"] = {"notifications": {"enabled": True, "events": ["post_output"]}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "NotificationHook" in hook_names

    def test_chain_notifications_disabled(self, tmp_path):
        cfg = _base_config()
        cfg["harness"] = {"notifications": {"enabled": False}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "NotificationHook" not in hook_names

    def test_chain_with_langfuse(self, tmp_path):
        cfg = _base_config()
        cfg["tracing"] = {"provider": "langfuse", "public_key": "pk", "secret_key": "sk"}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger)
        hook_names = [type(h).__name__ for h in chain._hooks]
        assert "LangfuseTracingHook" in hook_names

    def test_missing_optional_dependency_graceful(self, tmp_path):
        """Hooks with missing optional deps should be skipped, not crash."""
        cfg = _base_config()
        cfg["harness"] = {"telemetry": True}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        # Even if TelemetryCollector import fails, chain should still build
        chain = build_hook_chain(config, logger)
        assert isinstance(chain, HookChain)
        assert len(chain._hooks) >= 1  # At least LoggingHook

    def test_all_hooks_disabled(self, tmp_path):
        """With no config, only LoggingHook is added."""
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = build_hook_chain(config, logger)
        assert len(chain._hooks) == 1
        assert type(chain._hooks[0]).__name__ == "LoggingHook"


class TestNotificationEventMapping:
    def test_post_output(self):
        events = _build_notif_events(["post_output"])
        assert events == [HookEvent.POST_OUTPUT]

    def test_multiple_events(self):
        events = _build_notif_events(["post_output", "session_end", "doom_loop"])
        assert HookEvent.POST_OUTPUT in events
        assert HookEvent.SESSION_END in events
        assert HookEvent.DOOM_LOOP_DETECTED in events

    def test_unknown_event_filtered(self):
        events = _build_notif_events(["post_output", "unknown_event"])
        assert events == [HookEvent.POST_OUTPUT]

    def test_all_unknown_defaults_to_post_output(self):
        events = _build_notif_events(["unknown"])
        assert events == [HookEvent.POST_OUTPUT]

    def test_empty_list_defaults_to_post_output(self):
        events = _build_notif_events([])
        assert events == [HookEvent.POST_OUTPUT]


class TestRegisterHook:
    def test_register_custom_hook(self):
        initial_count = len(list_entries())
        entry = HookEntry(
            name="CustomHook",
            config_key="custom",
            should_add=lambda config, **kw: False,
            factory=lambda config, **kw: MagicMock(),
        )
        register_hook(entry)
        assert len(list_entries()) == initial_count + 1
        # Clean up
        _REGISTRY.remove(entry)
