"""Tests for koboi/modes.py -- Agent interaction modes."""

from __future__ import annotations

import pytest

from koboi.modes import AgentMode, ModeConfig, ModeManager


class TestAgentMode:
    def test_enum_values(self):
        assert AgentMode.CHAT.value == "chat"
        assert AgentMode.PLAN.value == "plan"
        assert AgentMode.ACT.value == "act"
        assert AgentMode.AUTO.value == "auto"
        assert AgentMode.YOLO.value == "yolo"

    def test_enum_members(self):
        assert len(AgentMode) == 5


class TestModeConfig:
    def test_chat_config(self):
        cfg = ModeManager.get_config(AgentMode.CHAT)
        assert cfg.allow_file_write is False
        assert cfg.allow_shell is False
        assert cfg.require_plan_approval is False
        assert cfg.permission_level == "always_ask"
        assert "CHAT mode" in cfg.system_prompt_suffix

    def test_plan_config(self):
        cfg = ModeManager.get_config(AgentMode.PLAN)
        assert cfg.allow_file_write is False
        assert cfg.allow_shell is False
        assert cfg.require_plan_approval is True
        assert "PLAN mode" in cfg.system_prompt_suffix

    def test_act_config(self):
        cfg = ModeManager.get_config(AgentMode.ACT)
        assert cfg.allow_file_write is True
        assert cfg.allow_shell is True
        assert cfg.require_plan_approval is False
        assert "ACT mode" in cfg.system_prompt_suffix

    def test_auto_config(self):
        cfg = ModeManager.get_config(AgentMode.AUTO)
        assert cfg.allow_file_write is True
        assert cfg.allow_shell is True
        assert cfg.permission_level == "graduated"
        assert "AUTO mode" in cfg.system_prompt_suffix

    def test_frozen_dataclass(self):
        cfg = ModeManager.get_config(AgentMode.CHAT)
        with pytest.raises(AttributeError):
            cfg.allow_file_write = True


class TestModeManager:
    def test_default_mode(self):
        mgr = ModeManager()
        assert mgr.current_mode == AgentMode.CHAT

    def test_custom_initial_mode(self):
        mgr = ModeManager(AgentMode.ACT)
        assert mgr.current_mode == AgentMode.ACT

    def test_switch_mode(self):
        mgr = ModeManager()
        mgr.switch_mode(AgentMode.ACT)
        assert mgr.current_mode == AgentMode.ACT

    def test_switch_same_mode_noop(self):
        mgr = ModeManager()
        calls = []
        mgr.on_mode_change(lambda o, n: calls.append((o, n)))
        mgr.switch_mode(AgentMode.CHAT)
        assert len(calls) == 0

    def test_cycle_mode(self):
        mgr = ModeManager()
        assert mgr.cycle_mode() == AgentMode.PLAN
        assert mgr.cycle_mode() == AgentMode.ACT
        assert mgr.cycle_mode() == AgentMode.AUTO
        assert mgr.cycle_mode() == AgentMode.YOLO
        assert mgr.cycle_mode() == AgentMode.CHAT

    def test_on_mode_change_listener(self):
        mgr = ModeManager()
        changes = []
        mgr.on_mode_change(lambda old, new: changes.append((old, new)))
        mgr.switch_mode(AgentMode.ACT)
        assert changes == [(AgentMode.CHAT, AgentMode.ACT)]

    def test_config_property(self):
        mgr = ModeManager(AgentMode.PLAN)
        assert mgr.config.require_plan_approval is True

    def test_from_string_valid(self):
        assert ModeManager.from_string("chat") == AgentMode.CHAT
        assert ModeManager.from_string("ACT") == AgentMode.ACT
        assert ModeManager.from_string("Auto") == AgentMode.AUTO

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            ModeManager.from_string("invalid")

    def test_get_config_static(self):
        cfg = ModeManager.get_config(AgentMode.AUTO)
        assert isinstance(cfg, ModeConfig)

    def test_from_string_yolo(self):
        assert ModeManager.from_string("yolo") == AgentMode.YOLO
        assert ModeManager.from_string("YOLO") == AgentMode.YOLO


class TestYoloMode:
    def test_yolo_config(self):
        cfg = ModeManager.get_config(AgentMode.YOLO)
        assert cfg.allow_file_write is True
        assert cfg.allow_shell is True
        assert cfg.require_plan_approval is False
        assert cfg.permission_level == "yolo"
        assert "YOLO mode" in cfg.system_prompt_suffix

    def test_yolo_in_cycle(self):
        mgr = ModeManager()
        mgr.switch_mode(AgentMode.AUTO)
        assert mgr.cycle_mode() == AgentMode.YOLO
        assert mgr.cycle_mode() == AgentMode.CHAT

    def test_switch_to_yolo(self):
        mgr = ModeManager()
        mgr.switch_mode(AgentMode.YOLO)
        assert mgr.current_mode == AgentMode.YOLO
        assert mgr.config.permission_level == "yolo"
