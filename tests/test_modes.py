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


class TestReadOnlyToolMatcher:
    """Snake_case builtin names must pass the read-only matcher (Wave 0 fix).

    Before the `_` separator was added, is_read_only_tool matched only exact
    names or `prefix.`-namespaced names, so CHAT/PLAN blocked every builtin --
    including read_file and grep_search.
    """

    def test_builtin_read_tools_are_read_only(self):
        from koboi.modes import is_read_only_tool

        for name in [
            "read_file",
            "list_files",
            "grep_search",
            "glob_find",
            "git_status",
            "git_log",
            "git_diff",
            "web_search",
            "web_fetch",
            "calculator",
            "delegate_tasks",
        ]:
            assert is_read_only_tool(name), f"{name} should be read-only"

    def test_mutating_tools_are_not_read_only(self):
        from koboi.modes import is_read_only_tool

        # Comprehensive: every shipped mutating tool must be gated in CHAT/PLAN.
        for name in [
            "write_file",
            "edit_file",
            "apply_patch",
            "delete_file",
            "run_shell",
            "git_add",
            "git_commit",
            "git_checkout",
            "git_push",
            "github_create_pr",
            "github_update_pr",
            "submit_background_shell",
            "kill_background_shell",
            "generate_image",
            "generate_video",
            "generate_music",
            "generate_speech",
            "memory_store",
            "ingest_url",
            "task_create",
            "task_update",
            "transfer_to_human",
        ]:
            assert not is_read_only_tool(name), f"{name} must NOT be read-only"

    def test_wave3_wave4_read_only_additions(self):
        from koboi.modes import is_read_only_tool

        for name in ["repo_map", "github_list_prs", "github_get_pr", "run_typecheck"]:
            assert is_read_only_tool(name), f"{name} should be read-only"

    def test_registry_snapshot_no_mutating_tool_is_read_only(self):
        """Durable guard for the ``prefix_`` footgun: ``is_read_only_tool``
        matches ``prefix + "_"`` for every read-only base, so a hypothetical
        future tool named ``git_diff_apply`` would inherit read-only status via
        the ``git_diff`` prefix and bypass CHAT/PLAN approval.

        This snapshot iterates the REAL registered tool inventory and fails at
        CI the moment any destructive tool (or any moderate tool that is
        read-only ONLY via the prefix rule, not by exact intent) collides.
        """
        from koboi.modes import _READ_ONLY_TOOLS, is_read_only_tool
        from koboi.tools.builtin import register_all
        from koboi.tools.registry import ToolRegistry
        from koboi.types import RiskLevel

        reg = ToolRegistry()
        register_all(reg)
        offenders: list[str] = []
        for name, td in reg.list_tools().items():
            ro = is_read_only_tool(name)
            if td.risk_level == RiskLevel.DESTRUCTIVE and ro:
                offenders.append(name)  # a destructive tool must NEVER be read-only
            elif (
                td.risk_level == RiskLevel.MODERATE
                and ro
                and name.lower() not in _READ_ONLY_TOOLS  # exact members (web_fetch, delegate_tasks) are intentional
            ):
                offenders.append(name)  # read-only only via the prefix footgun
        assert offenders == [], (
            f"mutating tools wrongly classified read-only (CHAT/PLAN would bypass approval): {offenders}"
        )

    def test_mode_manager_gates_edit_file_by_mode(self):
        mgr = ModeManager(AgentMode.CHAT)
        allowed, reason = mgr.is_tool_allowed("edit_file")
        assert allowed is False
        assert "CHAT mode" in reason

        read_allowed, _ = mgr.is_tool_allowed("read_file")
        assert read_allowed is True

        mgr.switch_mode(AgentMode.PLAN)
        allowed, reason = mgr.is_tool_allowed("edit_file")
        assert allowed is False
        assert "PLAN mode" in reason

        mgr.switch_mode(AgentMode.ACT)
        allowed, reason = mgr.is_tool_allowed("edit_file")
        assert allowed is True
        assert reason == ""
