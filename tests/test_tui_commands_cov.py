"""koboi/tui/commands.py -- branch coverage for the slash-command registry.

Most handlers are exercised in console mode (``app=None``) against a MagicMock
agent, which is where the bulk of the logic lives. Subprocess/editor paths are
mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koboi.tui.commands import (
    CommandContext,
    CommandResult,
    SlashCommand,
    SlashCommandRegistry,
    build_registry,
    _cmd_reset,
    _cmd_info,
    _cmd_history,
    _cmd_tools,
    _cmd_help,
    _cmd_theme,
    _cmd_sessions,
    _cmd_fork,
    _cmd_export,
    _cmd_skills,
    _cmd_mode,
    _cmd_tasks,
    _cmd_compact,
    _cmd_model,
    _cmd_undo,
    _cmd_vim,
    _cmd_copy,
    _cmd_run,
    _cmd_kill,
    _cmd_subagents,
    _cmd_diagnostics,
    _cmd_quit,
)


def _ctx(agent: MagicMock, args: str = "", out: list | None = None) -> CommandContext:
    return CommandContext(agent=agent, output=(out.append if out is not None else (lambda _x: None)), args=args)


def _fake_agent(messages=None, tools=None, guardrails=True, skills_reg=None) -> MagicMock:
    agent = MagicMock()
    agent.config.agent_name = "test-agent"
    agent.config.provider = "openai"
    agent.config.model = "gpt-test"
    agent.config.max_iterations = 5
    agent.config.rag_enabled = False
    agent.config.get = MagicMock(return_value={"web_hook": {}})
    agent.config.raw = {}

    core = MagicMock()
    core.input_guardrail = MagicMock() if guardrails else None
    core.output_guardrail = MagicMock() if guardrails else None
    core.rate_limiter = None
    core.approval_handler = None
    core.context_manager = MagicMock()
    core.client = MagicMock()
    core.client.provider = "openai"
    core.client.model = "gpt-test"
    core.skills = skills_reg

    # tools dict
    td = MagicMock()
    td.risk_level.value = "safe"
    core.tools.list_tools.return_value = tools if tools is not None else {"calc": td, "fs": td}

    core.memory.get_messages.return_value = messages if messages is not None else []
    core.memory.replace_messages = MagicMock()
    agent.core = core
    agent.mode_manager = MagicMock()
    agent.mode_manager.current_mode.value = "chat"
    agent._orchestrator = None
    return agent


class TestRegistry:
    async def test_dispatch_unknown_returns_none(self):
        reg = SlashCommandRegistry()
        assert await reg.dispatch("/nope", _ctx(_fake_agent())) is None

    async def test_dispatch_known_handler(self):
        reg = SlashCommandRegistry()

        async def handler(ctx):
            return CommandResult(message="ok")

        reg.register(SlashCommand("/x", "x"), handler)
        res = await reg.dispatch("/x", _ctx(_fake_agent()))
        assert res.message == "ok"

    def test_get_all_names_and_help(self):
        reg = SlashCommandRegistry()

        async def h(ctx):
            return CommandResult()

        reg.register(SlashCommand("/a", "desc a", aliases=["/alpha"]), h)
        reg.register(SlashCommand("/b", "desc b"), h)
        names = reg.get_all_names()
        assert "/a" in names and "/alpha" in names and "/b" in names
        help_text = reg.get_help_text()
        assert "desc a" in help_text and "alias: /alpha" in help_text

    def test_build_registry_has_all_commands(self):
        reg = build_registry()
        names = reg.get_all_names()
        for n in ["/reset", "/info", "/history", "/tools", "/help", "/mode", "/quit"]:
            assert n in names


class TestSimpleHandlers:
    async def test_reset(self):
        agent = _fake_agent()
        out: list = []
        res = await _cmd_reset(_ctx(agent, out=out))
        agent.reset.assert_called_once()
        assert res.clear_chat is True
        assert out and "reset" in out[0].lower()

    async def test_info_with_tools_and_guardrails(self):
        out: list = []
        await _cmd_info(_ctx(_fake_agent(guardrails=True), out=out))
        joined = "\n".join(out)
        assert "Agent: test-agent" in joined
        assert "Tools (2):" in joined
        assert "Guardrails: input, output" in joined

    async def test_info_no_core(self):
        agent = _fake_agent()
        agent.core = None
        out: list = []
        await _cmd_info(_ctx(agent, out=out))
        assert any("Agent: test-agent" in o for o in out)

    async def test_info_many_tools_truncates(self):
        tools = {f"tool_{i}": MagicMock(risk_level=MagicMock(value="safe")) for i in range(12)}
        out: list = []
        await _cmd_info(_ctx(_fake_agent(tools=tools), out=out))
        assert any("+4" in o for o in out)

    async def test_info_rag_enabled(self):
        agent = _fake_agent()
        agent.config.rag_enabled = True
        out: list = []
        await _cmd_info(_ctx(agent, out=out))
        assert any("RAG: enabled" in o for o in out)

    async def test_history_empty(self):
        out: list = []
        await _cmd_history(_ctx(_fake_agent(messages=[]), out=out))
        assert any("No messages" in o for o in out)

    async def test_history_with_messages(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x" * 200}]
        out: list = []
        await _cmd_history(_ctx(_fake_agent(messages=msgs), out=out))
        assert any("[0] user: hi" in o for o in out)
        assert any("..." in o for o in out)

    async def test_tools_empty(self):
        agent = _fake_agent(tools={})
        out: list = []
        await _cmd_tools(_ctx(agent, out=out))
        assert any("No tools" in o for o in out)

    async def test_tools_listed(self):
        out: list = []
        await _cmd_tools(_ctx(_fake_agent(), out=out))
        assert any("Registered Tools:" in o for o in out)

    async def test_help_console(self):
        out: list = []
        await _cmd_help(_ctx(_fake_agent(), out=out))
        assert any("/reset" in o for o in out)

    async def test_theme_requires_tui(self):
        out: list = []
        await _cmd_theme(_ctx(_fake_agent(), out=out))
        assert any("requires the TUI" in o for o in out)

    async def test_sessions_requires_tui(self):
        out: list = []
        await _cmd_sessions(_ctx(_fake_agent(), out=out))
        assert any("requires the TUI" in o for o in out)

    async def test_vim_requires_tui(self):
        out: list = []
        await _cmd_vim(_ctx(_fake_agent(), out=out))
        assert any("requires the TUI" in o for o in out)


class TestFork:
    async def test_fork_requires_sqlite(self):
        agent = _fake_agent()
        # spec-limited mock: hasattr(fork_session)/hasattr(db_path) are False
        agent.core.memory = MagicMock(spec=["get_messages"])
        out: list = []
        await _cmd_fork(_ctx(agent, out=out))
        assert any("SQLite" in o for o in out)

    async def test_fork_success(self):
        agent = _fake_agent()
        agent.core.memory.fork_session = MagicMock(return_value="abcdef1234567890")
        agent.core.memory.db_path = "/tmp/x.db"
        agent.core.memory.fork_and_switch = MagicMock(return_value="abcdef1234567890")
        out: list = []
        res = await _cmd_fork(_ctx(agent, out=out))
        assert res.clear_chat and res.repopulate_messages
        assert any("abcdef12" in o for o in out)


class TestExport:
    async def test_export_invalid_format(self):
        out: list = []
        await _cmd_export(_ctx(_fake_agent(messages=[{"role": "user", "content": "hi"}]), args="pdf", out=out))
        assert any("Usage" in o for o in out)

    async def test_export_markdown(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out: list = []
        res = await _cmd_export(_ctx(_fake_agent(messages=[{"role": "user", "content": "hi"}]), args="md", out=out))
        assert res.handled
        exported = list(tmp_path.glob("export_*.md"))
        assert exported and exported[0].read_text()

    async def test_export_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out: list = []
        await _cmd_export(_ctx(_fake_agent(messages=[{"role": "user", "content": "hi"}]), args="json", out=out))
        assert list(tmp_path.glob("export_*.json"))

    async def test_export_html(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out: list = []
        await _cmd_export(_ctx(_fake_agent(messages=[{"role": "user", "content": "hi"}]), args="html", out=out))
        assert list(tmp_path.glob("export_*.html"))


class TestSkills:
    async def test_no_skills_config(self):
        agent = _fake_agent(skills_reg=None)
        out: list = []
        await _cmd_skills(_ctx(agent, out=out))
        assert any("No skills configured" in o for o in out)

    async def test_skills_empty(self):
        reg = MagicMock()
        reg.list_skills.return_value = []
        out: list = []
        await _cmd_skills(_ctx(_fake_agent(skills_reg=reg), out=out))
        assert any("No skills discovered" in o for o in out)

    async def test_skills_listed(self):
        s = MagicMock()
        s.name = "review"
        s.description = "short"
        reg = MagicMock()
        reg.list_skills.return_value = [s]
        out: list = []
        await _cmd_skills(_ctx(_fake_agent(skills_reg=reg), out=out))
        assert any("review: short" in o for o in out)


class TestMode:
    async def test_mode_show_current(self):
        out: list = []
        await _cmd_mode(_ctx(_fake_agent(), out=out))
        assert any("Current mode: CHAT" in o for o in out)

    async def test_mode_invalid(self):
        out: list = []
        await _cmd_mode(_ctx(_fake_agent(), args="bogus", out=out))
        assert len(out) == 1

    async def test_mode_switch_non_yolo(self):
        agent = _fake_agent()
        out: list = []
        await _cmd_mode(_ctx(agent, args="act", out=out))
        agent.mode_manager.switch_mode.assert_called_once()
        assert any("ACT" in o for o in out)

    async def test_mode_yolo_console_decline(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *_a: "n")
        out: list = []
        await _cmd_mode(_ctx(_fake_agent(), args="yolo", out=out))
        assert any("cancelled" in o for o in out)

    async def test_mode_yolo_console_confirm(self, monkeypatch):
        agent = _fake_agent()
        monkeypatch.setattr("builtins.input", lambda *_a: "y")
        out: list = []
        await _cmd_mode(_ctx(agent, args="yolo", out=out))
        agent.mode_manager.switch_mode.assert_called_once()

    async def test_mode_yolo_console_eof(self, monkeypatch):
        def _raise(*_a):
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise)
        out: list = []
        await _cmd_mode(_ctx(_fake_agent(), args="yolo", out=out))
        assert any("cancelled" in o for o in out)


class TestTasks:
    async def test_no_task_manager(self):
        agent = _fake_agent()
        agent.core.tools.get_dep.return_value = None
        out: list = []
        await _cmd_tasks(_ctx(agent, out=out))
        assert any("not initialized" in o for o in out)

    async def test_tasks_empty(self):
        agent = _fake_agent()
        mgr = MagicMock()
        mgr.list_tasks.return_value = []
        agent.core.tools.get_dep.return_value = mgr
        out: list = []
        await _cmd_tasks(_ctx(agent, out=out))
        assert any("No tasks" in o for o in out)

    async def test_tasks_listed(self):
        agent = _fake_agent()
        t = MagicMock()
        t.id = "1"
        t.subject = "do"
        t.description = "desc"
        t.status = "pending"
        t.blocked_by = ["2"]
        mgr = MagicMock()
        mgr.list_tasks.return_value = [t]
        agent.core.tools.get_dep.return_value = mgr
        out: list = []
        await _cmd_tasks(_ctx(agent, args="pending", out=out))
        assert any("[pending] 1: do" in o for o in out)
        assert any("blocked by: 2" in o for o in out)


class TestCompact:
    async def test_no_context_manager(self):
        agent = _fake_agent()
        agent.core.context_manager = None
        out: list = []
        await _cmd_compact(_ctx(agent, out=out))
        assert any("No context strategy" in o for o in out)

    async def test_compact_runs(self):
        from unittest.mock import AsyncMock

        agent = _fake_agent(messages=[{"role": "user", "content": "hi"}])
        agent.core.context_manager.manage = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        out: list = []
        await _cmd_compact(_ctx(agent, out=out))
        agent.core.memory.replace_messages.assert_called_once()
        assert any("Compacted" in o for o in out)


class TestModel:
    async def test_no_core_or_orchestrator(self):
        agent = _fake_agent()
        agent.core = None
        agent._orchestrator = None
        out: list = []
        await _cmd_model(_ctx(agent, args="x", out=out))
        assert any("No agent core" in o for o in out)

    async def test_show_current_model(self):
        out: list = []
        await _cmd_model(_ctx(_fake_agent(), out=out))
        assert any("Current model: openai/gpt-test" in o for o in out)

    async def test_pool_disabled(self):
        agent = _fake_agent()
        # not a RetryClient -> disabled
        out: list = []
        await _cmd_model(_ctx(agent, args="gpt-4", out=out))
        assert any("disabled for provider pools" in o for o in out)

    async def test_switch_same_provider(self, monkeypatch):
        from koboi import client as client_mod

        agent = _fake_agent()
        real_client = MagicMock(spec=client_mod.RetryClient)
        real_client.provider = "openai"
        real_client.api_key = "k"
        real_client.base_url = "u"
        real_client.model = "gpt-test"
        real_client.logger = MagicMock()
        real_client.temperature = 0.7
        agent.core.client = real_client
        out: list = []
        await _cmd_model(_ctx(agent, args="gpt-4o", out=out))
        assert any("Switched model to: openai/gpt-4o" in o for o in out)

    async def test_switch_new_provider(self, monkeypatch):
        from koboi import client as client_mod

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        agent = _fake_agent()
        real_client = MagicMock(spec=client_mod.RetryClient)
        real_client.provider = "anthropic"
        real_client.model = "claude"
        real_client.logger = MagicMock()
        real_client.temperature = 0.5
        agent.core.client = real_client
        out: list = []
        await _cmd_model(_ctx(agent, args="openai/gpt-4o", out=out))
        assert any("Switched model to: openai/gpt-4o" in o for o in out)


class TestUndo:
    async def test_invalid_n(self):
        out: list = []
        await _cmd_undo(_ctx(_fake_agent(), args="abc", out=out))
        assert any("Usage" in o for o in out)

    async def test_n_out_of_range(self):
        out: list = []
        await _cmd_undo(_ctx(_fake_agent(), args="99", out=out))
        assert any("1-10" in o for o in out)

    async def test_git_log_error(self, monkeypatch):
        res = MagicMock(returncode=1, stderr="no git", stdout="")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: res)
        out: list = []
        await _cmd_undo(_ctx(_fake_agent(), args="1", out=out))
        assert any("Git error" in o for o in out)

    async def test_git_log_success_then_revert(self, monkeypatch):
        calls = {"n": 0}

        def fake_run(cmd, *a, **k):
            calls["n"] += 1
            if "log" in cmd:
                return MagicMock(returncode=0, stdout="abc1234 msg\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        out: list = []
        await _cmd_undo(_ctx(_fake_agent(), args="1", out=out))
        assert any("Reverted 1/1" in o for o in out)

    async def test_revert_failure_midway(self, monkeypatch):
        def fake_run(cmd, *a, **k):
            if "log" in cmd:
                return MagicMock(returncode=0, stdout="m1\nm2\n", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="conflict")

        monkeypatch.setattr("subprocess.run", fake_run)
        out: list = []
        await _cmd_undo(_ctx(_fake_agent(), args="2", out=out))
        assert any("Reverted 0/2" in o for o in out)


class TestCopy:
    async def test_no_assistant_message(self):
        out: list = []
        await _cmd_copy(_ctx(_fake_agent(messages=[{"role": "user", "content": "hi"}]), out=out))
        assert any("No assistant message" in o for o in out)

    async def test_copy_via_pyperclip(self, monkeypatch):
        import sys

        fake_mod = MagicMock()
        monkeypatch.setitem(sys.modules, "pyperclip", fake_mod)
        msgs = [{"role": "assistant", "content": "answer"}]
        out: list = []
        await _cmd_copy(_ctx(_fake_agent(messages=msgs), out=out))
        fake_mod.copy.assert_called_once_with("answer")
        assert any("Copied" in o for o in out)

    async def test_copy_no_backend(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "pyperclip", None)
        monkeypatch.setattr("shutil.which", lambda *_a: None)
        msgs = [{"role": "assistant", "content": "answer"}]
        out: list = []
        await _cmd_copy(_ctx(_fake_agent(messages=msgs), out=out))
        assert any("No clipboard backend" in o for o in out)


class TestRun:
    async def test_no_args(self):
        out: list = []
        await _cmd_run(_ctx(_fake_agent(), out=out))
        assert any("Usage" in o for o in out)

    async def test_config_not_found(self):
        out: list = []
        await _cmd_run(_ctx(_fake_agent(), args="/no/such.yaml", out=out))
        assert any("Config not found" in o for o in out)

    async def test_load_error(self, monkeypatch, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agent: {name: x}\n")
        monkeypatch.setattr("koboi.facade.KoboiAgent.from_config", MagicMock(side_effect=RuntimeError("boom")))
        out: list = []
        await _cmd_run(_ctx(_fake_agent(), args=str(cfg), out=out))
        assert any("Error loading config" in o for o in out)

    async def test_console_replace_success(self, monkeypatch, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agent: {name: x}\n")
        new_agent = MagicMock()
        new_agent.config.agent_name = "x"
        new_agent.config.provider = "openai"
        new_agent.config.model = "m"
        monkeypatch.setattr("koboi.facade.KoboiAgent.from_config", MagicMock(return_value=new_agent))
        agent = _fake_agent()
        out: list = []
        res = await _cmd_run(_ctx(agent, args=f"{cfg} hello", out=out))
        agent.replace_from.assert_called_once_with(new_agent)
        assert res.message == "hello"


class TestKillAndSubagents:
    async def test_kill_no_manager(self):
        agent = _fake_agent()
        agent.core.tools.get_dep.return_value = None
        out: list = []
        await _cmd_kill(_ctx(agent, out=out))
        assert any("not initialized" in o for o in out)

    async def test_kill_by_label_found(self):
        agent = _fake_agent()
        mgr = MagicMock()
        mgr.cancel_task.return_value = True
        agent.core.tools.get_dep.return_value = mgr
        out: list = []
        await _cmd_kill(_ctx(agent, args="worker1", out=out))
        assert any("Cancelled subagent: worker1" in o for o in out)

    async def test_kill_by_label_not_found_with_running(self):
        agent = _fake_agent()
        mgr = MagicMock()
        mgr.cancel_task.return_value = False
        mgr.list_running.return_value = ["other"]
        agent.core.tools.get_dep.return_value = mgr
        out: list = []
        await _cmd_kill(_ctx(agent, args="x", out=out))
        assert any("Running: other" in o for o in out)

    async def test_kill_by_label_not_found_none_running(self):
        agent = _fake_agent()
        mgr = MagicMock()
        mgr.cancel_task.return_value = False
        mgr.list_running.return_value = []
        agent.core.tools.get_dep.return_value = mgr
        out: list = []
        await _cmd_kill(_ctx(agent, args="x", out=out))
        assert any("No subagents active" in o for o in out)

    async def test_kill_all_running(self):
        agent = _fake_agent()
        mgr = MagicMock()
        mgr.list_running.return_value = ["a", "b"]
        agent.core.tools.get_dep.return_value = mgr
        out: list = []
        await _cmd_kill(_ctx(agent, out=out))
        assert any("Cancelled 2 subagent" in o for o in out)

    async def test_kill_all_none_running(self):
        agent = _fake_agent()
        mgr = MagicMock()
        mgr.list_running.return_value = []
        agent.core.tools.get_dep.return_value = mgr
        out: list = []
        await _cmd_kill(_ctx(agent, out=out))
        assert any("No running subagents to cancel" in o for o in out)

    async def test_subagents_no_manager(self):
        agent = _fake_agent()
        agent.core.tools.get_dep.return_value = None
        out: list = []
        await _cmd_subagents(_ctx(agent, out=out))
        assert any("not initialized" in o for o in out)

    async def test_subagents_running(self):
        agent = _fake_agent()
        mgr = MagicMock()
        mgr.list_running.return_value = ["w1"]
        agent.core.tools.get_dep.return_value = mgr
        out: list = []
        await _cmd_subagents(_ctx(agent, out=out))
        assert any("Running subagents (1)" in o for o in out)

    async def test_subagents_none(self):
        agent = _fake_agent()
        mgr = MagicMock()
        mgr.list_running.return_value = []
        agent.core.tools.get_dep.return_value = mgr
        out: list = []
        await _cmd_subagents(_ctx(agent, out=out))
        assert any("No subagents currently running" in o for o in out)


class TestDiagnostics:
    async def test_diagnostics_success(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("koboi.diagnostics.collect_diagnostics", lambda *_a: b"PK\x03\x04data")
        out: list = []
        await _cmd_diagnostics(_ctx(_fake_agent(), out=out))
        assert any("Diagnostics exported" in o for o in out)
        assert list(tmp_path.glob("diagnostics_*.zip"))

    async def test_diagnostics_error(self, monkeypatch):
        monkeypatch.setattr("koboi.diagnostics.collect_diagnostics", MagicMock(side_effect=RuntimeError("fail")))
        out: list = []
        await _cmd_diagnostics(_ctx(_fake_agent(), out=out))
        assert any("Error generating diagnostics" in o for o in out)


class TestQuit:
    async def test_quit_console_raises_systemexit(self):
        with pytest.raises(SystemExit):
            await _cmd_quit(_ctx(_fake_agent()))
