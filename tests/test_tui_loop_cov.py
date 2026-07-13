"""koboi/tui/loop.py -- branch coverage for build_slash_commands handlers + loop helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from koboi.events import (
    CompleteEvent,
    ErrorEvent,
    IterationEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from koboi.tui import loop as tui_loop
from koboi.tui.loop import _print_summary, _stream_response, build_slash_commands, interactive_loop


def _agent(*, messages=None, tools=None, core_none=False, orchestrator_none=True) -> MagicMock:
    agent = MagicMock()
    if core_none:
        agent.core = None
    else:
        td = MagicMock()
        td.risk_level.value = "safe"
        agent.core.tools.list_tools.return_value = tools if tools is not None else {"calc": td}
        agent.core.memory.get_messages.return_value = messages if messages is not None else []
        agent.core.memory.replace_messages = MagicMock()
        agent.core.context_manager = MagicMock()
        agent.core.client = MagicMock(provider="openai", model="gpt-test")
    agent._orchestrator = None if orchestrator_none else MagicMock()
    agent.config.raw = {}
    return agent


class TestHandlers:
    def test_reset_and_help(self):
        cmds = build_slash_commands(_agent())
        console = MagicMock()
        cmds["/reset"](_agent(), console)
        cmds["/help"](_agent(), console)
        assert console.print.call_count >= 2

    def test_history_empty_and_full(self):
        cmds = build_slash_commands(_agent())
        console = MagicMock()
        cmds["/history"](_agent(messages=[]), console)
        cmds["/history"](_agent(messages=[{"role": "user", "content": "x" * 200}]), console)
        assert console.print.call_count >= 2

    def test_tools_empty_and_full_with_styles(self):
        cmds = build_slash_commands(_agent())
        console = MagicMock()
        cmds["/tools"](_agent(tools={}), console)
        td = MagicMock(risk_level=MagicMock(value="destructive"))
        cmds["/tools"](_agent(tools={"rm": td}), console)
        assert console.print.call_count >= 2

    def test_info(self):
        cmds = build_slash_commands(_agent())
        cmds["/info"](_agent(), MagicMock())

    def test_run_no_args_notfound_error_success(self, monkeypatch, tmp_path):
        cmds = build_slash_commands(_agent())

        # no args -> usage hint
        c0 = MagicMock()
        cmds["/run"](_agent(), c0)
        assert any("Usage" in str(a) for a in c0.print.call_args_list)

        # file does not exist -> "Config not found" (checked before from_config)
        c1 = MagicMock()
        cmds["/run"](_agent(), c1, args="/no/such.yaml")
        assert any("Config not found" in str(a) for a in c1.print.call_args_list)

        # file EXISTS but from_config raises -> "Error loading config"
        err_cfg = tmp_path / "err.yaml"
        err_cfg.write_text("agent: {name: x}\n")
        monkeypatch.setattr("koboi.facade.KoboiAgent.from_config", MagicMock(side_effect=RuntimeError("boom")))
        c2 = MagicMock()
        cmds["/run"](_agent(), c2, args=str(err_cfg))
        assert any("Error loading config" in str(a) for a in c2.print.call_args_list)

        # success -> returns the run message and hot-swaps the agent
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agent: {name: x}\n")
        new_agent = MagicMock()
        monkeypatch.setattr("koboi.facade.KoboiAgent.from_config", MagicMock(return_value=new_agent))
        agent = _agent()
        ret = cmds["/run"](agent, MagicMock(), args=f"{cfg} hi")
        assert ret == "hi"
        agent.replace_from.assert_called_once_with(new_agent)

    async def test_compact_no_strategy_and_success(self):
        cmds = build_slash_commands(_agent())
        console = MagicMock()
        agent = _agent()
        agent.core.context_manager = None
        await cmds["/compact"](agent, console)  # no strategy
        agent2 = _agent(messages=[{"role": "user", "content": "hi"}])
        agent2.core.context_manager.manage = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        await cmds["/compact"](agent2, console)
        agent2.core.memory.replace_messages.assert_called_once()

    def test_model_variants(self, monkeypatch):
        cmds = build_slash_commands(_agent())
        console = MagicMock()
        # no core/orchestrator
        cmds["/model"](_agent(core_none=True), console, args="x")
        # show current
        cmds["/model"](_agent(), console)
        # pool disabled
        cmds["/model"](_agent(), console, args="gpt-4")
        # same provider switch
        from koboi import client as client_mod

        rc = MagicMock(spec=client_mod.RetryClient)
        rc.provider = "openai"
        rc.api_key = "k"
        rc.base_url = "u"
        rc.model = "gpt"
        rc.logger = MagicMock()
        rc.temperature = 0.5
        agent = _agent()
        agent.core.client = rc
        cmds["/model"](agent, console, args="gpt-4o")
        # new provider switch
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        rc2 = MagicMock(spec=client_mod.RetryClient)
        rc2.provider = "anthropic"
        rc2.model = "c"
        rc2.logger = MagicMock()
        rc2.temperature = 0.5
        agent2 = _agent()
        agent2.core.client = rc2
        cmds["/model"](agent2, console, args="openai/gpt-4o")

    def test_undo_variants(self, monkeypatch):
        cmds = build_slash_commands(_agent())
        console = MagicMock()
        cmds["/undo"](_agent(), console, args="abc")  # invalid
        cmds["/undo"](_agent(), console, args="99")  # out of range

        def fake_run(cmd, *a, **k):
            if "log" in cmd:
                return MagicMock(returncode=0, stdout="m1\n", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="conflict")

        monkeypatch.setattr("subprocess.run", fake_run)
        cmds["/undo"](_agent(), console, args="1")

        def fake_run2(cmd, *a, **k):
            if "log" in cmd:
                return MagicMock(returncode=1, stderr="no git", stdout="")
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run2)
        cmds["/undo"](_agent(), console, args="1")

        def fake_run3(cmd, *a, **k):
            if "log" in cmd:
                return MagicMock(returncode=0, stdout="m1\nm2\n", stderr="")
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run3)
        cmds["/undo"](_agent(), console, args="2")

    def test_copy_variants(self, monkeypatch):
        cmds = build_slash_commands(_agent())
        console = MagicMock()
        cmds["/copy"](_agent(messages=[{"role": "user", "content": "x"}]), console)  # no assistant
        import sys

        fake = MagicMock()
        monkeypatch.setitem(sys.modules, "pyperclip", fake)
        cmds["/copy"](_agent(messages=[{"role": "assistant", "content": "answer"}]), console)
        fake.copy.assert_called_once()
        monkeypatch.setitem(sys.modules, "pyperclip", None)
        monkeypatch.setattr("shutil.which", lambda *_a: None)
        cmds["/copy"](_agent(messages=[{"role": "assistant", "content": "answer"}]), console)

    def test_diagnostics_success_and_error(self, monkeypatch, tmp_path):
        cmds = build_slash_commands(_agent())
        console = MagicMock()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("koboi.diagnostics.collect_diagnostics", lambda *_a: b"PKzip")
        cmds["/diagnostics"](_agent(), console)
        monkeypatch.setattr("koboi.diagnostics.collect_diagnostics", MagicMock(side_effect=RuntimeError("x")))
        cmds["/diagnostics"](_agent(), console)


class TestPrintSummary:
    def test_with_agent_and_without(self):
        console = MagicMock()
        _print_summary(console, 3, 0.0, None)
        agent = _agent()
        agent.core.memory = [1, 2, 3]  # len works
        _print_summary(console, 3, 0.0, agent)
        # agent.core.memory raises -> swallowed
        agent2 = MagicMock()
        type(agent2).core = property(lambda *_a: (_ for _ in ()).throw(RuntimeError("x")))
        _print_summary(console, 3, 0.0, agent2)


class TestStreamResponse:
    async def test_all_event_types(self, monkeypatch):
        updates = []
        live_cm = MagicMock()
        live_cm.__enter__ = lambda self: live_cm
        live_cm.__exit__ = lambda *a: None
        live_cm.update = lambda payload: updates.append(payload)
        monkeypatch.setattr(tui_loop, "Live", lambda **kw: live_cm)

        events = [
            TextDeltaEvent(content="hel"),
            TextDeltaEvent(content="lo"),
            ToolCallEvent(tool_name="get_weather", tool_call_id="1", arguments="{}"),
            ToolResultEvent(tool_name="get_weather", tool_call_id="1", result="ok"),
            IterationEvent(iteration=1),
            CompleteEvent(response=MagicMock(), content="hello final"),
            ErrorEvent(error="boom"),
        ]

        async def gen(msg):
            for e in events:
                yield e

        agent = MagicMock()
        agent.run_stream = gen
        await _stream_response(agent, "hi", MagicMock(), "T")
        assert len(updates) >= len(events) - 2  # ToolResult/Iteration are pass


class TestInteractiveLoop:
    async def test_quit_and_summary(self, monkeypatch):
        console = MagicMock()
        console.input = lambda *_a: "quit"
        await interactive_loop(_agent(), console, stream=False)

    async def test_eof_breaks(self, monkeypatch):
        console = MagicMock()

        def _inp(*_a):
            raise EOFError

        console.input = _inp
        await interactive_loop(_agent(), console, stream=False)

    async def test_blank_then_quit(self, monkeypatch):
        seq = iter(["   ", "quit"])

        def _inp(*_a):
            return next(seq)

        console = MagicMock()
        console.input = _inp
        await interactive_loop(_agent(), console, stream=False)

    async def test_unknown_command_then_quit(self, monkeypatch):
        seq = iter(["/bogus", "quit"])

        def _inp(*_a):
            return next(seq)

        console = MagicMock()
        console.input = _inp
        await interactive_loop(_agent(), console, stream=False)

    async def test_message_non_stream_then_quit(self, monkeypatch):
        agent = _agent()
        agent.run = AsyncMock(return_value="answer")
        seq = iter(["hello", "quit"])

        def _inp(*_a):
            return next(seq)

        console = MagicMock()
        console.input = _inp
        await interactive_loop(agent, console, stream=False)

    async def test_message_stream_then_quit(self, monkeypatch):
        agent = _agent()

        async def gen(msg):
            yield TextDeltaEvent(content="chunk")
            yield CompleteEvent(response=MagicMock(), content="final")

        agent.run_stream = gen

        live_cm = MagicMock()
        live_cm.__enter__ = lambda self: live_cm
        live_cm.__exit__ = lambda *a: None
        live_cm.update = lambda *a, **k: None
        monkeypatch.setattr(tui_loop, "Live", lambda **kw: live_cm)

        seq = iter(["hi", "quit"])

        def _inp(*_a):
            return next(seq)

        console = MagicMock()
        console.input = _inp
        await interactive_loop(agent, console, stream=True)

    async def test_run_raises_handled(self, monkeypatch):
        agent = _agent()
        agent.run = AsyncMock(side_effect=RuntimeError("kaboom"))
        seq = iter(["hello", "quit"])

        def _inp(*_a):
            return next(seq)

        console = MagicMock()
        console.input = _inp
        await interactive_loop(agent, console, stream=False)
        # the loop must SURFACE the error, not silently swallow it
        assert any("kaboom" in str(a) for a in console.print.call_args_list)

    async def test_extra_commands_dispatch(self, monkeypatch):
        calls = []

        def my_cmd(agent, console, args):
            calls.append(args)
            return "injected message"

        agent = _agent()
        agent.run = AsyncMock(return_value="ok")
        seq = iter(["/my", "quit"])

        def _inp(*_a):
            return next(seq)

        console = MagicMock()
        console.input = _inp
        await interactive_loop(agent, console, extra_commands={"/my": my_cmd}, stream=False)
        assert calls == [""]

    async def test_extra_commands_no_return(self, monkeypatch):
        def my_cmd(agent, console, args):
            return None

        agent = _agent()
        agent.run = AsyncMock()
        seq = iter(["/my", "quit"])

        def _inp(*_a):
            return next(seq)

        console = MagicMock()
        console.input = _inp
        await interactive_loop(agent, console, extra_commands={"/my": my_cmd}, stream=False)
        agent.run.assert_not_awaited()
