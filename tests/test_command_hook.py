"""Tests for koboi/hooks/command_hook.py -- YAML-driven external command hooks.

Covers behavior (abort/inject/modify, exit codes, fire-and-forget vs awaited),
security (default-deny gate, secret-env scrubbing, sandbox consulted), the
non-blocking guarantee (asyncio.to_thread offload), and an end-to-end run that
forwards an LLM response to a file (the WhatsApp/Telegram use case).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from unittest.mock import patch

import pytest

from koboi.config import Config
from koboi.facade import _build_command_hooks
from koboi.hooks.chain import HookChain, HookContext, HookEvent
from koboi.hooks.command_hook import CommandHook
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.sandbox.passthrough import PassthroughBackend
from koboi.tools.registry import ToolRegistry
from koboi.types import AgentResponse, TokenUsage, ToolCall

_SB = PassthroughBackend()


def _responder(stdout_text: str, exit_code: int = 0) -> list[str]:
    """Portable python command: drain stdin, then write ``stdout_text`` and exit."""
    return [
        sys.executable,
        "-c",
        f"import sys\nsys.stdin.read()\nsys.stdout.write({stdout_text!r})\nsys.exit({exit_code})\n",
    ]


def _make_hook(command, events=None, **kw) -> CommandHook:
    return CommandHook(
        command=command,
        events=events or [HookEvent.PRE_TOOL_USE],
        sandbox=_SB,
        **kw,
    )


class TestBuildCommandHooksGate:
    def test_default_deny_skips_when_allow_exec_false(self, caplog):
        cfg = Config({"hooks": {"on_event": [{"command": ["echo", "x"], "events": ["post_output"]}]}})
        chain = HookChain()
        with caplog.at_level(logging.WARNING):
            _build_command_hooks(cfg, _SB, chain)
        assert not [h for h in chain._hooks if isinstance(h, CommandHook)]
        assert any("allow_exec is false" in r.message for r in caplog.records)

    def test_allow_exec_true_wires_hook(self):
        cfg = Config(
            {
                "hooks": {
                    "allow_exec": True,
                    "command_timeout": 7,
                    "on_event": [{"name": "fwd", "command": ["uvx", "x"], "events": ["post_output", "session_end"]}],
                }
            }
        )
        chain = HookChain()
        _build_command_hooks(cfg, _SB, chain)
        hooks = [h for h in chain._hooks if isinstance(h, CommandHook)]
        assert len(hooks) == 1
        assert hooks[0].handles() == [HookEvent.POST_OUTPUT, HookEvent.SESSION_END]
        assert hooks[0]._timeout == 7  # command_timeout flows through

    def test_unknown_event_raises(self):
        cfg = Config({"hooks": {"allow_exec": True, "on_event": [{"command": ["x"], "events": ["bogus"]}]}})
        with pytest.raises(ValueError, match="unknown event"):
            _build_command_hooks(cfg, _SB, HookChain())


class TestCommandHookControl:
    async def test_abort_via_exit_2_applies_json(self):
        h = _make_hook(_responder('{"abort": true, "inject_message": "no"}', 2), fire_and_forget=False)
        ctx = await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert ctx.abort is True
        assert "no" in ctx.inject_messages

    async def test_abort_via_json_exit_0(self):
        h = _make_hook(_responder('{"abort": true}'), fire_and_forget=False)
        ctx = await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert ctx.abort is True

    async def test_inject_messages(self):
        h = _make_hook(_responder('{"inject_messages": ["a", "b"]}'), fire_and_forget=False)
        ctx = await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert ctx.inject_messages == ["a", "b"]
        assert ctx.abort is False

    async def test_modified_tool_result_flows_through_pipeline(self):
        # POST_TOOL_USE mutation now honored (loop_pipeline one-line fix).
        tools = ToolRegistry()
        tools.register(
            name="t",
            description="d",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: "ORIGINAL",
        )
        h = CommandHook(
            command=_responder('{"modified_tool_result": "OVERRIDDEN"}'),
            events=[HookEvent.POST_TOOL_USE],
            sandbox=_SB,
            fire_and_forget=False,
        )
        pipe = ToolExecutionPipeline(tools=tools, memory=ConversationMemory(), hook_chain=HookChain([h]))
        pr = await pipe.execute_tool_call(ToolCall(id="1", name="t", arguments="{}"), iteration=0)
        assert pr.result == "OVERRIDDEN"


class TestFireAndForget:
    async def test_returns_immediately_and_ignores_mutations(self, tmp_path):
        # A command that would abort, but ff=True -> not awaited, no mutation.
        h = _make_hook(_responder('{"abort": true}', 2), events=[HookEvent.POST_OUTPUT], fire_and_forget=True)
        ctx = HookContext(event=HookEvent.POST_OUTPUT)
        t0 = time.monotonic()
        out = await h.execute(ctx)
        assert time.monotonic() - t0 < 0.3
        assert out.abort is False  # mutations ignored
        assert out is ctx  # unchanged

    async def test_control_honored_when_awaited(self):
        h = _make_hook(_responder('{"abort": true}', 2), events=[HookEvent.PRE_TOOL_USE], fire_and_forget=False)
        ctx = await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert ctx.abort is True


class TestFailSafe:
    async def test_timeout_continues_by_default(self):
        h = _make_hook(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            events=[HookEvent.PRE_TOOL_USE],
            fire_and_forget=False,
            timeout=0.3,
        )
        ctx = await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert ctx.abort is False  # fail-safe continue (abort_on_error defaults False)

    async def test_timeout_aborts_when_abort_on_error(self):
        h = _make_hook(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            events=[HookEvent.PRE_TOOL_USE],
            fire_and_forget=False,
            timeout=0.3,
            abort_on_error=True,
        )
        ctx = await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert ctx.abort is True

    async def test_nonzero_exit_continues_by_default(self):
        h = _make_hook(_responder("", 1), events=[HookEvent.PRE_TOOL_USE], fire_and_forget=False)
        ctx = await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert ctx.abort is False

    async def test_nonzero_exit_aborts_when_abort_on_error(self):
        h = _make_hook(_responder("", 1), events=[HookEvent.PRE_TOOL_USE], fire_and_forget=False, abort_on_error=True)
        ctx = await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert ctx.abort is True

    async def test_non_json_stdout_is_noop(self):
        h = _make_hook(_responder("not json{{"), events=[HookEvent.PRE_TOOL_USE], fire_and_forget=False)
        ctx = await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert ctx.abort is False
        assert ctx.inject_messages == []


class TestSecurity:
    async def test_secret_env_not_leaked_to_child(self, monkeypatch):
        # build_safe_env strips *_KEY; the child spawned by a CommandHook must NOT
        # see a planted secret. CommandHook builds env via sandbox.build_env(...).
        monkeypatch.setenv("MY_API_KEY", "super-secret")
        env = _SB.build_env({"env_passthrough": False})  # exactly what CommandHook uses
        assert "MY_API_KEY" not in env  # scrubbed before the child is spawned
        # Prove the child really receives the scrubbed env:
        res = _SB.run(
            [sys.executable, "-c", "import os; print(os.environ.get('MY_API_KEY','ABSENT'))"],
            env=env,
            timeout=5,
        )
        assert "ABSENT" in res.stdout
        assert "super-secret" not in res.stdout

    async def test_sandbox_run_is_consulted(self):
        h = _make_hook(_responder('{"abort":true}'), events=[HookEvent.PRE_TOOL_USE], fire_and_forget=False)
        with patch.object(_SB, "run", wraps=_SB.run) as spy:
            await h.execute(HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="t"))
        assert spy.called


class TestNonBlocking:
    async def test_offloads_to_thread_does_not_block_loop(self):
        # A 0.5s subprocess must NOT stall the event loop: while it runs in a worker
        # thread, control yields (asyncio.sleep(0)) return immediately. If sandbox.run
        # were called inline (blocking), these yields would be delayed ~0.5s.
        h = CommandHook(
            command=[sys.executable, "-c", "import time; time.sleep(0.5)"],
            events=[HookEvent.POST_OUTPUT],
            sandbox=_SB,
            fire_and_forget=False,
            timeout=5,
        )

        async def run_hook():
            await h.execute(HookContext(event=HookEvent.POST_OUTPUT))

        task = asyncio.create_task(run_hook())
        await asyncio.sleep(0.1)  # hook is mid-run in a thread now
        t0 = time.monotonic()
        for _ in range(5):
            await asyncio.sleep(0)  # would stall if the loop were blocked
        elapsed = time.monotonic() - t0
        assert elapsed < 0.2, f"event loop was blocked ({elapsed:.2f}s)"
        await task


class TestEndToEndForwarding:
    async def test_forwards_llm_response_to_file(self, tmp_path):
        from koboi.loop import AgentCore
        from tests.conftest import MockClient

        forwarder = tmp_path / "forwarder.py"
        forwarder.write_text(
            "import sys, json\n"
            "payload = json.load(sys.stdin)\n"
            "content = (payload.get('llm_response') or {}).get('content') or ''\n"
            "with open(sys.argv[1], 'a') as f:\n"
            "    f.write(content + '\\n')\n"
        )
        outfile = tmp_path / "out.txt"
        hook = CommandHook(
            command=[sys.executable, str(forwarder), str(outfile)],
            events=[HookEvent.POST_OUTPUT],
            sandbox=_SB,
            fire_and_forget=True,
            timeout=10,
        )
        client = MockClient([AgentResponse(content="Hello from LLM", tool_calls=[], usage=TokenUsage(1, 1))])
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=3,
            hook_chain=HookChain([hook]),
        )
        await agent.run("hi")
        # fire-and-forget: the forwarder ran as a background task; await it.
        if hook._bg_tasks:
            await asyncio.wait_for(asyncio.gather(*hook._bg_tasks, return_exceptions=True), timeout=5)
        assert "Hello from LLM" in outfile.read_text()
