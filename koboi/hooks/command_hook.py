"""koboi/hooks/command_hook.py -- External command hook (YAML-driven).

Spawns a configured command as a subprocess per lifecycle event, passing a JSON
HookContext on stdin. When awaited (``fire_and_forget=False``) it reads JSON
mutations back (``abort`` / ``inject_message(s)`` / ``modified_tool_result``); when
fire-and-forget (default) it runs the command off-loop without waiting -- zero
latency observe/side-effect (e.g. forwarding a response to a messaging channel).

The subprocess is always run through the wired ``sandbox`` backend (so it inherits
cwd/env/network/rlimit isolation + secret-hygiened env via ``build_env``), and the
synchronous ``sandbox.run`` is always offloaded via ``asyncio.to_thread`` so a slow
command can never block the asyncio event loop (critical on the server, where one
stalled hook would freeze every concurrent session).

See ``docs/custom-hooks.md`` for the full protocol + security model.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.sandbox.base import BaseSandbox

_logger = logging.getLogger("koboi.command_hook")

# Cap parsed stdout before json.loads to bound memory from a misbehaving hook.
_STDOUT_CAP = 65536


class CommandHook(Hook):
    """Hook that runs an external command per event (JSON over stdio)."""

    def __init__(
        self,
        *,
        command: list[str] | str,
        events: list[HookEvent],
        sandbox: BaseSandbox,
        logger: logging.Logger | None = None,
        fire_and_forget: bool = True,
        timeout: float = 10.0,
        priority: int = 50,
        pass_messages: bool = False,
        pass_metadata: bool = False,
        abort_on_error: bool = False,
        env_passthrough: bool = False,
        cwd: str | None = None,
        name: str | None = None,
    ):
        if not events:
            raise ValueError("CommandHook requires at least one event")
        for ev in events:
            if not isinstance(ev, HookEvent):
                raise TypeError(f"CommandHook events must be HookEvent values, got {type(ev).__name__}: {ev!r}")
        self._command = command
        self._events = list(events)
        self._sandbox = sandbox
        self._logger = logger or _logger
        self._fire_and_forget = fire_and_forget
        self._timeout = timeout
        self.priority = priority
        self._pass_messages = pass_messages
        self._pass_metadata = pass_metadata
        self._abort_on_error = abort_on_error
        self._env_passthrough = env_passthrough
        self._cwd = cwd
        self._name = name or "command-hook"
        # Hold strong refs to fire-and-forget tasks so CPython doesn't GC them mid-run.
        self._bg_tasks: set[asyncio.Task] = set()

    @property
    def name(self) -> str:
        return self._name

    def handles(self) -> list[HookEvent]:
        return list(self._events)

    # -- context serialization ------------------------------------------------

    def _build_payload(self, ctx: HookContext) -> dict[str, Any]:
        payload: dict[str, Any] = {"event": ctx.event.value, "iteration": ctx.iteration}
        if ctx.agent is not None:
            payload["agent"] = {
                "model": ctx.agent.model,
                "agent_name": ctx.agent.agent_name,
                "iteration": ctx.agent.iteration,
            }
        if ctx.tool_name is not None:
            payload["tool_name"] = ctx.tool_name
        if ctx.tool_arguments is not None:
            # tool_arguments is ALREADY a JSON string -- pass it raw, do NOT re-encode.
            payload["tool_arguments"] = ctx.tool_arguments
        if ctx.tool_result is not None:
            payload["tool_result"] = ctx.tool_result
        if ctx.user_message is not None:
            payload["user_message"] = ctx.user_message
        if ctx.llm_response is not None:
            payload["llm_response"] = self._serialize_llm_response(ctx.llm_response)
        if self._pass_messages and ctx.messages is not None:
            payload["messages"] = ctx.messages
        if self._pass_metadata and ctx.metadata:
            payload["metadata"] = self._safe_dict(ctx.metadata)
        return payload

    @staticmethod
    def _serialize_llm_response(resp: Any) -> dict[str, Any]:
        # AgentResponse -> {content, tool_calls, is_complete}; never the raw object
        # (it isn't JSON-serializable and would leak provider base_url/usage).
        out: dict[str, Any] = {}
        content = getattr(resp, "content", None)
        if content is not None:
            out["content"] = content
        tool_calls = getattr(resp, "tool_calls", None) or []
        out["tool_calls"] = [
            {
                "id": getattr(t, "id", None),
                "name": getattr(t, "name", None),
                "arguments": getattr(t, "arguments", None),
            }
            for t in tool_calls
        ]
        is_complete = getattr(resp, "is_complete", None)
        if is_complete is not None:
            out["is_complete"] = is_complete
        return out

    @staticmethod
    def _safe_dict(d: dict) -> dict[str, Any]:
        """Filter a MetadataBag to JSON-serializable values (some may be non-serializable)."""
        safe: dict[str, Any] = {}
        for k, v in dict(d).items():
            try:
                json.dumps(v)
            except (TypeError, ValueError):
                continue
            safe[k] = v
        return safe

    # -- execution ------------------------------------------------------------

    async def execute(self, ctx: HookContext) -> HookContext:
        payload = self._build_payload(ctx)
        env = self._sandbox.build_env({"env_passthrough": self._env_passthrough})
        shell = isinstance(self._command, str)
        stdin_data = json.dumps(payload)

        if self._fire_and_forget:
            # Spawn off-loop, do NOT wait -- observe/side-effect only. Mutations
            # are impossible (we never read stdout). Zero latency in the hot path.
            task = asyncio.create_task(
                asyncio.to_thread(
                    self._sandbox.run,
                    self._command,
                    cwd=self._cwd,
                    env=env,
                    timeout=self._timeout,
                    shell=shell,
                    input=stdin_data,
                )
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._on_bg_done)
            return ctx  # unchanged, immediately

        # Awaited branch: full control (abort / inject / modified_tool_result).
        try:
            result = await asyncio.to_thread(
                self._sandbox.run,
                self._command,
                cwd=self._cwd,
                env=env,
                timeout=self._timeout,
                shell=shell,
                input=stdin_data,
            )
        except Exception as exc:  # asyncio.to_thread propagates worker/sandbox crash
            self._logger.error("command hook %r crashed: %s", self._name, exc)
            if self._abort_on_error:
                ctx.abort = True
            return ctx

        if result.timed_out:
            self._logger.warning("command hook %r timed out (%ss)", self._name, self._timeout)
            if self._abort_on_error:
                ctx.abort = True
            return ctx

        if result.returncode == 2:
            # Explicit abort (Claude-Code convention). Optionally read a reason.
            self._apply_json(ctx, result.stdout, abort_default=True)
            return ctx

        if result.returncode != 0:
            self._logger.warning(
                "command hook %r exit=%s stderr=%r",
                self._name,
                result.returncode,
                (result.stderr or "")[:500],
            )
            if self._abort_on_error:
                ctx.abort = True
            return ctx

        # exit 0: apply any returned JSON mutations.
        self._apply_json(ctx, result.stdout, abort_default=False)
        return ctx

    def _apply_json(self, ctx: HookContext, stdout: str, *, abort_default: bool) -> None:
        """Parse the hook's stdout JSON and mutate ``ctx``. ``abort_default`` applies
        when there's no usable JSON (e.g. exit-2 with empty/garbage stdout)."""
        if not stdout or not stdout.strip():
            if abort_default:
                ctx.abort = True
            return
        text = stdout[:_STDOUT_CAP]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            self._logger.warning("command hook %r stdout not JSON: %r", self._name, text[:500])
            if abort_default:
                ctx.abort = True
            return
        if not isinstance(data, dict):
            if abort_default:
                ctx.abort = True
            return
        if data.get("abort") is True:
            ctx.abort = True
        msg = data.get("inject_message")
        if msg is not None:
            ctx.inject_messages.append(str(msg))
        msgs = data.get("inject_messages")
        if isinstance(msgs, list):
            ctx.inject_messages.extend(str(m) for m in msgs)
        modified = data.get("modified_tool_result")
        if modified is not None:
            ctx.tool_result = str(modified)

    def _on_bg_done(self, task: asyncio.Task) -> None:
        self._bg_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            self._logger.error("fire_and_forget command hook %r failed: %s", self._name, exc)
