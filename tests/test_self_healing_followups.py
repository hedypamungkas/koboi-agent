"""tests/test_self_healing_followups.py -- systemic idempotent fix + quick-win config/observability.

1. Side-effecting builtins are flagged idempotent=False (closes the replan SAFE-risk gap
   for delegate_tasks/memory_store + protects crash-resume from double-firing them).
2. pipeline_outcomes mirrors the P0-D errored/error_kind signal (eval/observability).
3. ReflectionHook prefers the structured tool_error_kind over the "Error:" prefix string-match.
"""

from __future__ import annotations

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.reflection_hook import ReflectionHook
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import AgentResponse
from tests.conftest import make_mock_response, make_mock_tool_call


class _CriticClient:
    def __init__(self, critique="try again"):
        self.critique = critique
        self.calls = 0

    async def complete(self, messages, tools=None, response_format=None):
        self.calls += 1
        return AgentResponse(content=self.critique)


# --------------------------------------------------------------------------- idempotent flags


def test_side_effecting_builtins_flagged_non_idempotent():
    from koboi.tools.builtin.calculator import calculate
    from koboi.tools.builtin.filesystem import delete_file, list_files, read_file, write_file
    from koboi.tools.builtin.ingest import ingest_url
    from koboi.tools.builtin.memory import memory_recall, memory_store
    from koboi.tools.builtin.shell import run_shell
    from koboi.tools.builtin.subagent import delegate_tasks

    # Side-effecting builtins must be non-idempotent (carry forward on replan / skip on resume).
    for fn in (run_shell, write_file, delete_file, ingest_url, delegate_tasks, memory_store):
        assert fn._tool_def.idempotent is False, f"{fn._tool_def.name} should be non-idempotent"
    # Read-only / pure builtins stay idempotent.
    for fn in (calculate, list_files, read_file, memory_recall):
        assert fn._tool_def.idempotent is True, f"{fn._tool_def.name} should stay idempotent"


# --------------------------------------------------------------------------- pipeline_outcomes mirror


class TestPipelineOutcomesMirror:
    async def test_error_outcome_has_structured_signal(self, mock_client):
        reg = ToolRegistry()

        def _boom():
            raise RuntimeError("kaboom")

        reg.register("boom", "boom", {"type": "object", "properties": {}}, _boom)
        client = mock_client(
            responses=[make_mock_response(None, [make_mock_tool_call("boom")]), make_mock_response("done")]
        )
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=reg, max_iterations=3)
        result = await agent.run("q")
        boom = next(o for o in result.pipeline_outcomes if o["tool_name"] == "boom")
        assert boom["errored"] is True
        assert boom["error_kind"] == "execution_error"


# --------------------------------------------------------------------------- reflection structured signal


class TestReflectionStructuredErrorKind:
    async def test_detects_error_via_structured_kind_not_just_prefix(self):
        # tool_error_kind set but the result text does NOT start with "Error:" -- must still
        # count as an error (the structured signal, not the fragile prefix).
        critic = _CriticClient()
        hook = ReflectionHook(client=critic, tool_error_threshold=2)
        for _ in range(2):
            ctx = HookContext(HookEvent.POST_TOOL_USE, tool_name="x", tool_arguments="{}", tool_result="looks fine")
            ctx.metadata["tool_error_kind"] = "execution_error"
            await hook.execute(ctx)
        assert critic.calls == 1  # 2nd identical -> critique fired via the structured signal

    async def test_no_error_kind_no_prefix_is_success(self):
        hook = ReflectionHook(client=_CriticClient(), tool_error_threshold=2)
        ctx = HookContext(HookEvent.POST_TOOL_USE, tool_name="x", tool_arguments="{}", tool_result="looks fine")
        await hook.execute(ctx)
        assert ctx.inject_messages == []  # neither signal -> success (resets counter)
