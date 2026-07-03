"""Integration tests: real agent-loop compaction -> hook reset (P3b).

These exercise the full path the unit tests don't reach:
``AgentCore._prepare_iteration`` -> ``_get_managed_messages`` (which stamps
``metadata["compacted"]`` from the real ``ContextManager.manage()`` trim) ->
``ReadBeforeWriteResetHook`` gating on that flag -> the filesystem tracker
actually being cleared (or preserved).

The hook unit tests (test_read_before_write.py) feed a hand-built HookContext;
these prove the *live loop* produces the right signal and drives the hook.
"""

from __future__ import annotations

import pytest

from koboi.context.manager import TruncationManager
from koboi.hooks.chain import HookChain
from koboi.hooks.read_before_write_reset_hook import ReadBeforeWriteResetHook
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.builtin import filesystem
from tests.conftest import MockClient, make_mock_response


@pytest.fixture(autouse=True)
def _clean_read_tracker():
    """The _read_paths module-global persists across tests in a process."""
    filesystem.reset_read_before_write()
    yield
    filesystem.reset_read_before_write()


def _populate(memory: ConversationMemory, n_pairs: int) -> ConversationMemory:
    for i in range(n_pairs):
        memory.add_user_message(f"User message {i} with enough padding text to cross the token budget " * 3)
        memory.add_assistant_message(f"Assistant reply {i} with padding text to grow the token count " * 3)
    return memory


def _agent(memory: ConversationMemory, max_context_tokens: int, hooks=None) -> AgentCore:
    return AgentCore(
        client=MockClient([make_mock_response("ok")]),
        memory=memory,
        max_iterations=2,
        context_manager=TruncationManager(keep_last=2),
        max_context_tokens=max_context_tokens,
        hook_chain=HookChain(hooks or []),
    )


class TestCompactionSignal:
    async def test_signals_compacted_when_trimmed(self):
        """When messages exceed the budget, manage() trims -> _last_compacted is True."""
        agent = _agent(_populate(ConversationMemory(), 6), max_context_tokens=5)
        await agent._prepare_iteration(0)
        assert getattr(agent, "_last_compacted", False) is True

    async def test_signals_not_compacted_when_under_budget(self):
        """When messages fit, manage() is a no-op -> _last_compacted is False (no reset)."""
        memory = ConversationMemory()
        memory.add_user_message("hi")
        agent = _agent(memory, max_context_tokens=1_000_000)
        await agent._prepare_iteration(0)
        assert getattr(agent, "_last_compacted", True) is False


class TestReadBeforeWriteResetOnCompaction:
    async def test_reset_fires_on_real_compaction(self, tmp_path):
        """A real trim through the live loop must clear the read-before-write tracker."""
        f = tmp_path / "a.txt"
        f.write_text("x")
        filesystem.read_file(str(f))
        assert filesystem.get_read_paths(), "precondition: tracker populated"

        agent = _agent(_populate(ConversationMemory(), 6), max_context_tokens=5, hooks=[ReadBeforeWriteResetHook()])
        await agent._prepare_iteration(0)
        assert filesystem.get_read_paths() == set()  # reset fired via compacted=True

    async def test_no_reset_when_not_compacted(self, tmp_path):
        """No trim -> tracker must survive the POST_COMPACT emission (the key regression)."""
        f = tmp_path / "a.txt"
        f.write_text("x")
        filesystem.read_file(str(f))
        before = filesystem.get_read_paths()

        memory = ConversationMemory()
        memory.add_user_message("hi")
        agent = _agent(memory, max_context_tokens=1_000_000, hooks=[ReadBeforeWriteResetHook()])
        await agent._prepare_iteration(0)
        assert filesystem.get_read_paths() == before  # preserved
