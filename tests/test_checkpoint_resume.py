"""Wave 2: crash-resume with shadow-repo checkpoints (loop + journal integration).

Simulates the crash window directly: a completed mutating call (checkpointed),
then an assistant message requesting a second mutating call whose result never
landed in memory, plus manual tree mutations standing in for that call's
partial effects. resume() must roll the tree back to the last checkpoint and
say so in the synthetic tool result.
"""

from __future__ import annotations

import shutil

import pytest

from koboi.checkpoint import WorkdirCheckpointer
from koboi.journal import StepJournal
from koboi.loop import AgentCore
from koboi.memory_sqlite import SQLiteMemory
from koboi.tools.registry import ToolRegistry

from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _registry(workdir) -> ToolRegistry:
    registry = ToolRegistry()

    def mutate_file(content: str = "v1") -> str:
        (workdir / "target.txt").write_text(content)
        return f"wrote {content}"

    registry.register(
        name="mutate_file",
        description="write target.txt",
        parameters={
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": [],
        },
        fn=mutate_file,
        idempotent=False,
    )
    return registry


def _core(db_path, session_id, responses, workdir, checkpointer):
    mem = SQLiteMemory(db_path=db_path, session_id=session_id)
    journal = StepJournal(mem._ensure_conn(), mem.session_id)
    core = AgentCore(
        client=MockClient(responses),
        memory=mem,
        tools=_registry(workdir),
        journal=journal,
        checkpointer=checkpointer,
        max_iterations=5,
    )
    return core, mem, journal


def _interrupted_call(mem, call_id="tc_crash"):
    """Persist an assistant message whose tool call has no result (crash window)."""
    mem.add_assistant_message(
        None,
        [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "mutate_file", "arguments": '{"content": "v2"}'},
            }
        ],
    )


class TestCheckpointResume:
    async def test_resume_rolls_back_partial_effects(self, tmp_path):
        db = str(tmp_path / "mem.db")
        ws = tmp_path / "ws"
        ws.mkdir()

        # Phase 1: a completed mutating call -> checkpoint committed.
        cp = WorkdirCheckpointer(str(ws))
        core, mem, journal = _core(
            db,
            "S1",
            [
                make_mock_response(tool_calls=[make_mock_tool_call("mutate_file", {"content": "v1"})]),
                make_mock_response("turn done"),
            ],
            ws,
            cp,
        )
        result = await core.run("write v1")
        assert result.success is True
        assert (ws / "target.txt").read_text() == "v1"
        rows = journal.list_steps()
        shas = [r["checkpoint_sha"] for r in rows if r["checkpoint_sha"]]
        assert shas, "mutating call must stamp a checkpoint sha"
        assert cp.head() == shas[-1]

        # Crash window: a second mutating call is requested but its result
        # never persists; its partial effects hit the tree.
        _interrupted_call(mem)
        (ws / "target.txt").write_text("v2-PARTIAL")
        (ws / "junk.tmp").write_text("half-written")

        # Phase 2: fresh core over the same session; resume.
        core2, mem2, _ = _core(db, "S1", [make_mock_response("resumed done")], ws, WorkdirCheckpointer(str(ws)))
        result2 = await core2.resume()
        assert result2.success is True
        # Tree rolled back to the post-call-1 checkpoint.
        assert (ws / "target.txt").read_text() == "v1"
        assert not (ws / "junk.tmp").exists()
        # The synthetic tool result tells the model about the rollback.
        tool_msgs = [m for m in mem2.get_messages() if m.get("role") == "tool"]
        synthetic = [m for m in tool_msgs if "skipped on resume" in (m.get("content") or "")]
        assert synthetic
        assert "rolled back to checkpoint" in synthetic[-1]["content"]

    async def test_resume_without_prior_shadow_leaves_tree_untouched(self, tmp_path):
        db = str(tmp_path / "mem.db")
        ws = tmp_path / "ws"
        ws.mkdir()

        # Phase 1 ran WITHOUT checkpointing (no shadow exists).
        core, mem, _ = _core(
            db,
            "S1",
            [
                make_mock_response(tool_calls=[make_mock_tool_call("mutate_file", {"content": "v1"})]),
                make_mock_response("turn done"),
            ],
            ws,
            checkpointer=None,
        )
        await core.run("write v1")
        _interrupted_call(mem)
        (ws / "target.txt").write_text("v2-PARTIAL")

        # Resume WITH checkpointing newly enabled: no baseline -> no restore.
        core2, mem2, _ = _core(db, "S1", [make_mock_response("resumed done")], ws, WorkdirCheckpointer(str(ws)))
        result2 = await core2.resume()
        assert result2.success is True
        assert (ws / "target.txt").read_text() == "v2-PARTIAL"  # tree untouched
        tool_msgs = [m for m in mem2.get_messages() if m.get("role") == "tool"]
        synthetic = [m for m in tool_msgs if "skipped on resume" in (m.get("content") or "")]
        assert synthetic
        assert "rolled back" not in synthetic[-1]["content"]  # legacy message

    async def test_llm_phase_crash_never_restores(self, tmp_path):
        """No missing tool calls -> restore must not run (operator edits survive)."""
        db = str(tmp_path / "mem.db")
        ws = tmp_path / "ws"
        ws.mkdir()
        cp = WorkdirCheckpointer(str(ws))
        core, mem, _ = _core(
            db,
            "S1",
            [
                make_mock_response(tool_calls=[make_mock_tool_call("mutate_file", {"content": "v1"})]),
                make_mock_response("turn done"),
            ],
            ws,
            cp,
        )
        await core.run("write v1")
        # Crash happened during an LLM call: no dangling tool call. The
        # operator hand-edits the tree before resuming.
        (ws / "target.txt").write_text("operator-edit")

        core2, _, _ = _core(db, "S1", [make_mock_response("resumed done")], ws, WorkdirCheckpointer(str(ws)))
        await core2.resume()
        assert (ws / "target.txt").read_text() == "operator-edit"  # never reset
