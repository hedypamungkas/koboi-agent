"""Integration tests for P0b (sandbox) + P2-A (journal/resume) via the facade.

These exercise the real user-facing paths that the unit tests (which build
backends / AgentCore directly) do not:
  - KoboiAgent.from_config(sandbox: restricted) -> tool actually runs contained;
  - the full `koboi run --resume` round-trip through KoboiAgent.resume();
  - run_stream() populating the steps journal;
  - orchestration sub-agents inheriting the parent sandbox.
"""

from __future__ import annotations


from koboi.config import Config
from koboi.facade import KoboiAgent
from koboi.memory_sqlite import SQLiteMemory
from koboi.sandbox.passthrough import PassthroughBackend
from koboi.sandbox.restricted import RestrictedProcessBackend

from tests.conftest import MockClient, make_mock_response, make_mock_tool_call


def _config(data: dict) -> Config:
    base = {
        "agent": {"name": "int-agent", "system_prompt": "helpful", "mode": "act"},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "base_url": "http://localhost:8080/v1",
        },
    }
    base.update(data)
    return Config.from_dict(base, validate=True)


def _swap_client(agent, responses):
    """Replace the facade-built RetryClient with a MockClient (no network)."""
    agent._core.client = MockClient(responses)
    return agent


# ---------------------------------------------------------------------------
# P0b: facade-level restricted sandbox
# ---------------------------------------------------------------------------


class TestFacadeRestrictedSandbox:
    async def test_write_outside_workdir_is_blocked(self, tmp_path):
        workdir = tmp_path / "ws"
        workdir.mkdir()
        outside = tmp_path / "outside.txt"  # sibling of workdir -> outside it

        cfg = _config(
            {
                "tools": {"builtin": ["write_file", "read_file", "list_files"]},
                "sandbox": {"backend": "restricted", "workdir": str(workdir)},
            }
        )
        agent = _swap_client(
            KoboiAgent._from_config(cfg),
            [
                make_mock_response(
                    tool_calls=[make_mock_tool_call("write_file", {"path": str(outside), "content": "x"})]
                ),
                make_mock_response(content="done"),
            ],
        )

        # The sandbox wired through the facade must be the restricted one.
        assert isinstance(agent.core.tools.get_dep("sandbox"), RestrictedProcessBackend)

        await agent.run("write a file outside the workspace")

        tool_msgs = [m for m in agent.core.memory.get_messages() if m.get("role") == "tool"]
        assert tool_msgs, "expected the write_file tool result in memory"
        assert "no access" in tool_msgs[0]["content"].lower() or "outside" in tool_msgs[0]["content"].lower()
        assert not outside.exists()  # containment held -- file never created

    async def test_write_inside_workdir_succeeds(self, tmp_path):
        workdir = tmp_path / "ws"
        workdir.mkdir()
        inside = workdir / "ok.txt"

        cfg = _config(
            {
                "tools": {"builtin": ["write_file"]},
                "sandbox": {"backend": "restricted", "workdir": str(workdir)},
            }
        )
        agent = _swap_client(
            KoboiAgent._from_config(cfg),
            [
                make_mock_response(
                    tool_calls=[make_mock_tool_call("write_file", {"path": str(inside), "content": "x"})]
                ),
                make_mock_response(content="done"),
            ],
        )
        await agent.run("write a file inside the workspace")
        assert inside.exists()

    def test_absent_sandbox_section_defaults_to_passthrough(self, tmp_path):
        cfg = _config({"tools": {"builtin": ["write_file"]}})
        agent = KoboiAgent._from_config(cfg)
        # No sandbox section -> passthrough (pre-P0b behavior preserved).
        assert isinstance(agent.core.tools.get_dep("sandbox"), PassthroughBackend)


# ---------------------------------------------------------------------------
# P2-A: full facade resume round-trip + run_stream journaling
# ---------------------------------------------------------------------------


class TestFacadeResumeRoundTrip:
    async def test_run_then_resume_continues_session(self, tmp_path):
        db = str(tmp_path / "rt.db")
        cfg = _config({"memory": {"backend": "sqlite", "db_path": db}})

        # Turn 1: run to completion.
        agent = _swap_client(KoboiAgent._from_config(cfg), [make_mock_response(content="first answer")])
        await agent.run("hello")
        sid = agent.core.memory.session_id
        assert agent.core.journal is not None  # sqlite -> journal wired
        first_steps = agent.core.journal.list_steps()
        assert any(s["is_terminal"] for s in first_steps)

        # "Crash": drop the agent, start a new process pointing at the same session.
        # Turn 2 (resume): rehydrate + continue without re-asking.
        agent2 = _swap_client(
            KoboiAgent._from_config(cfg, resume_session=sid),
            [make_mock_response(content="resumed answer")],
        )
        assert agent2.core.memory.session_id == sid
        # The first turn's messages were rehydrated.
        assert any("first answer" in (m.get("content") or "") for m in agent2.core.memory.get_messages())

        result = await agent2.resume()
        assert result.metadata["resumed"] is True
        assert "resumed answer" in result.content

    async def test_resume_message_is_ignored(self, tmp_path):
        """--resume must not add a new user message (rehydrate+continue)."""
        db = str(tmp_path / "rt2.db")
        cfg = _config({"memory": {"backend": "sqlite", "db_path": db}})
        agent = _swap_client(KoboiAgent._from_config(cfg), [make_mock_response(content="ok")])
        await agent.run("original question")
        sid = agent.core.memory.session_id
        n_before = len(SQLiteMemory.get_session_messages(db, sid))

        agent2 = _swap_client(
            KoboiAgent._from_config(cfg, resume_session=sid),
            [make_mock_response(content="follow-up")],
        )
        await agent2.resume()
        msgs = SQLiteMemory.get_session_messages(db, sid)
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 1  # only the original question, no new user msg
        assert len(msgs) == n_before + 1  # only the LLM's resume response was added


class TestRunStreamJournal:
    async def test_run_stream_records_terminal_step(self, tmp_path):
        db = str(tmp_path / "stream.db")
        cfg = _config(
            {
                "memory": {"backend": "sqlite", "db_path": db},
                "tools": {"builtin": ["write_file"]},
            }
        )
        agent = _swap_client(KoboiAgent._from_config(cfg), [make_mock_response(content="streamed answer")])

        # Drain the stream to completion.
        async for _ in agent.run_stream("hi"):
            pass

        steps = agent.core.journal.list_steps()
        # The 'running' marker is upserted to 'complete' for a finished step
        # (only interrupted steps keep 'running'); so we assert the terminal row.
        assert any(s["status"] == "complete" and s["is_terminal"] for s in steps)

    async def test_run_stream_tool_calls_step(self, tmp_path):
        db = str(tmp_path / "stream2.db")
        cfg = _config({"memory": {"backend": "sqlite", "db_path": db}, "tools": {"builtin": ["write_file"]}})
        agent = _swap_client(
            KoboiAgent._from_config(cfg),
            [
                make_mock_response(
                    tool_calls=[make_mock_tool_call("write_file", {"path": str(tmp_path / "x.txt"), "content": "y"})]
                ),
                make_mock_response(content="done"),
            ],
        )
        async for _ in agent.run_stream("write then finish"):
            pass
        statuses = [s["status"] for s in agent.core.journal.list_steps()]
        assert "tool_calls" in statuses
        assert "complete" in statuses


# ---------------------------------------------------------------------------
# Orchestration inherits the parent sandbox
# ---------------------------------------------------------------------------


class TestOrchestrationSandboxInheritance:
    def test_build_tools_from_config_inherits_sandbox(self):
        from koboi.orchestration.factory import AgentFactory

        sandbox = PassthroughBackend()
        tools = AgentFactory._build_tools_from_config({"builtin": ["run_shell"]}, sandbox=sandbox)
        assert tools is not None
        assert tools.get_dep("sandbox") is sandbox  # sub-agent tools inherit isolation

    def test_build_tools_from_config_without_sandbox_leaves_dep_unset(self):
        from koboi.orchestration.factory import AgentFactory

        tools = AgentFactory._build_tools_from_config({"builtin": ["run_shell"]})
        # No sandbox passed -> dep unset -> tools fall back to legacy behavior.
        assert tools.get_dep("sandbox") is None
