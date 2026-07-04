"""16.7: Partial-turn consistency verification.

Tests that when a multi-tool turn is interrupted mid-stream (client disconnect),
resume() re-executes ONLY the missing tool calls — not the ones whose results
were already persisted. This validates the P2-A journal/resume design.
"""

from __future__ import annotations

import copy

from koboi.config import Config
from koboi.events import ToolResultEvent
from koboi.facade import KoboiAgent
from koboi.types import RiskLevel
from tests.conftest import MockClient, make_mock_response, make_mock_tool_call


def _config(db_path: str) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "t", "system_prompt": "h", "max_iterations": 5, "mode": "act"},
            "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
            "memory": {"backend": "sqlite", "db_path": db_path},
            "sandbox": {"backend": "passthrough"},
        },
        validate=True,
    )


def _register_tools(agent: KoboiAgent, executed: list[str]) -> None:
    """Register 3 tracking tools that append their name to `executed`."""
    agent.add_tool(
        "tool_a",
        lambda: executed.append("A") or "result_a",
        "A",
        {"type": "object", "properties": {}, "required": []},
        risk_level=RiskLevel.SAFE,
    )
    agent.add_tool(
        "tool_b",
        lambda: executed.append("B") or "result_b",
        "B",
        {"type": "object", "properties": {}, "required": []},
        risk_level=RiskLevel.SAFE,
    )
    agent.add_tool(
        "tool_c",
        lambda: executed.append("C") or "result_c",
        "C",
        {"type": "object", "properties": {}, "required": []},
        risk_level=RiskLevel.SAFE,
    )


class TestPartialTurnResume:
    """Verify resume() re-executes only missing tool calls after a mid-stream cancel."""

    async def test_partial_turn_resume_re_executes_only_missing(self, tmp_path):
        db_path = str(tmp_path / "partial_turn.db")
        executed: list[str] = []

        # --- Phase 1: Start a 3-tool turn, cancel after tool A completes ---
        config = _config(db_path)
        agent = KoboiAgent.from_dict(config.raw)
        _register_tools(agent, executed)
        agent._core.client = MockClient(
            [
                make_mock_response(
                    tool_calls=[
                        make_mock_tool_call("tool_a"),
                        make_mock_tool_call("tool_b"),
                        make_mock_tool_call("tool_c"),
                    ]
                ),
                make_mock_response(content="done"),
            ]
        )
        session_id = agent._core.memory.session_id

        gen = agent.run_stream("go")
        async for ev in gen:
            if isinstance(ev, ToolResultEvent) and ev.tool_name == "tool_a":
                break  # simulate client disconnect after tool A
        await gen.aclose()

        # Tool A executed; B and C not.
        assert executed == ["A"]

        # --- Phase 2: Fresh agent on the same session, call resume() ---
        config2_data = copy.deepcopy(config.raw)
        config2_data["memory"]["session_id"] = session_id
        agent2 = KoboiAgent.from_dict(config2_data)
        _register_tools(agent2, executed)  # same closures → same `executed` list
        agent2._core.client = MockClient([make_mock_response(content="done")])

        result = await agent2.resume()

        # B and C re-executed; A NOT double-executed.
        assert "B" in executed
        assert "C" in executed
        assert executed.count("A") == 1  # no double-execution
        assert result.success is True

    async def test_completed_turn_resume_does_not_re_execute(self, tmp_path):
        """If the turn completed normally (all 3 tools), resume finds nothing missing."""
        db_path = str(tmp_path / "completed_turn.db")
        executed: list[str] = []

        config = _config(db_path)
        agent = KoboiAgent.from_dict(config.raw)
        _register_tools(agent, executed)
        agent._core.client = MockClient(
            [
                make_mock_response(
                    tool_calls=[
                        make_mock_tool_call("tool_a"),
                        make_mock_tool_call("tool_b"),
                        make_mock_tool_call("tool_c"),
                    ]
                ),
                make_mock_response(content="done"),
            ]
        )
        session_id = agent._core.memory.session_id

        # Let the full turn complete.
        async for _ in agent.run_stream("go"):
            pass

        assert sorted(executed) == ["A", "B", "C"]

        # Resume on fresh agent — no tools should re-execute.
        executed.clear()
        config2_data = copy.deepcopy(config.raw)
        config2_data["memory"]["session_id"] = session_id
        agent2 = KoboiAgent.from_dict(config2_data)
        _register_tools(agent2, executed)
        agent2._core.client = MockClient([make_mock_response(content="done")])

        await agent2.resume()

        # No tools re-executed (all results were persisted).
        assert executed == []
