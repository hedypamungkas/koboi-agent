"""Issue #8b: non-idempotent tools are skipped (not re-run) on crash-resume."""

from __future__ import annotations

import copy

from koboi.config import Config
from koboi.events import ToolResultEvent
from koboi.facade import KoboiAgent
from koboi.tools.registry import tool
from koboi.types import RiskLevel, ToolDefinition
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


class TestIdempotentFlagUnit:
    def test_default_idempotent_true(self):
        td = ToolDefinition(name="x", description="d", parameters={})
        assert td.idempotent is True

    def test_tool_decorator_threads_idempotent(self):
        @tool(name="charge", description="d", parameters={}, idempotent=False)
        def fn():
            return "ok"

        assert fn._tool_def.idempotent is False

    def test_get_definition_returns_none_for_unknown(self):
        from koboi.tools.registry import ToolRegistry

        reg = ToolRegistry()
        assert reg.get_definition("nope") is None


class TestNonIdempotentSkippedOnResume:
    async def test_non_idempotent_tool_skipped_on_resume(self, tmp_path):
        db_path = str(tmp_path / "idem.db")
        executed: list[str] = []

        config = _config(db_path)
        agent = KoboiAgent.from_dict(config.raw)
        agent.add_tool(
            "tool_safe",
            lambda: executed.append("SAFE") or "safe_ok",
            "safe tool",
            {"type": "object", "properties": {}, "required": []},
            risk_level=RiskLevel.SAFE,
        )
        agent.add_tool(
            "tool_charge",
            lambda: executed.append("CHARGE") or "charged",
            "side-effecting tool",
            {"type": "object", "properties": {}, "required": []},
            risk_level=RiskLevel.MODERATE,
            idempotent=False,
        )
        agent._core.client = MockClient(
            [
                make_mock_response(tool_calls=[make_mock_tool_call("tool_safe"), make_mock_tool_call("tool_charge")]),
                make_mock_response(content="done"),
            ]
        )
        session_id = agent._core.memory.session_id

        # Phase 1: run, cancel after tool_safe completes (tool_charge never runs).
        gen = agent.run_stream("go")
        async for ev in gen:
            if isinstance(ev, ToolResultEvent) and ev.tool_name == "tool_safe":
                break
        await gen.aclose()
        assert executed == ["SAFE"]

        # Phase 2: resume on a fresh agent.
        config2_data = copy.deepcopy(config.raw)
        config2_data["memory"]["session_id"] = session_id
        agent2 = KoboiAgent.from_dict(config2_data)
        agent2.add_tool(
            "tool_safe",
            lambda: executed.append("SAFE") or "safe_ok",
            "safe tool",
            {"type": "object", "properties": {}, "required": []},
            risk_level=RiskLevel.SAFE,
        )
        agent2.add_tool(
            "tool_charge",
            lambda: executed.append("CHARGE") or "charged",
            "side-effecting tool",
            {"type": "object", "properties": {}, "required": []},
            risk_level=RiskLevel.MODERATE,
            idempotent=False,
        )
        agent2._core.client = MockClient([make_mock_response(content="done")])

        result = await agent2.resume()

        # tool_charge was MISSING but must NOT have re-executed (non-idempotent).
        assert "CHARGE" not in executed
        assert result.success is True
        # A synthetic tool result was recorded for the skipped tool.
        msgs = " ".join(str(m.get("content", "")) for m in agent2._core.memory.get_messages())
        assert "skipped on resume" in msgs and "tool_charge" in msgs
