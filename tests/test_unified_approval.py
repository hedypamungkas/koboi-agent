"""Tests for unified approval resolution in ToolExecutionPipeline (M0 16.1).

Covers the Trust DB fast-path (auto-allow / deny rule), the ``no_handler`` inert
path (Q1), YOLO bypass, and direct ``_resolve_approval`` outcomes.
"""

from __future__ import annotations

import pytest

from koboi.guardrails.approval import ApprovalHandler
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.modes import AgentMode, ModeManager
from koboi.tools.registry import ToolRegistry
from koboi.trust import TrustDatabase
from koboi.types import RiskLevel, ToolCall


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {"message": {"type": "string"}}, "required": []},
        fn=lambda message="ok": f"result: {message}",
    )
    return registry


@pytest.fixture
def registry():
    return _make_registry()


@pytest.fixture
def memory():
    return ConversationMemory()


@pytest.fixture
def trust_db(tmp_path):
    db = TrustDatabase(str(tmp_path / "trust.db"))
    yield db
    db.close()


def _tc(name: str = "test_tool", args: str = '{"message": "hi"}') -> ToolCall:
    return ToolCall(id="1", name=name, arguments=args)


class _AlwaysDeny(ApprovalHandler):
    def should_approve(self, tool_name, arguments, risk_level):
        return False


class _AlwaysApprove(ApprovalHandler):
    def should_approve(self, tool_name, arguments, risk_level):
        return True


class TestTrustFastPath:
    async def test_trust_allow_skips_deny_handler(self, registry, memory, trust_db):
        # Trust allow rule short-circuits before the AlwaysDeny handler is consulted.
        trust_db.record_decision("test_tool", RiskLevel.SAFE, "allow", always=True)
        pipeline = ToolExecutionPipeline(
            tools=registry, memory=memory, approval_handler=_AlwaysDeny(), trust_db=trust_db
        )
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert not result.skipped
        assert "result: hi" in result.result

    async def test_trust_deny_rule_denies_without_handler(self, registry, memory, trust_db):
        trust_db.record_decision("test_tool", RiskLevel.SAFE, "deny", always=True)
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, trust_db=trust_db)
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert result.skipped
        assert result.skip_reason == "denied"

    async def test_no_trust_db_no_handler_executes(self, registry, memory):
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory)
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert not result.skipped
        assert "result: hi" in result.result

    async def test_yolo_bypasses_trust_deny(self, registry, memory, trust_db):
        # YOLO bypasses the whole approval resolution block (deny rule ignored).
        trust_db.record_decision("test_tool", RiskLevel.SAFE, "deny", always=True)
        mgr = ModeManager(AgentMode.YOLO)
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, trust_db=trust_db, mode_manager=mgr)
        result = await pipeline.execute_tool_call(_tc(), iteration=0)
        assert not result.skipped


class TestResolveApprovalOutcomes:
    async def test_no_handler_returns_no_handler_outcome(self, registry, memory):
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory)
        outcome = await pipeline._resolve_approval(_tc(), RiskLevel.SAFE)
        assert outcome.proceed is True
        assert outcome.reason == "no_handler"
        assert outcome.prompted is False

    async def test_trust_allow_outcome(self, registry, memory, trust_db):
        trust_db.record_decision("test_tool", RiskLevel.SAFE, "allow", always=True)
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, trust_db=trust_db)
        outcome = await pipeline._resolve_approval(_tc(), RiskLevel.SAFE)
        assert outcome.proceed is True
        assert outcome.reason == "skipped_via_trust"
        assert outcome.prompted is False
        assert outcome.trust_rule == "test_tool"

    async def test_trust_deny_outcome(self, registry, memory, trust_db):
        trust_db.record_decision("test_tool", RiskLevel.SAFE, "deny", always=True)
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, trust_db=trust_db)
        outcome = await pipeline._resolve_approval(_tc(), RiskLevel.SAFE)
        assert outcome.proceed is False
        assert outcome.reason == "denied"

    async def test_already_prompted_is_inert(self, registry, memory):
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, approval_handler=_AlwaysApprove())
        outcome = await pipeline._resolve_approval(_tc(), RiskLevel.SAFE, already_prompted=True)
        assert outcome.proceed is True
        assert outcome.reason == "inert"
        assert outcome.prompted is False

    async def test_handler_invoked_sets_prompted(self, registry, memory):
        pipeline = ToolExecutionPipeline(tools=registry, memory=memory, approval_handler=_AlwaysApprove())
        outcome = await pipeline._resolve_approval(_tc(), RiskLevel.SAFE)
        assert outcome.proceed is True
        assert outcome.reason == "approved"
        assert outcome.prompted is True
