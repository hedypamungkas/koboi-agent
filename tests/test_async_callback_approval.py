"""Tests for AsyncCallbackApprovalHandler (M0 16.3)."""

from __future__ import annotations

import asyncio

import pytest

from koboi.guardrails.approval import AsyncCallbackApprovalHandler
from koboi.guardrails.approval_types import ApprovalRequest, ApprovalResponse
from koboi.trust import TrustDatabase
from koboi.types import RiskLevel


@pytest.fixture
def trust_db(tmp_path):
    db = TrustDatabase(str(tmp_path / "trust.db"))
    yield db
    db.close()


def _cb(approved: bool = True, always_allow: bool = False, delay: float = 0.0):
    async def callback(request: ApprovalRequest) -> ApprovalResponse:
        if delay:
            await asyncio.sleep(delay)
        return ApprovalResponse(approved=approved, always_allow=always_allow)

    return callback


class TestAsyncCallbackApprovalHandler:
    async def test_approve(self):
        handler = AsyncCallbackApprovalHandler(callback=_cb(approved=True))
        assert await handler.should_approve("tool", "{}", RiskLevel.SAFE) is True

    async def test_deny(self):
        handler = AsyncCallbackApprovalHandler(callback=_cb(approved=False))
        assert await handler.should_approve("tool", "{}", RiskLevel.SAFE) is False

    async def test_timeout_denies(self):
        # Callback sleeps longer than the handler timeout -> deny (fail-closed).
        handler = AsyncCallbackApprovalHandler(callback=_cb(delay=0.2), timeout=0.05)
        assert await handler.should_approve("tool", "{}", RiskLevel.SAFE) is False

    async def test_callback_error_denies(self):
        async def boom(request: ApprovalRequest) -> ApprovalResponse:
            raise RuntimeError("boom")

        handler = AsyncCallbackApprovalHandler(callback=boom)
        assert await handler.should_approve("tool", "{}", RiskLevel.SAFE) is False

    async def test_always_allow_records_trust_rule(self, trust_db):
        handler = AsyncCallbackApprovalHandler(callback=_cb(approved=True, always_allow=True), trust_db=trust_db)
        await handler.should_approve("git.status", "{}", RiskLevel.SAFE)
        rules = trust_db.get_rules()
        assert len(rules) == 1
        assert rules[0].decision == "allow"
        assert rules[0].tool_pattern == "git.status"

    async def test_trust_auto_approve_skips_callback(self, trust_db):
        trust_db.record_decision("git.status", RiskLevel.SAFE, "allow", always=True)
        called = {"n": 0}

        async def callback(request):
            called["n"] += 1
            return ApprovalResponse(approved=True)

        handler = AsyncCallbackApprovalHandler(callback=callback, trust_db=trust_db)
        result = await handler.should_approve("git.status", "{}", RiskLevel.SAFE)
        assert result is True
        assert called["n"] == 0  # trust fast-path, callback never awaited

    async def test_callback_receives_typed_request(self):
        received: dict = {}

        async def callback(request: ApprovalRequest) -> ApprovalResponse:
            received["tool"] = request.tool_name
            received["risk"] = request.risk_level
            received["approval_id"] = request.approval_id
            return ApprovalResponse(approved=True)

        handler = AsyncCallbackApprovalHandler(callback=callback)
        await handler.should_approve("my_tool", '{"a": 1}', RiskLevel.MODERATE)
        assert received["tool"] == "my_tool"
        assert received["risk"] == RiskLevel.MODERATE
        assert received["approval_id"].startswith("ap_")
