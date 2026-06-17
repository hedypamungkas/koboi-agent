"""Tests for koboi/guardrails/tui_approval.py -- TUI approval handler."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from koboi.tui.approval import (
    TUIApprovalHandler,
    PermissionRequest,
    PermissionResponse,
)
from koboi.types import RiskLevel


class TestPermissionRequest:
    def test_init(self):
        future = MagicMock()
        msg = PermissionRequest("tool", "{}", "safe", future)
        assert msg.tool_name == "tool"
        assert msg.arguments == "{}"
        assert msg.risk_level == "safe"


class TestPermissionResponse:
    def test_init_approved(self):
        msg = PermissionResponse(approved=True, always_allow=True)
        assert msg.approved is True
        assert msg.always_allow is True

    def test_init_denied(self):
        msg = PermissionResponse(approved=False)
        assert msg.approved is False
        assert msg.always_allow is False


class TestTUIApprovalHandler:
    def test_init(self):
        app = MagicMock()
        handler = TUIApprovalHandler(app)
        assert handler._app is app
        assert handler._trust_db is None

    def test_resolve_pending(self):
        app = MagicMock()
        handler = TUIApprovalHandler(app)
        future = MagicMock()
        future.done.return_value = False
        handler._pending_future = future
        response = PermissionResponse(approved=True)
        handler.resolve_pending(response)
        future.set_result.assert_called_once_with(response)

    def test_resolve_no_pending(self):
        app = MagicMock()
        handler = TUIApprovalHandler(app)
        # Should not raise
        handler.resolve_pending(PermissionResponse(approved=True))

    def test_resolve_already_done(self):
        app = MagicMock()
        handler = TUIApprovalHandler(app)
        future = MagicMock()
        future.done.return_value = True
        handler._pending_future = future
        handler.resolve_pending(PermissionResponse(approved=True))
        future.set_result.assert_not_called()

    def test_cancel_pending(self):
        app = MagicMock()
        handler = TUIApprovalHandler(app)
        future = MagicMock()
        future.done.return_value = False
        handler._pending_future = future
        handler.cancel_pending()
        future.cancel.assert_called_once()

    def test_cancel_no_pending(self):
        app = MagicMock()
        handler = TUIApprovalHandler(app)
        # Should not raise
        handler.cancel_pending()

    def test_cancel_already_done(self):
        app = MagicMock()
        handler = TUIApprovalHandler(app)
        future = MagicMock()
        future.done.return_value = True
        handler._pending_future = future
        handler.cancel_pending()
        future.cancel.assert_not_called()

    def test_audit_records(self):
        app = MagicMock()
        trail = MagicMock()
        handler = TUIApprovalHandler(app, audit_trail=trail)
        handler._audit("tool", "{}", RiskLevel.SAFE, True, "test")
        trail.record.assert_called_once()

    def test_audit_no_trail(self):
        app = MagicMock()
        handler = TUIApprovalHandler(app)
        # Should not raise
        handler._audit("tool", "{}", RiskLevel.SAFE, True, "test")

    @pytest.mark.asyncio
    async def test_should_approve_with_trust_auto_approve(self):
        app = MagicMock()
        trust_db = MagicMock()
        trust_decision = MagicMock()
        trust_decision.auto_approve = True
        trust_decision.reason = "trusted"
        trust_db.should_auto_approve.return_value = trust_decision
        handler = TUIApprovalHandler(app, trust_db=trust_db)
        result = await handler.should_approve("read", "{}", RiskLevel.SAFE)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_approve_posts_message(self):
        app = MagicMock()

        # Create a handler that resolves the future immediately
        handler = TUIApprovalHandler(app)
        original_should_approve = handler.should_approve

        async def mock_should_approve(tool_name, arguments, risk_level):
            # Simulate what the app would do: resolve the future
            app.post_message.side_effect = lambda msg: (
                handler.resolve_pending(PermissionResponse(approved=True))
                if isinstance(msg, PermissionRequest)
                else None
            )
            return await original_should_approve(tool_name, arguments, risk_level)

        result = await mock_should_approve("tool", "{}", RiskLevel.MODERATE)
        assert result is True
        app.post_message.assert_called_once()
