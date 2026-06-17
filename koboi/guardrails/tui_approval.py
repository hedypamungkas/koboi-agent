"""koboi/guardrails/tui_approval.py -- Backward-compat re-export.

Moved to koboi.tui.approval to keep core guardrails free of TUI dependencies.
"""
from koboi.tui.approval import (
    PermissionRequest,
    PermissionResponse,
    TUIApprovalHandler,
)

__all__ = ["TUIApprovalHandler", "PermissionRequest", "PermissionResponse"]
