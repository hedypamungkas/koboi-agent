from koboi.guardrails.base import BaseGuardrail, PatternGuardrail
from koboi.guardrails.input import InputGuardrail
from koboi.guardrails.output import OutputGuardrail
from koboi.guardrails.grounding import GroundingGuardrail
from koboi.guardrails.scope import ScopeGuardrail
from koboi.guardrails.rate_limiter import RateLimiter
from koboi.guardrails.audit import AuditTrail, SQLiteAuditTrail
from koboi.guardrails.approval import ApprovalHandler, CLIApprovalHandler, CallbackApprovalHandler
from koboi.guardrails.registry import GuardrailRegistry, register_builtin_guardrails

# Register built-in guardrail factories at import time
register_builtin_guardrails()

__all__ = [
    "BaseGuardrail",
    "PatternGuardrail",
    "InputGuardrail",
    "OutputGuardrail",
    "GroundingGuardrail",
    "ScopeGuardrail",
    "RateLimiter",
    "AuditTrail",
    "SQLiteAuditTrail",
    "ApprovalHandler",
    "CLIApprovalHandler",
    "CallbackApprovalHandler",
    "GuardrailRegistry",
]
