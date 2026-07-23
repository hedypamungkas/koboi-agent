"""koboi/exceptions.py -- Agent-level exception hierarchy."""

from __future__ import annotations


class AgentError(Exception):
    """Base for all agent-level errors."""


class AgentMaxIterationsError(AgentError):
    def __init__(self, iterations: int, message: str = ""):
        self.iterations = iterations
        super().__init__(message or f"Agent reached maximum iterations ({iterations})")


class AgentBudgetExceededError(AgentError):
    """Raised when a run exceeds its configured token/cost budget (Wave 2).

    Configured via ``agent.max_total_tokens`` / ``agent.max_cost_usd``; with
    ``self_healing.graceful_max_iter`` on, the loop degrades to a summary
    instead of raising.
    """

    def __init__(
        self,
        *,
        spent_tokens: int,
        spent_usd: float,
        limit: str,
        message: str = "",
    ):
        self.spent_tokens = spent_tokens
        self.spent_usd = spent_usd
        self.limit = limit
        super().__init__(message or f"Agent budget exceeded ({limit}): spent {spent_tokens} tokens (~${spent_usd:.4f})")


class AgentGuardrailError(AgentError):
    def __init__(
        self,
        reason: str,
        direction: str = "input",
        sanitized_content: str | None = None,
    ):
        self.reason = reason
        self.direction = direction
        # When set on an INPUT-direction block, the engine surfaces this as the graceful
        # response instead of a hard block -> generic fallback (see loop.py
        # `_graceful_input_deflection`). Only INPUT guardrails populate it -- today
        # `injection_detector` with `deflection_text` (and a custom input guardrail that
        # returns sanitized_content on a block); empty/length input blocks leave it None
        # -> raise as before. NOTE: distinct from `GuardrailResult.sanitized_content`,
        # which OUTPUT guardrails (Grounding/ScopeGuardrail) set on the `abstain`/
        # `handover` *result* consumed by `_process_output` (a different type + path);
        # output blocks (`direction="output"`) never set it here.
        self.sanitized_content = sanitized_content
        super().__init__(f"Guardrail blocked ({direction}): {reason}")

    @property
    def is_graceful_deflection(self) -> bool:
        """True when the engine should surface ``sanitized_content`` as a graceful
        in-character reply instead of raising -> generic fallback. Centralizes the
        gate so the two call sites in ``run``/``run_stream`` stay in sync: input
        direction + a non-empty (after strip) deflection payload. None / "" / "  "
        all fall through to "raise as before"."""
        return self.direction == "input" and bool(self.sanitized_content and self.sanitized_content.strip())


class AgentToolError(AgentError):
    def __init__(self, tool_name: str, detail: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {detail}")


class AgentTimeoutError(AgentError):
    pass


class AgentStreamError(AgentError):
    pass


class AgentAbortedError(AgentError):
    """Raised when a hook aborts execution."""


class AgentHandoverError(AgentError):
    """Raised (by the ``transfer_to_human`` tool or a handover hook) to yield the
    conversation to a human operator (B1). Propagates out of ``run_stream``/``run``
    uncaught (exactly like ``AgentGuardrailError`` from ``_process_output``) and is
    caught in ``_run_agent`` (SSE -> ``HandoverEvent``) or ``run_job`` (->
    ``awaiting_human``). Stop-then-resume: NO Future is awaited, so ``pool.session_lock``
    releases when the run ends -- a human operator can then take over the session.
    """

    def __init__(self, reason: str, summary: str = ""):
        self.reason = reason
        self.summary = summary
        super().__init__(f"Handover: {reason}")
