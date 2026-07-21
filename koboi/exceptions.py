"""koboi/exceptions.py -- Agent-level exception hierarchy."""

from __future__ import annotations


class AgentError(Exception):
    """Base for all agent-level errors."""


class AgentMaxIterationsError(AgentError):
    def __init__(self, iterations: int, message: str = ""):
        self.iterations = iterations
        super().__init__(message or f"Agent reached maximum iterations ({iterations})")


class AgentGuardrailError(AgentError):
    def __init__(
        self,
        reason: str,
        direction: str = "input",
        sanitized_content: str | None = None,
    ):
        self.reason = reason
        self.direction = direction
        # When set, the engine surfaces this as the graceful response instead of a
        # hard block -> generic fallback (see loop.py `_graceful_input_deflection` /
        # the output `abstain` path). Opt-in per guardrail -- only guardrails that
        # supply a deflection (e.g. injection_detector with deflection_text, or
        # ScopeGuardrail) set it; empty/length blocks leave it None -> raise as before.
        self.sanitized_content = sanitized_content
        super().__init__(f"Guardrail blocked ({direction}): {reason}")


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
