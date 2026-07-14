"""koboi/exceptions.py -- Agent-level exception hierarchy."""

from __future__ import annotations


class AgentError(Exception):
    """Base for all agent-level errors."""


class AgentMaxIterationsError(AgentError):
    def __init__(self, iterations: int, message: str = ""):
        self.iterations = iterations
        super().__init__(message or f"Agent reached maximum iterations ({iterations})")


class AgentGuardrailError(AgentError):
    def __init__(self, reason: str, direction: str = "input"):
        self.reason = reason
        self.direction = direction
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
