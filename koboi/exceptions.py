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
