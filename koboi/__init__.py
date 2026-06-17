"""koboi — Universal configurable AI agent framework.

Usage:
    from koboi import KoboiAgent

    agent = KoboiAgent.from_config("configs/sales_agent.yaml")
    result = await agent.run("What products are available?")
"""
from __future__ import annotations

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("koboi-agent")
except Exception:
    __version__ = "0.1.0"

# --- Eager imports: lightweight, always needed ---
from koboi.config import Config, ConfigBuilder
from koboi.facade import KoboiAgent
from koboi.types import (
    RiskLevel, ToolDefinition, ToolCall, ToolResult,
    TokenUsage, AgentResponse, RunResult,
    GuardrailResult, AuditEntry, RateLimitConfig,
    RoutingDecision, AgentResult, OrchestratorResult, AgentBlueprint, AgentDef,
    MCPToolInfo, SkillDefinition,
    EvalScore, EvalResult, EvalCase,
)
from koboi.client import RetryClient, Client, ClientError, RetryClientError
from koboi.memory import ConversationMemory, MemoryBackend
from koboi.tools.registry import ToolRegistry, tool, register_decorated
from koboi.logger import AgentLogger
from koboi.tokens import estimate_tokens, estimate_single
from koboi.exceptions import (
    AgentError, AgentMaxIterationsError, AgentGuardrailError,
    AgentToolError, AgentTimeoutError, AgentStreamError, AgentAbortedError,
)
from koboi.events import (
    StreamEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent,
    IterationEvent, CompleteEvent, ErrorEvent,
    RoutingDecisionEvent, AgentDispatchEvent, AgentResultEvent, OrchestrationCompleteEvent,
)
from koboi.hooks import HookEvent, HookContext, Hook, HookChain, HookOutcome, AgentInfo
from koboi.hooks.callback_hook import CallbackHook

# --- Lazy imports: heavier subsystems, loaded on first access ---
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # LLM
    "LLMClient": ("koboi.llm.base", "LLMClient"),
    "LLMError": ("koboi.llm.base", "LLMError"),
    "LLMConnectionError": ("koboi.llm.base", "LLMConnectionError"),
    "LLMAuthenticationError": ("koboi.llm.base", "LLMAuthenticationError"),
    "LLMRateLimitError": ("koboi.llm.base", "LLMRateLimitError"),
    "LLMServerError": ("koboi.llm.base", "LLMServerError"),
    "LLMInvalidRequestError": ("koboi.llm.base", "LLMInvalidRequestError"),
    "LLMResponseParseError": ("koboi.llm.base", "LLMResponseParseError"),
    "create_client": ("koboi.llm.factory", "create_client"),
    # Orchestration
    "BaseRouter": ("koboi.orchestration.router", "BaseRouter"),
    "KeywordRouter": ("koboi.orchestration.router", "KeywordRouter"),
    "LLMRouter": ("koboi.orchestration.router", "LLMRouter"),
    "HybridRouter": ("koboi.orchestration.router", "HybridRouter"),
    "Orchestrator": ("koboi.orchestration.orchestrator", "Orchestrator"),
    "QualityEvaluator": ("koboi.orchestration.orchestrator", "QualityEvaluator"),
    "AgentFactory": ("koboi.orchestration.factory", "AgentFactory"),
    "DynamicAgentBuilder": ("koboi.orchestration.factory", "DynamicAgentBuilder"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'koboi' has no attribute {name!r}")


# Discover and load external plugins at import time
try:
    from koboi.plugins import discover_plugins as _discover_plugins
    _discover_plugins()
except Exception:
    pass  # Plugin discovery is best-effort

__all__ = [
    # Core
    "__version__", "KoboiAgent", "Config", "ConfigBuilder",
    # Orchestration
    "BaseRouter", "KeywordRouter", "LLMRouter", "HybridRouter",
    "Orchestrator", "QualityEvaluator", "AgentFactory", "DynamicAgentBuilder",
    # Types
    "RiskLevel", "ToolDefinition", "ToolCall", "ToolResult",
    "TokenUsage", "AgentResponse", "RunResult",
    "GuardrailResult", "AuditEntry", "RateLimitConfig",
    "RoutingDecision", "AgentResult", "OrchestratorResult", "AgentBlueprint", "AgentDef",
    "MCPToolInfo", "SkillDefinition",
    "EvalScore", "EvalResult", "EvalCase",
    # Client
    "RetryClient", "Client", "ClientError", "RetryClientError",
    # Memory
    "ConversationMemory", "MemoryBackend",
    # Tools
    "ToolRegistry", "tool", "register_decorated",
    # Logging
    "AgentLogger",
    # Tokens
    "estimate_tokens", "estimate_single",
    # Exceptions
    "AgentError", "AgentMaxIterationsError", "AgentGuardrailError",
    "AgentToolError", "AgentTimeoutError", "AgentStreamError", "AgentAbortedError",
    # Events
    "StreamEvent", "TextDeltaEvent", "ToolCallEvent", "ToolResultEvent",
    "IterationEvent", "CompleteEvent", "ErrorEvent",
    "RoutingDecisionEvent", "AgentDispatchEvent", "AgentResultEvent", "OrchestrationCompleteEvent",
    # Hooks
    "HookEvent", "HookContext", "Hook", "HookChain", "HookOutcome", "AgentInfo", "CallbackHook",
    # LLM
    "LLMClient", "LLMError", "LLMConnectionError", "LLMAuthenticationError",
    "LLMRateLimitError", "LLMServerError", "LLMInvalidRequestError", "LLMResponseParseError",
    "create_client",
]
