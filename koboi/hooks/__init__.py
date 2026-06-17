"""koboi/hooks -- Observer-pattern lifecycle hook system."""

from koboi.hooks.chain import HookEvent, HookContext, Hook, HookChain, HookOutcome, AgentInfo
from koboi.hooks.callback_hook import CallbackHook
from koboi.hooks.builtin import LoggingHook, AuditHook
from koboi.hooks.registry import HookEntry, build_hook_chain, register_hook

__all__ = [
    "HookEvent",
    "HookContext",
    "Hook",
    "HookChain",
    "HookOutcome",
    "AgentInfo",
    "CallbackHook",
    "LoggingHook",
    "AuditHook",
    "HookEntry",
    "build_hook_chain",
    "register_hook",
]
