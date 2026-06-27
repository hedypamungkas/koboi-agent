"""koboi/hooks/registry.py -- Hook registry for declarative hook construction.

Replaces the procedural _build_hooks() factory with a registry where each
hook entry declares its config key, enablement predicate, and factory callable.
Adding a new hook is zero-touch on the facade -- just append to _REGISTRY.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookChain

if TYPE_CHECKING:
    from koboi.config import Config
    from koboi.guardrails.audit import AuditTrail
    from koboi.harness.policy import PolicyEngine
    from koboi.logger import AgentLogger
    from koboi.modes import ModeManager
    from koboi.tools.registry import ToolRegistry

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HookEntry:
    """Metadata for a single hook registration.

    Attributes:
        name: Human-readable hook name (for logging).
        config_key: Dot-separated config path for logging (e.g. "harness.telemetry").
        should_add: Predicate(Config, **runtime_kwargs) -> bool.
        factory: Callable(Config, **runtime_kwargs) -> Hook.
    """

    name: str
    config_key: str
    should_add: Callable[..., bool]
    factory: Callable[..., Hook]


def _build_notif_events(raw_events: list[str]) -> list:
    """Map string event names to HookEvent values."""
    from koboi.hooks.chain import HookEvent

    event_map = {
        "post_output": HookEvent.POST_OUTPUT,
        "session_end": HookEvent.SESSION_END,
        "doom_loop": HookEvent.DOOM_LOOP_DETECTED,
        "agent_completed": HookEvent.AGENT_COMPLETED,
    }
    events = [event_map[e] for e in raw_events if e in event_map]
    return events or [HookEvent.POST_OUTPUT]


# ---------------------------------------------------------------------------
# Hook registry -- each entry is (name, config_key, should_add, factory)
# ---------------------------------------------------------------------------

_REGISTRY: list[HookEntry] = [
    HookEntry(
        name="AuditHook",
        config_key="guardrails.audit",
        should_add=lambda config, **kw: kw.get("audit_trail") is not None,
        factory=lambda config, **kw: _create_audit_hook(kw["audit_trail"]),
    ),
    HookEntry(
        name="PolicyHook",
        config_key="harness.policy",
        should_add=lambda config, **kw: kw.get("policy_engine") is not None,
        factory=lambda config, **kw: _create_policy_hook(kw["policy_engine"], kw.get("tool_registry"), config=config),
    ),
    HookEntry(
        name="ModeHook",
        config_key="agent.mode",
        should_add=lambda config, **kw: kw.get("mode_manager") is not None,
        factory=lambda config, **kw: _create_mode_hook(kw["mode_manager"]),
    ),
    HookEntry(
        name="TelemetryHook",
        config_key="harness.telemetry",
        should_add=lambda config, **kw: bool(config.get("harness", "telemetry", default=False)),
        factory=lambda config, **kw: _create_telemetry_hook(config),
    ),
    HookEntry(
        name="CarryoverHook",
        config_key="harness.carryover",
        should_add=lambda config, **kw: bool(config.get("harness", "carryover", default=False)),
        factory=lambda config, **kw: _create_carryover_hook(config),
    ),
    HookEntry(
        name="DoomLoopHook",
        config_key="harness.doom_loop",
        should_add=lambda config, **kw: config.get("harness", "doom_loop", default=None) is not None,
        factory=lambda config, **kw: _create_doom_loop_hook(config),
    ),
    HookEntry(
        name="TaskHook",
        config_key="harness.tasks",
        should_add=lambda config, **kw: config.get("harness", "tasks", default=None) is not None,
        factory=lambda config, **kw: _create_task_hook(config),
    ),
    HookEntry(
        name="NotificationHook",
        config_key="harness.notifications",
        should_add=lambda config, **kw: _should_add_notifications(config),
        factory=lambda config, **kw: _create_notification_hook(config),
    ),
    HookEntry(
        name="LangfuseTracingHook",
        config_key="tracing.langfuse",
        should_add=lambda config, **kw: config.tracing.get("provider") == "langfuse",
        factory=lambda config, **kw: _create_langfuse_hook(config),
    ),
]


# ---------------------------------------------------------------------------
# Factory helpers -- each returns a Hook or raises ImportError
# ---------------------------------------------------------------------------


def _create_audit_hook(audit_trail: AuditTrail | None) -> Hook:
    from koboi.hooks.builtin import AuditHook

    return AuditHook(audit_trail=audit_trail)


def _create_policy_hook(
    policy_engine: PolicyEngine | None,
    tool_registry: ToolRegistry | None = None,
    config: Config | None = None,
) -> Hook:
    from koboi.hooks.policy_hook import PolicyHook

    risk_lookup = {}
    if tool_registry is not None:
        risk_lookup = {name: tool.risk_level for name, tool in tool_registry._tools.items()}
    audit_log = None
    policy_conf = config.get("policy", default={}) if config else {}
    audit_path = policy_conf.get("audit_log")
    if audit_path:
        from koboi.harness.policy_audit import PolicyAuditLog

        audit_log = PolicyAuditLog(file_path=audit_path)
    return PolicyHook(policy_engine=policy_engine, risk_lookup=risk_lookup, audit_log=audit_log)


def _create_mode_hook(mode_manager: ModeManager | None) -> Hook:
    from koboi.hooks.mode_hook import ModeHook

    return ModeHook(mode_manager=mode_manager)


def _create_telemetry_hook(config: Config) -> Hook:
    from koboi.hooks.telemetry_hook import TelemetryHook
    from koboi.harness.telemetry import TelemetryCollector

    health_weights = config.get("harness", "health_weights")
    return TelemetryHook(telemetry=TelemetryCollector(health_weights=health_weights))


def _create_carryover_hook(config: Config) -> Hook:
    from koboi.hooks.carryover_hook import CarryoverHook
    from koboi.harness.carryover import CarryoverState

    carryover_limits = config.get("harness", "carryover_limits", default={})
    if carryover_limits:
        state = CarryoverState(
            max_log_entries=carryover_limits.get("max_log_entries", 50),
            max_goals=carryover_limits.get("max_goals", 10),
            max_artifacts=carryover_limits.get("max_artifacts", 20),
            max_verified=carryover_limits.get("max_verified", 20),
        )
    else:
        state = CarryoverState()
    return CarryoverHook(state=state)


def _create_doom_loop_hook(config: Config) -> Hook:
    from koboi.hooks.doom_loop_hook import DoomLoopHook
    from koboi.harness.doom_loop import DoomLoopConfig

    raw = config.harness.get("doom_loop", {})
    if isinstance(raw, dict):
        if "consecutive_threshold" in raw:
            raw["consecutive_identical_threshold"] = raw.pop("consecutive_threshold")
        doom_config = DoomLoopConfig(**raw)
    else:
        doom_config = DoomLoopConfig()
    return DoomLoopHook(config=doom_config)


def _create_task_hook(config: Config) -> Hook:
    from koboi.hooks.task_hook import TaskHook

    tasks_conf = config.get("harness", "tasks", default={})
    reminder_interval = tasks_conf.get("reminder_interval", 3) if isinstance(tasks_conf, dict) else 3
    return TaskHook(reminder_interval=reminder_interval)


def _should_add_notifications(config: Config) -> bool:
    notif_conf = config.get("harness", "notifications", default=None)
    return bool(notif_conf and notif_conf.get("enabled", True))


def _create_notification_hook(config: Config) -> Hook:
    from koboi.hooks.notification_hook import NotificationHook

    notif_conf = config.get("harness", "notifications", default={})
    raw_events = notif_conf.get("events", ["post_output"])
    events = _build_notif_events(raw_events)
    return NotificationHook(
        events=events,
        sound=notif_conf.get("sound", False),
        sound_name=notif_conf.get("sound_name", "Ping"),
    )


def _create_langfuse_hook(config: Config) -> Hook:
    from koboi.hooks.langfuse_hook import LangfuseTracingHook

    tracing_conf = config.tracing
    return LangfuseTracingHook(
        public_key=tracing_conf.get("public_key", ""),
        secret_key=tracing_conf.get("secret_key", ""),
        base_url=tracing_conf.get("base_url", "http://localhost:3300"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_hook_chain(
    config: Config,
    logger: AgentLogger,
    audit_trail: AuditTrail | None = None,
    mode_manager: ModeManager | None = None,
    verbose: bool = False,
    policy_engine: PolicyEngine | None = None,
    tool_registry: ToolRegistry | None = None,
) -> HookChain:
    """Build a HookChain from config using the registry.

    This replaces the procedural _build_hooks() function. Each registered
    hook entry is checked against config and runtime dependencies; matching
    entries are instantiated and added to the chain.

    Always adds LoggingHook (infrastructure, unconditional).
    """
    from koboi.hooks.builtin import LoggingHook

    chain = HookChain()
    chain.add(LoggingHook(logger=logger, verbose=verbose))

    runtime_kwargs = {
        "audit_trail": audit_trail,
        "mode_manager": mode_manager,
        "policy_engine": policy_engine,
        "tool_registry": tool_registry,
    }

    for entry in _REGISTRY:
        if entry.should_add(config, **runtime_kwargs):
            try:
                hook = entry.factory(config, **runtime_kwargs)
                chain.add(hook)
            except ImportError:
                _logger.debug("Optional hook %s not available (missing dependency)", entry.name)
            except Exception:
                _logger.warning("Failed to create hook %s", entry.name, exc_info=True)

    return chain


def register_hook(entry: HookEntry) -> None:
    """Register a new hook entry. Call this from hook modules at import time."""
    _REGISTRY.append(entry)


def list_entries() -> list[HookEntry]:
    """Return a copy of all registered hook entries."""
    return list(_REGISTRY)
