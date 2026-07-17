"""tests/test_self_healing_facade.py -- facade-wiring tests (config → hook construction).

Verifies that KoboiAgent.from_dict() with self_healing.enabled=True constructs the
correct hooks on agent._core.hooks (ReflectionHook, FailureClassifierHook,
LadderRouterHook, HandoverDetectionHook) — the config → AgentAssembler.build() →
hook_chain path that no other test exercises.
"""

from __future__ import annotations

from koboi.facade import KoboiAgent
from koboi.hooks.failure_classifier_hook import FailureClassifierHook
from koboi.hooks.handover_detection_hook import HandoverDetectionHook
from koboi.hooks.ladder_router_hook import LadderRouterHook
from koboi.hooks.reflection_hook import ReflectionHook

_BASE = {
    "agent": {"name": "test-sh", "max_iterations": 5},
    "llm": {"model": "gpt-4o-mini", "api_key": "test-key"},
    "guardrails": {
        "output": [{"name": "grounding_check", "provider": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}]
    },
}


class TestFacadeWiring:
    def test_enabled_wires_all_hooks(self):
        config = {
            **_BASE,
            "handover": {"detection": {"enabled": True}},
            "self_healing": {"enabled": True, "max_turns": 3},
        }
        agent = KoboiAgent.from_dict(config)
        hooks = agent._core.hooks
        assert hooks.find_hook(lambda h: isinstance(h, ReflectionHook)) is not None
        assert hooks.find_hook(lambda h: isinstance(h, FailureClassifierHook)) is not None
        assert hooks.find_hook(lambda h: isinstance(h, LadderRouterHook)) is not None
        assert hooks.find_hook(lambda h: isinstance(h, HandoverDetectionHook)) is not None

    def test_disabled_wires_no_self_healing_hooks(self):
        agent = KoboiAgent.from_dict(_BASE)
        hooks = agent._core.hooks
        assert hooks.find_hook(lambda h: isinstance(h, ReflectionHook)) is None
        assert hooks.find_hook(lambda h: isinstance(h, FailureClassifierHook)) is None
        assert hooks.find_hook(lambda h: isinstance(h, LadderRouterHook)) is None

    def test_shared_budget_between_router_and_reflection(self):
        config = {
            **_BASE,
            "handover": {"detection": {"enabled": True}},
            "self_healing": {"enabled": True, "max_turns": 3},
        }
        agent = KoboiAgent.from_dict(config)
        router = agent._core.hooks.find_hook(lambda h: isinstance(h, LadderRouterHook))
        reflection = agent._core.hooks.find_hook(lambda h: isinstance(h, ReflectionHook))
        assert router is not None and reflection is not None
        # The shared RecoveryBudget is the SAME object on both hooks.
        assert router._budget is reflection._budget

    def test_tool_verification_enabled_passes_tools(self):
        config = {
            **_BASE,
            "handover": {"detection": {"enabled": True}},
            "self_healing": {
                "enabled": True,
                "tool_verification": {"enabled": True},
            },
        }
        agent = KoboiAgent.from_dict(config)
        reflection = agent._core.hooks.find_hook(lambda h: isinstance(h, ReflectionHook))
        assert reflection is not None
        assert reflection._tools is not None  # tool registry injected

    def test_tool_verification_disabled_no_tools(self):
        config = {
            **_BASE,
            "handover": {"detection": {"enabled": True}},
            "self_healing": {"enabled": True},
        }
        agent = KoboiAgent.from_dict(config)
        reflection = agent._core.hooks.find_hook(lambda h: isinstance(h, ReflectionHook))
        assert reflection is not None
        assert reflection._tools is None  # no tool registry injected
