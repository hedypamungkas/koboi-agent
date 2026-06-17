"""Tests for the PolicyHook in koboi.hooks.policy_hook."""
from __future__ import annotations

import pytest

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.policy_hook import PolicyHook
from koboi.harness.policy import PolicyEngine, PolicyRule, PolicyAction, PolicyDecision
from koboi.types import RiskLevel


class TestPolicyHookInit:
    def test_imports(self):
        hook = PolicyHook(PolicyEngine())
        assert isinstance(hook, PolicyHook)

    def test_handles_pre_tool_use(self):
        hook = PolicyHook(PolicyEngine())
        assert HookEvent.PRE_TOOL_USE in hook.handles()
        assert len(hook.handles()) == 1

    def test_default_risk_is_safe(self):
        hook = PolicyHook(PolicyEngine())
        assert hook.default_risk == RiskLevel.SAFE

    def test_custom_risk_lookup(self):
        lookup = {"run_shell": RiskLevel.MODERATE}
        hook = PolicyHook(PolicyEngine(), risk_lookup=lookup)
        assert hook.risk_lookup == lookup


class TestPolicyHookDeny:
    async def test_deny_sets_abort(self):
        engine = PolicyEngine([
            PolicyRule(
                name="deny_all",
                action=PolicyAction.DENY,
                tool_pattern="*",
                description="Block everything",
            ),
        ])
        hook = PolicyHook(engine)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="test_tool")
        result = await hook.execute(ctx)
        assert result.abort is True

    async def test_deny_injects_message(self):
        engine = PolicyEngine([
            PolicyRule(
                name="deny_shell",
                action=PolicyAction.DENY,
                tool_pattern="run_shell",
                description="Shell is blocked",
            ),
        ])
        hook = PolicyHook(engine)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="run_shell")
        result = await hook.execute(ctx)
        assert result.inject_message is not None
        assert "Policy denied" in result.inject_message
        assert "Shell is blocked" in result.inject_message

    async def test_deny_stores_decision_in_metadata(self):
        engine = PolicyEngine([
            PolicyRule(
                name="deny_calc",
                action=PolicyAction.DENY,
                tool_pattern="calculate",
                description="Calculator blocked",
            ),
        ])
        hook = PolicyHook(engine)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="calculate")
        result = await hook.execute(ctx)
        assert "policy_decision" in result.metadata
        decision = result.metadata["policy_decision"]
        assert decision["action"] == "deny"
        assert decision["matched_rule"] == "deny_calc"

    async def test_sensitive_path_denied(self):
        engine = PolicyEngine()
        hook = PolicyHook(engine)
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="run_shell",
            tool_arguments='{"command": "cat /etc/shadow"}',
        )
        result = await hook.execute(ctx)
        assert result.abort is True
        assert "sensitive path" in result.inject_message.lower() or "sensitive" in result.inject_message.lower()

    async def test_command_deny_pattern(self):
        engine = PolicyEngine()
        hook = PolicyHook(engine)
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="run_shell",
            tool_arguments='{"command": "rm -rf /"}',
        )
        result = await hook.execute(ctx)
        assert result.abort is True


class TestPolicyHookAllow:
    async def test_allow_passes_through(self):
        engine = PolicyEngine()
        hook = PolicyHook(engine)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="calculate")
        result = await hook.execute(ctx)
        assert result.abort is False

    async def test_allow_stores_decision_in_metadata(self):
        engine = PolicyEngine()
        hook = PolicyHook(engine)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="calculate")
        result = await hook.execute(ctx)
        assert "policy_decision" in result.metadata
        decision = result.metadata["policy_decision"]
        assert decision["action"] == "allow"

    async def test_explicit_allow_rule(self):
        engine = PolicyEngine([
            PolicyRule(
                name="allow_calc",
                action=PolicyAction.ALLOW,
                tool_pattern="calculate",
                description="Calculator is fine",
            ),
        ])
        hook = PolicyHook(engine)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="calculate")
        result = await hook.execute(ctx)
        assert result.abort is False
        assert result.metadata["policy_decision"]["matched_rule"] == "allow_calc"


class TestPolicyHookConfirm:
    async def test_confirm_flags_metadata(self):
        engine = PolicyEngine([
            PolicyRule(
                name="confirm_shell",
                action=PolicyAction.CONFIRM,
                tool_pattern="run_shell",
                description="Needs user approval",
            ),
        ])
        hook = PolicyHook(engine, risk_lookup={"run_shell": RiskLevel.MODERATE})
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="run_shell")
        result = await hook.execute(ctx)
        assert result.abort is False
        assert result.metadata.get("policy_needs_confirmation") is True
        assert "policy_reason" in result.metadata

    async def test_destructive_fallback_confirms(self):
        engine = PolicyEngine()
        hook = PolicyHook(engine, risk_lookup={"nuke": RiskLevel.DESTRUCTIVE})
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="nuke")
        result = await hook.execute(ctx)
        assert result.abort is False
        assert result.metadata.get("policy_needs_confirmation") is True


class TestPolicyHookMissingToolName:
    async def test_missing_tool_name_passes_through(self):
        engine = PolicyEngine([
            PolicyRule(
                name="deny_all",
                action=PolicyAction.DENY,
                tool_pattern="*",
                description="Block everything",
            ),
        ])
        hook = PolicyHook(engine)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name=None)
        result = await hook.execute(ctx)
        assert result.abort is False
        assert "policy_decision" not in result.metadata

    async def test_empty_tool_name_passes_through(self):
        engine = PolicyEngine([
            PolicyRule(
                name="deny_all",
                action=PolicyAction.DENY,
                tool_pattern="*",
                description="Block everything",
            ),
        ])
        hook = PolicyHook(engine)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name=None)
        result = await hook.execute(ctx)
        assert result.abort is False


class TestPolicyHookRiskLookup:
    async def test_risk_lookup_overrides_default(self):
        engine = PolicyEngine()
        hook = PolicyHook(
            engine,
            risk_lookup={"dangerous_tool": RiskLevel.DESTRUCTIVE},
        )
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="dangerous_tool")
        result = await hook.execute(ctx)
        assert result.metadata.get("policy_needs_confirmation") is True

    async def test_unknown_tool_uses_default_risk(self):
        engine = PolicyEngine()
        hook = PolicyHook(engine, default_risk=RiskLevel.SAFE)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="unknown_tool")
        result = await hook.execute(ctx)
        assert result.abort is False
        assert result.metadata["policy_decision"]["action"] == "allow"


class TestPolicyHookWithContextArguments:
    async def test_arguments_passed_to_engine(self):
        engine = PolicyEngine()
        hook = PolicyHook(engine)
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="run_shell",
            tool_arguments='{"command": "echo hello"}',
        )
        result = await hook.execute(ctx)
        assert result.abort is False

    async def test_none_arguments_handled(self):
        engine = PolicyEngine([
            PolicyRule(
                name="allow_all",
                action=PolicyAction.ALLOW,
                tool_pattern="*",
            ),
        ])
        hook = PolicyHook(engine)
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="test",
            tool_arguments=None,
        )
        result = await hook.execute(ctx)
        assert result.abort is False
