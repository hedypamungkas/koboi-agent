"""tests/test_policy_argument_patterns.py -- #4 policy.rules argument matching.

Pre-#4 ``_build_policy`` hardcoded ``argument_patterns={"command": pattern}``, so a
policy rule could only ever match an argument literally named ``command`` -- custom
tools with args like ``filename``/``path``/``query`` silently never matched. The
PolicyEngine already supported arbitrary arg-name globs; only the facade hardcoded
the key. Now ``argument_patterns: {arg: glob}`` is honored, with the legacy
``pattern:`` shorthand preserved for back-compat.
"""

from __future__ import annotations

from koboi.config import Config
from koboi.facade import _build_policy
from koboi.harness.policy import PolicyAction
from koboi.types import RiskLevel


def test_argument_patterns_match_custom_arg():
    config = Config(
        {
            "policy": {
                "rules": [
                    {
                        "tool": "set_tag",
                        "argument_patterns": {"tag": "blocked-*"},
                        "action": "deny",
                    }
                ]
            }
        }
    )
    engine = _build_policy(config)

    denied = engine.evaluate("set_tag", '{"tag": "blocked-xyz"}', RiskLevel.MODERATE)
    assert denied.action == PolicyAction.DENY

    # Non-matching value -> not denied by this rule
    allowed = engine.evaluate("set_tag", '{"tag": "ok"}', RiskLevel.MODERATE)
    assert allowed.action == PolicyAction.ALLOW


def test_argument_patterns_multiple_args_all_must_match():
    config = Config(
        {
            "policy": {
                "rules": [
                    {
                        "tool": "set_tag",
                        "argument_patterns": {"tag": "blocked-*", "env": "live"},
                        "action": "deny",
                    }
                ]
            }
        }
    )
    engine = _build_policy(config)

    both = engine.evaluate("set_tag", '{"tag": "blocked-xyz", "env": "live"}', RiskLevel.MODERATE)
    assert both.action == PolicyAction.DENY

    # Only one of two patterns matches -> rule does not fire
    one = engine.evaluate("set_tag", '{"tag": "blocked-xyz", "env": "test"}', RiskLevel.MODERATE)
    assert one.action == PolicyAction.ALLOW


def test_legacy_pattern_shorthand_still_matches_command_arg():
    """Back-compat: ``pattern:`` still maps to the ``command`` arg for run_shell configs."""
    config = Config({"policy": {"rules": [{"tool": "run_shell", "pattern": "forbidden-cmd", "action": "deny"}]}})
    engine = _build_policy(config)

    denied = engine.evaluate("run_shell", '{"command": "forbidden-cmd"}', RiskLevel.DESTRUCTIVE)
    assert denied.action == PolicyAction.DENY
