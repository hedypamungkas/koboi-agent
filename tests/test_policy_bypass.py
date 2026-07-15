"""Tests for issue #45 -- deny-list bypass via trivial command variants.

The hardcoded ``COMMAND_DENY_PATTERNS`` are exact-anchored regexes and
``SENSITIVE_PATHS`` use a plain substring match, so trivial variants slip past
them (``python3 -W ignore -c``, ``bash -ic``, ``rm -fr /``, ``cat /etc/pass*``).
These tests exercise both gates (``PolicyEngine.evaluate`` and the shell tool's
``_check_command_blocked``) plus a false-positive guard list that MUST stay
allowed.
"""

from __future__ import annotations

import json

import pytest

from koboi.harness.policy import PolicyAction, PolicyEngine
from koboi.tools.builtin.shell import _check_command_blocked
from koboi.types import RiskLevel

BYPASS_COMMANDS = [
    "python3 -W ignore -c 'import os'",
    "python3 <<< 'import os'",
    "bash -ic 'echo pwned'",
    "cat /etc/pass''wd",
    "cat /etc/pass*",
    "rm -fr /",
    "rm --recursive --force /",
]


@pytest.mark.parametrize("cmd", BYPASS_COMMANDS)
def test_evaluate_denies_bypass(cmd):
    engine = PolicyEngine()
    decision = engine.evaluate("run_shell", json.dumps({"command": cmd}), RiskLevel.DESTRUCTIVE)
    assert decision.action == PolicyAction.DENY, f"{cmd!r} was NOT denied: {decision}"


@pytest.mark.parametrize("cmd", BYPASS_COMMANDS)
def test_check_command_blocked_catches_bypass(cmd):
    reason = _check_command_blocked(cmd)
    assert reason is not None, f"{cmd!r} was NOT blocked by the shell gate"


# FP guards -- these must stay ALLOWED
ALLOWED_COMMANDS = [
    "python3 --version",
    "python3 myscript.py",
    "echo hello | tr a-z A-Z",
    "cat README.md",
    "rm -rf build/",
]


@pytest.mark.parametrize("cmd", ALLOWED_COMMANDS)
def test_allowed_commands_stay_allowed(cmd):
    reason = _check_command_blocked(cmd)
    assert reason is None, f"{cmd!r} was WRONGLY blocked (false positive): {reason}"
