"""koboi/harness/policy.py -- Policy-based permission engine for tool execution.

Replaces simple risk-level-based approval with an engine supporting:
- Glob pattern matching on tool name and arguments
- Hardcoded sensitive path protection (non-overridable)
- Command deny patterns (regex-based)
- Composable rules with first-match-wins priority
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch

from koboi.types import RiskLevel


class PolicyAction(Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass
class PolicyRule:
    name: str
    action: PolicyAction
    tool_pattern: str = "*"
    argument_patterns: dict[str, str] | None = None
    risk_levels: list[RiskLevel] | None = None
    description: str = ""


@dataclass
class PolicyDecision:
    action: PolicyAction
    matched_rule: str | None = None
    reason: str = ""


SENSITIVE_PATHS = [
    "/.ssh/",
    "/.aws/credentials",
    "/.gnupg/",
    "/.env",
    ".env",
    "/credentials",
    "/credentials.json",
    "/etc/shadow",
    "/etc/passwd",
    "/id_rsa",
    "/id_ed25519",
]

COMMAND_DENY_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r"rm\s+-[a-zA-Z]*f\s+\*"),
    re.compile(r"rm\s+-rf\s+\."),
    re.compile(r"mkfs\."),
    re.compile(r"dd\s+if="),
    re.compile(r"curl\b.*\|\s*bash"),
    re.compile(r"curl\b.*\|\s*sh"),
    re.compile(r"wget\b.*\|\s*bash"),
    re.compile(r"wget\b.*\|\s*sh"),
    re.compile(r":\(\)\{.*\}"),  # fork bomb
    re.compile(r"chmod\s+-R\s+777\s+/"),
    re.compile(r"shutdown\b"),
    re.compile(r"reboot\b"),
    # C2: interpreter-exec / exfil-evasion vectors (defense-in-depth). Inline
    # interpreters bypass file-based arguments and are the primary prompt-injection
    # exfil path (python3 -c, perl -e, bash -c, ...); /dev/tcp is bash net-exfil;
    # base64-decode-into-shell hides payloads. Blocked even with a Trust rule.
    re.compile(r"\bpython[0-9.]*\s+-c\b"),
    re.compile(r"\bperl\s+-e\b"),
    re.compile(r"\b(?:bash|sh|dash|zsh)\s+-c\b"),
    re.compile(r"\bnode\s+-e\b"),
    re.compile(r"\bruby\s+-e\b"),
    re.compile(r"/dev/tcp"),
    re.compile(r"base64\b[^|]*\|\s*(?:bash|sh|python)"),
]


class PolicyEngine:
    def __init__(self, rules: list[PolicyRule] | None = None):
        self._rules: list[PolicyRule] = list(rules) if rules else []

    def add_rule(self, rule: PolicyRule) -> None:
        self._rules.append(rule)

    def evaluate(self, tool_name: str, arguments: str, risk_level: RiskLevel) -> PolicyDecision:
        # 1. Hardcoded safety -- always checked first, cannot be overridden
        sensitive = self._check_sensitive_paths(arguments)
        if sensitive:
            return sensitive

        denied = self._check_command_deny(arguments)
        if denied:
            return denied

        # 2. User-defined rules -- first match wins
        for rule in self._rules:
            if self._match_rule(rule, tool_name, arguments, risk_level):
                return PolicyDecision(
                    action=rule.action,
                    matched_rule=rule.name,
                    reason=rule.description or f"Matched rule: {rule.name}",
                )

        # 3. Fallback: use risk level
        if risk_level == RiskLevel.DESTRUCTIVE:
            return PolicyDecision(
                action=PolicyAction.CONFIRM,
                matched_rule="__fallback_destructive",
                reason="Destructive tool requires confirmation",
            )

        return PolicyDecision(
            action=PolicyAction.ALLOW,
            matched_rule="__fallback_safe",
            reason="Default: safe tool allowed",
        )

    def _check_sensitive_paths(self, arguments: str) -> PolicyDecision | None:
        args_lower = arguments.lower()
        for path in SENSITIVE_PATHS:
            if path.lower() in args_lower:
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    matched_rule="__sensitive_path",
                    reason=f"Blocked: references sensitive path ({path})",
                )
        return None

    def _check_command_deny(self, arguments: str) -> PolicyDecision | None:
        for pattern in COMMAND_DENY_PATTERNS:
            match = pattern.search(arguments.lower())
            if match:
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    matched_rule="__command_deny",
                    reason=f"Blocked: matches denied command pattern ({match.group()[:50]})",
                )
        return None

    def _match_rule(self, rule: PolicyRule, tool_name: str, arguments: str, risk_level: RiskLevel) -> bool:
        if not fnmatch(tool_name, rule.tool_pattern):
            return False

        if rule.risk_levels and risk_level not in rule.risk_levels:
            return False

        if rule.argument_patterns:
            for arg_name, arg_pattern in rule.argument_patterns.items():
                if arg_name.lower() not in arguments.lower():
                    return False
                import json

                try:
                    args_dict = json.loads(arguments)
                    val = str(args_dict.get(arg_name, ""))
                    if not fnmatch(val.lower(), arg_pattern.lower()):
                        return False
                except (json.JSONDecodeError, AttributeError):
                    if not fnmatch(arguments.lower(), f"*{arg_pattern.lower()}*"):
                        return False

        return True
