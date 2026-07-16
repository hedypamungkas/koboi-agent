"""koboi/harness/policy.py -- Policy-based permission engine for tool execution.

Replaces simple risk-level-based approval with an engine supporting:
- Glob pattern matching on tool name and arguments
- Hardcoded sensitive path protection (non-overridable)
- Command deny patterns (regex-based)
- Composable rules with first-match-wins priority
"""

from __future__ import annotations

import json
import os
import re
import shlex
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

# Interpreters that take an inline-code flag. ``-c`` for the first set, ``-e``
# for the second (perl/ruby accept both). Version-suffixed python binaries
# (python3.11) are normalized to ``python`` by ``_interp_kind``.
C_INTERPRETERS = {"python", "perl", "ruby", "bash", "sh", "dash", "zsh"}
E_INTERPRETERS = {"perl", "ruby", "node"}
_PYTHON_RE = re.compile(r"^python[0-9.]*$")
# Shell separators that terminate a single command word (for token scanning).
_SEPARATORS = {"|", ";", "&&", "||"}
# stdin / heredoc redirection tokens fed to an interpreter -- the primary
# prompt-injection exfil vector (``python3 <<< 'code'``).
_STDIN_REDIRECTS = {"<<<", "<<", "<<-", "<"}
# Targets that, when force-removed recursively, wipe root or the workdir root.
_ROOTISH = {"/", "/*", "/.", ".", ".*"}


def _split_tokens(command: str) -> list[str]:
    """Tokenize with shlex (which also concatenates ``pass''wd`` -> ``passwd``).

    Falls back to a naive whitespace split on unbalanced quotes -- the same
    pattern as ``koboi/sandbox/restricted.py:_first_network_binary``.
    """
    try:
        return shlex.split(command)
    except ValueError:
        return command.replace(";", " ").replace("|", " ").split()


def _interp_kind(base: str) -> str | None:
    """Return ``"c"`` if the interpreter takes ``-c``, ``"e"`` for ``-e``.

    ``None`` for non-interpreters. Normalizes ``python3.11`` -> ``python``.
    """
    name = base.lower()
    if _PYTHON_RE.match(name):
        name = "python"
    if name in C_INTERPRETERS:
        return "c"
    if name in E_INTERPRETERS:
        return "e"
    return None


def _sensitive_path_reason(command: str) -> str | None:
    """Reason string if ``command`` references a sensitive path, else None.

    Substring match (broad, preserves prior behavior) PLUS a prefix-anchored
    glob match over shlex tokens so ``/etc/pass*`` and shlex-joined forms like
    ``/etc/pass''wd`` are caught. The glob is anchored to a non-trivial literal
    prefix (>= 4 chars) so a bare ``*`` (e.g. ``echo *``) cannot match every
    sensitive path -- only globs that name a sensitive directory (``/etc/*``,
    ``/etc/pass*``).
    """
    cmd_lower = command.lower()
    for path in SENSITIVE_PATHS:
        if path.lower() in cmd_lower:
            return f"Blocked: command references sensitive path ({path})"
    for tok in _split_tokens(command):
        tok_l = tok.lower()
        # (a) exact/substring on the token (catches shlex-joined forms).
        for path in SENSITIVE_PATHS:
            if path.lower() in tok_l:
                return f"Blocked: command references sensitive path ({path})"
        # (b) prefix-anchored glob: /etc/pass* -> prefix "/etc/pass" is a prefix
        # of "/etc/passwd". A bare "*" has an empty prefix and is skipped.
        if any(ch in tok_l for ch in "*?["):
            prefix = re.split(r"[*?\[]", tok_l, maxsplit=1)[0]
            if len(prefix) >= 4:
                for path in SENSITIVE_PATHS:
                    if path.lower().startswith(prefix):
                        return f"Blocked: command references sensitive path ({path})"
    return None


def _interpreter_deny_reason(tokens: list[str]) -> str | None:
    """Block any interpreter with an inline-code flag (-c/-e) or stdin redirect.

    Scans the whole token stream (not just argv[0]) so ``echo x | python3 -c``
    is still caught -- preserving the coverage of the original anchored regexes
    while also catching variant spellings (``python3 -W ignore -c``,
    ``bash -ic``).
    """
    for i, tok in enumerate(tokens):
        kind = _interp_kind(os.path.basename(tok))
        if kind is None:
            continue
        for nxt in tokens[i + 1 :]:
            if nxt in _SEPARATORS:
                break
            if nxt in _STDIN_REDIRECTS:
                return f"Blocked: interpreter stdin redirection ({tok})"
            if nxt.startswith("--"):
                continue  # long flags are never inline-code flags
            if nxt.startswith("-") and len(nxt) > 1:
                flag_chars = set(nxt[1:])
                if kind == "c" and "c" in flag_chars:
                    return f"Blocked: inline interpreter code execution ({tok})"
                if kind == "e" and "e" in flag_chars:
                    return f"Blocked: inline interpreter code execution ({tok})"
    return None


def _rm_deny_reason(tokens: list[str]) -> str | None:
    """Block ``rm`` that is BOTH recursive AND force on a root-ish target.

    Normalizes combined short flags (``-fr``/``-Rf``) and long forms
    (``--recursive``/``--force``) so ``rm -fr /`` and ``rm --recursive --force /``
    are caught in addition to the existing ``rm -rf /``/``rm -rf .`` regexes.
    """
    for i, tok in enumerate(tokens):
        if os.path.basename(tok).lower() != "rm":
            continue
        recursive = False
        force = False
        for nxt in tokens[i + 1 :]:
            if nxt in _SEPARATORS:
                break
            if nxt.startswith("--"):
                low = nxt.lower()
                if low == "--recursive":
                    recursive = True
                elif low == "--force":
                    force = True
                continue
            if nxt.startswith("-") and len(nxt) > 1:
                flag_chars = set(nxt[1:])
                if "r" in flag_chars or "R" in flag_chars:
                    recursive = True
                if "f" in flag_chars:
                    force = True
                continue
            # positional target
            if recursive and force and nxt in _ROOTISH:
                return f"Blocked: recursive forced rm of root ({nxt})"
    return None


def _command_deny_reason(command: str) -> str | None:
    """Reason string if ``command`` matches a deny rule, else None.

    Combines the existing payload-shape regexes (kept as defense-in-depth) with
    token-based interpreter and rm classification so trivial flag-ordering /
    combined-flag / long-form variants can no longer bypass the gate.
    """
    cmd_lower = command.lower()
    for pattern in COMMAND_DENY_PATTERNS:
        match = pattern.search(cmd_lower)
        if match:
            return f"Blocked: command matches deny pattern ({match.group()[:50]})"
    tokens = _split_tokens(command)
    reason = _interpreter_deny_reason(tokens)
    if reason:
        return reason
    return _rm_deny_reason(tokens)


def check_command_blocked(command: str) -> str | None:
    """Shared command gate. Returns a reason string if blocked, else None.

    Single implementation reused by ``PolicyEngine`` (hardcoded safety) and the
    shell tool / skill ``!`cmd`` `` path. Checks sensitive paths first, then
    command-deny (payload regexes + token-based interpreter/rm classification).
    """
    reason = _sensitive_path_reason(command)
    if reason:
        return reason
    return _command_deny_reason(command)


def _extract_command(arguments: str) -> str:
    """Best-effort extract the ``command``/``cmd`` value from a JSON args blob.

    Returns the raw ``arguments`` unchanged when it is not a JSON object so the
    token-aware checks can still run on plain command strings.
    """
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments
    if isinstance(parsed, dict):
        for key in ("command", "cmd"):
            val = parsed.get(key)
            if isinstance(val, str):
                return val
    return arguments


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
        # Run on the raw args blob (broad substring, catches any arg name) and
        # on the extracted command value (token-aware reverse-glob match).
        cmd = _extract_command(arguments)
        for text in (arguments, cmd):
            reason = _sensitive_path_reason(text)
            if reason:
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    matched_rule="__sensitive_path",
                    reason=reason,
                )
        return None

    def _check_command_deny(self, arguments: str) -> PolicyDecision | None:
        # The regex scan works on the raw blob (catches commands embedded under
        # any arg name); the token-based interpreter/rm classification runs on
        # the extracted command value. Shared helpers = ONE implementation.
        cmd = _extract_command(arguments)
        for text in (arguments, cmd):
            reason = _command_deny_reason(text)
            if reason:
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    matched_rule="__command_deny",
                    reason=reason,
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
