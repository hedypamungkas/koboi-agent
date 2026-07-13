"""koboi/redact -- Shared secret redaction for persisted/observed data.

Consolidates the two proven redaction techniques already in-repo so the step
journal (and future callers) never durable-store leaked credentials:

* **value-shape masking** -- regex for known secret *values*
  (sk-..., AKIA..., bearer ..., key=value) -- copied from ``server/jobs.py``.
* **key-name masking** -- mask a dict value whose *key* looks sensitive
  (password/token/secret/api_key/...) -- copied from ``diagnostics.py`` +
  the fnmatch globs in ``harness/env.py``.

Existing callers (``server/jobs.py``/``diagnostics.py``) are untouched to avoid
churn; this module copies the patterns. A future cleanup can re-point them here.
"""

from __future__ import annotations

import fnmatch
import json
import re

REDACTED = "***REDACTED***"

# Max nesting depth for _redact_nested (guards against RecursionError on deeply
# nested untrusted JSON in the durability-critical journal write path).
_REDACT_MAX_DEPTH = 32

# Value-shape patterns (copied from koboi/server/jobs.py:47-52).
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),  # OpenAI-style keys
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key IDs
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),  # bearer tokens
    re.compile(r"(?i)(api[_-]?key|token|password|passwd|secret)[=:]\s*\S+"),
)

# Exact sensitive key names (lowercased) -- extends diagnostics.py:128.
SENSITIVE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "secret_key",
        "secret",
        "auth_token",
        "authtoken",
        "password",
        "passwd",
        "token",
        "access_key",
        "accesskey",
        "credential",
        "credentials",
        "private_key",
        "privatekey",
        "session_token",
        "refresh_token",
        # Payment / PII (not in the original diagnostics/env sources, but
        # clearly secret -- tool args commonly carry these).
        "credit_card",
        "creditcard",
        "card_number",
        "cardnumber",
        "cvv",
        "cvc",
        "pan",
        "ssn",
    }
)

# fnmatch globs (uppercased) -- copied from harness/env.py:58-74 SECRET_BLOCKLIST.
_SECRET_KEY_GLOBS: tuple[str, ...] = (
    "*_KEY",
    "*_SECRET",
    "*_TOKEN",
    "*PASSWORD*",
    "*PASSPHRASE*",
    "*_DB_URL",
    "*_CREDENTIALS",
    "*_CREDENTIAL",
    "PRIVATE_KEY*",
)


def _is_sensitive_key(key: object) -> bool:
    """True if a dict key name looks like it holds a secret."""
    if not isinstance(key, str) or not key:
        return False
    if key.lower() in SENSITIVE_KEY_NAMES:
        return True
    upper = key.upper()
    return any(fnmatch.fnmatchcase(upper, glob) for glob in _SECRET_KEY_GLOBS)


def redact_value(text: str) -> str:
    """Mask known secret-value shapes in an arbitrary string (no truncation)."""
    if not isinstance(text, str) or not text:
        return text
    redacted = text
    for pat in _SECRET_VALUE_PATTERNS:
        redacted = pat.sub(REDACTED, redacted)
    return redacted


def _redact_nested(obj: object, _depth: int = 0) -> object:
    """Recursively mask dict values by sensitive key name + leaf value shapes.

    Depth-capped (``_REDACT_MAX_DEPTH``) so a pathologically/hallucinatorily
    nested JSON argument from an untrusted LLM cannot cause a RecursionError in
    the durability-critical journal write path; past the cap, leaves are masked
    via value-shape redaction rather than recursed into.
    """
    if _depth > _REDACT_MAX_DEPTH:
        return redact_value(obj) if isinstance(obj, str) else obj
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            out[k] = REDACTED if _is_sensitive_key(k) else _redact_nested(v, _depth + 1)
        return out
    if isinstance(obj, list):
        return [_redact_nested(v, _depth + 1) for v in obj]
    if isinstance(obj, str):
        return redact_value(obj)
    return obj


# A value that is entirely a ``${VAR}`` / ``${VAR:default}`` env template. Such
# values are KEPT on export (under sensitive keys) so an exported workflow bundle
# stays re-runnable via environment credentials instead of carrying a real secret.
_ENV_TEMPLATE_RE = re.compile(r"^\$\{[^}]+\}$")


def _redact_export_value(value: object, _depth: int = 0) -> object:
    """Export-time handling for a SENSITIVE-key value.

    Keep a whole ``${VAR:default}`` template (re-runnable); otherwise recurse via
    :func:`redact_config_for_export` so a concrete secret (``sk-...``) is masked.
    """
    if isinstance(value, str) and _ENV_TEMPLATE_RE.match(value):
        return value
    return redact_config_for_export(value, _depth + 1)


def redact_config_for_export(obj: object, _depth: int = 0) -> object:
    """Export-time redaction that PRESERVES ``${VAR:default}`` env placeholders.

    Unlike :func:`_redact_nested` (which masks any value under a sensitive key),
    this keeps a value that is entirely a ``${VAR}`` / ``${VAR:default}`` template
    on sensitive keys, so an exported workflow bundle stays re-runnable via env.
    A concrete secret (``sk-...``, ``bearer ...``) under a sensitive key, and any
    secret-shaped leaf under a non-sensitive key, are still masked via
    :func:`redact_value`. Depth-capped like :func:`_redact_nested`.
    """
    if _depth > _REDACT_MAX_DEPTH:
        return redact_value(obj) if isinstance(obj, str) else obj
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if _is_sensitive_key(k):
                out[k] = _redact_export_value(v, _depth)
            else:
                out[k] = redact_config_for_export(v, _depth + 1)
        return out
    if isinstance(obj, list):
        return [redact_config_for_export(v, _depth + 1) for v in obj]
    if isinstance(obj, str):
        return redact_value(obj)
    return obj


def redact_tool_arguments(arguments_json: str) -> str:
    """Redact secrets from a tool-call ``arguments`` JSON string.

    Walks the parsed JSON masking (a) values whose key name is sensitive and
    (b) secret *value shapes* in any leaf string. If the input is not valid
    JSON, falls back to value-shape redaction on the raw string. Always returns
    a string (round-trippable for the journal's ``tool_calls_json`` column).
    """
    if not arguments_json:
        return arguments_json
    try:
        parsed = json.loads(arguments_json)
    except (json.JSONDecodeError, ValueError):
        return redact_value(arguments_json)
    redacted = _redact_nested(parsed)
    return json.dumps(redacted, ensure_ascii=False)
