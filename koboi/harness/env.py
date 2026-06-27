"""koboi/harness/env.py -- Secret-hygiene env filtering for subprocess tools.

By default, subprocess-spawning tools (run_shell, git_*, skill ``!`cmd````
preprocessing) receive a SANITIZED environment: only a generous allow-list of
non-secret process/locale vars passes through, plus anything matching the
``KOBOI_*`` glob, minus anything that looks secret-shaped. This closes the
worst secret-exfiltration exposure (a model-authored command reading
``OPENAI_API_KEY`` / ``DATABASE_URL`` from the inherited env).

Escape hatches (restore the full env, e.g. for trusted CI):
  - per-agent YAML:   ``tools.defaults.env_passthrough: true``
  - process-wide:     ``KOBOI_ENV_PASSTHROUGH=1``

Filtering order (block-list WINS over allow-list, so a var like
``KOBOI_DB_TOKEN`` matches the ``KOBOI_*`` allow-glob but is stripped by the
``*_TOKEN`` block-glob):
  1. start from ``os.environ.copy()`` (preserves values, not just keys);
  2. if passthrough is set, return the base env unchanged;
  3. keep only allow-listed keys (default set + KOBOI_* glob + user list);
  4. drop any key matching the secret block-list (default + user list).
"""

from __future__ import annotations

import os
from fnmatch import fnmatch

# Exact-match allow-list: process essentials, locale, and koboi control vars.
# Intentionally EXCLUDES anything secret-shaped (handled by SECRET_BLOCKLIST).
DEFAULT_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SHELL",
        "TERM",
        "TERM_PROGRAM",
        "PWD",
        "NODE_PATH",  # preserved for shell.py npm-root logic
        "KOBOI_VERBOSE",
    }
)

# Glob allow-list patterns (matched case-insensitively against env var names).
ENV_GLOB_ALLOWLIST: tuple[str, ...] = ("KOBOI_*",)

# Secret-shaped block-list. Each entry is an fnmatch glob, matched
# case-insensitively. Catches API keys, DB URLs, etc. even if they slip
# through the allow-list via the KOBOI_* glob.
SECRET_BLOCKLIST: tuple[str, ...] = (
    "*_KEY",
    "*_SECRET",
    "*_TOKEN",
    "*PASSWORD*",
    "*PASSPHRASE*",
    "DATABASE_URL",
    "*_DB_URL",
    "*_CREDENTIALS",
    "*_CREDENTIAL",
    "PRIVATE_KEY*",
    "AWS_SECRET_ACCESS_KEY",  # explicit; *_KEY already covers it
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)

# Module-level defaults, populated once at agent startup by
# configure_env_defaults(). Used by call sites that lack _tool_config
# (the skill-shell preprocessing path).
_env_defaults: dict = {}


def configure_env_defaults(defaults: dict | None) -> None:
    """Set the module-level env config (called once from facade._build_tools).

    Call sites without _tool_config (skills/registry._preprocess_shell_commands)
    fall back to these defaults so all three leak sites share one config.
    """
    global _env_defaults
    _env_defaults = dict(defaults or {})


def _matches_any(key_upper: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch(key_upper, pat.upper()) for pat in patterns)


def _is_allowed(key: str, extra_allow: tuple[str, ...]) -> bool:
    if key in DEFAULT_ENV_ALLOWLIST:
        return True
    k = key.upper()
    if _matches_any(k, ENV_GLOB_ALLOWLIST):
        return True
    return _matches_any(k, extra_allow)


def _is_blocked(key: str, extra_block: tuple[str, ...]) -> bool:
    k = key.upper()
    if _matches_any(k, SECRET_BLOCKLIST):
        return True
    return _matches_any(k, extra_block)


def build_safe_env(tool_config: dict | None = None) -> dict[str, str]:
    """Return a sanitized copy of ``os.environ`` for ``subprocess.run(env=...)``.

    Args:
        tool_config: merged ``tools.defaults`` + ``tools.overrides.<name>`` dict.
            Recognized keys:
              - ``env_passthrough`` (bool): escape hatch, restores full env.
              - ``env_allowlist`` (list[str]): extra key/glob patterns to allow.
              - ``env_blocklist`` (list[str]): extra key/glob patterns to strip.
            When ``None``/empty, falls back to the module-level defaults set by
            :func:`configure_env_defaults` (used by the skills path).
    """
    base = os.environ.copy()

    cfg = tool_config or _env_defaults or {}
    passthrough = bool(cfg.get("env_passthrough") or os.environ.get("KOBOI_ENV_PASSTHROUGH") in ("1", "true", "yes"))
    if passthrough:
        return base

    extra_allow = tuple(cfg.get("env_allowlist") or [])
    extra_block = tuple(cfg.get("env_blocklist") or [])

    return {k: v for k, v in base.items() if _is_allowed(k, extra_allow) and not _is_blocked(k, extra_block)}
