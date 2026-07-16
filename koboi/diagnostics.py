"""koboi/diagnostics.py -- Session diagnostics bundle generator.

Collects session data, config, telemetry, logs, and conversation history
into a ZIP file for debugging and support purposes.
"""

from __future__ import annotations

import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from koboi.redact import redact_config_for_export

if TYPE_CHECKING:
    from koboi.facade import KoboiAgent


def collect_diagnostics(agent: KoboiAgent) -> bytes:
    """Generate a ZIP bundle of session diagnostics.

    Returns the ZIP file contents as bytes.
    """
    buf = BytesIO()
    session_id = "unknown"

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # -- Session metadata --
        meta: dict[str, object] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "koboi_version": _get_version(),
        }
        try:
            config = agent.config
            meta["agent_name"] = config.agent_name
            meta["model"] = f"{config.provider}/{config.model}"
            meta["max_iterations"] = config.max_iterations
            meta["rag_enabled"] = config.rag_enabled
            meta["mode"] = config.mode
        except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
            pass

        # -- Session ID --
        try:
            mem = agent.core.memory
            if hasattr(mem, "_session_id"):
                session_id = mem._session_id
                meta["session_id"] = session_id
        except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
            pass

        zf.writestr("metadata.json", json.dumps(meta, indent=2, default=str))

        # -- Config dump (sanitized) --
        try:
            config_data = _sanitize_config(agent.config._data)
            zf.writestr("config.json", json.dumps(config_data, indent=2, default=str))
        except Exception as e:
            zf.writestr("config_error.txt", str(e))

        # -- Conversation messages --
        try:
            messages = agent.core.memory.get_messages()
            zf.writestr("messages.json", json.dumps(messages, indent=2, default=str))
        except Exception as e:
            zf.writestr("messages_error.txt", str(e))

        # -- Telemetry --
        try:
            hook_chain = agent.core.hooks
            telemetry_hook = hook_chain.find_hook(lambda h: type(h).__name__ == "TelemetryHook")
            if telemetry_hook and hasattr(telemetry_hook, "_telemetry"):
                tc = telemetry_hook._telemetry
                zf.writestr("telemetry.json", json.dumps(tc.report(), indent=2, default=str))
        except Exception as e:
            zf.writestr("telemetry_error.txt", str(e))

        # -- Harness state --
        try:
            hook_chain = agent.core.hooks
            carryover_hook = hook_chain.find_hook(lambda h: type(h).__name__ == "CarryoverHook")
            if carryover_hook and hasattr(carryover_hook, "_state"):
                zf.writestr("carryover.txt", carryover_hook._state.summary())
        except Exception as e:
            zf.writestr("carryover_error.txt", str(e))

        # -- Log file --
        try:
            log_path = Path(".logs") / f"{session_id}.log"
            if log_path.exists():
                zf.writestr("session.log", log_path.read_text(errors="replace"))
        except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
            pass

        # -- Registered tools --
        try:
            tools_dict = agent.core.tools._tools
            tools_info = {
                name: {
                    "risk_level": str(td.risk_level.value) if hasattr(td, "risk_level") else "safe",
                    "description": td.description[:200] if hasattr(td, "description") else "",
                }
                for name, td in tools_dict.items()
            }
            zf.writestr("tools.json", json.dumps(tools_info, indent=2))
        except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
            pass

        # -- Hooks --
        try:
            hooks = agent.core.hooks.list_hooks()
            hooks_info = [{"name": h["name"], "events": h["events"]} for h in hooks]
            zf.writestr("hooks.json", json.dumps(hooks_info, indent=2))
        except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
            pass

    return buf.getvalue()


def _sanitize_config(data: dict) -> dict:
    """Remove sensitive values from config before export.

    Delegates to the shared :func:`koboi.redact.redact_config_for_export`
    redactor, which recurses into lists, matches sensitive key names
    case-insensitively (plus fnmatch globs), masks secret value shapes, and
    preserves ``${VAR:default}`` env templates. This closes the issue #55 leak
    where list-valued auth sections (``mcp.servers[].auth``,
    ``mcp.servers[].headers["Authorization"]``, ``server.api_keys``,
    ``jobs.webhooks[].secret``) survived verbatim because the old local redactor
    only descended into ``dict`` (never ``list``), knew a handful of key names,
    and did no value-shape masking.

    Fail-safe: on any error returns a fully-redacted placeholder so a diagnostics
    export never crashes and never leaks a secret (mirrors the ``_safe_redact``
    wrapper in ``journal.py``).
    """
    try:
        # Coerce non-JSON-native values (PyYAML-parsed datetimes, sets, Paths,
        # custom objects) to strings FIRST so a value under a sensitive key
        # becomes a redactable string rather than passing through
        # redact_config_for_export unchanged (its ``return obj`` fall-through)
        # and leaking via the downstream ``json.dumps(..., default=str)``.
        coerced = json.loads(json.dumps(data, default=str))
        redacted = redact_config_for_export(coerced)
        if isinstance(redacted, dict):
            return redacted
        return {"_redacted": "config was not a dict after redaction"}
    except Exception:  # nosec B110 - fail-safe: never crash export, never leak
        return {"_redacted": "config redacted (fail-safe on error)"}


def _redact_nested(d: dict, keys: set[str]) -> None:
    """Recursively redact sensitive keys."""
    for k, v in d.items():
        if isinstance(v, dict):
            _redact_nested(v, keys)
        elif k.lower() in keys and isinstance(v, str) and v:
            d[k] = "***REDACTED***"


def _get_version() -> str:
    """Get koboi version from package metadata."""
    try:
        from importlib.metadata import version

        return version("koboi-agent")
    except Exception:
        return "unknown"
