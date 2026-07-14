"""koboi/workflows/capture.py -- capture-from-run pipeline (record -> freeze -> bundle).

Capture turns a successful run into a reusable, byte-deterministic workflow
bundle: the config (redacted) + provenance (source_run_id) + optionally the
run's response cache frozen as a sidecar. Re-running the captured bundle in
cache mode loads the sidecar -> every response is a cache hit -> byte-identical
+ offline (no API key for cached calls). This is the "beyond Claude Code" wedge.

Pure: returns a ``WorkflowDefinition`` + a list of ``(key, payload)`` cache
entries; persistence (file dir / SQLite) is the caller's job (the store owns it).
"""

from __future__ import annotations

import yaml

from koboi import redact
from koboi.llm.cache import ResponseCache
from koboi.workflows.definition import (
    WorkflowDefinition,
    build_from_config_text,
    validate_workflow,
)


def _redact_payload(payload: dict) -> dict:
    """Mask secret-shaped content + tool args in a cache entry's response."""
    resp = payload.get("response")
    if isinstance(resp, dict):
        if isinstance(resp.get("content"), str):
            resp["content"] = redact.redact_value(resp["content"])
        for tc in resp.get("tool_calls") or []:
            if isinstance(tc, dict) and isinstance(tc.get("arguments"), str):
                tc["arguments"] = redact.redact_tool_arguments(tc["arguments"])
    return payload


def capture_from_run(
    *,
    config_text: str,
    name: str,
    source_run_id: str | None = None,
    source_session_id: str | None = None,
    description: str = "",
    with_cache: bool = False,
    cache_dir: str | None = None,
    redact_cache: bool = False,
) -> tuple[WorkflowDefinition, list[tuple[str, dict]] | None]:
    """Capture a run into a ``(WorkflowDefinition, cache_entries)`` pair.

    ``config_text`` is the run's config YAML (a path-loaded config should use
    :func:`build_from_config_path` first; the server passes ``Config.to_yaml()``).
    When ``with_cache`` + ``cache_dir``, the run's response cache is frozen into
    a list of ``(key, payload)`` entries for the caller to write to a sidecar.
    ``redact_cache`` masks secret-shaped content/args (breaks byte-identical
    replay but makes the bundle share-safe).
    """
    wd = build_from_config_text(
        config_text,
        name=name,
        description=description,
        source_run_id=source_run_id,
        source_session_id=source_session_id,
    )
    entries: list[tuple[str, dict]] | None = None
    if with_cache and cache_dir:
        raw = list(ResponseCache(cache_dir).iter_entries())
        entries = [(k, (_redact_payload(p) if redact_cache else p)) for k, p in raw]
    wd.provenance.with_cache = with_cache
    wd.provenance.cache_entries = len(entries) if entries else 0
    wd.provenance.cache_redacted = bool(redact_cache and entries)
    return wd, entries


def prepare_captured_bundle(bundle_yaml: str, cache_dir: str | None = None) -> str:
    """Inject ``replay.mode=cache`` + ``replay.cache_dir`` into a bundle YAML.

    The unifying re-run helper: a captured bundle re-run points its ``cache_dir``
    at the sidecar (or a per-job dir) so every cached response is a hit. Returns
    the bundle unchanged when ``cache_dir`` is None.
    """
    if cache_dir is None:
        return bundle_yaml
    data = yaml.safe_load(bundle_yaml) or {}
    replay = data.setdefault("replay", {})
    replay["mode"] = "cache"
    replay["cache_dir"] = str(cache_dir)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def validate_capture(definition: WorkflowDefinition, entries: list[tuple[str, dict]] | None) -> list[str]:
    """Warnings specific to a captured bundle (on top of :func:`validate_workflow`)."""
    warnings = list(validate_workflow(definition))
    if definition.provenance.with_cache and not entries:
        warnings.append(
            "with_cache was set but no cache entries were captured (the run may not have run in cache mode)."
        )
    if definition.provenance.cache_redacted:
        warnings.append(
            "cache_redacted=True: the sidecar was redacted, so a re-run may diverge "
            "from the original (redaction can mask load-bearing content)."
        )
    return warnings
