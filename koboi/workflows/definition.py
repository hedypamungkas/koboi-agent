"""koboi/workflows/definition -- serializable workflow definition + export/import.

A *workflow bundle* is a self-contained koboi config YAML carrying a ``workflow:``
metadata envelope (schema version, name, description, provenance) on top of the
usual config sections (``agent`` / ``llm`` / ``orchestration`` / ...). It is
re-runnable directly via ``KoboiAgent.from_config_string`` / ``koboi run <file>``.

This module is stdlib + PyYAML + ``koboi.redact`` + ``koboi.__version__`` only (no
facade/server import) so it stays importable on a bare install for the CLI. v1
ships the ``live`` replay mode (sampling pinning); ``cache`` / ``replay`` arrive
later (v2/v3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from koboi import redact

if TYPE_CHECKING:
    from koboi.config import Config

WORKFLOW_SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class WorkflowProvenance:
    """Where a workflow bundle came from."""

    source_run_id: str | None = None
    source_session_id: str | None = None  # v2: the session the run lived in
    captured_at: str | None = None  # ISO-8601 UTC
    koboi_version: str | None = None  # from koboi.__version__
    with_cache: bool = False  # v2: a frozen response-cache sidecar accompanies the bundle
    cache_entries: int = 0  # v2: number of frozen cache entries
    cache_redacted: bool = False  # v2: sidecar was redacted (may diverge on replay)


@dataclass
class DeterminismProfile:
    """Sampling/pinning knobs that narrow LLM output variance.

    ``temperature`` / ``top_p`` / ``seed`` flow into the per-node ``llm_config``
    as forward-as-is generation params (``seed`` is dropped on Anthropic, which
    has no seed parameter). ``model_pin`` maps to ``llm.model`` (a dated snapshot
    for bounded reproducibility). ``replay_mode`` is metadata (not an LLM param);
    v1 only supports ``"live"``.
    """

    temperature: float | None = None
    seed: int | None = None
    top_p: float | None = None
    model_pin: str | None = None
    replay_mode: str = "live"

    def merge(self, override: DeterminismProfile | None) -> DeterminismProfile:
        """Node override wins for non-None fields; workflow default fills the rest."""
        if override is None:
            return self
        return DeterminismProfile(
            temperature=override.temperature if override.temperature is not None else self.temperature,
            seed=override.seed if override.seed is not None else self.seed,
            top_p=override.top_p if override.top_p is not None else self.top_p,
            model_pin=override.model_pin if override.model_pin is not None else self.model_pin,
            replay_mode=override.replay_mode if override.replay_mode != "live" else self.replay_mode,
        )

    def to_llm_overrides(self) -> dict:
        """Map the profile onto ``llm_config`` keys (``replay_mode`` excluded)."""
        out: dict = {}
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.seed is not None:
            out["seed"] = self.seed
        if self.top_p is not None:
            out["top_p"] = self.top_p
        if self.model_pin is not None:
            out["model"] = self.model_pin
        return out

    @classmethod
    def from_dict(cls, det: dict | None) -> DeterminismProfile | None:
        """Build a profile from a raw determinism dict (or None when empty)."""
        if not det:
            return None
        return cls(
            temperature=det.get("temperature"),
            seed=det.get("seed"),
            top_p=det.get("top_p"),
            model_pin=det.get("model_pin"),
            replay_mode=det.get("replay_mode", "live"),
        )


@dataclass
class WorkflowDefinition:
    """A serializable, re-runnable workflow = envelope + a frozen config body."""

    schema_version: str = WORKFLOW_SCHEMA_VERSION
    name: str = ""
    description: str = ""
    provenance: WorkflowProvenance = field(default_factory=WorkflowProvenance)
    config: dict = field(default_factory=dict)

    @property
    def determinism(self) -> DeterminismProfile | None:
        return parse_determinism(self.config)

    def to_bundle_dict(self) -> dict:
        envelope = {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "provenance": {
                "source_run_id": self.provenance.source_run_id,
                "source_session_id": self.provenance.source_session_id,
                "captured_at": self.provenance.captured_at,
                "koboi_version": self.provenance.koboi_version,
                "with_cache": self.provenance.with_cache,
                "cache_entries": self.provenance.cache_entries,
                "cache_redacted": self.provenance.cache_redacted,
            },
        }
        out: dict = {"workflow": envelope}
        out.update(self.config)
        return out

    def to_bundle_yaml(self) -> str:
        return yaml.safe_dump(self.to_bundle_dict(), sort_keys=False, allow_unicode=True)

    def to_bundle_json(self) -> str:
        import json

        return json.dumps(self.to_bundle_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_bundle_dict(cls, data: dict) -> WorkflowDefinition:
        envelope = data.get("workflow") or {}
        prov = envelope.get("provenance") or {}
        body = {k: v for k, v in data.items() if k != "workflow"}
        return cls(
            schema_version=envelope.get("schema_version", WORKFLOW_SCHEMA_VERSION),
            name=envelope.get("name", ""),
            description=envelope.get("description", ""),
            provenance=WorkflowProvenance(
                source_run_id=prov.get("source_run_id"),
                source_session_id=prov.get("source_session_id"),
                captured_at=prov.get("captured_at"),
                koboi_version=prov.get("koboi_version"),
                with_cache=bool(prov.get("with_cache", False)),
                cache_entries=int(prov.get("cache_entries", 0) or 0),
                cache_redacted=bool(prov.get("cache_redacted", False)),
            ),
            config=body,
        )

    @classmethod
    def from_bundle_yaml(cls, text: str) -> WorkflowDefinition:
        data = yaml.safe_load(text) or {}
        return cls.from_bundle_dict(data)


# --------------------------------------------------------------------------- #
# Build / parse helpers
# --------------------------------------------------------------------------- #
def parse_determinism(config_data: dict) -> DeterminismProfile | None:
    """Read ``orchestration.determinism`` from a config dict (or None when unset)."""
    return DeterminismProfile.from_dict((config_data.get("orchestration") or {}).get("determinism"))


def build_from_config_path(
    path: str | Path,
    *,
    name: str,
    description: str = "",
    source_run_id: str | None = None,
) -> WorkflowDefinition:
    """Build a :class:`WorkflowDefinition` from a config YAML path.

    Reads the UN-interpolated merged source (so ``${VAR:default}`` templates
    survive) and redacts secrets via
    :func:`koboi.redact.redact_config_for_export` (templates kept on sensitive
    keys; concrete secrets masked), producing a share-safe, re-runnable bundle.
    """
    from koboi.config import _load_yaml_with_extends
    import koboi
    from datetime import datetime, timezone

    raw = _load_yaml_with_extends(Path(path))
    body = cast("dict", redact.redact_config_for_export(raw))
    provenance = WorkflowProvenance(
        source_run_id=source_run_id,
        captured_at=datetime.now(timezone.utc).isoformat(),
        koboi_version=getattr(koboi, "__version__", None),
    )
    return WorkflowDefinition(
        schema_version=WORKFLOW_SCHEMA_VERSION,
        name=name,
        description=description,
        provenance=provenance,
        config=body,
    )


def build_from_config_text(
    config_text: str,
    *,
    name: str,
    description: str = "",
    source_run_id: str | None = None,
    source_session_id: str | None = None,
) -> WorkflowDefinition:
    """Build a :class:`WorkflowDefinition` from a config YAML STRING.

    Sibling of :func:`build_from_config_path` for the capture path (the server
    has a config string from ``Config.to_yaml()``, not a path). Parses the text
    and redacts secrets via :func:`koboi.redact.redact_config_for_export`.
    """
    import koboi
    from datetime import datetime, timezone

    raw = yaml.safe_load(config_text) or {}
    body = cast("dict", redact.redact_config_for_export(raw))
    provenance = WorkflowProvenance(
        source_run_id=source_run_id,
        source_session_id=source_session_id,
        captured_at=datetime.now(timezone.utc).isoformat(),
        koboi_version=getattr(koboi, "__version__", None),
    )
    return WorkflowDefinition(
        schema_version=WORKFLOW_SCHEMA_VERSION,
        name=name,
        description=description,
        provenance=provenance,
        config=body,
    )


def validate_workflow(definition: WorkflowDefinition) -> list[str]:
    """Return human-readable determinism warnings for a workflow bundle.

    Honest about provider limits: ``seed`` is dropped on Anthropic (no seed
    parameter); ``sliding_window`` context summarizes via an LLM call
    (non-deterministic); an unpinned model can drift across provider updates.
    """
    warnings: list[str] = []
    cfg = definition.config
    det = definition.determinism
    llm = cfg.get("llm") or {}
    provider = str(llm.get("provider", "openai")).lower()

    if det and det.seed is not None and provider == "anthropic":
        warnings.append(
            "determinism.seed is set but the Anthropic Messages API has no seed "
            "parameter; it will be dropped (not forwarded). temperature/top_p still apply."
        )
    if det and det.model_pin is None and llm.get("model"):
        warnings.append(
            f"determinism is configured but no model_pin is set; llm.model={llm.get('model')!r} "
            "may drift if the provider updates/deprecates it. Set determinism.model_pin to a "
            "dated snapshot for bounded reproducibility."
        )
    if str((cfg.get("context") or {}).get("strategy", "")).lower() == "sliding_window":
        warnings.append(
            "context.strategy=sliding_window summarizes old messages via an LLM call, which is "
            "non-deterministic; a fresh re-run can produce a different prompt window. Use "
            "smart_truncation (rule-based) for replay-grade determinism."
        )
    if det and det.temperature is not None and "temperature" in llm and llm["temperature"] != det.temperature:
        warnings.append(
            f"orchestration.determinism.temperature={det.temperature} conflicts with the explicit "
            f"llm.temperature={llm['temperature']}; the workflow-level determinism is ignored on "
            "nodes that set llm.temperature explicitly."
        )
    return warnings


def _orchestration_view(config: object) -> dict:
    if hasattr(config, "orchestration"):
        return config.orchestration or {}
    if hasattr(config, "get"):
        return config.get("orchestration") or {}
    return {}


def build_graph_snapshot(agent_defs: list, config: Config | dict) -> dict:
    """Build a NON-LOSSY but backward-compatible graph JSON snapshot.

    Keeps the legacy top-level ``nodes`` (list[str]) and ``edges``
    (list[{from, to}]) keys and ADDS ``conditionals``, ``execution_mode``,
    ``router``, and ``agents`` (``AgentDef.to_dict()`` per node) so the output is
    a faithful, re-importable view of the workflow (fixes the lossy
    ``koboi graph --format json``).
    """
    nodes = [ad.name for ad in agent_defs]
    edges = [{"from": dep, "to": ad.name} for ad in agent_defs for dep in ad.depends_on]
    conditionals = []
    for ad in agent_defs:
        for c in ad.conditionals or []:
            conditionals.append({"from": ad.name, "to": c.get("to"), "when": c.get("when")})
    orch = _orchestration_view(config)
    exec_mode = (orch.get("execution") or {}).get("mode", "sequential")
    router = orch.get("router") or {}
    return {
        "nodes": nodes,
        "edges": edges,
        "conditionals": conditionals,
        "execution_mode": exec_mode,
        "router": router,
        "agents": [ad.to_dict() for ad in agent_defs],
    }
