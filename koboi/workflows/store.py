"""koboi/workflows/store -- filesystem-backed workflow store for the CLI.

One YAML bundle per file in a scope-resolved directory (project
``cwd/.koboi/workflows``, user ``~/.koboi/workflows``, or
``KOBOI_WORKFLOWS_DIR`` override). Atomic writes via ``.tmp`` + ``os.replace``
(mirrors the ``~/.koboi/keys.json`` managed-file pattern).
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _slugify(name: str) -> str:
    """Lowercase, ``[a-z0-9_-]`` only; falls back to ``workflow`` when empty."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", (name or "").lower()).strip("-")
    return slug or "workflow"


def resolve_workflows_dir(scope: str = "project", *, user: bool = False) -> Path:
    """Resolve the workflow directory for a scope.

    ``KOBOI_WORKFLOWS_DIR`` wins; else ``~/.koboi/workflows`` for the user scope
    (or when ``user=True``); else ``cwd/.koboi/workflows`` for the project scope.
    """
    env = os.environ.get("KOBOI_WORKFLOWS_DIR")
    if env:
        return Path(env).expanduser()
    if user or scope == "user":
        return Path.home() / ".koboi" / "workflows"
    return Path.cwd() / ".koboi" / "workflows"


class FileWorkflowStore:
    """A directory of ``<slug>.yaml`` workflow bundles."""

    def __init__(self, scope: str = "project", *, user: bool = False) -> None:
        self.scope = "user" if user else scope
        self._dir = resolve_workflows_dir(self.scope, user=user)

    @property
    def dir(self) -> Path:
        return self._dir

    def path_for(self, name: str) -> Path:
        return self._dir / f"{_slugify(name)}.yaml"

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def cache_dir_for(self, name: str) -> Path:
        """The sibling ``<slug>.cache/`` sidecar dir for a captured workflow."""
        return self._dir / f"{_slugify(name)}.cache"

    def save(self, name: str, bundle_yaml: str, *, sidecar_entries=None) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(name)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(bundle_yaml, encoding="utf-8")
        os.replace(str(tmp), str(path))
        if sidecar_entries is not None:
            from koboi.workflows.cache_sidecar import DirectoryCacheSidecar

            DirectoryCacheSidecar(self.cache_dir_for(name)).write(sidecar_entries)
        return path

    def load(self, name: str) -> str:
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"Workflow {name!r} not found at {path}")
        return path.read_text(encoding="utf-8")

    def load_with_cache(self, name: str) -> tuple[str, Path | None]:
        """Return ``(bundle_yaml, cache_dir)``; ``cache_dir`` is None when no sidecar."""
        bundle = self.load(name)
        cache_dir = self.cache_dir_for(name)
        return bundle, (cache_dir if cache_dir.exists() else None)

    def list(self) -> list[dict]:
        import yaml

        out: list[dict] = []
        if not self._dir.exists():
            return out
        for p in sorted(self._dir.glob("*.yaml")):
            # Surface unreadable bundles with a marker rather than silently dropping
            # them (avoids bare pass/continue; matches diagnostics.py's per-section
            # error-surfacing pattern).
            entry: dict = {"name": p.stem, "path": str(p), "description": ""}
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                envelope = data.get("workflow") or {}
                entry["name"] = envelope.get("name", p.stem)
                entry["description"] = envelope.get("description", "")
                entry["schema_version"] = envelope.get("schema_version")
            except Exception as exc:
                entry["description"] = f"(unreadable: {type(exc).__name__})"
            out.append(entry)
        return out

    def delete(self, name: str) -> bool:
        path = self.path_for(name)
        removed = path.exists()
        if removed:
            path.unlink()
        cache_dir = self.cache_dir_for(name)
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)
            removed = True
        return removed
