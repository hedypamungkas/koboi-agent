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

    def save(self, name: str, bundle_yaml: str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(name)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(bundle_yaml, encoding="utf-8")
        os.replace(str(tmp), str(path))
        return path

    def load(self, name: str) -> str:
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"Workflow {name!r} not found at {path}")
        return path.read_text(encoding="utf-8")

    def list(self) -> list[dict]:
        import yaml

        out: list[dict] = []
        if not self._dir.exists():
            return out
        for p in sorted(self._dir.glob("*.yaml")):
            entry: dict = {"name": p.stem, "path": str(p), "description": ""}
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                envelope = data.get("workflow") or {}
                entry["name"] = envelope.get("name", p.stem)
                entry["description"] = envelope.get("description", "")
                entry["schema_version"] = envelope.get("schema_version")
            except Exception:
                pass
            out.append(entry)
        return out

    def delete(self, name: str) -> bool:
        path = self.path_for(name)
        if path.exists():
            path.unlink()
            return True
        return False
