"""file_suggester.py -- @ file path autocomplete with fuzzy matching."""
from __future__ import annotations

import os
from pathlib import Path

from textual.suggester import Suggester

# Directories to skip during recursive scan
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info",
}


class FileSuggester(Suggester):
    """Suggests file paths when input contains '@' prefix.

    Uses rapidfuzz for fuzzy matching and recursive directory scanning.
    """

    def __init__(self, base_dir: str = ".", **kwargs) -> None:
        super().__init__(use_cache=False, case_sensitive=True, **kwargs)
        self._base_dir = Path(base_dir).resolve()
        self._file_cache: list[str] | None = None

    def _scan_files(self) -> list[str]:
        """Recursively scan files and directories, skipping common non-source dirs."""
        if self._file_cache is not None:
            return self._file_cache

        entries: list[str] = []
        try:
            for root, dirs, filenames in os.walk(self._base_dir):
                # Prune skipped directories in-place
                dirs[:] = [
                    d for d in dirs
                    if d not in _SKIP_DIRS and not d.endswith(".egg-info")
                ]
                # Include directories (with trailing /)
                for d in dirs:
                    full = Path(root) / d
                    rel = str(full.relative_to(self._base_dir))
                    entries.append(rel + "/")
                # Include files
                for fname in filenames:
                    full = Path(root) / fname
                    rel = str(full.relative_to(self._base_dir))
                    entries.append(rel)
        except (OSError, ValueError):
            pass

        self._file_cache = entries
        return entries

    def invalidate_cache(self) -> None:
        """Force re-scan on next suggestion request."""
        self._file_cache = None

    async def get_suggestion(self, value: str) -> str | None:
        at_idx = value.rfind("@")
        if at_idx == -1:
            return None

        partial = value[at_idx + 1:]
        if not partial:
            return None

        try:
            from rapidfuzz import fuzz, process

            files = self._scan_files()
            if not files:
                return None

            # Use rapidfuzz to find the best fuzzy match
            result = process.extractOne(
                partial,
                files,
                scorer=fuzz.WRatio,
                score_cutoff=50,
            )
            if result is None:
                return None

            match_path = result[0]
            return value[:at_idx] + "@" + match_path

        except ImportError:
            # Fallback to prefix matching if rapidfuzz not installed
            return await self._fallback_get_suggestion(value, at_idx, partial)
        except (OSError, ValueError):
            return None

    async def _fallback_get_suggestion(
        self, value: str, at_idx: int, partial: str
    ) -> str | None:
        """Prefix-matching fallback when rapidfuzz is unavailable."""
        try:
            if "/" in partial:
                parent = self._base_dir / partial.rsplit("/", 1)[0]
                prefix = partial.rsplit("/", 1)[1]
            else:
                parent = self._base_dir
                prefix = partial

            if not parent.is_dir():
                return None

            for entry in sorted(parent.iterdir()):
                name = entry.name
                if name.startswith(prefix) and name != prefix:
                    suffix = "/" if entry.is_dir() else ""
                    rel = str(entry.relative_to(self._base_dir))
                    return value[:at_idx] + "@" + rel + suffix
        except (OSError, ValueError):
            pass
        return None


class CompositeSuggester(Suggester):
    """Delegates to different suggesters based on input prefix."""

    def __init__(
        self,
        slash_suggester: Suggester,
        file_suggester: Suggester,
        **kwargs,
    ) -> None:
        super().__init__(use_cache=False, case_sensitive=True, **kwargs)
        self._slash = slash_suggester
        self._file = file_suggester

    async def get_suggestion(self, value: str) -> str | None:
        stripped = value.lstrip()
        if stripped.startswith("/"):
            return await self._slash.get_suggestion(value)
        if "@" in value:
            return await self._file.get_suggestion(value)
        return None
