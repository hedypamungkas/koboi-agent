"""koboi/tools/builtin/memory -- Persistent key-value memory store."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time

from koboi.tools.registry import tool

_logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:
    fcntl = None

MEMORY_FILE = ".agent_memory.json"
MAX_KEY_LEN = 256
MAX_VALUE_LEN = 50000
LOCK_RETRIES = 3
LOCK_RETRY_INTERVAL = 0.1


class _MemoryStore:
    _instance = None

    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data = {str(k): str(v) for k, v in data.items()}
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            self._data = {}

    def _save(self) -> bool:
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, dir=os.path.dirname(self.filepath) or "."
            )
            json.dump(self._data, tmp, ensure_ascii=False, indent=2)
            tmp.close()
            os.replace(tmp.name, self.filepath)
            return True
        except (OSError, PermissionError, IOError) as e:
            _logger.error("Memory save failed: %s", e)
            return False

    def _acquire_lock(self):
        if fcntl is None:
            return None
        lock_path = self.filepath + ".lock"
        for attempt in range(LOCK_RETRIES):
            try:
                fd = open(lock_path, "w")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except (IOError, OSError):
                try:
                    fd.close()
                except Exception:
                    pass
                _logger.debug("Lock attempt %d/%d failed for %s", attempt + 1, LOCK_RETRIES, lock_path)
                if attempt < LOCK_RETRIES - 1:
                    time.sleep(LOCK_RETRY_INTERVAL)
        _logger.warning("Could not acquire lock for %s after %d retries", lock_path, LOCK_RETRIES)
        return None

    def _release_lock(self, fd) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
        except Exception as e:
            _logger.debug("Lock release failed: %s", e)

    def store(self, key: str, value: str) -> str:
        fd = self._acquire_lock()
        if fd is None and fcntl is not None:
            return "Error: could not acquire lock for memory file"

        try:
            self._load()
            self._data[key] = value
            if not self._save():
                return f"Error: failed to persist '{key}' to disk"
        finally:
            if fd is not None:
                self._release_lock(fd)

        return f"Successfully saved '{key}' ({len(value)} characters)"

    def recall(self, key: str = "", query: str = "") -> str:
        if key:
            if key in self._data:
                return f"Value for '{key}': {self._data[key]}"
            return f"Error: key '{key}' not found"

        if query:
            q = query.lower()
            matches = {k: v for k, v in self._data.items() if q in k.lower() or q in v.lower()}
            if not matches:
                return f"No entry found matching '{query}'"
            lines = [f"  {k}: {v[:100]}" for k, v in matches.items()]
            return "\n".join(lines)

        if not self._data:
            return "Memory is empty, no data saved yet"
        lines = [f"  {k}: {v[:100]}" for k, v in list(self._data.items())[:200]]
        return "\n".join(lines)


_store = _MemoryStore()


@tool(
    name="memory_store",
    group="memory",
    description="Store key-value pair to persistent memory. Data persists across sessions.",
    parameters={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key to store the value, e.g. 'user_preference_theme'",
            },
            "value": {
                "type": "string",
                "description": "Value to store, e.g. 'dark'",
            },
        },
        "required": ["key", "value"],
    },
    deps=["memory_store_ref"],
)
def memory_store(key: str, value: str, _deps: dict | None = None) -> str:
    if not key or not key.strip():
        return "Error: key cannot be empty"
    if "/" in key or "\\" in key:
        return "Error: key cannot contain path separator"
    if len(key) > MAX_KEY_LEN:
        return f"Error: key too long (max {MAX_KEY_LEN} characters)"
    if len(value) > MAX_VALUE_LEN:
        value = value[:MAX_VALUE_LEN]
    store = _deps.get("memory_store_ref") if _deps else _store
    return store.store(key.strip(), value)


@tool(
    name="memory_recall",
    group="memory",
    description="Retrieve value from memory by key or search query.",
    parameters={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Specific key to recall.",
            },
            "query": {
                "type": "string",
                "description": "Search query to find in keys and values.",
            },
        },
        "required": [],
    },
    deps=["memory_store_ref"],
)
def memory_recall(key: str = "", query: str = "", _deps: dict | None = None) -> str:
    store = _deps.get("memory_store_ref") if _deps else _store
    return store.recall(key=key, query=query)
