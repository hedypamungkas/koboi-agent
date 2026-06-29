"""koboi/server/keys_cli -- ``koboi keys`` CLI for API key management (M3).

Manages a JSON file of hashed API keys (``~/.koboi/keys.json`` by default).
Keys are stored as SHA-256 hashes; the plaintext is shown only once at creation.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path

DEFAULT_KEYS_FILE = "~/.koboi/keys.json"


def _load_keys(file_path: str) -> list[dict]:
    """Load keys from JSON file. Returns [] on missing/corrupt (graceful degradation)."""
    p = Path(file_path).expanduser()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_keys(file_path: str, keys: list[dict]) -> None:
    """Atomically write keys with restrictive permissions (0600)."""
    p = Path(file_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(keys, indent=2))
    os.replace(str(tmp), str(p))  # atomic on POSIX
    os.chmod(str(p), 0o600)


def _generate_key() -> tuple[str, str]:
    """Returns (plaintext, sha256_hash)."""
    plaintext = f"koboi_{secrets.token_hex(32)}"
    return plaintext, hashlib.sha256(plaintext.encode()).hexdigest()


def create_key(file_path: str = DEFAULT_KEYS_FILE, label: str = "") -> str:
    """Create a new API key. Returns the plaintext (shown once)."""
    keys = _load_keys(file_path)
    plaintext, h = _generate_key()
    key_id = f"key_{len(keys) + 1:04d}"
    keys.append({"id": key_id, "hash": h, "label": label, "created_at": time.time()})
    _save_keys(file_path, keys)
    return plaintext


def list_keys(file_path: str = DEFAULT_KEYS_FILE) -> list[dict]:
    """List all keys (id, label, created_at, revoked — never the hash)."""
    return [
        {
            "id": k["id"],
            "label": k.get("label", ""),
            "created_at": k.get("created_at", 0),
            "revoked": k.get("revoked", False),
        }
        for k in _load_keys(file_path)
    ]


def revoke_key(key_id: str, file_path: str = DEFAULT_KEYS_FILE) -> bool:
    """Mark a key as revoked. Returns True if found."""
    keys = _load_keys(file_path)
    for k in keys:
        if k["id"] == key_id:
            k["revoked"] = True
            _save_keys(file_path, keys)
            return True
    return False


def rotate_key(key_id: str, file_path: str = DEFAULT_KEYS_FILE, label: str = "") -> str | None:
    """Revoke old key + create new. Returns new plaintext or None if key_id not found."""
    keys = _load_keys(file_path)
    found = any(k["id"] == key_id for k in keys)
    if not found:
        return None
    for k in keys:
        if k["id"] == key_id:
            k["revoked"] = True
            break
    plaintext, h = _generate_key()
    new_id = f"key_{len(keys) + 1:04d}"
    keys.append({"id": new_id, "hash": h, "label": label or f"rotated from {key_id}", "created_at": time.time()})
    _save_keys(file_path, keys)
    return plaintext
