"""koboi/server/auth -- API-key auth middleware + key store (M3).

Reads keys from a JSON file (SHA-256 hashed) + ``${KOBOI_API_KEYS}`` env
(plaintext, back-compat). When no keys are configured, auth is OFF (dev mode)
with a warning. Health endpoints (``/healthz``, ``/readyz``) are always open.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Any

from fastapi.responses import JSONResponse

_logger = logging.getLogger(__name__)

#: Endpoints that bypass auth (health probes don't send Bearer tokens).
OPEN_PATHS = frozenset({"/healthz", "/readyz"})


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class KeyStore:
    """In-memory map of ``token_hash → key_id``. Loaded from file + env at startup."""

    def __init__(self) -> None:
        self._keys: dict[str, str] = {}

    def load_from_file(self, path: str | None) -> int:
        """Load non-revoked hashed keys from a JSON file. Returns count loaded."""
        if not path:
            return 0
        p = Path(path).expanduser()
        if not p.exists():
            return 0
        try:
            data = json.loads(p.read_text())
            count = 0
            for entry in data:
                if entry.get("revoked"):
                    continue
                h = entry.get("hash")
                if h:
                    self._keys[h] = entry.get("id", h[:12])
                    count += 1
            _logger.info("Loaded %d API key(s) from %s", count, p)
            return count
        except Exception as exc:
            _logger.warning("Failed to load keys file %s: %s", p, exc)
            return 0

    def load_from_env(self, env_value: str) -> int:
        """Load plaintext keys from env (comma-separated). Hashes them for uniform comparison."""
        if not env_value or not env_value.strip():
            return 0
        count = 0
        for raw in env_value.split(","):
            token = raw.strip()
            if token:
                h = _hash_token(token)
                self._keys[h] = f"env:{h[:12]}"
                count += 1
        if count:
            _logger.info("Loaded %d API key(s) from KOBOI_API_KEYS env", count)
        return count

    def validate(self, token: str) -> str | None:
        """Returns ``key_id`` if the token is valid, ``None`` otherwise.

        Uses ``hmac.compare_digest`` for constant-time comparison (standard
        practice for credential validation — avoids timing side-channels).
        """
        candidate = _hash_token(token)
        for stored_hash, key_id in self._keys.items():
            if hmac.compare_digest(candidate, stored_hash):
                return key_id
        return None

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def has_keys(self) -> bool:
        return len(self._keys) > 0


def make_auth_middleware(key_store: KeyStore, *, auth_required: bool = True, peer_registry: Any | None = None):
    """Build a Starlette HTTP middleware that validates ``Bearer`` tokens.

    ``auth_required`` (default True, mirrors ``server.auth_required``): when no
    API keys are configured, True **fails closed** (401) so a production deploy
    with a missing/unreadable keys file is never left open. Set
    ``auth_required: false`` to opt into dev mode (no auth) -- intended only for
    trusted loopback local development.

    ``peer_registry`` (A2A): when provided, an ``Authorization: Bearer`` token
    matching a configured inbound peer token authenticates the caller as a
    same-org peer (``request.state.peer_id``), distinct from a tenant API key.
    Tried before ``key_store.validate`` so peer tokens never collide with API keys.
    """

    _warned = False

    async def auth_middleware(request, call_next):  # type: ignore[no-untyped-def]
        nonlocal _warned
        if request.url.path in OPEN_PATHS:
            return await call_next(request)
        has_api_keys = key_store.has_keys
        has_peer_auth = peer_registry is not None and peer_registry.has_peers
        # No credentials of any kind configured: fail closed (or dev mode).
        if not has_api_keys and not has_peer_auth:
            if auth_required:
                # Fail closed: no keys + auth_required=true → never serve open.
                return _unauthorized(request, "no API keys configured (auth_required is true)")
            if not _warned:
                _logger.warning(
                    "No API keys configured — auth disabled (dev mode, auth_required=false). "
                    "Configure keys via 'koboi keys create' or KOBOI_API_KEYS env."
                )
                _warned = True
            request.state.api_key_id = "dev"
            return await call_next(request)
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _unauthorized(request, "missing or invalid Authorization header")
        token = auth_header[7:]
        # A2A: a valid inbound peer token authenticates as a same-org peer caller
        # (distinct from a tenant API key). Tried before key_store.validate so peer
        # tokens never collide with API keys; stamps request.state.peer_id.
        if peer_registry is not None:
            peer_id = peer_registry.validate_inbound_token(token)
            if peer_id is not None:
                request.state.peer_id = peer_id
                request.state.api_key_id = peer_id  # caller identity for ownership checks
                return await call_next(request)
        if not has_api_keys:
            # Peer-only instance: the token was not a valid peer token.
            return _unauthorized(request, "invalid peer token")
        key_id = key_store.validate(token)
        if key_id is None:
            return _unauthorized(request, "invalid API key")
        request.state.api_key_id = key_id
        return await call_next(request)

    return auth_middleware


def _unauthorized(request, message: str) -> JSONResponse:
    from koboi.server.schema import ErrorDetail, ErrorResponse

    return JSONResponse(
        status_code=401,
        content=ErrorResponse(
            error=ErrorDetail(
                code="unauthorized",
                message=message,
                request_id=getattr(request.state, "request_id", None),
            )
        ).model_dump(),
        headers={"WWW-Authenticate": "Bearer"},
    )
