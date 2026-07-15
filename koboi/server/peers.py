"""koboi/server/peers -- PeerRegistry for cross-instance agent-to-agent (A2A) calls.

Mirrors :class:`koboi.server.auth.KeyStore`: in-memory, config-driven, no SQLite.
Holds outbound peer definitions (name -> config, with the PLAINTEXT token presented
to the peer) and hashed inbound tokens (accepted from peer callers). Each registered
peer URL is trusted as same-org/owner (static Bearer per peer).

Opt-in via the ``peers:`` config section; inert by default.
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from typing import Any

from koboi.server.auth import _hash_token

_logger = logging.getLogger(__name__)

#: P0 identity stamped on ``request.state.peer_id`` for any valid inbound peer
#: token. All registered peers are same-org trusted, so a single shared identity
#: suffices; P3 (agent-card discovery) may map inbound tokens to named peers.
_PEER_ID = "peer"


@dataclass
class PeerConfig:
    """Outbound peer definition (runtime shape; mirrors config_models.PeerConfig)."""

    name: str
    url: str
    token: str = ""
    agent_name: str = ""
    org: str = ""
    timeout: float = 30.0


class PeerRegistry:
    """Outbound ``name -> PeerConfig`` + inbound ``token_hash -> peer_id``.

    Outbound tokens are stored PLAINTEXT (they are sent as ``Authorization: Bearer
    <token>`` to the remote instance). Inbound tokens are stored SHA-256 hashed and
    compared constant-time via :func:`hmac.compare_digest` (mirrors KeyStore.validate).
    """

    def __init__(self) -> None:
        self._peers: dict[str, PeerConfig] = {}
        self._inbound: dict[str, str] = {}  # token_hash -> peer_id

    def load_from_config(self, peers_cfg: Any) -> int:
        """Load outbound peers + inbound tokens from a ``peers:`` config block.

        SSRF-checks each peer URL at load time; a bad/private URL is skipped with a
        warning (boot must never crash on one misconfigured peer). Returns the number
        of outbound peers loaded.
        """
        cfg = self._as_dict(peers_cfg)
        allow_private = bool(cfg.get("allow_private_network", False))

        # Outbound peers
        loaded = 0
        for raw in cfg.get("peers", []) or []:
            try:
                peer = PeerConfig(
                    name=str(raw["name"]),
                    url=str(raw["url"]),
                    token=str(raw.get("token", "")),
                    agent_name=str(raw.get("agent_name", "")),
                    org=str(raw.get("org", "")),
                    timeout=float(raw.get("timeout", 30.0) or 30.0),
                )
            except (KeyError, TypeError, ValueError) as exc:
                _logger.warning("Skipping malformed peer entry %r: %s", raw, exc)
                continue
            if not self._url_ok(peer.url, peer.name, allow_private):
                continue
            self._peers[peer.name] = peer
            loaded += 1
            _logger.info("Loaded A2A peer '%s' -> %s", peer.name, peer.url)

        # Inbound tokens (hashed; compared constant-time at request time)
        for token in cfg.get("inbound_tokens", []) or []:
            token = str(token).strip()
            if token:
                self._inbound[_hash_token(token)] = _PEER_ID

        return loaded

    def get(self, name: str) -> PeerConfig | None:
        """Resolve an outbound peer by name (returns None if unknown)."""
        return self._peers.get(name)

    def validate_inbound_token(self, token: str) -> str | None:
        """Return the peer_id if ``token`` matches a configured inbound token, else None.

        Constant-time comparison via :func:`hmac.compare_digest` (no timing oracle).
        """
        candidate = _hash_token(token)
        for stored_hash, peer_id in self._inbound.items():
            if hmac.compare_digest(candidate, stored_hash):
                return peer_id
        return None

    @property
    def has_peers(self) -> bool:
        """True if any outbound peer OR inbound token is configured."""
        return bool(self._peers) or bool(self._inbound)

    @staticmethod
    def _as_dict(peers_cfg: Any) -> dict:
        if peers_cfg is None:
            return {}
        # Pydantic model -> dict (defense-in-depth; runtime truth is dotted-path dicts)
        dump = getattr(peers_cfg, "model_dump", None)
        if callable(dump):
            return dump()
        return peers_cfg if isinstance(peers_cfg, dict) else {}

    @staticmethod
    def _url_ok(url: str, peer_name: str, allow_private: bool) -> bool:
        """Validate a peer URL; log + return False on rejection (never raise).

        Always requires an ``http``/``https`` URL with a hostname. When
        ``allow_private`` is False (default), also runs the strict SSRF gate that
        rejects private/loopback/CGNAT IPs. When True, private/localhost URLs are
        permitted (operator vouches for same-org internal peers).
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"invalid peer URL (need http(s)://host): {url!r}")
            if not allow_private:
                # Lazy import avoids coupling server -> tools.builtin at module load.
                from koboi.tools.builtin.web import _check_url_ssrf

                _check_url_ssrf(url)
            return True
        except Exception as exc:  # noqa: BLE001 -- any validation failure = skip peer
            _logger.warning(
                "Skipping A2A peer '%s': URL %s failed validity/SSRF check: %s",
                peer_name,
                url,
                exc,
            )
            return False
