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
    verified: bool = False  # P3: True once the peer's agent-card org-claim is verified


@dataclass
class PeerInvokeResult:
    """Result of an A2A invoke: the peer's answer + its Langfuse trace-id (if configured).

    ``trace_id`` is the receiver's LANGFUSE trace-id (for direct lookup in its Langfuse
    project; empty if Langfuse isn't configured) -- NOT the W3C correlation key. The
    shared W3C trace-id (which both instances stamp in their step journals) is identical
    on both sides; a caller already holds it via ``tracing_context.current_trace_id()``.
    """

    content: str
    trace_id: str = ""


class PeerRegistry:
    """Outbound ``name -> PeerConfig`` + inbound ``token_hash -> peer_id``.

    Outbound tokens are stored PLAINTEXT (they are sent as ``Authorization: Bearer
    <token>`` to the remote instance). Inbound tokens are stored SHA-256 hashed and
    compared constant-time via :func:`hmac.compare_digest` (mirrors KeyStore.validate).
    """

    def __init__(self) -> None:
        self._peers: dict[str, PeerConfig] = {}
        self._inbound: dict[str, str] = {}  # token_hash -> peer_id
        # P3: when an org_secret is set, declared peers must be org-verified at startup.
        self._org_secret: str = ""
        self._require_verification: bool = False

    def load_from_config(self, peers_cfg: Any) -> int:
        """Load outbound peers + inbound tokens from a ``peers:`` config block.

        SSRF-checks each peer URL at load time; a bad/private URL is skipped with a
        warning (boot must never crash on one misconfigured peer). Returns the number
        of outbound peers loaded.
        """
        cfg = self._as_dict(peers_cfg)
        allow_private = bool(cfg.get("allow_private_network", False))
        # P3: org_secret enables verified-only -- declared peers start unverified and
        # must pass agent-card org-claim verification (verify_all) before they're callable.
        self._org_secret = str(cfg.get("org_secret", "") or "")
        self._require_verification = bool(self._org_secret)

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
                    verified=not self._require_verification,
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
        """Resolve a CALLABLE outbound peer by name.

        Returns None if unknown or -- when org-claim verification is enabled
        (``org_secret`` set) -- if the peer's agent-card has not yet been verified.
        """
        peer = self._peers.get(name)
        if peer is None:
            return None
        if self._require_verification and not peer.verified:
            return None
        return peer

    async def verify_all(self) -> int:
        """P3: fetch + HMAC-verify each declared peer's agent-card (call at startup).

        Only does work when an ``org_secret`` is configured. Concurrent
        (``asyncio.gather``), per-peer timeout, non-fatal -- a peer that fails
        verification stays ``verified=False`` (uncallable via :meth:`get`) but never
        crashes boot. Returns the number of peers verified.
        """
        if not self._require_verification or not self._peers:
            return 0
        import asyncio

        results = await asyncio.gather(
            *(self._verify_one(p) for p in list(self._peers.values())),
            return_exceptions=True,
        )
        verified = sum(1 for r in results if r is True)
        if verified != len(self._peers):
            _logger.warning(
                "A2A org-claim verification: %d/%d peers verified; unverified peers are uncallable",
                verified,
                len(self._peers),
            )
        return verified

    async def _verify_one(self, peer: PeerConfig) -> bool:
        import httpx

        from koboi.server.agent_card import CARD_PATH, verify_card

        url = peer.url.rstrip("/") + CARD_PATH
        try:
            async with httpx.AsyncClient(timeout=peer.timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                card = resp.json()
        except Exception as exc:  # noqa: BLE001 -- unreachable/bad peer -> not verified
            _logger.warning("A2A peer '%s' card fetch failed: %s", peer.name, exc)
            return False
        if not isinstance(card, dict) or not verify_card(card, self._org_secret):
            _logger.warning("A2A peer '%s' failed org-claim verification (rejected)", peer.name)
            return False
        peer.verified = True
        # Fill agent_name from the card if the operator left it blank.
        if not peer.agent_name:
            peer.agent_name = str(card.get("agent_name", "")) or peer.agent_name
        return True

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

    @property
    def requires_verification(self) -> bool:
        """True when an org_secret is set (declared peers must be org-verified)."""
        return self._require_verification

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


async def invoke_peer(peer: PeerConfig, message: str) -> PeerInvokeResult:
    """POST a peer instance's ``/v1/peer/invoke`` receiver and return its answer + trace-id.

    The single A2A HTTP path, shared by the ``call_peer_agent`` tool and
    :class:`koboi.orchestration.remote_proxy.RemoteAgentProxy`. Propagates the current
    W3C trace as a child ``traceparent`` header (P4). Raises ``httpx.HTTPStatusError``
    on a non-2xx response and ``ValueError`` on a malformed body; callers surface failures.
    """
    import httpx  # lazy: peers.py stays importable without httpx at module load

    from koboi import tracing_context

    url = peer.url.rstrip("/") + "/v1/peer/invoke"
    headers = {"Authorization": f"Bearer {peer.token}"}
    # P4: carry the W3C trace as a child traceparent (same trace-id, fresh parent-id).
    tc = tracing_context.current()
    if tc is not None:
        headers["traceparent"] = tracing_context.child(tc).as_traceparent()
    body: dict = {"message": message}
    if peer.agent_name:
        body["agent_name"] = peer.agent_name  # routing hint (informational)
    async with httpx.AsyncClient(timeout=peer.timeout) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    content = data.get("content")
    if not isinstance(content, str):
        raise ValueError(f"peer returned no string content: {data!r}")
    return PeerInvokeResult(content=content, trace_id=str(data.get("trace_id") or ""))
