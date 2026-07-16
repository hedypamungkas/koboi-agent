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


@dataclass(frozen=True)
class PeerConfig:
    """Immutable outbound peer definition (runtime shape; mirrors config_models.PeerDef)."""

    name: str
    url: str
    token: str = ""
    agent_name: str = ""
    org: str = ""
    timeout: float = 30.0


@dataclass(frozen=True)
class PeerInvokeResult:
    """Result of an A2A invoke: the peer's answer + the receiver's Langfuse trace-id.

    ``receiver_trace_id`` is the receiver's LANGFUSE trace-id (for direct lookup in its
    Langfuse project; empty if Langfuse isn't configured) -- NOT the W3C correlation key.
    The shared W3C trace-id (which both instances stamp in their step journals) is
    identical on both sides; a caller already holds it via
    ``tracing_context.current_trace_id()``.
    """

    content: str
    receiver_trace_id: str = ""


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
        self._freshness_seconds: float | None = None  # W2: configurable card freshness
        # Names of peers whose agent-card org-claim has been verified. Registry-owned
        # state (NOT a mutable flag on PeerConfig) -- verification is a time-varying
        # property of the registry's relationship to the peer.
        self._verified: set[str] = set()

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
        self._freshness_seconds = float(cfg.get("card_freshness_seconds", 0) or 0) or None

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
                    timeout=max(1.0, float(raw.get("timeout", 30.0) or 30.0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                _logger.warning("Skipping malformed peer entry %r: %s", raw, exc)
                continue
            if not self._url_ok(peer.url, peer.name, allow_private):
                continue
            if peer.name in self._peers:
                _logger.warning("Duplicate A2A peer name '%s' — overwriting previous entry", peer.name)
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
        (``org_secret`` set) -- if the peer's agent-card has not been verified.
        """
        peer = self._peers.get(name)
        if peer is None:
            return None
        if self._require_verification and name not in self._verified:
            _logger.debug(
                "A2A peer '%s' is known but not org-verified (uncallable); see startup/refresh logs",
                name,
            )
            return None
        return peer

    async def verify_all(self) -> int:
        """P3: fetch + HMAC-verify each declared peer's agent-card (startup + refresh).

        Concurrent (``asyncio.gather``), per-peer timeout, non-fatal. REBUILDS the
        verified set from the results -- a previously-verified peer whose card has
        expired/rotated is downgraded (dropped), so "verified-only" holds over time,
        not just at first success.
        """
        if not self._require_verification or not self._peers:
            return 0
        import asyncio

        peers = list(self._peers.values())
        results = await asyncio.gather(*(self._verify_one(p) for p in peers), return_exceptions=True)
        self._verified = {p.name for p, ok in zip(peers, results, strict=True) if ok is True}
        if len(self._verified) != len(peers):
            _logger.warning(
                "A2A org-claim verification: %d/%d peers verified; unverified peers are uncallable",
                len(self._verified),
                len(peers),
            )
        return len(self._verified)

    async def _verify_one(self, peer: PeerConfig) -> bool:
        import dataclasses
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
        if not isinstance(card, dict) or not verify_card(
            card, self._org_secret, freshness_seconds=self._freshness_seconds
        ):
            _logger.warning("A2A peer '%s' failed org-claim verification (rejected)", peer.name)
            return False
        # Audience binding: the card must advertise the URL we fetched it from (anti-replay).
        from urllib.parse import urlparse

        fetched_host = urlparse(peer.url).hostname
        card_host = urlparse(str(card.get("peer_invoke_url", ""))).hostname
        if not card_host or card_host != fetched_host:
            _logger.warning(
                "A2A peer '%s' card URL mismatch (fetched %s, card advertises %s) -- rejected",
                peer.name,
                fetched_host,
                card_host,
            )
            return False
        # Fill agent_name from the card if the operator left it blank (PeerConfig is frozen).
        if not peer.agent_name:
            card_agent = str(card.get("agent_name", "") or "")
            if card_agent:
                self._peers[peer.name] = dataclasses.replace(peer, agent_name=card_agent)
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

    @property
    def peer_count(self) -> int:
        """Number of declared outbound peers."""
        return len(self._peers)

    def disable_verification(self) -> None:
        """Disable verified-only gating (all loaded peers become callable).

        Used for the CLI/local build path, which has no lifespan to run
        :meth:`verify_all` -- gating there would make every peer uncallable.
        The ``koboi serve`` path keeps verification on (peers verified at startup).
        """
        self._require_verification = False

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


class PeerRateLimiter:
    """Per-peer-token rate limiter + concurrency cap (in-memory, instance-scoped).

    Bounds how many ``/v1/peer/invoke`` calls a single peer token can make per minute
    (rate) + how many can run simultaneously (concurrency), preventing a malicious or
    compromised peer from draining the receiver's LLM budget or monopolizing its pool.
    """

    def __init__(self, max_per_minute: int = 60, max_concurrent: int = 10) -> None:
        self._max = max_per_minute
        self._hits: dict[str, list[float]] = {}
        self._max_concurrent = max_concurrent
        self._active: dict[str, int] = {}

    def allow(self, peer_id: str) -> bool:
        """True if the call is within the per-minute rate limit; False if throttled."""
        if self._max <= 0:
            return True  # unlimited
        import time

        now = time.time()
        hits = self._hits.setdefault(peer_id, [])
        hits[:] = [t for t in hits if now - t < 60.0]  # prune old
        if len(hits) >= self._max:
            return False
        hits.append(now)
        return True

    def try_acquire(self, peer_id: str) -> bool:
        """Acquire a concurrency slot. True if allowed, False if at the cap."""
        if self._max_concurrent <= 0:
            return True  # unlimited
        count = self._active.get(peer_id, 0)
        if count >= self._max_concurrent:
            return False
        self._active[peer_id] = count + 1
        return True

    def release(self, peer_id: str) -> None:
        """Release a concurrency slot."""
        self._active[peer_id] = max(0, self._active.get(peer_id, 0) - 1)


def build_peer_registry(peers_cfg: Any, *, verified_registry: PeerRegistry | None = None) -> PeerRegistry | None:
    """Resolve the PeerRegistry for an agent build (single source for the facade paths).

    If a ``verified_registry`` is provided (the server's ``app.state.peer_registry``,
    verified at lifespan), return it as-is so agents share the SAME verified registry
    (without this, each agent would build its own unverified registry and -- with an
    ``org_secret`` set -- every peer would be gated out). Otherwise build a fresh one
    from config with verification DISABLED (the CLI/local path has no lifespan to run
    ``verify_all``; verified-only is a ``koboi serve`` feature). Returns None when A2A
    is not enabled.
    """
    if verified_registry is not None:
        return verified_registry
    if not peers_cfg:
        return None
    cfg = PeerRegistry._as_dict(peers_cfg)
    if not cfg.get("enabled"):
        return None
    reg = PeerRegistry()
    reg.load_from_config(cfg)
    reg.disable_verification()
    return reg


#: Max chars in a peer's response content (matches PeerInvokeRequest.message max_length).
_MAX_PEER_CONTENT = 65536
#: Retry attempts on transient failures (3 total tries: initial + 2 retries).
_MAX_RETRIES = 2


async def invoke_peer(peer: PeerConfig, message: str) -> PeerInvokeResult:
    """POST a peer instance's ``/v1/peer/invoke`` receiver and return its answer + trace-id.

    The single A2A HTTP path, shared by the ``call_peer_agent`` tool and
    :class:`koboi.orchestration.remote_proxy.RemoteAgentProxy`. Retries on transient
    failures (5xx, connection errors) with exponential backoff; raises on 4xx, malformed
    responses, or responses exceeding ``_MAX_PEER_CONTENT``. Callers surface failures.
    """
    import asyncio

    import httpx  # lazy: peers.py stays importable without httpx at module load

    from koboi import tracing_context

    url = peer.url.rstrip("/") + "/v1/peer/invoke"
    headers = {"Authorization": f"Bearer {peer.token}"}
    # P4: carry the W3C trace as a child traceparent (same trace-id, fresh parent-id).
    trace_ctx = tracing_context.current()
    if trace_ctx is not None:
        headers["traceparent"] = tracing_context.child(trace_ctx).as_traceparent()
    body: dict = {"message": message}
    if peer.agent_name:
        body["agent_name"] = peer.agent_name  # routing hint (informational)

    _retriable = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=peer.timeout) as client:
                resp = await client.post(url, json=body, headers=headers)
                if resp.status_code >= 500 and attempt < _MAX_RETRIES:
                    await asyncio.sleep(2**attempt)  # backoff: 1s, 2s
                    continue
                resp.raise_for_status()
                try:
                    data = resp.json()
                except Exception as exc:
                    raise ValueError(
                        f"peer returned non-JSON (status {resp.status_code}): {resp.text[:200]!r}"
                    ) from exc
            content = data.get("content")
            if not isinstance(content, str):
                raise ValueError(f"peer returned no string content: {data!r}")
            if len(content) > _MAX_PEER_CONTENT:
                raise ValueError(f"peer response too large: {len(content)} chars (max {_MAX_PEER_CONTENT})")
            return PeerInvokeResult(content=content, receiver_trace_id=str(data.get("trace_id") or ""))
        except _retriable:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(2**attempt)  # backoff: 1s, 2s
                continue
            raise
    # Unreachable: the loop always returns or raises on its last iteration
    # (attempt == _MAX_RETRIES makes every `continue` guard above false).
    raise AssertionError("invoke_peer: retry loop exited without return/raise")
