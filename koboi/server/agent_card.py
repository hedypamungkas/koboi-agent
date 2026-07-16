"""koboi/server/agent_card -- self-describing agent-card + HMAC org-claim (P3).

Each instance serves a signed JSON agent-card at :data:`CARD_PATH` describing
itself (org, agent name(s), capabilities, peer-invoke URL). The card is HMAC-SHA256
-signed with a shared ``peers.org_secret`` so a peer can *prove* same-org membership
("self-observing" trust) instead of *assuming* it. Open endpoint (no Bearer) --
trust comes from the signature, which only same-org instances can produce.

Used two ways:
- outbound: :func:`build_agent_card` assembles + signs the local card (called once
  in ``create_app``, served at ``GET CARD_PATH``).
- inbound: :func:`verify_card` checks a fetched peer's card; ``PeerRegistry.verify_all``
  calls it at startup to gate which peers are callable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from koboi.config import Config

_logger = logging.getLogger(__name__)

#: Well-known open endpoint serving the agent-card (added to auth OPEN_PATHS).
CARD_PATH = "/.well-known/agent-card"

#: Card freshness window -- a fetched card's ``issued_at`` must be within this of now.
#: Kept at 6h (not days): the serving instance refreshes its card hourly
#: (``_a2a_refresh_loop``), so a live peer's card is always <1h old; 6h tolerates
#: clock skew + a missed refresh, while bounding the replay window for a captured
#: open card to hours, not days.
FRESHNESS_SECONDS = 6 * 3600

_SIGNATURE_PREFIX = "sha256="


def _canonical(card: dict) -> bytes:
    """Stable byte encoding of the card minus its ``signature`` field."""
    body = {k: v for k, v in card.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def sign_card(card: dict, org_secret: str) -> str:
    """HMAC-SHA256 over the canonical card, hex, ``sha256=``-prefixed (mirrors webhook signing)."""
    digest = hmac.new(org_secret.encode(), _canonical(card), hashlib.sha256).hexdigest()
    return _SIGNATURE_PREFIX + digest


def verify_card(card: dict, org_secret: str) -> bool:
    """True iff ``card``'s HMAC org-claim matches ``org_secret`` AND it is fresh.

    No secret / missing or malformed signature / tampered body / stale ``issued_at``
    all return False. Constant-time signature compare via :func:`hmac.compare_digest`.
    Each failure emits a DEBUG reason so the verifying operator (who holds the
    ``org_secret`` and reads their own logs) can diagnose why a peer was rejected --
    the rejection itself stays opaque to a remote attacker.
    """
    if not org_secret:
        _logger.debug("agent-card verify: no org_secret configured")
        return False
    sig = card.get("signature")
    if not isinstance(sig, str) or not sig.startswith(_SIGNATURE_PREFIX):
        _logger.debug("agent-card verify: missing or malformed signature")
        return False
    expected = sign_card(card, org_secret)
    if not hmac.compare_digest(expected, sig):
        _logger.debug("agent-card verify: signature mismatch (tampered body or wrong org_secret)")
        return False
    try:
        issued_at = float(card.get("issued_at", 0))
    except (TypeError, ValueError):
        _logger.debug("agent-card verify: issued_at not parseable")
        return False
    if issued_at <= 0 or abs(time.time() - issued_at) > FRESHNESS_SECONDS:
        _logger.debug("agent-card verify: stale or non-positive issued_at")
        return False
    return True


def build_agent_card(config: Config, org_secret: str, public_base_url: str) -> dict:
    """Assemble + sign this instance's agent-card from ``config`` (no live agent needed).

    Works uniformly for single-agent and orchestrated configs (and degrades to the
    mode label for dynamic/deep_research, which can't enumerate agents statically).
    """
    from koboi.skills.registry import discover_skills

    org = str(config.get("peers", "org", default="") or "")
    agent_name = config.agent_name

    # Agent identity: orchestrated -> the configured node names; else the single agent.
    agents: list[dict] = []
    if config.get("orchestration", "enabled", default=False):
        for a in config.get("orchestration", "agents", default=[]) or []:
            if isinstance(a, dict) and a.get("name"):
                agents.append({"name": str(a["name"]), "description": str(a.get("description", ""))})
    if not agents:
        agents = [{"name": agent_name, "description": str(config.get("agent", "description", default=""))}]

    skills = [
        {"name": s.name, "description": str(s.description or "")}
        for s in discover_skills(config.get("skills", "search_paths", default=[]))
    ]

    card: dict = {
        "version": 1,
        "org": org,
        "agent_name": agent_name,
        "agents": agents,
        "peer_invoke_url": public_base_url.rstrip("/") + "/v1/peer/invoke",
        "skills": skills,
        "issued_at": time.time(),
    }
    if org_secret:
        card["signature"] = sign_card(card, org_secret)
    return card
