"""Tests for the P3 agent-card: signing/verification, the open route, and verified-only gating."""

from __future__ import annotations

import asyncio
import copy
import time

import httpx
from httpx import ASGITransport

from koboi.config import Config
from koboi.server.agent_card import CARD_PATH, build_agent_card, sign_card, verify_card
from koboi.server.app import _a2a_refresh_once, create_app
from koboi.server.peers import PeerRegistry
from tests.conftest import MockClient, make_mock_response


def _card(org_secret: str = "s3cr3t") -> dict:
    cfg = Config.from_dict(
        {
            "agent": {"name": "C", "description": "reviewer"},
            "llm": {"provider": "openai", "model": "x", "api_key": "x"},
            "peers": {"org": "acme", "org_secret": org_secret, "public_base_url": "http://c.local:8000"},
        }
    )
    return build_agent_card(cfg, org_secret, "http://c.local:8000")


class TestAgentCardSigning:
    def test_build_shape_and_signed(self):
        card = _card()
        assert card["org"] == "acme"
        assert card["agent_name"] == "C"
        assert card["peer_invoke_url"] == "http://c.local:8000/v1/peer/invoke"
        assert card["agents"] == [{"name": "C", "description": "reviewer"}]
        assert "signature" in card and card["signature"].startswith("sha256=")

    def test_unsigned_when_no_secret(self):
        card = _card(org_secret="")
        assert "signature" not in card

    def test_verify_round_trip(self):
        assert verify_card(_card(), "s3cr3t") is True

    def test_tamper_detected(self):
        bad = copy.deepcopy(_card())
        bad["org"] = "evil"
        assert verify_card(bad, "s3cr3t") is False

    def test_wrong_secret_rejected(self):
        assert verify_card(_card(), "other") is False

    def test_unsigned_card_rejected(self):
        assert verify_card(_card(org_secret=""), "s3cr3t") is False

    def test_stale_card_rejected_even_if_signed(self):
        card = _card()
        card["issued_at"] = time.time() - (8 * 24 * 3600)  # older than the 6h freshness window
        card["signature"] = sign_card(card, "s3cr3t")  # re-sign the stale body
        assert verify_card(card, "s3cr3t") is False

    def test_missing_signature_rejected(self):
        card = _card()
        del card["signature"]
        assert verify_card(card, "s3cr3t") is False


def _app(*, org_secret: str, api_keys=None):
    cfg = Config.from_dict(
        {
            "agent": {"name": "C", "mode": "chat", "system_prompt": "C"},
            "llm": {"provider": "openai", "model": "x", "api_key": "x"},
            "memory": {"backend": "memory"},
            "peers": {
                "enabled": True,
                "org": "acme",
                "org_secret": org_secret,
                "public_base_url": "http://c.local:8000",
                "inbound_tokens": ["tok-y"],
            },
        }
    )
    return create_app(cfg, client_factory=lambda: MockClient([make_mock_response(content="ok")]), api_keys=api_keys)


class TestAgentCardRoute:
    async def test_card_route_open_and_signed(self):
        # auth is ON (api_keys set), yet the card route is open + signed.
        app = _app(org_secret="s3cr3t", api_keys=["admin"])
        async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
            r = await c.get(CARD_PATH)  # no Bearer
        assert r.status_code == 200
        card = r.json()
        assert card["org"] == "acme"
        assert card["signature"].startswith("sha256=")
        assert verify_card(card, "s3cr3t") is True

    async def test_other_routes_still_require_auth(self):
        app = _app(org_secret="s3cr3t", api_keys=["admin"])
        async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
            r = await c.post("/v1/peer/invoke", json={"message": "hi"})  # no Bearer
        assert r.status_code == 401

    async def test_card_unsigned_when_no_secret(self):
        app = _app(org_secret="", api_keys=["admin"])
        async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
            r = await c.get(CARD_PATH)
        assert r.status_code == 200
        assert "signature" not in r.json()


# --- PeerRegistry.verify_all (verified-only gating) ---


class _FakeResp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    def json(self):
        return self._p


class _FakeClient:
    def __init__(self, payload, **kw):
        self._p = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        return _FakeResp(self._p)


def _registry_with_secret(peers):
    reg = PeerRegistry()
    reg.load_from_config({"enabled": True, "org_secret": "s3cr3t", "allow_private_network": True, "peers": peers})
    return reg


class TestVerifyAll:
    async def test_verified_peer_becomes_callable(self, monkeypatch):
        reg = _registry_with_secret([{"name": "C", "url": "http://localhost:8002", "token": "t"}])
        assert reg.get("C") is None  # unverified -> uncallable
        # Serve a valid card from the peer.
        good_card = _card()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _FakeClient(good_card))
        n = await reg.verify_all()
        assert n == 1
        assert reg.get("C") is not None  # callable == in the verified set

    async def test_bad_signature_peer_stays_gated(self, monkeypatch):
        reg = _registry_with_secret(
            [
                {"name": "good", "url": "http://localhost:8002", "token": "t"},
                {"name": "bad", "url": "http://localhost:8003", "token": "t"},
            ]
        )
        good_card = _card()
        evil_card = copy.deepcopy(good_card)
        evil_card["org"] = "evil"  # tampered -> signature invalid

        class _RoutingClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                return _FakeResp(good_card if "8002" in url else evil_card)

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _RoutingClient())
        n = await reg.verify_all()
        assert n == 1  # only 'good' verified
        assert reg.get("good") is not None
        assert reg.get("bad") is None  # gated

    async def test_no_secret_skips_verification(self):
        reg = PeerRegistry()
        reg.load_from_config(
            {"enabled": True, "allow_private_network": True, "peers": [{"name": "C", "url": "http://localhost:8002"}]}
        )
        assert reg._require_verification is False
        assert (await reg.verify_all()) == 0
        assert reg.get("C") is not None  # usable without verification (P0-P2 behavior)

    async def test_refresh_downgrades_peer_on_card_rotation(self, monkeypatch):
        # C2 regression: a previously-verified peer whose card rotates (now tampered)
        # is re-gated by the next verify_all -- "verified-only" holds over time.
        reg = _registry_with_secret([{"name": "C", "url": "http://localhost:8002", "token": "t"}])
        good_card = _card()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _FakeClient(good_card))
        await reg.verify_all()
        assert reg.get("C") is not None  # verified

        evil_card = copy.deepcopy(good_card)
        evil_card["org"] = "evil"  # tampered -> signature invalid
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _FakeClient(evil_card))
        n = await reg.verify_all()
        assert n == 0
        assert reg.get("C") is None  # downgraded (re-gated)


class TestA2ARefresh:
    async def test_refresh_advances_issued_at_and_reverifies(self, monkeypatch):
        # C1/I1: the background refresh re-stamps the card (never ages out) + re-verifies peers.
        cfg = Config.from_dict(
            {
                "agent": {"name": "C", "mode": "chat"},
                "llm": {"provider": "openai", "model": "x", "api_key": "x"},
                "memory": {"backend": "memory"},
                "peers": {
                    "enabled": True,
                    "org_secret": "s3cr3t",
                    "allow_private_network": True,
                    "peers": [{"name": "Y", "url": "http://localhost:8002", "token": "t"}],
                },
            }
        )
        app = create_app(cfg, client_factory=lambda: MockClient([make_mock_response(content="ok")]), api_keys=["admin"])
        old_issued = app.state.agent_card["issued_at"]

        calls = {"n": 0}

        async def fake_verify():
            calls["n"] += 1

        monkeypatch.setattr(app.state.peer_registry, "verify_all", fake_verify)
        await asyncio.sleep(0.01)
        await _a2a_refresh_once(app, cfg, "s3cr3t", "http://c.local:8000", app.state.peer_registry)

        assert app.state.agent_card["issued_at"] > old_issued  # card re-stamped (C1)
        assert verify_card(app.state.agent_card, "s3cr3t") is True  # still validly signed
        assert calls["n"] == 1  # peers re-verified (I1)
