"""Unit tests for PeerRegistry (cross-instance A2A)."""

from __future__ import annotations

from koboi.server.peers import PeerRegistry


def _cfg(**kw):
    base = {"enabled": True, "peers": [], "inbound_tokens": []}
    base.update(kw)
    return base


class TestPeerRegistry:
    def test_get_returns_configured_peer(self):
        r = PeerRegistry()
        r.load_from_config(_cfg(peers=[{"name": "C", "url": "http://example.com:8000", "token": "t"}]))
        peer = r.get("C")
        assert peer is not None
        assert peer.url == "http://example.com:8000"
        assert peer.token == "t"

    def test_get_unknown_returns_none(self):
        assert PeerRegistry().get("nope") is None

    def test_inbound_token_accept_configured(self):
        r = PeerRegistry()
        r.load_from_config(_cfg(inbound_tokens=["tok-x"]))
        assert r.validate_inbound_token("tok-x") == "peer"

    def test_inbound_token_reject_unknown(self):
        r = PeerRegistry()
        r.load_from_config(_cfg(inbound_tokens=["tok-x"]))
        assert r.validate_inbound_token("wrong") is None

    def test_inbound_empty_tokens_ignored(self):
        r = PeerRegistry()
        r.load_from_config(_cfg(inbound_tokens=["", "  "]))
        assert r.validate_inbound_token("") is None
        assert r.has_peers is False

    def test_has_peers_false_when_empty(self):
        assert PeerRegistry().has_peers is False

    def test_has_peers_true_with_only_inbound(self):
        r = PeerRegistry()
        r.load_from_config(_cfg(inbound_tokens=["tok"]))
        assert r.has_peers is True

    def test_ssrf_rejects_private_ip(self):
        r = PeerRegistry()
        n = r.load_from_config(_cfg(peers=[{"name": "BAD", "url": "http://127.0.0.1:8000"}]))
        assert n == 0
        assert r.get("BAD") is None
        assert r.has_peers is False

    def test_ssrf_rejects_localhost(self):
        r = PeerRegistry()
        n = r.load_from_config(_cfg(peers=[{"name": "L", "url": "http://localhost:9"}]))
        assert n == 0
        assert r.get("L") is None

    def test_ssrf_rejects_internal_range(self):
        r = PeerRegistry()
        n = r.load_from_config(_cfg(peers=[{"name": "I", "url": "http://10.0.0.1"}]))
        assert n == 0
        assert r.get("I") is None

    def test_bad_scheme_rejected_even_with_allow_private(self):
        r = PeerRegistry()
        n = r.load_from_config({"enabled": True, "allow_private_network": True, "peers": [{"name": "F", "url": "ftp://x"}]})
        assert n == 0
        assert r.get("F") is None

    def test_allow_private_accepts_localhost(self):
        r = PeerRegistry()
        n = r.load_from_config(
            {"enabled": True, "allow_private_network": True, "peers": [{"name": "C", "url": "http://localhost:8002"}]}
        )
        assert n == 1
        assert r.get("C") is not None

    def test_skip_bad_keep_good(self):
        r = PeerRegistry()
        n = r.load_from_config(
            _cfg(
                peers=[
                    {"name": "BAD", "url": "http://10.0.0.1"},  # private, rejected
                    {"name": "GOOD", "url": "http://example.com"},  # ok
                    {"name": "NOPROMPT"},  # malformed (no url)
                    {"name": "FTP", "url": "ftp://x"},  # bad scheme
                ]
            )
        )
        assert n == 1
        assert r.get("GOOD") is not None
        assert r.get("BAD") is None
        assert r.get("FTP") is None

    def test_timeout_parsed(self):
        r = PeerRegistry()
        r.load_from_config(
            {"enabled": True, "allow_private_network": True, "peers": [{"name": "C", "url": "http://localhost:1", "timeout": 5}]}
        )
        assert r.get("C").timeout == 5.0

    def test_default_timeout(self):
        r = PeerRegistry()
        r.load_from_config(
            {"enabled": True, "allow_private_network": True, "peers": [{"name": "C", "url": "http://localhost:1"}]}
        )
        assert r.get("C").timeout == 30.0
