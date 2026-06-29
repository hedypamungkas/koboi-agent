"""Unit tests for koboi/server/auth.py KeyStore (no FastAPI)."""

from __future__ import annotations

import json

from koboi.server.auth import KeyStore, _hash_token


class TestKeyStore:
    def test_empty_has_no_keys(self):
        ks = KeyStore()
        assert not ks.has_keys
        assert len(ks) == 0

    def test_load_from_env(self):
        ks = KeyStore()
        ks.load_from_env("key1,key2,key3")
        assert len(ks) == 3
        assert ks.validate("key1") is not None
        assert ks.validate("key2") is not None
        assert ks.validate("invalid") is None

    def test_load_from_file(self, tmp_path):
        keys_file = tmp_path / "keys.json"
        h = _hash_token("my-secret")
        keys_file.write_text(json.dumps([{"id": "k1", "hash": h, "label": "test"}]))
        ks = KeyStore()
        assert ks.load_from_file(str(keys_file)) == 1
        assert ks.validate("my-secret") == "k1"

    def test_load_from_file_skips_revoked(self, tmp_path):
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(
            json.dumps(
                [
                    {"id": "k1", "hash": _hash_token("active")},
                    {"id": "k2", "hash": _hash_token("revoked"), "revoked": True},
                ]
            )
        )
        ks = KeyStore()
        assert ks.load_from_file(str(keys_file)) == 1
        assert ks.validate("active") is not None
        assert ks.validate("revoked") is None

    def test_load_from_missing_file(self):
        ks = KeyStore()
        assert ks.load_from_file("/nonexistent/path") == 0
        assert not ks.has_keys

    def test_validate_returns_key_id(self):
        ks = KeyStore()
        ks.load_from_env("secret123")
        kid = ks.validate("secret123")
        assert kid is not None
        assert kid.startswith("env:")

    def test_hash_is_deterministic(self):
        assert _hash_token("test") == _hash_token("test")
        assert _hash_token("test") != _hash_token("test2")

    def test_env_and_file_keys_coexist(self, tmp_path):
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(json.dumps([{"id": "file-key", "hash": _hash_token("from-file")}]))
        ks = KeyStore()
        ks.load_from_file(str(keys_file))
        ks.load_from_env("from-env")
        assert ks.validate("from-file") == "file-key"
        assert ks.validate("from-env") is not None
        assert len(ks) == 2
