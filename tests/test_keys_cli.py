"""Unit tests for koboi/server/keys_cli.py (no FastAPI)."""

from __future__ import annotations

import json

from koboi.server.keys_cli import create_key, list_keys, revoke_key, rotate_key


class TestKeysCli:
    def test_create_returns_plaintext_and_stores_hash(self, tmp_path):
        f = str(tmp_path / "keys.json")
        plaintext = create_key(f, label="test")
        assert plaintext.startswith("koboi_")
        keys = json.loads((tmp_path / "keys.json").read_text())
        assert len(keys) == 1
        assert keys[0]["hash"] != plaintext
        assert keys[0]["label"] == "test"

    def test_list_shows_no_hash(self, tmp_path):
        f = str(tmp_path / "keys.json")
        create_key(f, label="prod")
        create_key(f, label="dev")
        listed = list_keys(f)
        assert len(listed) == 2
        assert all("hash" not in k for k in listed)
        assert listed[0]["label"] == "prod"

    def test_revoke(self, tmp_path):
        f = str(tmp_path / "keys.json")
        create_key(f)
        assert revoke_key("key_0001", f) is True
        listed = list_keys(f)
        assert listed[0]["revoked"] is True
        assert revoke_key("nonexistent", f) is False

    def test_rotate(self, tmp_path):
        f = str(tmp_path / "keys.json")
        create_key(f, label="original")
        new = rotate_key("key_0001", f, label="rotated")
        assert new is not None
        assert new.startswith("koboi_")
        listed = list_keys(f)
        assert len(listed) == 2
        assert listed[0]["revoked"] is True
        assert listed[1]["revoked"] is False
        assert listed[1]["label"] == "rotated"

    def test_rotate_unknown_returns_none(self, tmp_path):
        f = str(tmp_path / "keys.json")
        assert rotate_key("nonexistent", f) is None
