"""Tests for ``serve_app`` bind resolution: CLI flag > YAML (server.host/port) > defaults.

Gated on the ``api`` extra: ``pytest.importorskip("fastapi")`` skips cleanly when
fastapi isn't installed (app.py imports fastapi at module top).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from koboi.config import Config  # noqa: E402
from koboi.server.app import _resolve_bind  # noqa: E402


def _cfg(server: dict | None = None) -> Config:
    data = {"agent": {"name": "t"}, "llm": {"model": "m"}}
    if server:
        data["server"] = server
    return Config.from_dict(data, validate=True)


class TestResolveBind:
    def test_defaults_when_nothing_set(self):
        host, port = _resolve_bind(_cfg(), None, None)
        assert host == "127.0.0.1"
        assert port == 8000

    def test_yaml_honored_when_no_cli_flag(self):
        host, port = _resolve_bind(_cfg(server={"host": "0.0.0.0", "port": 9000}), None, None)
        assert host == "0.0.0.0"
        assert port == 9000

    def test_cli_overrides_yaml(self):
        host, port = _resolve_bind(_cfg(server={"host": "0.0.0.0", "port": 9000}), "1.2.3.4", 7000)
        assert host == "1.2.3.4"
        assert port == 7000

    def test_partial_cli_override(self):
        # host via CLI, port falls through to YAML
        host, port = _resolve_bind(_cfg(server={"port": 9000}), "1.2.3.4", None)
        assert host == "1.2.3.4"
        assert port == 9000
