"""Tests for server/jobs config schema + sandbox.network_isolation (M0 16.6)."""

from __future__ import annotations

import logging

from koboi.config import Config


def _base() -> dict:
    return {"agent": {"name": "t"}, "llm": {"model": "m"}}


class TestServerJobsConfig:
    def test_defaults_when_absent(self):
        cfg = Config.from_dict(_base(), validate=True)
        assert cfg.schema.server.enabled is False
        assert cfg.schema.jobs.enabled is False
        assert cfg.server == {}
        assert cfg.jobs == {}

    def test_server_section_parses(self):
        cfg = Config.from_dict(
            {
                **_base(),
                "server": {"enabled": True, "host": "0.0.0.0", "port": 9000, "auth_required": False},
            },
            validate=True,
        )
        assert cfg.schema.server.enabled is True
        assert cfg.schema.server.host == "0.0.0.0"
        assert cfg.schema.server.port == 9000
        assert cfg.schema.server.auth_required is False
        # dict accessor too
        assert cfg.server["host"] == "0.0.0.0"

    def test_jobs_section_parses(self):
        cfg = Config.from_dict(
            {**_base(), "jobs": {"enabled": True, "max_concurrent": 16, "ttl_seconds": 3600}},
            validate=True,
        )
        assert cfg.schema.jobs.enabled is True
        assert cfg.schema.jobs.max_concurrent == 16
        assert cfg.schema.jobs.ttl_seconds == 3600

    def test_server_invalid_port_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            Config.from_dict({**_base(), "server": {"port": 99999}}, validate=True)

    def test_server_unknown_key_ignored(self):
        cfg = Config.from_dict({**_base(), "server": {"bogus": 1}}, validate=True)
        assert cfg.schema.server.enabled is False  # parsed fine, unknown key dropped

    def test_builder_server(self):
        cfg = Config.builder().agent(name="t").llm(model="m").server(host="0.0.0.0", port=9000, enabled=True).build()
        assert cfg.server["host"] == "0.0.0.0"
        assert cfg.server["port"] == 9000
        assert cfg.server["enabled"] is True

    def test_builder_jobs(self):
        cfg = (
            Config.builder()
            .agent(name="t")
            .llm(model="m")
            .jobs(max_concurrent=16, ttl_seconds=3600)
            .build()
        )
        assert cfg.jobs["max_concurrent"] == 16
        assert cfg.jobs["ttl_seconds"] == 3600
        # default_dedicated_session was removed (dead -- dedicated is unconditional at app.py submit)
        assert "default_dedicated_session" not in cfg.jobs

    def test_sandbox_network_isolation_default(self):
        cfg = Config.from_dict(_base(), validate=True)
        assert cfg.schema.sandbox.network_isolation is None

    def test_sandbox_network_isolation_parses(self):
        cfg = Config.from_dict({**_base(), "sandbox": {"network_isolation": "seccomp"}}, validate=True)
        assert cfg.schema.sandbox.network_isolation == "seccomp"
        cfg2 = Config.from_dict({**_base(), "sandbox": {"network_isolation": "seccomp_strict"}}, validate=True)
        assert cfg2.schema.sandbox.network_isolation == "seccomp_strict"

    def test_sandbox_network_isolation_invalid_value_rejected(self):
        # Fail-closed: a typo value (e.g. ``seccop``) MUST raise, not silently fall back.
        import pytest

        with pytest.raises(ValueError):
            Config.from_dict({**_base(), "sandbox": {"network_isolation": "seccop"}}, validate=True)

    def test_sandbox_unknown_key_raises(self):
        # A key-name typo (e.g. ``network_isolaton``) MUST raise (fail-closed to
        # match value-typo behavior; extra='ignore' would otherwise silently drop it).
        import pytest

        with pytest.raises(ValueError, match=r"(network_isolat|Unknown sandbox)"):
            Config.from_dict({**_base(), "sandbox": {"network_isolaton": "seccomp"}}, validate=True)

    def test_no_unknown_key_warning_for_server_jobs(self, caplog):
        # server/jobs are now declared top-level keys -> no "Unknown config key" warning.
        with caplog.at_level(logging.WARNING):
            Config.from_dict({**_base(), "server": {"enabled": True}, "jobs": {"enabled": True}}, validate=True)
        unknown = [r for r in caplog.records if "Unknown" in r.message]
        assert not any("server" in r.message or "jobs" in r.message for r in unknown)
