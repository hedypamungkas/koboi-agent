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

    def test_jobs_shell_allowlist_parses(self):
        cfg = Config.from_dict(
            {**_base(), "jobs": {"enabled": True, "shell_allowlist": ["pytest*", "git commit*"]}},
            validate=True,
        )
        assert cfg.schema.jobs.shell_allowlist == ["pytest*", "git commit*"]
        # Runtime read path (dotted-path over raw data) sees it too.
        assert cfg.get("jobs", "shell_allowlist", default=[]) == ["pytest*", "git commit*"]
        # Default: empty (deny-by-default preserved).
        cfg2 = Config.from_dict({**_base(), "jobs": {"enabled": True}}, validate=True)
        assert cfg2.schema.jobs.shell_allowlist == []

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
        cfg = Config.builder().agent(name="t").llm(model="m").jobs(max_concurrent=16, ttl_seconds=3600).build()
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

    def test_background_shell_unknown_key_raises(self):
        # I-3: issue #79 parity -- a typo'd bg-shell key must raise, not silently
        # fall back to the default lifetime. (background_shell nests under agent:.)
        import pytest

        with pytest.raises(ValueError, match=r"(max_lifetime_secnds|Unknown BackgroundShellConfig)"):
            Config.from_dict(
                {**_base(), "agent": {"name": "t", "background_shell": {"max_lifetime_secnds": 60}}},
                validate=True,
            )

    def test_github_unknown_key_raises(self):
        # I-3: a misspelled token key (``tokin``) would leave the token empty and
        # fail opaquely at runtime -- fail closed at load instead.
        import pytest

        with pytest.raises(ValueError, match=r"(tokin|Unknown GithubConfig)"):
            Config.from_dict({**_base(), "github": {"tokin": "ghp_x"}}, validate=True)

    def test_github_timeout_must_be_positive(self):
        import pytest

        with pytest.raises(ValueError):
            Config.from_dict({**_base(), "github": {"timeout": 0}}, validate=True)

    def test_journal_checkpoint_unknown_key_raises(self):
        # I-3: nested journal.checkpoint typo must raise too.
        import pytest

        with pytest.raises(ValueError, match=r"(git_timout|Unknown JournalCheckpointConfig)"):
            Config.from_dict({**_base(), "journal": {"checkpoint": {"git_timout": 30}}}, validate=True)

    def test_parallel_tools_inner_typo_raises(self):
        # agent.parallel_tools is a dict read via raw .get() at runtime, so a typo'd
        # inner key (max_concurency) would silently fall back to default. Fail closed.
        import pytest

        with pytest.raises(ValueError, match=r"(max_concurency|Unknown _ParallelToolsShape)"):
            Config.from_dict(
                {**_base(), "agent": {"name": "t", "parallel_tools": {"enabled": True, "max_concurency": 8}}},
                validate=True,
            )

    def test_token_prices_inner_typo_raises(self):
        import pytest

        with pytest.raises(ValueError, match=r"(imput_per_1k|Unknown _TokenPricesShape)"):
            Config.from_dict(
                {**_base(), "agent": {"name": "t", "token_prices": {"imput_per_1k": 1.0}}},
                validate=True,
            )

    def test_no_unknown_key_warning_for_server_jobs(self, caplog):
        # server/jobs are now declared top-level keys -> no "Unknown config key" warning.
        with caplog.at_level(logging.WARNING):
            Config.from_dict({**_base(), "server": {"enabled": True}, "jobs": {"enabled": True}}, validate=True)
        unknown = [r for r in caplog.records if "Unknown" in r.message]
        assert not any("server" in r.message or "jobs" in r.message for r in unknown)
