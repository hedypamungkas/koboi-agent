"""Tests for koboi.harness.env secret-hygiene filtering (P0a)."""

from __future__ import annotations

from unittest.mock import patch

from koboi.harness.env import (
    DEFAULT_ENV_ALLOWLIST,
    SECRET_BLOCKLIST,
    build_safe_env,
    configure_env_defaults,
)


class TestBuildSafeEnvDefaults:
    def test_returns_dict(self):
        assert isinstance(build_safe_env(), dict)

    def test_path_always_present(self):
        with patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
            assert build_safe_env()["PATH"] == "/usr/bin"

    def test_home_allowed(self):
        with patch.dict("os.environ", {"HOME": "/h", "PATH": "/p"}, clear=True):
            assert build_safe_env()["HOME"] == "/h"

    def test_secret_key_stripped(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-xxx", "PATH": "/x"}, clear=True):
            assert "OPENAI_API_KEY" not in build_safe_env()

    def test_secret_key_lowercase_stripped(self):
        with patch.dict("os.environ", {"openai_api_key": "sk-xxx", "PATH": "/x"}, clear=True):
            assert "openai_api_key" not in build_safe_env()

    def test_token_stripped(self):
        with patch.dict("os.environ", {"GH_TOKEN": "t", "PATH": "/x"}, clear=True):
            assert "GH_TOKEN" not in build_safe_env()

    def test_database_url_stripped(self):
        with patch.dict("os.environ", {"DATABASE_URL": "postgres://x", "PATH": "/x"}, clear=True):
            assert "DATABASE_URL" not in build_safe_env()

    def test_credentials_stripped(self):
        with patch.dict("os.environ", {"MY_SERVICE_CREDENTIALS": "c", "PATH": "/x"}, clear=True):
            assert "MY_SERVICE_CREDENTIALS" not in build_safe_env()

    def test_koboi_glob_allowed(self):
        with patch.dict("os.environ", {"KOBOI_CUSTOM": "v", "PATH": "/x"}, clear=True):
            assert build_safe_env()["KOBOI_CUSTOM"] == "v"

    def test_koboi_secret_token_blocked_despite_glob(self):
        # KOBOI_* matches the allow-glob, but *_TOKEN block-glob WINS.
        with patch.dict("os.environ", {"KOBOI_DB_TOKEN": "x", "PATH": "/x"}, clear=True):
            assert "KOBOI_DB_TOKEN" not in build_safe_env()

    def test_unknown_var_stripped(self):
        with patch.dict("os.environ", {"SOMETHING_RANDOM": "v", "PATH": "/x"}, clear=True):
            assert "SOMETHING_RANDOM" not in build_safe_env()


class TestEnvPassthrough:
    def test_config_passthrough_restores_full_env(self):
        with patch.dict("os.environ", {"SECRET_VAR": "x", "PATH": "/x"}, clear=True):
            env = build_safe_env({"env_passthrough": True})
            assert "SECRET_VAR" in env

    def test_env_var_passthrough_restores_full_env(self):
        with patch.dict(
            "os.environ",
            {"SECRET_VAR": "x", "KOBOI_ENV_PASSTHROUGH": "1", "PATH": "/x"},
            clear=True,
        ):
            assert "SECRET_VAR" in build_safe_env()

    def test_env_var_passthrough_false_does_not_restore(self):
        with patch.dict(
            "os.environ",
            {"SECRET_VAR": "x", "KOBOI_ENV_PASSTHROUGH": "0", "PATH": "/x"},
            clear=True,
        ):
            assert "SECRET_VAR" not in build_safe_env()


class TestEnvAllowlistExtensions:
    def test_user_allowlist_adds_var(self):
        with patch.dict("os.environ", {"CARGO_HOME": "/x", "PATH": "/p"}, clear=True):
            env = build_safe_env({"env_allowlist": ["CARGO_HOME"]})
            assert env["CARGO_HOME"] == "/x"

    def test_user_allowlist_glob(self):
        with patch.dict("os.environ", {"RUSTUP_HOME": "/x", "PATH": "/p"}, clear=True):
            env = build_safe_env({"env_allowlist": ["RUSTUP_*"]})
            assert "RUSTUP_HOME" in env

    def test_user_allowlist_cannot_override_secret_blocklist(self):
        # Explicitly allow-listing a secret name must NOT bypass the block-list.
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-x", "PATH": "/p"}, clear=True):
            env = build_safe_env({"env_allowlist": ["OPENAI_API_KEY"]})
            assert "OPENAI_API_KEY" not in env


class TestEnvBlocklistExtensions:
    def test_user_blocklist_strips_var(self):
        with patch.dict("os.environ", {"MY_CUSTOM_VAR": "x", "PATH": "/p"}, clear=True):
            env = build_safe_env({"env_blocklist": ["MY_CUSTOM_*"]})
            assert "MY_CUSTOM_VAR" not in env


class TestConfigureEnvDefaults:
    def test_module_defaults_used_when_no_arg(self):
        configure_env_defaults({"env_allowlist": ["EXTRA_VAR"]})
        try:
            with patch.dict("os.environ", {"EXTRA_VAR": "v", "PATH": "/p"}, clear=True):
                assert build_safe_env()["EXTRA_VAR"] == "v"
        finally:
            configure_env_defaults(None)  # reset to avoid polluting other tests

    def test_explicit_tool_config_overrides_module_defaults(self):
        configure_env_defaults({"env_allowlist": ["FROM_DEFAULTS"]})
        try:
            with patch.dict(
                "os.environ",
                {"FROM_DEFAULTS": "d", "FROM_OVERRIDE": "o", "PATH": "/p"},
                clear=True,
            ):
                env = build_safe_env({"env_allowlist": ["FROM_OVERRIDE"]})
                # module defaults are NOT merged when an explicit tool_config is passed
                assert "FROM_OVERRIDE" in env
                assert "FROM_DEFAULTS" not in env
        finally:
            configure_env_defaults(None)


class TestConstants:
    def test_allowlist_is_frozenset(self):
        assert isinstance(DEFAULT_ENV_ALLOWLIST, frozenset)

    def test_blocklist_is_tuple(self):
        assert isinstance(SECRET_BLOCKLIST, tuple)

    def test_path_in_allowlist(self):
        assert "PATH" in DEFAULT_ENV_ALLOWLIST

    def test_key_pattern_in_blocklist(self):
        assert "*_KEY" in SECRET_BLOCKLIST
