"""Tests for the sandbox abstraction (P0b) in koboi.sandbox."""

from __future__ import annotations

import copy
import os
import sys
from unittest.mock import patch

import pytest

from koboi.sandbox.base import BaseSandbox, SandboxResult
from koboi.sandbox.passthrough import PassthroughBackend
from koboi.sandbox.restricted import (
    DEFAULT_SAFE_PATH_DIRS,
    NETWORK_ENV_BLOCKLIST,
    RestrictedProcessBackend,
    _HAS_SECCOMP,
)
from koboi.sandbox.registry import build_sandbox, register_sandbox, sandbox_registry


@pytest.fixture(autouse=True)
def _isolate_sandbox_registry():
    """Deep-copy the registry entries so test registrations never leak."""
    saved = copy.deepcopy(sandbox_registry._entries)
    yield
    sandbox_registry._entries = saved


POSIX = sys.platform != "win32"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestSandboxRegistry:
    def test_builtins_registered_at_import(self):
        assert "passthrough" in sandbox_registry.list_available()
        assert "restricted" in sandbox_registry.list_available()

    def test_register_decorator_returns_class_unchanged(self):
        @register_sandbox("temp-test-backend")
        class _Dummy(BaseSandbox):
            def run(self, command, **kw):
                return SandboxResult(0, "", "", False)

            def validate_path(self, path, **kw):
                return path

            def build_env(self, tool_config=None):
                return {}

        assert _Dummy.__name__ == "_Dummy"  # decorator is transparent
        assert sandbox_registry.get("temp-test-backend") is not None

    def test_build_sandbox_absent_config_defaults_passthrough(self):
        sb = build_sandbox(None)
        assert isinstance(sb, PassthroughBackend)

    def test_build_sandbox_empty_dict_defaults_passthrough(self):
        sb = build_sandbox({})
        assert isinstance(sb, PassthroughBackend)

    def test_build_sandbox_explicit_passthrough(self):
        sb = build_sandbox({"backend": "passthrough"})
        assert isinstance(sb, PassthroughBackend)

    def test_build_sandbox_unknown_backend_raises(self):
        # C3: a typo'd backend name fails closed (ValueError) instead of silently
        # downgrading to passthrough (no isolation).
        with pytest.raises(ValueError, match="Unknown sandbox backend"):
            build_sandbox({"backend": "does-not-exist"})

    def test_build_sandbox_restricted_with_kwargs(self):
        sb = build_sandbox({"backend": "restricted", "workdir": "/tmp", "rlimits": {"cpu": 5}})
        assert isinstance(sb, RestrictedProcessBackend)
        assert sb._workdir == os.path.realpath("/tmp")
        assert sb._rlimits == {"cpu": 5}


# ---------------------------------------------------------------------------
# Passthrough backend
# ---------------------------------------------------------------------------


class TestPassthroughBackend:
    def test_run_string_shell(self):
        sb = PassthroughBackend()
        r = sb.run("echo hi", shell=True)
        assert isinstance(r, SandboxResult)
        assert r.returncode == 0
        assert r.stdout.strip() == "hi"

    def test_run_argv_no_shell(self):
        sb = PassthroughBackend()
        r = sb.run(["echo", "argv"], shell=False)
        assert r.stdout.strip() == "argv"

    def test_validate_path_no_enforcement_without_env(self, monkeypatch):
        monkeypatch.delenv("KOBOI_SANDBOX_DIR", raising=False)
        sb = PassthroughBackend()
        # No sandbox dir -> any path resolves without raising.
        assert sb.validate_path("/etc/hosts") == os.path.realpath("/etc/hosts")

    def test_validate_path_koboi_sandbox_dir_backcompat(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KOBOI_SANDBOX_DIR", str(tmp_path))
        sb = PassthroughBackend()
        inside = sb.validate_path(str(tmp_path / "child"))
        assert inside.startswith(str(tmp_path))
        with pytest.raises(PermissionError):
            sb.validate_path("/etc/hosts")

    def test_build_env_delegates_to_safe_env(self):
        sb = PassthroughBackend()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-x"}, clear=False):
            env = sb.build_env()
        assert "OPENAI_API_KEY" not in env
        assert "PATH" in env

    def test_network_allowed_default_true(self):
        assert PassthroughBackend().network_allowed("curl https://x") is True


# ---------------------------------------------------------------------------
# Restricted backend
# ---------------------------------------------------------------------------


@pytest.fixture
def restricted(tmp_path):
    return RestrictedProcessBackend(workdir=str(tmp_path))


class TestRestrictedValidatePath:
    def test_rejects_path_outside_workdir(self, restricted):
        with pytest.raises(PermissionError):
            restricted.validate_path("/etc/passwd")

    def test_accepts_path_inside_workdir(self, restricted, tmp_path):
        child = tmp_path / "sub"
        child.mkdir()
        resolved = restricted.validate_path(str(child))
        assert resolved.startswith(str(tmp_path))

    def test_symlink_escape_is_rejected(self, restricted, tmp_path):
        link = tmp_path / "escape"
        link.symlink_to("/etc")
        with pytest.raises(PermissionError):
            restricted.validate_path(str(link))


class TestRestrictedRun:
    def test_run_cwd_enforced_to_workdir(self, restricted, tmp_path):
        # No cwd passed -> runs inside workdir.
        r = restricted.run("pwd", shell=True)
        assert str(tmp_path) in r.stdout or os.path.realpath(str(tmp_path)) in r.stdout

    def test_run_cwd_outside_workdir_blocked(self, restricted):
        r = restricted.run("pwd", cwd="/tmp", shell=True)
        assert r.returncode != 0
        assert "outside the sandbox" in r.stderr

    def test_network_binary_soft_denied(self, restricted):
        r = restricted.run("curl https://example.com", shell=True)
        assert r.returncode == 126
        assert "network binary" in r.stderr

    def test_network_allowed_when_config_allow(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path), network="allow")
        # network_allowed flips to True; command tokens not scanned.
        assert sb.network_allowed("curl https://example.com") is True

    def test_network_scan_basename_match(self, restricted):
        assert restricted.network_allowed("wget http://x") is False
        assert restricted.network_allowed("/usr/bin/nc -z host") is False
        assert restricted.network_allowed("echo hello") is True

    def test_run_inside_workdir_succeeds(self, restricted, tmp_path):
        r = restricted.run("echo ok > out.txt && cat out.txt", shell=True)
        assert r.returncode == 0
        assert "ok" in r.stdout

    def test_env_path_restricted_to_safe_dirs(self, restricted):
        env = restricted.build_env()
        dirs = env["PATH"].split(os.pathsep)
        for d in dirs:
            assert d in DEFAULT_SAFE_PATH_DIRS

    def test_env_strips_proxy_vars(self, restricted):
        proxies = {k: "http://proxy" for k in NETWORK_ENV_BLOCKLIST}
        with patch.dict(os.environ, proxies, clear=False):
            env = restricted.build_env()
        for k in NETWORK_ENV_BLOCKLIST:
            assert k not in env

    def test_output_truncated_at_max(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path), max_output=64)
        r = sb.run("echo " + ("x" * 500), shell=True)
        # truncate_text caps at max_output + a notice suffix.
        assert len(r.stdout) <= 64 + 100
        assert "truncated" in r.stdout


@pytest.mark.skipif(not POSIX, reason="POSIX rlimits")
class TestRestrictedRlimits:
    def test_fsize_limit_kills_large_write(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path), rlimits={"fsize_mb": 1})
        # Writing 4MB under a 1MB FSIZE limit trips SIGXFSZ -> non-zero exit.
        r = sb.run("dd if=/dev/zero of=big bs=1048576 count=4 2>/dev/null; true", shell=True)
        # The file should not have grown to 4MB (rlimit killed the write).
        big = tmp_path / "big"
        if big.exists():
            assert big.stat().st_size < 4 * 1024 * 1024
        # Non-negative sanity (dd may report partial write).
        assert r.returncode >= 0


# ---------------------------------------------------------------------------
# Wiring / back-compat
# ---------------------------------------------------------------------------


class TestWiringBackcompat:
    async def test_run_shell_legacy_path_without_deps(self):
        # Calling run_shell directly (no _deps) must keep pre-P0b behavior.
        from koboi.tools.builtin.shell import run_shell

        result = run_shell("echo legacy")
        assert "legacy" in result

    async def test_run_shell_uses_sandbox_when_provided(self):
        from koboi.tools.builtin.shell import run_shell

        class _FakeSandbox(BaseSandbox):
            def __init__(self):
                self.called = False

            def run(self, command, **kw):
                self.called = True
                return SandboxResult(0, "from-sandbox", "", False)

            def validate_path(self, path, **kw):
                return path

            def build_env(self, tool_config=None):
                return {}

        fake = _FakeSandbox()
        result = run_shell("echo hi", _deps={"sandbox": fake})
        assert fake.called is True
        assert "from-sandbox" in result

    async def test_filesystem_validate_path_uses_sandbox(self):
        from koboi.tools.builtin.filesystem import _validate_path

        class _FakeSandbox(BaseSandbox):
            def __init__(self):
                self.seen = None

            def run(self, command, **kw):
                return SandboxResult(0, "", "", False)

            def validate_path(self, path, **kw):
                self.seen = path
                return "/resolved/" + path

            def build_env(self, tool_config=None):
                return {}

        fake = _FakeSandbox()
        out = _validate_path("/some/path", sandbox=fake)
        assert fake.seen == "/some/path"
        assert out == "/resolved//some/path"

    async def test_registry_execute_routes_through_sandbox(self):

        from koboi.tools.builtin import register_all
        from koboi.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register_all(registry)
        registry.set_dep("sandbox", build_sandbox(None))
        out = await registry.execute("run_shell", '{"command": "echo routed"}')
        assert "routed" in out


class TestConfigSurface:
    def test_sandbox_section_defaults_to_passthrough(self):
        from koboi.config import Config

        cfg = Config.from_dict(
            {"agent": {"name": "t"}, "llm": {"model": "m"}},
            validate=True,
        )
        # No sandbox section -> raw dict empty -> passthrough.
        assert cfg.sandbox == {}
        assert build_sandbox(cfg.sandbox).__class__.__name__ == "PassthroughBackend"

    def test_sandbox_section_parses_restricted(self):
        from koboi.config import Config

        cfg = Config.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"model": "m"},
                "sandbox": {
                    "backend": "restricted",
                    "workdir": ".",
                    "network": "deny",
                    "rlimits": {"cpu": 10, "fsize_mb": 20},
                },
            },
            validate=True,
        )
        sb = build_sandbox(cfg.sandbox)
        assert isinstance(sb, RestrictedProcessBackend)
        assert sb._rlimits["cpu"] == 10
        assert sb._rlimits["fsize_mb"] == 20
        # Pydantic schema also accepts the section without error.
        assert cfg.schema.sandbox.backend == "restricted"

    def test_builder_sandbox_method(self):
        from koboi.config import Config

        cfg = Config.builder().agent(name="t").llm(model="m").sandbox(backend="restricted", network="deny").build()
        assert cfg.sandbox["backend"] == "restricted"
        assert cfg.sandbox["network"] == "deny"

    def test_network_isolation_seccomp_resolves(self):
        """The network_isolation knob flows YAML -> build_sandbox -> backend (_resolve_kwargs)."""
        from koboi.config import Config

        cfg = Config.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"model": "m"},
                "sandbox": {"backend": "restricted", "network": "deny", "network_isolation": "seccomp"},
            },
            validate=True,
        )
        sb = build_sandbox(cfg.sandbox)
        assert isinstance(sb, RestrictedProcessBackend)
        assert sb._network_isolation == "seccomp"

    def test_server_deploy_yaml_sandbox_resolves(self):
        """Shipped production config parses and resolves to a restricted backend
        with HARD (seccomp) network isolation. Guards the shipped YAML against
        typos / schema rejection of the new field in the fast unit suite."""
        from koboi.config import Config

        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "server_deploy.yaml")
        cfg = Config.from_yaml(cfg_path)
        sb = build_sandbox(cfg.sandbox)
        assert isinstance(sb, RestrictedProcessBackend)
        assert sb._network_isolation == "seccomp"


# ---------------------------------------------------------------------------
# seccomp hard network isolation (network_isolation: seccomp)
# ---------------------------------------------------------------------------


class TestSeccompFallback:
    """When seccomp is requested but unavailable, degrade to soft deny (no crash).

    Runs on every platform (verifies the graceful fallback path that macOS and
    extra-less installs hit).
    """

    def test_requested_but_unavailable_yields_no_loader(self, tmp_path):
        with patch("koboi.sandbox.restricted._HAS_SECCOMP", False):
            sb = RestrictedProcessBackend(workdir=str(tmp_path), network="deny", network_isolation="seccomp")
        assert sb._seccomp_preexec is None  # graceful fallback, not a crash

    def test_unavailable_keeps_soft_token_deny(self, tmp_path):
        with patch("koboi.sandbox.restricted._HAS_SECCOMP", False):
            sb = RestrictedProcessBackend(workdir=str(tmp_path), network="deny", network_isolation="seccomp")
        # curl as a command token is still soft-blocked.
        r = sb.run("curl -s http://127.0.0.1:1", shell=True)
        assert r.returncode == 126
        assert "network binary" in r.stderr

    def test_not_requested_engages_no_seccomp(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path), network="deny")
        assert sb._seccomp_preexec is None
        assert sb.run("echo ok", shell=True).returncode == 0


def _fresh_listener():
    """Spin up a localhost TCP listener; return (port, received_box, thread).

    The listener accepts exactly one connection, records what it receives, then
    closes. Used to prove (no-)egress: if the sandboxed process connects + sends,
    the bytes appear in ``received_box[0]``.
    """
    import socket
    import threading

    received: list[bytes] = []
    ready = threading.Event()
    port = [0]

    def _run():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port[0] = srv.getsockname()[1]
        ready.set()
        srv.settimeout(6)
        try:
            conn, _ = srv.accept()
            received.append(conn.recv(64))
            conn.close()
        except socket.timeout:
            received.append(b"<timeout: no connection>")
        srv.close()

    t = threading.Thread(target=_run)
    t.start()
    ready.wait(5)
    return port[0], received, t


@pytest.mark.skipif(not _HAS_SECCOMP, reason="seccomp hard isolation is Linux-only")
class TestSeccompEgress:
    """Reproduces the 4 empirical bypass vectors -- all must be blocked under seccomp.

    These are the exact egress paths proven permeable against the SOFT token-scan
    (python3 socket, bash /dev/tcp, absolute-path interpreter, real outbound TCP).
    Under ``network_isolation='seccomp'`` the syscall-layer filter must deny them.
    """

    def test_seccomp_active_on_this_host(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path), network="deny", network_isolation="seccomp")
        assert sb._seccomp_preexec is not None  # sanity: the CI host has python3-seccomp

    def test_interpreter_socket_blocked(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path), network="deny", network_isolation="seccomp")
        port, received, thr = _fresh_listener()
        code = f"import socket; s=socket.create_connection(('127.0.0.1',{port}),timeout=3); s.send(b'EXFIL'); s.close()"
        r = sb.run(["python3", "-c", code], timeout=10)
        thr.join(timeout=7)
        assert r.returncode != 0  # seccomp denied connect() -> python raised
        assert received and b"EXFIL" not in received[0]

    def test_bash_dev_tcp_blocked(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path), network="deny", network_isolation="seccomp")
        port, received, thr = _fresh_listener()
        r = sb.run(["bash", "-c", f"echo BASH >/dev/tcp/127.0.0.1/{port}"], timeout=10)
        thr.join(timeout=7)
        assert r.returncode != 0
        assert received and b"BASH" not in received[0]

    def test_absolute_path_interpreter_blocked(self, tmp_path):
        import shutil

        if not shutil.which("python3"):
            pytest.skip("python3 not on PATH")
        py = shutil.which("python3")
        sb = RestrictedProcessBackend(workdir=str(tmp_path), network="deny", network_isolation="seccomp")
        port, received, thr = _fresh_listener()
        code = f"import socket; s=socket.create_connection(('127.0.0.1',{port}),timeout=3); s.send(b'ABS'); s.close()"
        r = sb.run([py, "-c", code], timeout=10)
        thr.join(timeout=7)
        assert r.returncode != 0
        assert received and b"ABS" not in received[0]

    def test_soft_still_blocks_curl_token(self, tmp_path):
        """The soft token layer still fires for curl under seccomp (defense in depth)."""
        sb = RestrictedProcessBackend(workdir=str(tmp_path), network="deny", network_isolation="seccomp")
        r = sb.run("curl -s http://127.0.0.1:1", shell=True)
        assert r.returncode == 126

    def test_echo_still_works(self, tmp_path):
        """Positive control: non-network commands are unaffected by the filter."""
        sb = RestrictedProcessBackend(workdir=str(tmp_path), network="deny", network_isolation="seccomp")
        r = sb.run("echo ok", shell=True)
        assert r.returncode == 0
        assert "ok" in r.stdout
