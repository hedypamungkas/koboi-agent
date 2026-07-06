"""koboi/sandbox/restricted -- Process-level isolation without a container.

Enforces (best-effort, no root required):
  - cwd containment via :meth:`validate_path` (symlink-safe realpath check);
  - env hygiene via ``build_safe_env`` plus proxy/NETRC stripping;
  - PATH restriction to a safe-bin allowlist;
  - soft network-binary denial (command-token scan);
  - rlimits (RLIMIT_CPU/AS/FSIZE/NOFILE) applied in the *child* via preexec_fn;
  - wall-clock timeout with full process-group kill (``start_new_session``).

This is a SOFT boundary by default: it raises the bar for accidental exfiltration
but cannot stop a determined attacker (e.g. ``python3 -c 'import urllib'`` or
``bash -c 'echo > /dev/tcp/...'``). For HARD network isolation on Linux set
``network_isolation: seccomp`` (requires the ``python3-seccomp`` system package) -- the
seccomp filter blocks egress at the syscall layer so interpreters and shell
builtins cannot connect out. For full OS-level isolation (filesystem too), use
the Docker backend (P0c).

Rlimit correctness: ``setrlimit`` affects the calling process, so we apply it
in ``preexec_fn`` (runs in the child between fork and exec), NOT in the worker
thread or parent -- otherwise the host agent's own resources would be capped.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys

from koboi.sandbox.base import BaseSandbox, SandboxResult
from koboi.tools.registry import truncate_text

_logger = logging.getLogger(__name__)

# M10: warn once per process that the restricted sandbox's network denial is a
# soft boundary (see module docstring) -- real isolation needs the Docker backend.
_network_soft_boundary_warned = False

# Network-capable binaries denied at the command-token layer (soft boundary).
# True network isolation requires a container (P0c); this list blocks the
# obvious egress tools so a model can't trivially phone home.
DEFAULT_NETWORK_BINARIES: tuple[str, ...] = (
    "curl",
    "wget",
    "nc",
    "ncat",
    "ssh",
    "scp",
    "sftp",
    "telnet",
    "ftp",
    "ftpget",
)

# Env vars stripped on top of the secret block-list when network is denied:
# proxy settings and per-tool netrc/credential files.
NETWORK_ENV_BLOCKLIST: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
    "NETRC",
    "CURL_HOME",
    "WGETRC",
    "REQUESTS_CA_BUNDLE",
)

DEFAULT_SAFE_PATH_DIRS: tuple[str, ...] = (
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/opt/homebrew/bin",
)

# POSIX-only: rlimits need the ``resource`` module + preexec_fn.
try:  # pragma: no cover - platform guard
    import resource as _resource  # type: ignore[import-not-found]

    _HAS_RLIMIT = True
except ImportError:  # Windows
    _resource = None  # type: ignore[assignment]
    _HAS_RLIMIT = False

_POSIX = hasattr(os, "setsid")

# seccomp (HARD network isolation): Linux-only + libseccomp binding via the
# ``seccomp`` module (provided by the ``python3-seccomp`` system package). When active it blocks
# egress at the syscall layer, so interpreters (python3 urllib) and shell
# builtins (bash /dev/tcp) -- which evade the token-scan soft layer -- cannot
# connect out. Filter is applied in the child via preexec_fn and persists across
# execve, so the exec'd binary inherits the deny list.
try:  # pragma: no cover - platform/dep guard
    import seccomp as _seccomp  # type: ignore[import-not-found]

    _HAS_SECCOMP_LIB = True
except ImportError:
    _seccomp = None  # type: ignore[assignment]
    _HAS_SECCOMP_LIB = False
_HAS_SECCOMP = _HAS_SECCOMP_LIB and sys.platform == "linux"

# Egress syscalls blocked under network_isolation="seccomp". Denying connect /
# connectat / sendto / sendmsg blocks TCP/UDP egress. v1 does NOT arg-filter by
# socket family, so AF_UNIX connect() is blocked too (over-blocking is safe for
# network=deny; add family arg-filtering later if local unix-socket IPC must work).
# socket() creation itself is allowed (harmless without connect). The action enum
# (seccomp.Action.ALLOW / .ERRNO) is resolved inside the child at install time so
# we use the canonical names of whatever python3-seccomp version is present (the
# libseccomp bindings ship as a system package, not on PyPI).
_SECCOMP_EGRESS_SYSCALLS: tuple[str, ...] = ("connect", "connectat", "sendto", "sendmsg")

# One-time warning when seccomp is requested but unavailable (non-Linux / system
# package python3-seccomp not installed) -- degrade to soft deny rather than crash.
_seccomp_unavailable_warned = False


def _make_preexec_fn(rlimits: dict, seccomp_preexec=None):
    """Return a preexec_fn that applies rlimits + seccomp filter in the child only.

    Running ``setrlimit`` / ``seccomp_preexec`` in preexec_fn (between fork and
    exec) confines the limits + filter to the subprocess; the host agent keeps
    its own. ``seccomp_preexec`` is a no-arg callable that builds AND loads the
    filter entirely in the child (canonical seccomp+subprocess pattern: no
    parent-built filter context crosses the fork, and the filter persists across
    the subsequent execve). Returns ``None`` when there is nothing to apply
    (keeps Popen kwargs clean).
    """
    rlimits = rlimits or {}
    has_rlimits = _HAS_RLIMIT and rlimits
    if not has_rlimits and seccomp_preexec is None:
        return None

    def _apply() -> None:
        if has_rlimits:
            if rlimits.get("cpu"):
                _resource.setrlimit(_resource.RLIMIT_CPU, (rlimits["cpu"], rlimits["cpu"]))
            if rlimits.get("as_mb"):
                bytes_ = int(rlimits["as_mb"]) * 1024 * 1024
                _resource.setrlimit(_resource.RLIMIT_AS, (bytes_, bytes_))
            if rlimits.get("fsize_mb"):
                bytes_ = int(rlimits["fsize_mb"]) * 1024 * 1024
                _resource.setrlimit(_resource.RLIMIT_FSIZE, (bytes_, bytes_))
            if rlimits.get("nofile"):
                _resource.setrlimit(_resource.RLIMIT_NOFILE, (rlimits["nofile"], rlimits["nofile"]))
        if seccomp_preexec is not None:
            seccomp_preexec()

    return _apply


class RestrictedProcessBackend(BaseSandbox):
    """Containment without a container. See module docstring for guarantees."""

    name = "restricted"

    def __init__(
        self,
        *,
        workdir: str = ".",
        network: str = "deny",
        network_binaries: list[str] | None = None,
        network_isolation: str | None = None,
        safe_path: list[str] | None = None,
        env_passthrough: bool = False,
        rlimits: dict | None = None,
        timeout: float = 30.0,
        max_output: int = 10000,
    ):
        self._workdir = os.path.realpath(workdir)
        self._network = network
        self._network_isolation = network_isolation
        # HARD network isolation via seccomp when requested + available. Returns a
        # preexec_fn callable that builds+loads the filter in the forked child, or
        # None (with a warning) if requested but unavailable.
        self._seccomp_preexec = self._build_seccomp_preexec()
        # M10: one-time SOFT-boundary warning -- only when hard isolation is NOT
        # active (seccomp off/unavailable). When seccomp is active the network
        # boundary is HARD for the blocked syscall set, so no soft caveat applies.
        global _network_soft_boundary_warned
        if network == "deny" and self._seccomp_preexec is None and not _network_soft_boundary_warned:
            _network_soft_boundary_warned = True
            _logger.warning(
                "sandbox.backend='restricted' network=deny is a SOFT boundary -- it "
                "blocks obvious egress tools but not interpreters (e.g. python3 -c "
                "'import urllib'). For HARD network isolation set "
                "sandbox.network_isolation: seccomp on a Linux host with the "
                "python3-seccomp system package, or use the Docker backend (P0c)."
            )
        self._network_binaries = set(network_binaries) if network_binaries else set(DEFAULT_NETWORK_BINARIES)
        self._safe_path = list(safe_path) if safe_path else list(DEFAULT_SAFE_PATH_DIRS)
        self._env_passthrough = env_passthrough
        self._rlimits = dict(rlimits) if rlimits else {}
        self._timeout = timeout
        self._max_output = max_output

    # -- public API --------------------------------------------------------

    def run(self, command, *, cwd=None, env=None, timeout=None, shell=False) -> SandboxResult:
        effective_timeout = timeout if timeout is not None else self._timeout

        # 1. Resolve cwd with containment (caller's cwd or the workdir root).
        try:
            resolved_cwd = self.validate_path(cwd) if cwd else self._workdir
        except PermissionError as exc:
            return SandboxResult(returncode=126, stdout="", stderr=str(exc), timed_out=False)

        # 2. Env: caller may pass a pre-built one (e.g. shell NODE_PATH augment).
        run_env = env if env is not None else self.build_env()

        # 3. Soft network deny: scan command tokens for network binaries.
        if self._network == "deny" and not self.network_allowed(str(command)):
            blocked = self._first_network_binary(str(command))
            return SandboxResult(
                returncode=126,
                stdout="",
                stderr=f"blocked: network binary '{blocked}' is not permitted in the restricted sandbox",
                timed_out=False,
            )

        return self._run_subprocess(command, resolved_cwd, run_env, effective_timeout, shell)

    def validate_path(self, path: str) -> str:
        # Anchor RELATIVE paths to the workdir before resolving. A tool passing
        # "hello.txt" means "inside my workdir" — but os.path.realpath() alone
        # resolves it against the server process's cwd (typically /app), which
        # is always outside the workdir, so every relative write_file/read_file
        # was wrongly rejected with "no access".
        if not os.path.isabs(path):
            path = os.path.join(self._workdir, path)
        resolved = os.path.realpath(path)
        if resolved == self._workdir or resolved.startswith(self._workdir + os.sep):
            return resolved
        raise PermissionError(f"Path '{path}' (resolved '{resolved}') is outside the sandbox workdir '{self._workdir}'")

    def build_env(self, tool_config: dict | None = None) -> dict[str, str]:
        from koboi.harness.env import build_safe_env

        cfg = dict(tool_config or {})
        cfg.setdefault("env_passthrough", self._env_passthrough)
        env = build_safe_env(cfg)

        if self._network == "deny":
            for k in NETWORK_ENV_BLOCKLIST:
                env.pop(k, None)
        # Restrict PATH to the safe-bin allowlist (best-effort lookup isolation).
        env["PATH"] = os.pathsep.join(self._safe_path)
        return env

    def network_allowed(self, command: str) -> bool:
        if self._network != "deny" or not self._network_binaries:
            return True
        return self._first_network_binary(command) is None

    # -- internals ---------------------------------------------------------

    def _first_network_binary(self, command) -> str | None:
        if isinstance(command, str):
            try:
                tokens = shlex.split(command)
            except ValueError:
                # Unbalanced quotes: fall back to a naive split on separators.
                tokens = command.replace(";", " ").replace("|", " ").split()
        else:
            tokens = list(command)
        for tok in tokens:
            base = os.path.basename(tok)
            if base in self._network_binaries:
                return base
        return None

    def _build_seccomp_preexec(self):
        """Return a preexec_fn callable that builds+loads the egress-deny filter in
        the forked child, or None when seccomp is off/unavailable.

        Activated only when ``network == "deny"`` AND
        ``network_isolation == "seccomp"`` AND seccomp is available (Linux host
        with the ``python3-seccomp`` system package). When requested but unavailable,
        logs a one-time warning and returns None so the backend degrades to the
        soft token-deny rather than crashing.

        The filter is built AND loaded entirely inside the child (between fork and
        exec) -- the canonical seccomp+subprocess pattern: no parent-built filter
        context crosses the fork, and the filter persists across the subsequent
        execve so the exec'd binary (python3/bash/curl) inherits the deny list.
        Blocks ``connect``/``connectat``/``sendto``/``sendmsg`` (TCP/UDP egress).
        """
        if not (self._network == "deny" and self._network_isolation == "seccomp"):
            return None
        if not _HAS_SECCOMP:
            global _seccomp_unavailable_warned
            if not _seccomp_unavailable_warned:
                _seccomp_unavailable_warned = True
                _logger.warning(
                    "sandbox.network_isolation='seccomp' requested but unavailable "
                    "(non-Linux host or python3-seccomp system package not installed); "
                    "falling back to SOFT network deny. Install on a Linux host with: "
                    "apt install python3-seccomp (Debian/Ubuntu)."
                )
            return None
        egress = _SECCOMP_EGRESS_SYSCALLS

        def _install() -> None:
            # Runs in the forked child (preexec_fn), before exec. Build + load
            # here so no parent-built context crosses the fork. NOTE: python3-seccomp
            # exposes actions as MODULE-LEVEL constants (seccomp.ALLOW / .ERRNO),
            # NOT under a seccomp.Action namespace; and ERRNO must be CONSTRUCTED
            # with an errno value: seccomp.ERRNO(errno.EPERM) -- the bare seccomp.ERRNO
            # raises TypeError("an integer is required").
            import errno as _errno
            import seccomp

            f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
            try:
                deny = seccomp.ERRNO(_errno.EPERM)
            except (TypeError, AttributeError):
                # Older/different bindings -- fall back to KILL (terminate on call).
                deny = getattr(seccomp, "KILL", None)
            if deny is not None:
                for sc in egress:
                    try:
                        f.add_rule(deny, sc)
                    except (RuntimeError, ValueError, TypeError):
                        # Syscall name not recognized on this kernel/arch -- best effort.
                        pass
            f.load()

        return _install

    def _run_subprocess(self, command, cwd, env, timeout, shell) -> SandboxResult:
        popen_kwargs: dict = {
            "shell": shell,
            "cwd": cwd,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        # start_new_session puts the child in its own process group so a timeout
        # can kill piped children (e.g. `sleep 30 | cat`). POSIX only.
        if _POSIX:
            popen_kwargs["start_new_session"] = True
        preexec = _make_preexec_fn(self._rlimits, self._seccomp_preexec)
        if preexec is not None:
            popen_kwargs["preexec_fn"] = preexec

        proc = subprocess.Popen(command, **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                stdout, stderr = ("", "killed after timeout")
            return SandboxResult(
                returncode=proc.returncode if proc.returncode is not None else -9,
                stdout=stdout or "",
                stderr=stderr or "",
                timed_out=True,
            )
        return SandboxResult(
            returncode=proc.returncode,
            stdout=truncate_text(stdout or "", self._max_output),
            stderr=stderr or "",
            timed_out=False,
        )

    @staticmethod
    def _kill_group(proc) -> None:
        # Prefer killing the whole process group (catches piped children).
        # Safe only because we set start_new_session on POSIX, so the group is
        # the child's own -- never the agent's. Fall back to the direct child.
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        proc.kill()
