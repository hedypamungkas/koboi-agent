"""koboi/sandbox/base -- Sandbox backend ABC + SandboxResult."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """Outcome of a sandboxed subprocess run."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class BaseSandbox(ABC):
    """Sandbox backend interface for subprocess + filesystem tools.

    Tools obtain their handle via ``_deps["sandbox"]`` (declared with
    ``deps=["sandbox"]`` on the ``@tool`` decorator). Two implementations ship:
    :class:`~koboi.sandbox.passthrough.PassthroughBackend` (default,
    behavior-preserving) and
    :class:`~koboi.sandbox.restricted.RestrictedProcessBackend`
    (cwd/env/PATH/network/rlimit containment).

    The sandbox is the *during-execution* isolation layer; the policy engine
    (``koboi.harness.policy``) stays the *pre-execution* deny layer. Both run.
    """

    name: str = "base"

    @abstractmethod
    def run(
        self,
        command: str | list[str],
        *,
        cwd: str | None = None,
        env: dict | None = None,
        timeout: float | None = None,
        shell: bool = False,
        input: str | None = None,
    ) -> SandboxResult:
        """Run ``command`` and return a :class:`SandboxResult`.

        ``shell=True`` supports pipe/redirect chaining (string command);
        ``shell=False`` runs an argv list (used by git). ``input`` (when given)
        is fed to the child's stdin as text -- used by command hooks to pass a
        JSON payload (see :class:`koboi.hooks.command_hook.CommandHook`).
        """

    @abstractmethod
    def validate_path(self, path: str) -> str:
        """Resolve ``path`` and enforce containment. Returns the resolved path.

        Raises ``PermissionError`` if the path escapes the sandbox root.
        """

    @abstractmethod
    def build_env(self, tool_config: dict | None = None) -> dict[str, str]:
        """Return an environment dict for ``subprocess.run(env=...)``.

        Delegates secret hygiene to ``koboi.harness.env.build_safe_env``.
        """

    def network_allowed(self, command: str) -> bool:
        """Soft signal: does ``command`` look network-capable?

        Default permits; restricted backends override to flag network binaries.
        Enforcement is best-effort and documented as a soft boundary.
        """
        return True
