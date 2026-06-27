"""koboi/sandbox/passthrough -- Default backend, preserves pre-P0b behavior."""

from __future__ import annotations

import os
import subprocess

from koboi.sandbox.base import BaseSandbox, SandboxResult


class PassthroughBackend(BaseSandbox):
    """Behavior-preserving default sandbox.

    When no ``sandbox:`` section is configured (or ``backend: passthrough``),
    subprocess tools behave exactly as before P0b. The legacy
    ``KOBOI_SANDBOX_DIR`` env var is still honored for filesystem containment
    (back-compat) so existing user setups keep working.
    """

    name = "passthrough"

    def __init__(self):
        self._legacy_sandbox_dir = os.environ.get("KOBOI_SANDBOX_DIR")

    def run(
        self,
        command,
        *,
        cwd=None,
        env=None,
        timeout=None,
        shell=False,
    ) -> SandboxResult:
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or None,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(returncode=-1, stdout="", stderr="", timed_out=True)
        return SandboxResult(
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            timed_out=False,
        )

    def validate_path(self, path: str) -> str:
        # Exact reproduction of the pre-P0b filesystem._validate_path behavior.
        resolved = os.path.realpath(path)
        if self._legacy_sandbox_dir is None:
            return resolved
        sandbox = os.path.realpath(self._legacy_sandbox_dir)
        if not (resolved.startswith(sandbox + os.sep) or resolved == sandbox):
            raise PermissionError(f"Path '{path}' is outside the sandbox directory")
        return resolved

    def build_env(self, tool_config: dict | None = None) -> dict[str, str]:
        from koboi.harness.env import build_safe_env

        return build_safe_env(tool_config)
