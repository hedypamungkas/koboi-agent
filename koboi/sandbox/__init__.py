"""koboi/sandbox -- Pluggable subprocess/filesystem isolation backends.

A sandbox wraps subprocess execution and path validation for the built-in
shell/git/filesystem tools. The default
:class:`~koboi.sandbox.passthrough.PassthroughBackend` preserves pre-P0b
behavior exactly;
:class:`~koboi.sandbox.restricted.RestrictedProcessBackend` adds
cwd/env/PATH/network/rlimit containment without a container. Docker (P0c) is
deferred.

Configure via the top-level ``sandbox:`` YAML section::

    sandbox:
      backend: restricted
      workdir: ./workspace
      network: deny
      rlimits: {cpu: 30, fsize_mb: 50}

When the section is absent, agents keep their current (passthrough) behavior.
"""

from __future__ import annotations

from koboi.sandbox.base import BaseSandbox, SandboxResult
from koboi.sandbox.passthrough import PassthroughBackend
from koboi.sandbox.restricted import RestrictedProcessBackend
from koboi.sandbox.registry import (
    build_sandbox,
    register_builtin_sandboxes,
    register_sandbox,
    sandbox_registry,
)

# Register shipped backends at import time (mirrors koboi/guardrails/__init__.py).
register_builtin_sandboxes()

__all__ = [
    "BaseSandbox",
    "SandboxResult",
    "PassthroughBackend",
    "RestrictedProcessBackend",
    "sandbox_registry",
    "register_sandbox",
    "build_sandbox",
]
