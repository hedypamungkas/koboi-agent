"""Regression test for the idempotent-forwarding fix (S9).

``register_decorated`` previously dropped ``idempotent`` when calling ``register()``,
so ``@tool(idempotent=False)`` was silently ignored for every builtin. After the fix,
a tool that declares ``idempotent=False`` lands with ``idempotent=False``, while tools
that leave it at default stay ``True`` (behavior unchanged).
"""

from __future__ import annotations

from koboi.tools.builtin import register_all
from koboi.tools.registry import ToolRegistry


class TestIdempotentForwarding:
    def test_call_peer_agent_is_not_idempotent(self):
        r = ToolRegistry()
        register_all(r)
        assert r.get_definition("call_peer_agent").idempotent is False

    def test_default_tools_stay_idempotent(self):
        """Read-only/safe builtins keep re-running on resume (idempotent=True by default)."""
        r = ToolRegistry()
        register_all(r)
        for name in ("delegate_tasks", "calculate", "read_file"):
            td = r.get_definition(name)
            assert td is not None, f"{name} missing"
            assert td.idempotent is True, f"{name} should remain idempotent=True"

    def test_destructive_shipped_builtins_are_not_idempotent(self):
        """PR #48 (main): DESTRUCTIVE/side-effecting builtins must not double-execute on resume."""
        r = ToolRegistry()
        register_all(r)
        for name in ("run_shell", "write_file", "delete_file"):
            td = r.get_definition(name)
            assert td is not None, f"{name} missing"
            assert td.idempotent is False, f"{name} should be idempotent=False"
