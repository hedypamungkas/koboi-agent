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
        """Existing builtins keep re-running on resume (idempotent=True by default)."""
        r = ToolRegistry()
        register_all(r)
        for name in ("delegate_tasks", "calculate", "read_file", "run_shell"):
            td = r.get_definition(name)
            assert td is not None, f"{name} missing"
            assert td.idempotent is True, f"{name} should remain idempotent=True"
