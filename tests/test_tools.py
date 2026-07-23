"""Tests for koboi.tools.registry module."""

from __future__ import annotations

from koboi.tools.registry import ToolRegistry, tool, register_decorated, apply_tool_selection
from koboi.types import RiskLevel


class TestToolRegistry:
    async def test_register_and_execute(self):
        registry = ToolRegistry()
        registry.register(
            "echo", "Echo input", {"type": "object", "properties": {"msg": {"type": "string"}}}, lambda msg: msg
        )
        result = await registry.execute("echo", '{"msg": "hello"}')
        assert result == "hello"

    def test_get_definitions(self):
        registry = ToolRegistry()
        registry.register("test", "A test", {"type": "object", "properties": {}}, lambda: "ok")
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "test"

    async def test_tool_not_found(self):
        registry = ToolRegistry()
        result = await registry.execute("missing", "{}")
        assert "not found" in result

    async def test_invalid_json_args(self):
        registry = ToolRegistry()
        registry.register("test", "test", {"type": "object", "properties": {}}, lambda: "ok")
        result = await registry.execute("test", "not json")
        assert "Error" in result

    async def test_unknown_params_stripped(self):
        calls = []
        registry = ToolRegistry()
        registry.register(
            "test",
            "test",
            {"type": "object", "properties": {"a": {"type": "string"}}},
            lambda a: calls.append(a) or f"got {a}",
        )
        await registry.execute("test", '{"a": "val", "b": "extra"}')
        assert calls == ["val"]

    def test_risk_level(self):
        registry = ToolRegistry()
        registry.register("safe", "safe", {}, lambda: "", risk_level=RiskLevel.SAFE)
        registry.register("destructive", "destructive", {}, lambda: "", risk_level=RiskLevel.DESTRUCTIVE)
        assert registry.get_risk_level("safe") == RiskLevel.SAFE
        assert registry.get_risk_level("destructive") == RiskLevel.DESTRUCTIVE
        assert registry.get_risk_level("missing") is None

    async def test_timeout(self):
        import time

        registry = ToolRegistry()
        registry.register(
            "slow",
            "slow tool",
            {},
            lambda: time.sleep(10),
            timeout=0.1,
        )
        result = await registry.execute("slow", "{}")
        assert "timed out" in result


class TestToolDecorator:
    def test_decorator(self):
        @tool(name="my_tool", description="My tool", parameters={"type": "object", "properties": {}})
        def my_fn():
            return "ok"

        assert hasattr(my_fn, "_tool_def")
        assert my_fn._tool_def.name == "my_tool"

    async def test_register_decorated(self):
        import types

        mod = types.ModuleType("test_mod")

        @tool(name="mod_tool", description="Module tool", parameters={"type": "object", "properties": {}})
        def mod_fn():
            return "from module"

        mod.mod_fn = mod_fn

        registry = ToolRegistry()
        register_decorated(registry, mod)
        result = await registry.execute("mod_tool", "{}")
        assert result == "from module"


class TestToolSelection:
    """Tests for disable / groups / override-alias selection (P3g)."""

    def _registry_with(self, *specs):
        """Build a registry from (name, group) tuples; each tool returns '<name>-ran'."""
        registry = ToolRegistry()
        for name, group in specs:
            registry.register(
                name,
                f"{name} tool",
                {"type": "object", "properties": {}},
                (lambda captured=name: f"{captured}-ran"),
                group=group,
            )
        return registry

    def test_disable_removes_from_definitions(self):
        registry = self._registry_with(("a", None), ("b", None))
        registry.disable(["a"])
        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert names == {"b"}

    async def test_disable_removes_from_execute(self):
        registry = self._registry_with(("a", None))
        registry.disable(["a"])
        result = await registry.execute("a", "{}")
        assert "not found" in result

    def test_disable_clears_overrides(self):
        registry = self._registry_with(("a", None))
        registry.set_tool_config({}, {"a": {"timeout": 9}})
        registry.disable(["a"])
        # re-register a with the same name -- stale override must not leak back
        registry.register("a", "a tool", {"type": "object", "properties": {}}, lambda: "ok")
        assert "timeout" not in registry.get_tool_config("a")

    def test_disable_unknown_name_non_fatal(self):
        registry = self._registry_with(("a", None))
        registry.disable(["ghost"])  # must not raise
        assert "a" in registry

    def test_apply_tool_selection_disable(self):
        registry = self._registry_with(("a", None), ("b", None), ("c", None))
        apply_tool_selection(registry, {"disabled": ["b"]})
        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert names == {"a", "c"}

    def test_apply_tool_selection_disable_alias(self):
        """Alias keys in 'disabled' must resolve (shell -> run_shell), like overrides."""
        registry = self._registry_with(("run_shell", None), ("calc", None))
        apply_tool_selection(registry, {"disabled": ["shell"]})
        assert "run_shell" not in registry
        assert "calc" in registry

    async def test_apply_tool_selection_groups_hides_but_keeps_executable(self):
        registry = self._registry_with(("calc", "math"), ("search", "web"))
        apply_tool_selection(registry, {"groups": ["math"]})
        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert names == {"calc"}
        # hidden from the LLM, but still callable programmatically
        result = await registry.execute("search", "{}")
        assert result == "search-ran"

    def test_apply_tool_selection_order_disable_then_groups(self):
        registry = self._registry_with(("a", "g1"), ("b", "g1"), ("c", "g2"))
        apply_tool_selection(registry, {"disabled": ["b"], "groups": ["g1"]})
        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert names == {"a"}  # b disabled, c hidden by groups
        assert "b" not in registry

    def test_apply_tool_selection_overrides_alias(self):
        registry = self._registry_with(("run_shell", None))
        apply_tool_selection(registry, {"overrides": {"shell": {"timeout": 99}}})
        assert registry.get_tool_config("run_shell").get("timeout") == 99

    def test_apply_tool_selection_git_ambiguous_warns(self, caplog):
        import logging

        registry = self._registry_with(("git_status", None), ("git_log", None))
        with caplog.at_level(logging.WARNING):
            apply_tool_selection(registry, {"overrides": {"git": {"timeout": 5}}})
        assert any("ambiguous" in r.message for r in caplog.records)
        # not applied to any git_* tool
        assert "timeout" not in registry.get_tool_config("git_status")

    def test_apply_tool_selection_none_noop(self):
        registry = self._registry_with(("a", None))
        apply_tool_selection(registry, None)
        assert "a" in registry

    def test_apply_tool_selection_empty_noop(self):
        registry = self._registry_with(("a", None))
        apply_tool_selection(registry, {})
        assert "a" in registry


class TestBuiltinRegistration:
    """Guard against a builtin tool module being added but NOT registered.

    Regression for run_typecheck (Wave 2.4): the module existed and the tool
    carried @tool metadata, but was missing from register_all() -> the tool was
    dead in production (never in the LLM spec, never executable). apply_patch
    is covered too (it lives in filesystem.py which IS registered, but assert
    both to be safe).
    """

    def test_apply_patch_and_run_typecheck_register(self):
        from koboi.tools.builtin import register_all

        registry = ToolRegistry()
        register_all(registry)
        for name in ("apply_patch", "run_typecheck"):
            assert name in registry, f"{name} not registered by register_all()"
