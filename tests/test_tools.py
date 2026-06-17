"""Tests for koboi.tools.registry module."""
from __future__ import annotations

from koboi.tools.registry import ToolRegistry, tool, register_decorated
from koboi.types import RiskLevel


class TestToolRegistry:
    async def test_register_and_execute(self):
        registry = ToolRegistry()
        registry.register("echo", "Echo input", {"type": "object", "properties": {"msg": {"type": "string"}}}, lambda msg: msg)
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
            "test", "test",
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
            "slow", "slow tool", {},
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
