"""koboi/tools/registry.py -- Tool registry with async execution."""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from collections.abc import Callable
from typing import Any

from koboi.types import ToolDefinition, RiskLevel


class ToolRegistry:
    def __init__(self, default_timeout: float | None = None):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self._default_timeout = default_timeout
        self._tool_defaults: dict = {}
        self._tool_overrides: dict = {}
        self._deps: dict[str, Any] = {}
        self._active_groups: list[str] | None = None

    def list_tools(self) -> dict[str, ToolDefinition]:
        """Return a copy of registered tools (name -> ToolDefinition)."""
        return dict(self._tools)

    def set_tool_config(self, defaults: dict, overrides: dict) -> None:
        self._tool_defaults = defaults or {}
        self._tool_overrides = overrides or {}

    def get_tool_config(self, tool_name: str) -> dict:
        merged = dict(self._tool_defaults)
        merged.update(self._tool_overrides.get(tool_name, {}))
        return merged

    def set_dep(self, name: str, value: Any) -> None:
        """Set a named dependency that tool closures can access."""
        self._deps[name] = value

    def get_dep(self, name: str) -> Any:
        """Get a named dependency. Returns None if not set."""
        return self._deps.get(name)

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: Callable,
        risk_level: RiskLevel = RiskLevel.SAFE,
        timeout: float | None = None,
        group: str | None = None,
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            risk_level=risk_level,
            timeout=timeout,
            group=group,
        )
        self._handlers[name] = fn

    def set_active_groups(self, groups: list[str] | None) -> None:
        """Set which tool groups are exposed to the LLM.

        When set to None (default), all tools are shown.
        When set to a list of group names, only tools belonging to those groups
        are included in get_definitions(). Tools outside active groups remain
        registered for execution but are not presented to the LLM.
        """
        self._active_groups = groups

    def keep_only(self, names: list[str]) -> None:
        """Remove all tools except those in the given names list."""
        name_set = set(names)
        self._tools = {k: v for k, v in self._tools.items() if k in name_set}
        self._handlers = {k: v for k, v in self._handlers.items() if k in name_set}

    def get_definitions(self) -> list[dict]:
        tools = self._tools.values()
        # Filter by active groups if set (non-destructive)
        if self._active_groups is not None:
            tools = [t for t in tools if t.group is None or t.group in self._active_groups]
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def get_risk_level(self, name: str) -> RiskLevel | None:
        if name in self._tools:
            return self._tools[name].risk_level
        return None

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    async def execute(self, name: str, arguments_json: str) -> str:
        if name not in self._handlers:
            return f"Error: tool '{name}' not found"

        try:
            args = json.loads(arguments_json)
        except json.JSONDecodeError as e:
            return f"Error: invalid arguments JSON: {e}"

        # Strip unknown params
        schema_props = self._tools[name].parameters.get("properties", {})
        if isinstance(args, dict) and schema_props:
            args = {k: v for k, v in args.items() if k in schema_props}

        effective_timeout = (
            self._tools[name].timeout if self._tools[name].timeout is not None else self._default_timeout
        )

        try:
            handler = self._handlers[name]

            if asyncio.iscoroutinefunction(handler):
                coro = handler(**args)
            else:
                coro = asyncio.to_thread(handler, **args)

            if effective_timeout is not None:
                result = await asyncio.wait_for(coro, timeout=effective_timeout)
            else:
                result = await coro if asyncio.iscoroutinefunction(handler) else await coro

            return str(result)
        except asyncio.TimeoutError:
            return f"Error: tool '{name}' timed out after {effective_timeout}s"
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            tb = traceback.format_exc() if os.environ.get("KOBOI_VERBOSE") else ""
            return f"Error executing '{name}': {e}" + (f"\n{tb}" if tb else "")


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending a truncation notice if shortened."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (output truncated, total {len(text)} chars)"


def tool(
    name: str,
    description: str,
    parameters: dict,
    risk_level: RiskLevel = RiskLevel.SAFE,
    timeout: float | None = None,
    group: str | None = None,
    deps: list[str] | None = None,
):
    """Decorator to register a function as a tool.

    Args:
        group: Tool group name for namespace filtering (e.g., "math", "file", "web").
        deps: List of dependency names this tool requires. At registration time,
              the tool is wrapped with a closure that injects these dependencies
              from the registry's dep store as a ``_deps`` dict parameter.
    """

    def decorator(fn: Callable) -> Callable:
        fn._tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            risk_level=risk_level,
            timeout=timeout,
            group=group,
        )
        fn._tool_deps = deps or []
        return fn

    return decorator


def _wrap_with_deps(
    fn: Callable,
    dep_names: list[str],
    registry: ToolRegistry,
) -> Callable:
    """Wrap a tool function to inject dependencies and config from the registry at call time."""
    import functools
    import inspect

    tool_name = fn._tool_def.name
    sig = inspect.signature(fn)
    accepts_config = "_tool_config" in sig.parameters

    if not dep_names and not accepts_config:
        return fn

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if dep_names:
                kwargs["_deps"] = {d: registry.get_dep(d) for d in dep_names}
            if accepts_config:
                kwargs["_tool_config"] = registry.get_tool_config(tool_name)
            return await fn(*args, **kwargs)

        return async_wrapper
    else:

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if dep_names:
                kwargs["_deps"] = {d: registry.get_dep(d) for d in dep_names}
            if accepts_config:
                kwargs["_tool_config"] = registry.get_tool_config(tool_name)
            return fn(*args, **kwargs)

        return sync_wrapper


def register_decorated(registry: ToolRegistry, module: Any) -> None:
    """Scan a module for functions with @tool decorator and register them."""
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if callable(obj) and hasattr(obj, "_tool_def"):
            td = obj._tool_def
            dep_names = getattr(obj, "_tool_deps", [])
            fn = _wrap_with_deps(obj, dep_names, registry)
            registry.register(
                td.name, td.description, td.parameters, fn, risk_level=td.risk_level, timeout=td.timeout, group=td.group
            )
