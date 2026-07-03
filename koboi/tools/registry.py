"""koboi/tools/registry.py -- Tool registry with async execution.

Tool selection semantics (applied in order during _build_tools via
``apply_tool_selection``):

  - tools.builtin   : ALLOWLIST. Applied via keep_only() BEFORE other
                      selection. If set, only listed tools remain.
  - tools.disabled  : DENYLIST. disable() removes the tool entirely from
                      BOTH the LLM view (get_definitions) and execution
                      (execute). The tool ceases to exist for this agent.
  - tools.groups    : HIDE FROM LLM. set_active_groups() filters
                      get_definitions() but the tool stays executable -- handy
                      for tools you invoke programmatically without advertising
                      them to the model.
  - tools.overrides : PER-TOOL CONFIG. Merged onto defaults at call time via
                      get_tool_config(). Keys are canonical tool names; the
                      legacy 1:1 alias 'shell' is mapped to 'run_shell'. 'git'
                      is N:1 (git_status/git_log/git_diff) and therefore
                      ambiguous -- it is warned and dropped.
  - policy.rules    : RUNTIME BLOCK. PolicyHook evaluates per-invocation and can
                      deny/confirm a specific call. Orthogonal to all of the above.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from collections.abc import Callable
from typing import Any

from koboi.types import ToolDefinition, RiskLevel

# Short YAML override keys -> canonical registered tool names.
# Only 1:1 aliases are safe here; N:1 (e.g. git -> git_status/git_log/git_diff)
# is intentionally NOT aliased because the target is ambiguous.
_TOOL_CONFIG_ALIASES: dict[str, str] = {
    "shell": "run_shell",
}


def _resolve_tool_config_key(tool_name: str) -> str:
    """Resolve short YAML override keys (e.g. 'shell') to canonical names."""
    return _TOOL_CONFIG_ALIASES.get(tool_name, tool_name)


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

    def disable(self, names: list[str]) -> None:
        """Remove specific tools entirely from both LLM view and execution.

        Unlike set_active_groups (which only hides from the LLM), disable()
        makes a tool cease to exist: get_definitions() will not list it and
        execute() will return 'tool not found'. A denylist complement to
        keep_only -- useful to drop one tool from an otherwise-full set
        (e.g. ``disabled: [run_shell]``) without enumerating the rest.

        Any matching per-tool overrides are cleared so stale config does not
        leak into get_tool_config() for a tool that no longer exists.
        Unknown names are logged at debug level and skipped (non-fatal) so
        disabling an optional tool that failed to register does not crash.
        """
        import logging

        _log = logging.getLogger(__name__)
        for n in names:
            if n in self._tools:
                del self._tools[n]
                self._handlers.pop(n, None)
                self._tool_overrides.pop(n, None)
            else:
                _log.debug("disable: tool '%s' not registered, skipping", n)

    def get_definitions(self) -> list[dict]:
        tools = list(self._tools.values())
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
        td = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            risk_level=risk_level,
            timeout=timeout,
            group=group,
        )
        fn._tool_def = td  # type: ignore[attr-defined]  # attrs attached by the @tool decorator
        fn._tool_deps = deps or []  # type: ignore[attr-defined]
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

    tool_name = fn._tool_def.name  # type: ignore[attr-defined]  # set by the @tool decorator
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


def apply_tool_selection(registry: ToolRegistry, tools_config: dict | None) -> ToolRegistry:
    """Apply ``tools.*`` selection/config keys to a registry.

    Single source of truth for tool selection, called from both the facade
    (``_build_tools``) and the orchestration factory (``_build_tools_from_config``)
    so the two paths never drift. Call order is enforced here:

      1. ``set_tool_config(defaults, overrides)`` -- short alias keys normalized
         to canonical names so legacy YAML (``shell:``) still resolves.
      2. ``disable(names)`` -- denylist, removes from LLM view AND execution.
      3. ``set_active_groups(groups)`` -- hides from LLM view, keeps executable.

    Preconditions: the caller has already run ``register_all()``,
    ``keep_only(builtin_list)``, and any custom-module registration. This
    helper only post-processes selection/config on an already-populated registry.

    Args:
        tools_config: the raw ``tools:`` config dict (may be None/empty).

    Returns:
        The same registry (for chaining).
    """
    if not tools_config:
        return registry

    defaults = tools_config.get("defaults") or {}
    raw_overrides = tools_config.get("overrides") or {}

    # 'git' is N:1 (git_status/git_log/git_diff) -> ambiguous, warn and drop
    # rather than guess which tool the override was meant for.
    if "git" in raw_overrides:
        import logging

        logging.getLogger(__name__).warning(
            "tools.overrides.git is ambiguous (git_status/git_log/git_diff); "
            "use explicit per-tool keys. This override will be ignored."
        )
        raw_overrides = {k: v for k, v in raw_overrides.items() if k != "git"}

    # Normalize short alias keys (shell -> run_shell) to canonical names.
    overrides = {_resolve_tool_config_key(k): v for k, v in raw_overrides.items()}
    if defaults or overrides:
        registry.set_tool_config(defaults, overrides)

    # Mirror the defaults onto the env-hygiene module config so subprocess tools
    # (which read _tool_config) AND skill-shell preprocessing (which has none)
    # share one config. Called here -- the single chokepoint for both the facade
    # and orchestration build paths -- so neither path forgets to set it.
    # NOTE: module-global; in a multi-agent process the last-built config wins.
    from koboi.harness.env import configure_env_defaults

    configure_env_defaults(defaults)

    disabled = tools_config.get("disabled") or []
    if isinstance(disabled, list) and disabled:
        # Normalize alias keys (shell -> run_shell) so 'disabled: [shell]' works,
        # matching the override-key normalization above. Without this, disabling
        # via the documented alias would silently no-op -- bad for a security knob.
        registry.disable([_resolve_tool_config_key(n) for n in disabled])

    groups = tools_config.get("groups")
    if groups is not None:
        registry.set_active_groups(groups)

    return registry
