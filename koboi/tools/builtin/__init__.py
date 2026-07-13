"""koboi/tools/builtin -- Built-in tool implementations."""

from koboi.tools.registry import ToolRegistry, register_decorated


def register_all(registry: ToolRegistry) -> None:
    from koboi.tools.builtin import calculator, filesystem, shell, web, memory, search, git, subagent, task, handover

    for mod in [calculator, filesystem, shell, web, memory, search, git, subagent, task, handover]:
        register_decorated(registry, mod)
