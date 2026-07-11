"""koboi/tools/builtin -- Built-in tool implementations."""

from koboi.tools.registry import ToolRegistry, register_decorated


def register_all(registry: ToolRegistry) -> None:
    from koboi.tools.builtin import (
        calculator,
        filesystem,
        git,
        ingest,
        memory,
        search,
        shell,
        subagent,
        task,
        web,
    )

    for mod in [calculator, filesystem, shell, web, memory, search, git, subagent, task, ingest]:
        register_decorated(registry, mod)
