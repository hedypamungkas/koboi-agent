"""koboi/tools/builtin -- Built-in tool implementations."""

from koboi.tools.registry import ToolRegistry, register_decorated


def register_all(registry: ToolRegistry) -> None:
    from koboi.tools.builtin import (
        calculator,
        filesystem,
        git,
        handover,
        ingest,
        media,
        memory,
        peer,
        repo_map,
        search,
        shell,
        subagent,
        task,
        web,
    )

    for mod in [
        calculator,
        filesystem,
        git,
        handover,
        ingest,
        media,
        memory,
        peer,
        repo_map,
        search,
        shell,
        subagent,
        task,
        web,
    ]:
        register_decorated(registry, mod)
