"""koboi/tools/builtin -- Built-in tool implementations."""

from koboi.tools.registry import ToolRegistry, register_decorated


def register_all(registry: ToolRegistry) -> None:
    from koboi.tools.builtin import (
        background_shell,
        calculator,
        filesystem,
        git,
        github,
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
        typecheck,
        web,
        bitbucket,
    )

    for mod in [
        background_shell,
        calculator,
        filesystem,
        git,
        github,
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
        typecheck,
        web,
        bitbucket,
    ]:
        register_decorated(registry, mod)
