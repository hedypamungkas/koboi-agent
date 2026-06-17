"""koboi/plugins.py -- Plugin discovery via entry_points.

External packages can register providers, guardrails, scorers, and tools
by declaring entry points in their pyproject.toml:

    [project.entry-points."koboi.providers"]
    my_provider = "my_package.llm:register"

    [project.entry-points."koboi.guardrails"]
    my_guardrail = "my_package.guardrails:register"

    [project.entry-points."koboi.scorers"]
    my_scorer = "my_package.scorers:register"

    [project.entry-points."koboi.tools"]
    my_tool = "my_package.tools:register"

Each entry point should be a callable that takes no arguments and registers
itself with the appropriate registry.
"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points

_logger = logging.getLogger(__name__)

# Entry point group names
_GROUPS = (
    "koboi.providers",
    "koboi.guardrails",
    "koboi.scorers",
    "koboi.tools",
)


def discover_plugins() -> dict[str, list[str]]:
    """Discover and load all registered plugins.

    Returns a dict mapping group name to list of loaded plugin names.
    """
    loaded: dict[str, list[str]] = {g: [] for g in _GROUPS}

    for group in _GROUPS:
        try:
            eps = entry_points(group=group)
        except TypeError:
            # Python 3.9 compat: entry_points() returns dict
            eps = entry_points().get(group, [])  # type: ignore[assignment]

        for ep in eps:
            try:
                factory = ep.load()
                factory()
                loaded[group].append(ep.name)
                _logger.info("Loaded plugin '%s' from group '%s'", ep.name, group)
            except Exception:
                _logger.warning("Failed to load plugin '%s' from group '%s'", ep.name, group, exc_info=True)

    return loaded
