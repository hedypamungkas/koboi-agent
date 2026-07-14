"""koboi/guardrails/registry.py -- Guardrail factory registry."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from koboi.guardrails.base import BaseGuardrail

_logger = logging.getLogger(__name__)


class GuardrailRegistry:
    """Registry of named guardrail factories for config-driven composition."""

    _factories: dict[str, Callable[..., BaseGuardrail]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseGuardrail]) -> None:
        cls._factories[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseGuardrail:
        if name not in cls._factories:
            raise ValueError(f"Unknown guardrail '{name}'. Available: {cls.list_available()}")
        return cls._factories[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._factories.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered factories. Useful for test isolation."""
        cls._factories.clear()

    @classmethod
    def from_config(cls, slot_configs: list[dict[str, Any]]) -> list[BaseGuardrail]:
        """Build guardrail list from config dicts.

        Each dict must have a 'name' key. Remaining keys are passed as kwargs.
        Accepts both a list of dicts and a single dict (auto-wrapped).
        """
        if isinstance(slot_configs, dict):
            slot_configs = [slot_configs]

        guardrails: list[BaseGuardrail] = []
        for cfg in slot_configs:
            if not isinstance(cfg, dict):
                continue
            name = cfg.get("name")
            if not name:
                _logger.warning("Guardrail config missing 'name', skipping: %s", cfg)
                continue
            kwargs = {k: v for k, v in cfg.items() if k != "name"}
            try:
                guardrails.append(cls.create(name, **kwargs))
            except (ValueError, TypeError) as e:
                _logger.warning("Failed to create guardrail '%s': %s", name, e)
        return guardrails


def register_builtin_guardrails() -> None:
    """Register built-in guardrails. Called once at import time."""
    from koboi.guardrails.input import InputGuardrail
    from koboi.guardrails.output import OutputGuardrail

    GuardrailRegistry.register(
        "injection_detector",
        lambda **kw: InputGuardrail(**kw),
    )
    GuardrailRegistry.register(
        "content_filter",
        lambda **kw: OutputGuardrail(**kw),
    )
    # A3: runtime faithfulness guardrail (opt-in via config; side-LLM NLI judge).
    from koboi.guardrails.grounding import GroundingGuardrail

    GuardrailRegistry.register(
        "grounding_check",
        lambda **kw: GroundingGuardrail(**kw),
    )
