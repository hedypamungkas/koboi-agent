"""koboi/harness/recovery_budget.py -- shared per-run recovery budget (self-healing P2a).

A single counter shared across recovery rungs so reflection, doom, etc. cannot
together consume unbounded iterations (the "reflection amplifies doom-loop" risk,
docs/self-healing-feasibility.md §6.2). The ``LadderRouterHook`` owns and consumes
it; the rungs honor the router's choice. Reset on SESSION_START (per run).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecoveryBudget:
    """Per-run cap on recovery (reflect/doom) turns. Owned by the ladder router."""

    max_turns: int = 3

    def __post_init__(self) -> None:
        self._used = 0

    def can_consume(self) -> bool:
        """True if a recovery turn is still within the per-run budget."""
        return self._used < self.max_turns

    def consume(self) -> None:
        """Spend one recovery turn (called by the router when it picks a non-terminal rung)."""
        self._used += 1

    def reset(self) -> None:
        """Per-run reset (called on SESSION_START)."""
        self._used = 0

    @property
    def used(self) -> int:
        return self._used
