"""Rate-limiting throttler for E2E test runs.

Prevents overwhelming the upstream LLM provider during large
multi-scenario runs. The delay is applied *between* turns within a scenario
and *between* scenarios. Non-LLM tests (security/auth/404) set their
``throttle_seconds`` to 0 to skip the wait.

Configurable via env:
    E2E_THROTTLE_SECONDS        delay between turns (default 1.0)
    E2E_INTER_SCENARIO_SECONDS  delay between scenarios (default 2.0)
"""

from __future__ import annotations

import asyncio
import os


class Throttler:
    """Await ``wait(seconds)`` to sleep at least ``seconds`` since the last wait.

    The effective sleep is ``max(requested, 0)``; a 0 (or negative) request is
    a no-op so security/edge scenarios pay no latency tax.
    """

    def __init__(self, default_delay: float | None = None) -> None:
        self._default_delay = (
            default_delay if default_delay is not None else float(os.environ.get("E2E_THROTTLE_SECONDS", "1.0"))
        )

    async def wait(self, seconds: float | None = None) -> None:
        delay = self._default_delay if seconds is None else seconds
        if delay and delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    async def between_scenarios() -> None:
        """Sleep the inter-scenario delay (env ``E2E_INTER_SCENARIO_SECONDS``)."""
        delay = float(os.environ.get("E2E_INTER_SCENARIO_SECONDS", "2.0"))
        if delay > 0:
            await asyncio.sleep(delay)
