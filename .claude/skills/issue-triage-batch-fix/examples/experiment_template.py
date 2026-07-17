#!/usr/bin/env python3
"""experiment_<topic>.py -- TEMPLATE for a reproducible bug harness.

Hard rules (see references/empirical-red-tests.md):
  - drive REAL production classes (mock only external I/O: LLM/HTTP/platform module)
  - no network, no real API keys, deterministic (gates/pre-seeds, never timing)
  - prints CHECK -> VERDICT (OPEN|FIXED) + concrete EVIDENCE
  - sys.exit(1) if any CHECK is OPEN (bug present), 0 if all FIXED

Replace the TODOs with the real system-under-test. Run: `python experiment_<topic>.py`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # repo root on path

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

# TODO: import the real entry point(s) under test, e.g.:
# from myapp.server import create_app
# from myapp.config import Config
# from tests.conftest import MockClient, make_mock_response   # mock only the LLM


def _config(**overrides):
    """Build a real config dict/object for the system under test."""
    cfg = {
        # ...minimal config to construct the real app...
    }
    cfg.update(overrides)
    return cfg  # or Config.from_dict(cfg, validate=True)


async def check_1_main_bug(tmp):
    """TODO: the headline CHECK. Drives the real route/class; asserts the FIXED
    behavior (fails today = OPEN, passes after fix = FIXED)."""
    # app = create_app(_config(), client_factory=lambda: MockClient([...]), ...)
    # async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
    #     r = await c.post("/v1/...", json={...}, headers={...})
    # bug_present = (r.status_code == 200)   # today's buggy behavior
    bug_present = True  # TODO replace with real assertion
    verdict = "OPEN" if bug_present else "FIXED"
    evidence = "TODO: concrete status/counts that prove the bug (or its absence)"
    return verdict, evidence


async def check_2_edge(tmp):
    """TODO: an adjacent CHECK (related code path, boundary, regression guard)."""
    bug_present = True  # TODO
    return ("OPEN" if bug_present else "FIXED", "TODO: evidence")


async def main() -> int:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="exp-"))
    checks = [
        ("CHECK 1: <headline claim>", check_1_main_bug),
        ("CHECK 2: <edge/related>", check_2_edge),
    ]
    print("=" * 78)
    print("experiment_<topic>.py — <one-line bug summary>")
    print("=" * 78)
    any_open = False
    for title, fn in checks:
        verdict, evidence = await fn(tmp)
        any_open = any_open or verdict == "OPEN"
        print(f"\n{title}\n  VERDICT: {verdict}\n  EVIDENCE: {evidence}")
    print("\n" + "=" * 78)
    print("SUMMARY:", "OPEN — bug reproduces on this build" if any_open else "all FIXED")
    print("=" * 78)
    return 1 if any_open else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
