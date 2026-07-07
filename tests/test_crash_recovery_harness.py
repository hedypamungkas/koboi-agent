"""tests/test_crash_recovery_harness.py -- #8 CI smoke for the crash-recovery benchmark.

Loads benchmarks/crash_recovery/run.py by path (it is a standalone script, not a
package) and runs a single in-process trial to prove the resume mechanism works in
CI (no API key -- the benchmark uses a scripted client).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "crash_recovery" / "run.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("crash_recovery_bench", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_one_trial_recovers_correctly(bench, tmp_path):
    result = await bench.one_trial(tmp_path)
    assert result["ok"] is True
    assert result["no_double_execution"] is True
    assert result["missing_tools_reran"] is True
    assert result["pre_resume_running_rows"] >= 1  # a 'running' crash-marker was left
    assert result["post_resume_status_counts"].get("running", 0) == 0  # cleared after resume
