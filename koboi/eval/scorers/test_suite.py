"""koboi/eval/scorers/test_suite -- run the repo's real test suite; gate on exit code.

The ground-truth scorer for coding evals: instead of comparing the agent's
output text to a golden answer, execute ``case.test_command`` inside the
case's materialized workspace via a restricted sandbox and score 1.0 iff the
suite exits 0.

Boundaries (documented, accepted): the restricted sandbox's ``network: deny``
is a soft token-scan gate, and its env is scrubbed -- test commands should use
interpreter-module form (``python3 -m unittest`` / an absolute interpreter),
not bare PATH lookups like ``pytest``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from koboi.eval.scorers.base import BaseScorer
from koboi.sandbox.registry import build_sandbox
from koboi.types import EvalCase, EvalScore


class TestSuiteScorer(BaseScorer):
    """Runs a test command in the case workspace; 1.0 iff exit code 0.

    Resolution order: ``case.test_command`` -> ``case.metadata['test_command']``
    -> constructor kwarg; workspace: ``context['workspace']`` ->
    ``case.metadata['workspace']`` -> constructor kwarg. Constructor kwargs are
    the ``t.judge("test_suite", test_command=..., workspace=...)`` path.
    """

    __test__ = False  # class name starts with "Test" -- keep pytest from collecting it

    def __init__(
        self,
        *,
        test_command: str | None = None,
        workspace: str | None = None,
        timeout: float = 900.0,
        network: str = "deny",
        backend: str = "restricted",
        tail_chars: int = 800,
    ):
        self.test_command = test_command
        self.workspace = workspace
        self.timeout = timeout
        self.network = network
        self.backend = backend
        self.tail_chars = tail_chars

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        # getattr keeps this duck-type-safe for bare-mock cases in tests.
        cmd = getattr(case, "test_command", None) or case.metadata.get("test_command") or self.test_command
        ws = context.get("workspace") or case.metadata.get("workspace") or self.workspace
        if not cmd and not ws:
            # N/A convention (matches ToolUsageScorer): a 0.0 here would poison
            # the mean for mixed suites containing non-coding cases.
            return EvalScore("test_suite", 1.0, "not applicable: no test_command/workspace")
        if not cmd:
            return EvalScore("test_suite", 0.0, "workspace present but no test_command")
        if not ws or not Path(ws).is_dir():
            return EvalScore("test_suite", 0.0, f"workspace missing or not a directory: {ws!r}")

        sandbox = build_sandbox({"backend": self.backend, "workdir": str(ws), "network": self.network})
        res = await asyncio.to_thread(sandbox.run, cmd, shell=True, cwd=str(ws), timeout=self.timeout)

        if getattr(res, "timed_out", False):
            return EvalScore("test_suite", 0.0, f"timed out after {self.timeout}s: {cmd!r}")
        tail = f"{res.stdout}\n{res.stderr}".strip()[-self.tail_chars :]
        # 126 is ALSO a normal POSIX exit code ("not executable"). Only attribute
        # it to a sandbox-escape when the SANDBOX emitted it (empty stdout + a
        # sandbox message in stderr) -- otherwise a test command's own exit 126
        # must read as a plain failure, not "cwd escaped sandbox workdir".
        if res.returncode == 126 and not (res.stdout or "").strip() and "sandbox" in (res.stderr or "").lower():
            return EvalScore("test_suite", 0.0, f"exit=126 (cwd escaped sandbox workdir) {tail}")
        value = 1.0 if res.returncode == 0 else 0.0
        return EvalScore("test_suite", value, f"exit={res.returncode} {tail}".strip())
