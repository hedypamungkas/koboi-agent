"""benchmarks/crash_recovery/run.py -- crash/redeploy resume benchmark (#8).

Proves koboi's library-level crash/redeploy resume (StepJournal + AgentCore.resume)
and measures its wall-clock cost. A multi-tool turn is interrupted mid-stream --
the exact state a SIGKILL/redeploy leaves behind (a ``running`` step row plus a
trailing assistant message whose tool_calls outnumber persisted ``role=tool``
results) -- then a FRESH agent built on the same SQLite session calls ``resume()``
and continues, re-executing ONLY the missing tool calls.

HONEST SCOPE
  - Benchmarks the CLI/facade ``resume()`` path (``AgentCore.resume`` +
    ``_repair_interrupted_turn``), NOT ``jobs.resume_on_startup`` (separate
    subsystem; marks running-jobs failed, does not rehydrate).
  - Uses a scripted (no-API-key) client -> deterministic and CI-friendly.
  - The crash is simulated in-process by breaking the stream mid-turn. Because
    StepJournal commits the ``running`` marker BEFORE each LLM call (eager, WAL),
    a literal SIGKILL leaves the identical row, so this faithfully exercises the
    recovery mechanism. A cross-process SIGKILL harness (needs a live LLM) is a
    documented follow-up, not a different mechanism.

Usage:
    python benchmarks/crash_recovery/run.py [--trials N] [--output report.json]

Comparison vs LangGraph/CrewAI/AutoGen/OpenAI Agents SDK: those frameworks have
no library-level mid-run durable checkpoint equivalent (LangGraph markets
"durable execution" only at the platform/LangSmith tier). This harness measures
koboi's number; running each competitor's equivalent is a separate exercise --
the point is the mechanism exists here at the library core.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

from koboi.events import ToolResultEvent
from koboi.facade import KoboiAgent
from koboi.llm.base import LLMClient
from koboi.types import AgentResponse, RiskLevel, ToolCall

_TOOL_NAMES = ("tool_a", "tool_b", "tool_c")
_TOOL_MARKS = {"tool_a": "A", "tool_b": "B", "tool_c": "C"}


def _config_dict(db_path: str) -> dict:
    return {
        "agent": {"name": "crash-recovery", "system_prompt": "h", "max_iterations": 5, "mode": "act"},
        "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
        "memory": {"backend": "sqlite", "db_path": db_path},
        "sandbox": {"backend": "passthrough"},
    }


class _ScriptedClient(LLMClient):
    """Deterministic LLM double (no API key) returning scripted responses."""

    def __init__(self, responses: list[AgentResponse]):
        self._responses = list(responses)
        self._i = 0

    async def complete(self, messages, tools=None, response_format=None):
        r = self._responses[self._i] if self._i < len(self._responses) else AgentResponse(content="done")
        self._i += 1
        return r

    async def get_embeddings(self, text):
        return None


def _tool_calls_response(*names: str) -> AgentResponse:
    return AgentResponse(
        content=None,
        tool_calls=[ToolCall(id=f"tc_{n}", name=n, arguments="{}") for n in names],
    )


def _register_tools(agent: KoboiAgent, executed: list[str]) -> None:
    """Three idempotent tracking tools that append a mark to `executed`."""
    for name in _TOOL_NAMES:
        mark = _TOOL_MARKS[name]
        agent.add_tool(
            name,
            lambda m=mark: executed.append(m) or f"result_{m}",
            f"tracking tool {mark}",
            {"type": "object", "properties": {}, "required": []},
            risk_level=RiskLevel.SAFE,
        )


def _steps_by_status(db_path: str, session_id: str) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM steps WHERE session_id=? GROUP BY status", (session_id,)
        ).fetchall()
    finally:
        conn.close()
    return {status: count for status, count in rows}


async def one_trial(workdir: Path, crash_after: str = "tool_a") -> dict:
    """Run one crash+resume trial; return a measurements dict."""
    db_path = str(workdir / "crash.db")
    if Path(db_path).exists():
        Path(db_path).unlink()
    executed: list[str] = []

    # Phase 1: start a 3-tool turn, cancel after `crash_after` completes (mid-stream crash).
    agent = KoboiAgent.from_dict(_config_dict(db_path))
    _register_tools(agent, executed)
    agent._core.client = _ScriptedClient([_tool_calls_response(*_TOOL_NAMES), AgentResponse(content="done")])
    session_id = agent._core.memory.session_id

    gen = agent.run_stream("go")
    async for ev in gen:
        if isinstance(ev, ToolResultEvent) and ev.tool_name == crash_after:
            break  # simulate client disconnect / process death
    await gen.aclose()
    after_phase1 = executed[:]
    pre_running = _steps_by_status(db_path, session_id).get("running", 0)

    # Phase 2: fresh agent on the same SQLite session -> resume() (timed).
    cfg2 = copy.deepcopy(_config_dict(db_path))
    cfg2["memory"]["session_id"] = session_id
    agent2 = KoboiAgent.from_dict(cfg2)
    _register_tools(agent2, executed)
    agent2._core.client = _ScriptedClient([AgentResponse(content="done")])

    t0 = time.perf_counter()
    result = await agent2.resume()
    resume_ms = (time.perf_counter() - t0) * 1000.0

    post = _steps_by_status(db_path, session_id)
    no_double_exec = executed.count(_TOOL_MARKS[crash_after]) == 1
    missing_reran = all(_TOOL_MARKS[n] in executed for n in _TOOL_NAMES if n != crash_after)
    no_open_running = post.get("running", 0) == 0
    has_complete = post.get("complete", 0) >= 1
    ok = bool(
        result.success and no_double_exec and missing_reran and no_open_running and has_complete and pre_running >= 1
    )
    return {
        "ok": ok,
        "resume_ms": resume_ms,
        "pre_resume_running_rows": pre_running,
        "post_resume_status_counts": post,
        "phase1_executed": after_phase1,
        "no_double_execution": no_double_exec,
        "missing_tools_reran": missing_reran,
    }


async def run_baseline(trials: int = 10) -> dict:
    """Run N crash+resume trials in fresh temp dirs and aggregate."""
    timings: list[float] = []
    passed = 0
    sample: dict | None = None
    for i in range(trials):
        workdir = Path(tempfile.mkdtemp(prefix=f"crash_recovery_{i}_"))
        try:
            res = await one_trial(workdir)
        finally:
            for f in workdir.glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                workdir.rmdir()
            except OSError:
                pass
        if res["ok"]:
            passed += 1
            timings.append(res["resume_ms"])
        sample = res
    timings_sorted = sorted(timings)
    return {
        "mechanism": "StepJournal + AgentCore.resume (CLI/facade path)",
        "trials": trials,
        "passed": passed,
        "correctness_rate": passed / trials if trials else 0.0,
        "resume_ms_mean": statistics.mean(timings) if timings else None,
        "resume_ms_median": statistics.median(timings) if timings else None,
        "resume_ms_min": timings_sorted[0] if timings else None,
        "resume_ms_max": timings_sorted[-1] if timings else None,
        "sample_trial": sample,
        "note": (
            "koboi re-executes only the missing tool calls after a mid-turn crash; "
            "no double-execution; the 'running' crash-marker is cleared and a "
            "terminal 'complete' row is written. Competitor comparison "
            "(LangGraph/CrewAI/AutoGen/OpenAI Agents SDK) is a separate exercise."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="koboi crash/redeploy resume benchmark")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--output", help="write JSON report to this path")
    args = p.parse_args()

    report = asyncio.run(run_baseline(trials=args.trials))

    print(f"koboi crash-recovery benchmark ({report['trials']} trials)")
    print(f"  correctness:        {report['passed']}/{report['trials']} ({report['correctness_rate'] * 100:.0f}%)")
    if report["resume_ms_mean"] is not None:
        print(
            f"  resume wall-clock:  mean={report['resume_ms_mean']:.2f}ms "
            f"median={report['resume_ms_median']:.2f}ms "
            f"[min={report['resume_ms_min']:.2f}, max={report['resume_ms_max']:.2f}]"
        )
    print(f"  mechanism:          {report['mechanism']}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"  report written:     {args.output}")
    return 0 if report["passed"] == report["trials"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
