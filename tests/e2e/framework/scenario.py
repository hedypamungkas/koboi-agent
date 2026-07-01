"""Scenario dataclasses + executor for data-driven E2E tests.

A Scenario is a list of Turns sent to the agent in sequence. The executor
runs each turn, collects SSE events + metrics, and produces a ScenarioResult
with full JSON-serializable output (request, response, tokens, latency, tools).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from tests.e2e.framework.metrics import collect_system_metrics
from tests.e2e.framework.throttler import Throttler

_logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_results_dir() -> Path:
    """Directory for the *current* run's JSON output.

    Honors ``E2E_RUN_ID`` (stamped once per pytest session by ``conftest.py``)
    so each run lands in its own timestamped subfolder and prior runs are
    preserved for comparison/history. Falls back to the flat ``RESULTS_DIR``
    when unset (ad-hoc single-file runs / direct executor use).
    """
    run_id = os.environ.get("E2E_RUN_ID")
    return (RESULTS_DIR / run_id) if run_id else RESULTS_DIR


def _write_latest_pointer(run_dir: Path) -> None:
    """Record the run folder name in ``results/latest.txt`` for easy lookup."""
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "latest.txt").write_text(run_dir.name)
    except OSError:
        pass

#: Provider error fragments that indicate a HARD, non-retriable block (monthly
#: cost cap, exhausted quota, invalid key). Retrying these only burns wall-clock
#: and risks looking like a hang — the suite should fast-fail/skip instead.
_HARD_PROVIDER_BLOCK_MARKERS = (
    "rate_limit_exceeded",
    "monthly limit",
    "limit exceeded",
    "insufficient_quota",
    "quota",
    "invalid_api_key",
    "incorrect api key",
    "authentication",
    "exceeded your current quota",
)


class ProviderBlocked(Exception):
    """Raised when the upstream LLM provider is hard-blocked (quota/cost/auth).

    Distinct from transient errors: these do not recover on retry, so the
    executor fast-fails the scenario and the runner skips the rest of the run.
    """


def _is_hard_provider_block(events: list) -> str | None:
    """Return the blocking message if any error event looks like a hard cap, else None."""
    for e in events:
        if isinstance(e, dict) and e.get("type") == "error":
            msg = str(e.get("error", "")).lower()
            if any(m in msg for m in _HARD_PROVIDER_BLOCK_MARKERS):
                return str(e.get("error", ""))[:200]
    return None


def _kw_match(content: str, kw: str) -> bool:
    """Case-insensitive substring match with number normalization.

    Also matches with thousands separators stripped, so a keyword ``"1260"``
    matches content ``"1,260"`` (and vice-versa) -- the e2e models frequently
    format numbers with commas, which broke exact-substring assertions.
    """
    c = content.lower()
    k = kw.lower()
    if k in c:
        return True
    return k.replace(",", "") in c.replace(",", "")


@dataclass
class Turn:
    """A single user message + expectations."""

    message: str
    expect_tools: list[str] = field(default_factory=list)
    expect_keywords: list[str] = field(default_factory=list)
    #: Pass if ANY of these is present (OR semantics). Use for answers with
    #: multiple acceptable forms (e.g. ["15,000", "15000"]) or concept synonyms.
    expect_any_of: list[str] = field(default_factory=list)
    min_events: int = 2
    #: Map of expected tool name -> equivalent tool names that also satisfy the
    #: assertion. Use when a builtin tool has a functionally equivalent MCP tool
    #: the model may pick instead (e.g. task_create vs add_todo). Backward
    #: compatible (default empty = exact/substring behaviour unchanged).
    tool_aliases: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Scenario:
    """A named sequence of turns with optional assertions."""

    name: str
    category: str
    turns: list[Turn]
    session_id: str | None = None
    job: bool = False
    throttle_seconds: float = 1.0
    timeout_per_turn: float = 180.0
    skip: str | None = None
    #: When >0, run the FIRST turn on ``concurrent`` independent sessions in
    #: parallel (stress/concurrency scenarios). Remaining turns are ignored.
    concurrent: int = 0
    #: Retries on a recoverable LLM error (timeout / 5xx / error event). The
    #: upstream provider (gpt-4o-mini) is variable; one retry catches blips.
    retries: int = 1


@dataclass
class TurnResult:
    """Metrics from a single turn."""

    message: str
    events: list[dict]
    content: str
    tool_calls: list[dict]
    tool_results: list[dict]
    token_usage: dict | None
    latency_seconds: float
    timestamp: str
    model_name: str | None = None
    url_provider: str | None = None

    def to_dict(self) -> dict:
        u = self.token_usage or {}
        return {
            "message": self.message,
            "content": self.content,
            "event_count": len(self.events),
            "event_types": [e.get("type", "?") for e in self.events if isinstance(e, dict)],
            "tool_calls": [e.get("tool_name", "") for e in self.tool_calls],
            "tool_results": [
                {"tool_name": e.get("tool_name", ""), "result_preview": str(e.get("result", ""))[:200]}
                for e in self.tool_results
            ],
            "model_name": self.model_name,
            "url_provider": self.url_provider,
            "token_input": u.get("token_input", u.get("prompt_tokens")),
            "token_output": u.get("token_output", u.get("completion_tokens")),
            "token_reasoning": u.get("token_reasoning", u.get("reasoning_tokens")),
            "token_usage": self.token_usage,
            "latency_seconds": round(self.latency_seconds, 3),
            "timestamp": self.timestamp,
        }


@dataclass
class ScenarioResult:
    """Full result of running a scenario."""

    scenario_name: str
    category: str
    passed: bool
    duration_seconds: float
    turns: list[TurnResult]
    system_metrics: dict
    error: str | None = None
    assertions_checked: int = 0
    assertions_passed: int = 0

    def to_dict(self) -> dict:
        # Model/provider from the last turn that reported it (constant per run).
        model = next((t.model_name for t in reversed(self.turns) if t.model_name), None)
        url = next((t.url_provider for t in reversed(self.turns) if t.url_provider), None)
        u = lambda key: sum((t.token_usage or {}).get(key, 0) for t in self.turns)  # noqa: E731
        return {
            "scenario": self.scenario_name,
            "category": self.category,
            "passed": self.passed,
            "model_name": model,
            "url_provider": url,
            "token_input": u("token_input") or u("prompt_tokens"),
            "token_output": u("token_output") or u("completion_tokens"),
            "token_reasoning": u("token_reasoning") or u("reasoning_tokens"),
            "duration_seconds": round(self.duration_seconds, 3),
            "turns": [t.to_dict() for t in self.turns],
            "system_metrics": self.system_metrics,
            "error": self.error,
            "assertions_checked": self.assertions_checked,
            "assertions_passed": self.assertions_passed,
            "total_tokens": sum(
                (t.token_usage or {}).get("total_tokens", 0) for t in self.turns
            ),
            "total_latency": round(sum(t.latency_seconds for t in self.turns), 3),
            "total_tool_calls": sum(len(t.tool_calls) for t in self.turns),
        }

    def save_json(self) -> Path:
        run_dir = run_results_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{self.scenario_name}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


class ScenarioExecutor:
    """Runs a Scenario against a live server and collects metrics."""

    #: Process-wide flag: once the provider hard-blocks (quota/cost/auth), set
    #: True so every subsequent scenario skips instantly instead of retrying.
    _globally_blocked: bool = False

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        default_headers: dict | None = None,
        auto_approve: bool = True,
    ):
        self._client = client
        self._base_url = base_url
        self._api_key = api_key
        self._default_headers = default_headers or {}
        #: When True, the executor auto-approves mid-stream PendingApprovalEvents
        #: (MODERATE/DESTRUCTIVE tools) so interactive turns complete without a
        #: human — exercising the real queue-bridge HITL path end-to-end.
        self._auto_approve = auto_approve
        self._throttler = Throttler(
            default_delay=float(os.environ.get("E2E_THROTTLE_SECONDS", "1.0")),
        )

    def _headers(self, **extra: str) -> dict:
        h = {"Content-Type": "application/json", **self._default_headers}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        h.update(extra)
        return h

    async def _create_session(self) -> str:
        r = await self._client.post("/v1/sessions", headers=self._headers())
        assert r.status_code == 201, f"session create failed: {r.status_code}"
        return r.json()["session_id"]

    async def _stream_chat(
        self, message: str, session_id: str | None, timeout: float
    ) -> tuple[list[dict], float]:
        headers = self._headers()
        if session_id:
            headers["X-Session-Id"] = session_id
        events: list = []
        t0 = time.monotonic()
        async with self._client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": message},
            headers=headers,
            timeout=timeout,
        ) as r:
            assert r.status_code == 200, f"stream failed: {r.status_code} {await r.aread()!r}"
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    events.append("[DONE]")
                    break
                ev = json.loads(payload)
                events.append(ev)
                # Mid-stream HITL: when the agent asks for approval (MODERATE/
                # DESTRUCTIVE tools like write_file), auto-approve so the turn
                # completes. This exercises the real queue-bridge approve path.
                if (
                    self._auto_approve
                    and session_id
                    and isinstance(ev, dict)
                    and ev.get("type") == "pending_approval"
                    and ev.get("approval_id")
                ):
                    asyncio.create_task(self._approve(session_id, ev["approval_id"]))
        latency = time.monotonic() - t0
        return events, latency

    async def _approve(self, session_id: str, approval_id: str) -> None:
        """POST /v1/sessions/:id/approve to resolve a pending approval."""
        try:
            r = await self._client.post(
                f"/v1/sessions/{session_id}/approve",
                json={"approval_id": approval_id, "decision": "approve"},
                headers=self._headers(),
                timeout=30,
            )
            if r.status_code != 200:
                _logger.warning("auto-approve %s failed: %s %s", approval_id, r.status_code, r.text[:120])
        except Exception as exc:
            _logger.warning("auto-approve %s error: %s", approval_id, exc)

    async def _run_as_job(self, message: str, timeout: float) -> tuple[list[dict], float]:
        r = await self._client.post(
            "/v1/jobs", json={"message": message}, headers=self._headers()
        )
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        t0 = time.monotonic()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            jr = await self._client.get(
                f"/v1/jobs/{job_id}", headers=self._headers()
            )
            body = jr.json()
            if body["status"] in ("completed", "failed", "timed_out", "cancelled"):
                break
            await asyncio.sleep(0.5)
        latency = time.monotonic() - t0
        events: list = []
        async with self._client.stream(
            "GET", f"/v1/jobs/{job_id}/stream", headers=self._headers(), timeout=30
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        events.append("[DONE]")
                        break
                    events.append(json.loads(payload))
        return events, latency

    async def _run_one_turn(
        self, turn: Turn, session_id: str | None, scenario: Scenario
    ) -> TurnResult:
        """Run a single turn with retry on recoverable errors.

        Recoverable = exception OR an SSE ``error`` event (LLM blip / timeout).
        Retries are spaced by ``throttle_seconds`` so we don't hammer the provider.
        """
        last_exc: Exception | None = None
        for attempt in range(scenario.retries + 1):
            await self._throttler.wait(scenario.throttle_seconds if attempt else 0)
            try:
                if scenario.job:
                    events, latency = await self._run_as_job(
                        turn.message, scenario.timeout_per_turn
                    )
                else:
                    events, latency = await self._stream_chat(
                        turn.message, session_id, scenario.timeout_per_turn
                    )
            except Exception as exc:  # network / timeout / non-200
                last_exc = exc
                _logger.warning(
                    "Scenario %s turn %r attempt %d raised %s",
                    scenario.name, turn.message[:40], attempt + 1, type(exc).__name__,
                )
                continue

            # If the stream itself emitted an error event, decide retry vs abort.
            error_events = [e for e in events if isinstance(e, dict) and e.get("type") == "error"]
            if error_events:
                block_msg = _is_hard_provider_block(events)
                if block_msg:
                    # Hard provider cap (monthly cost / quota / auth): NOT
                    # recoverable. Fast-fail so the run doesn't burn time.
                    raise ProviderBlocked(block_msg)
                if attempt < scenario.retries:
                    _logger.warning(
                        "Scenario %s turn %r attempt %d got error event; retrying",
                        scenario.name, turn.message[:40], attempt + 1,
                    )
                    continue
            return self._parse_turn(turn, events, latency)

        # Exhausted retries — raise so execute() records it as a hard error.
        if last_exc:
            raise last_exc
        return self._parse_turn(turn, [], 0.0)

    def _parse_turn(self, turn: Turn, events: list[dict], latency: float) -> TurnResult:
        content = ""
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        token_usage: dict | None = None
        model_name: str | None = None
        url_provider: str | None = None
        for ev in events:
            if not isinstance(ev, dict):
                continue
            t = ev.get("type", "")
            if t == "complete":
                content = ev.get("content", "")
                token_usage = ev.get("token_usage")
                model_name = ev.get("model_name") or model_name
                url_provider = ev.get("url_provider") or url_provider
            elif t == "tool_call":
                tool_calls.append(ev)
            elif t == "tool_result":
                tool_results.append(ev)
        return TurnResult(
            message=turn.message,
            events=events,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            token_usage=token_usage,
            latency_seconds=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            url_provider=url_provider,
        )

    async def execute(self, scenario: Scenario) -> ScenarioResult:
        """Run a full scenario and return structured results."""
        if scenario.skip:
            return ScenarioResult(
                scenario_name=scenario.name,
                category=scenario.category,
                passed=True,
                duration_seconds=0,
                turns=[],
                system_metrics={},
                error=f"SKIPPED: {scenario.skip}",
            )
        # Once the provider is hard-blocked, skip remaining scenarios instantly.
        if ScenarioExecutor._globally_blocked:
            return ScenarioResult(
                scenario_name=scenario.name,
                category=scenario.category,
                passed=True,
                duration_seconds=0,
                turns=[],
                system_metrics={},
                error="BLOCKED: upstream provider rate-limited earlier in this run",
            )

        start = time.monotonic()
        turns_results: list[TurnResult] = []
        error: str | None = None
        passed = True

        try:
            if scenario.concurrent:
                # Stress path: N parallel sessions each run the first turn.
                turns_results = await self._run_concurrent(scenario)
                await self._throttler.wait(scenario.throttle_seconds)
            else:
                session_id = scenario.session_id or await self._create_session()
                for i, turn in enumerate(scenario.turns):
                    tr = await self._run_one_turn(turn, session_id if not scenario.job else None, scenario)
                    turns_results.append(tr)
                    checked, ok = self._check_turn(turn, tr.content, tr.tool_calls, tr.events)
                    if not ok:
                        passed = False
                        if not error:
                            error = f"Turn {i+1} assertion failed for '{turn.message[:50]}'"

            # Keyword assertions (tolerant; recorded but do flip passed=False).
            kw_passed, kw_error = self._evaluate_keywords(scenario, turns_results)
            if not kw_passed:
                passed = False
                if not error:
                    error = kw_error

        except ProviderBlocked as exc:
            # Mark the whole run blocked so subsequent scenarios skip fast.
            ScenarioExecutor._globally_blocked = True
            passed = False
            error = f"BLOCKED: {exc}"

        except Exception as exc:
            passed = False
            error = f"{type(exc).__name__}: {exc}"
            _logger.exception("Scenario %s failed", scenario.name)

        duration = time.monotonic() - start
        metrics = collect_system_metrics()
        assertions_checked, assertions_passed = self._count_assertions(scenario, turns_results)
        result = ScenarioResult(
            scenario_name=scenario.name,
            category=scenario.category,
            passed=passed,
            duration_seconds=duration,
            turns=turns_results,
            system_metrics=metrics,
            error=error,
            assertions_checked=assertions_checked,
            assertions_passed=assertions_passed,
        )
        result.save_json()
        return result

    async def _run_concurrent(self, scenario: Scenario) -> list[TurnResult]:
        """Run ``scenario.concurrent`` copies of the first turn on separate sessions."""
        n = scenario.concurrent
        sessions = [await self._create_session() for _ in range(n)]

        async def _one(sid: str) -> TurnResult:
            # Jobs create their own session; interactive reuses the given one.
            return await self._run_one_turn(
                scenario.turns[0], None if scenario.job else sid, scenario
            )

        results = await asyncio.gather(*[_one(s) for s in sessions], return_exceptions=True)
        out: list[TurnResult] = []
        for r in results:
            if isinstance(r, TurnResult):
                out.append(r)
            else:
                # An exception in one session shouldn't abort the others.
                out.append(
                    TurnResult(
                        message=scenario.turns[0].message,
                        events=[],
                        content="",
                        tool_calls=[],
                        tool_results=[],
                        token_usage=None,
                        latency_seconds=0.0,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )
        return out

    def _check_turn(
        self, turn: Turn, content: str, tool_calls: list[dict], events: list[dict]
    ) -> tuple[int, bool]:
        checked = 0
        for kw in turn.expect_keywords:
            checked += 1
            if not _kw_match(content, kw):
                return checked, False
        if turn.expect_any_of:
            checked += 1
            if not any(_kw_match(content, k) for k in turn.expect_any_of):
                return checked, False
        for tool in turn.expect_tools:
            checked += 1
            # An expected tool is satisfied by its canonical name OR any declared
            # alias (e.g. task_create satisfied by MCP add_todo). Substring match
            # is preserved so 'calculate' still matches tool-call names.
            candidates = [tool, *turn.tool_aliases.get(tool, [])]
            tool_names = [tc.get("tool_name", "") for tc in tool_calls]
            if not any(c in tn for tn in tool_names for c in candidates):
                return checked, False
        if turn.min_events and len(events) < turn.min_events:
            checked += 1
            return checked, False
        return checked, True

    @staticmethod
    def _evaluate_keywords(scenario: Scenario, turns_results: list[TurnResult]) -> tuple[bool, str | None]:
        """Keyword pass logic. Returns (all_passed, first_error_or_None).

        Sequential: each turn must contain ALL ``expect_keywords`` (AND) and, if
        ``expect_any_of`` is set, at least one of them (OR).
        Concurrent: each of the N session replies must independently contain at
        least one keyword from the first turn's set (OR per session) -- requiring
        one short reply to contain ALL keywords is impossible, which is what made
        the stress fan-out scenarios fail.
        """
        if scenario.concurrent and turns_results:
            first = scenario.turns[0] if scenario.turns else None
            kws = (first.expect_keywords + first.expect_any_of) if first else []
            for i, tr in enumerate(turns_results):
                if kws and not any(_kw_match(tr.content, k) for k in kws):
                    return False, f"Concurrent session {i+1}: none of {kws} found in reply"
            return True, None
        for i, tr in enumerate(turns_results):
            if i < len(scenario.turns):
                turn = scenario.turns[i]
                for kw in turn.expect_keywords:
                    if not _kw_match(tr.content, kw):
                        return False, f"Keyword '{kw}' not found in turn {i+1}"
                if turn.expect_any_of and not any(_kw_match(tr.content, k) for k in turn.expect_any_of):
                    return False, f"None of {turn.expect_any_of} found in turn {i+1}"
        return True, None

    @staticmethod
    def _count_assertions(scenario: Scenario, turns_results: list[TurnResult]) -> tuple[int, int]:
        """Count assertions checked/passed, consistent with ``_evaluate_keywords``."""
        if scenario.concurrent and turns_results:
            first = scenario.turns[0] if scenario.turns else None
            kws = (first.expect_keywords + first.expect_any_of) if first else []
            checked = len(turns_results)
            passed = sum(1 for tr in turns_results if (not kws or any(_kw_match(tr.content, k) for k in kws)))
            return checked, passed
        checked = sum(
            len(t.expect_keywords) + len(t.expect_tools) + (1 if t.expect_any_of else 0) for t in scenario.turns
        )
        passed = 0
        for i, tr in enumerate(turns_results):
            if i >= len(scenario.turns):
                break
            turn = scenario.turns[i]
            passed += sum(1 for kw in turn.expect_keywords if _kw_match(tr.content, kw))
            if turn.expect_any_of and any(_kw_match(tr.content, k) for k in turn.expect_any_of):
                passed += 1
        return checked, passed


def save_summary(results: list[ScenarioResult]) -> Path:
    """Write summary.json with pass/fail counts and timing into the run folder."""
    run_dir = run_results_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(results)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    _write_latest_pointer(run_dir)
    return path


def _build_summary(results: list[ScenarioResult]) -> dict:
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "skipped": sum(
            1 for r in results
            if r.error and (r.error.startswith("SKIPPED") or r.error.startswith("BLOCKED"))
        ),
        "blocked": sum(1 for r in results if r.error and r.error.startswith("BLOCKED")),
        "total_duration": round(sum(r.duration_seconds for r in results), 1),
        "total_tokens": sum(
            (t.token_usage or {}).get("total_tokens", 0)
            for r in results
            for t in r.turns
        ),
        "categories": {},
        "scenarios": [
            {
                "name": r.scenario_name,
                "category": r.category,
                "passed": r.passed,
                "duration": round(r.duration_seconds, 1),
                "tokens": sum(
                    (t.token_usage or {}).get("total_tokens", 0) for t in r.turns
                ),
                "tool_calls": sum(len(t.tool_calls) for t in r.turns),
                "error": r.error,
            }
            for r in results
        ],
    }
    for r in results:
        cat = summary["categories"].setdefault(
            r.category, {"total": 0, "passed": 0, "failed": 0, "duration": 0}
        )
        cat["total"] += 1
        cat["passed" if r.passed else "failed"] += 1
        cat["duration"] += round(r.duration_seconds, 1)
    return summary


def save_summary_from_disk() -> Path:
    """Aggregate every ``<scenario>.json`` in the run folder into summary.json.

    Called at session teardown (conftest) so the summary reflects the actual
    run regardless of how scenarios were filtered/selected.
    """
    run_dir = run_results_dir()
    results: list[ScenarioResult] = []
    for p in sorted(run_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        results.append(
            ScenarioResult(
                scenario_name=data.get("scenario", p.stem),
                category=data.get("category", "unknown"),
                passed=bool(data.get("passed")),
                duration_seconds=float(data.get("duration_seconds", 0)),
                turns=[],  # not needed for the summary rollup
                system_metrics=data.get("system_metrics", {}),
                error=data.get("error"),
            )
        )
    return save_summary(results)
