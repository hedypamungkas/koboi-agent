"""koboi/eval/t/runner.py -- Run ``*.eval.py`` tests and fold assertions into EvalResults."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from koboi.eval.t.assertions import Severity
from koboi.eval.t.context import TestContext
from koboi.eval.t.loader import LoadedTest
from koboi.eval.t.mock import ScriptedClient
from koboi.types import EvalResult, EvalScore

if TYPE_CHECKING:
    from koboi.facade import KoboiAgent
    from rich.console import Console


class TestRunner:
    """Runs discovered ``t`` tests, producing real :class:`~koboi.types.EvalResult` objects.

    Drives the agent directly per ``t.send()`` (so multi-turn tests work naturally)
    and reuses ``EvalResult`` / ``EvalRunner.format_results`` /
    :class:`~koboi.eval.regression.RegressionTracker` for output and regression
    tracking. A failed gate assertion forces ``EvalResult.passed = False``
    regardless of ``overall_score``.
    """

    # Not a pytest test class despite the ``Test`` prefix.
    __test__ = False

    def __init__(
        self,
        *,
        threshold: float = 0.6,
        default_severity: Severity = Severity.GATE,
        console: Console | None = None,
    ):
        self.threshold = threshold
        self.default_severity = default_severity
        self._console = console

    async def run_test(
        self,
        test: LoadedTest,
        *,
        config: str | dict | None = None,
        mock: bool | None = None,
        mock_responses: list | None = None,
    ) -> EvalResult:
        agent = await self._build_agent(test, config=config, mock=mock, mock_responses=mock_responses)
        ctx = TestContext(agent, default_severity=self.default_severity)
        start = time.monotonic()
        error: str | None = None
        try:
            await self._invoke(test, ctx)
        except Exception as exc:  # uncaught error in the test body -> hard gate failure
            error = f"{type(exc).__name__}: {exc}"
            ctx.record_gate_error(f"test body raised: {error}")
        finally:
            await self._close(agent)
        elapsed = time.monotonic() - start
        return self._fold(test, ctx, elapsed=elapsed, error=error)

    async def run_tests(
        self,
        tests: list[LoadedTest],
        *,
        parallel: bool = False,
        max_concurrency: int = 5,
        tags: list[str] | None = None,
        config: str | dict | None = None,
        mock: bool | None = None,
        mock_responses: list | None = None,
    ) -> list[EvalResult]:
        selected = self._filter_by_tags(tests, tags)
        if not selected:
            return []

        if parallel:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _bounded(t: LoadedTest) -> EvalResult:
                async with semaphore:
                    self._progress(t)
                    return await self._safe_run(t, config=config, mock=mock, mock_responses=mock_responses)

            return list(await asyncio.gather(*[_bounded(t) for t in selected]))

        results: list[EvalResult] = []
        for t in selected:
            self._progress(t)
            results.append(await self._safe_run(t, config=config, mock=mock, mock_responses=mock_responses))
        return results

    async def _safe_run(
        self,
        test: LoadedTest,
        *,
        config: str | dict | None,
        mock: bool | None,
        mock_responses: list | None,
    ) -> EvalResult:
        """Run one test; fold build/config errors into a failed EvalResult (batch isolation).

        A single misconfigured test (e.g. missing CONFIG) must not abort the whole
        suite -- it becomes a failed result so the batch and `--strict` exit code
        keep reflecting per-test outcomes.
        """
        try:
            return await self.run_test(test, config=config, mock=mock, mock_responses=mock_responses)
        except Exception as exc:
            return EvalResult(
                case_name=test.case_name,
                output="",
                scores=[EvalScore("test:error", 0.0, f"[gate] setup failed: {type(exc).__name__}: {exc}")],
                overall_score=0.0,
                passed=False,
                metadata={
                    "tags": list(test.tags),
                    "framework": "eval-test",
                    "file": str(test.file),
                    "turns": 0,
                    "gate_failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    # --------------------------------------------------------------- internals
    async def _invoke(self, test: LoadedTest, ctx: TestContext) -> None:
        if test.timeout:
            await asyncio.wait_for(test.func(ctx), timeout=test.timeout)
        else:
            await test.func(ctx)

    async def _build_agent(
        self,
        test: LoadedTest,
        *,
        config: str | dict | None,
        mock: bool | None,
        mock_responses: list | None,
    ) -> KoboiAgent:
        use_mock = mock if mock is not None else (bool(test.mock_responses) or test.use_mock)
        responses = mock_responses if mock_responses is not None else test.mock_responses

        if use_mock:
            return await self._build_mock_agent(test, responses=responses, config=config)

        cfg = config or test.config
        if cfg is None:
            raise ValueError(
                f"Test {test.case_name!r} has no CONFIG and --config was not provided; "
                "set CONFIG in the .eval.py, pass --config, or supply mock responses."
            )
        return self._build_live_agent(cfg)

    def _build_live_agent(self, cfg: str | dict) -> KoboiAgent:
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_config(cfg) if isinstance(cfg, (str, Path)) else KoboiAgent.from_dict(cfg)
        agent.ensure_telemetry_hook()
        return agent

    async def _build_mock_agent(
        self,
        test: LoadedTest,
        *,
        responses: list | None,
        config: str | dict | None,
    ) -> KoboiAgent:
        scripted = ScriptedClient(responses or [])
        cfg = config or test.config
        if cfg is not None:
            # High fidelity: borrow the production tools/hooks/guardrails and swap
            # only the LLM transport. AgentCore.client is a public attribute
            # (loop.py), so this stays off the hot path and off private API.
            agent = self._build_live_agent(cfg)
            # W6.1: orchestration configs (deep_research) have core=None but orchestrator set.
            # Swap the orchestrator's LLM client with a content-dispatching mock (not ScriptedClient,
            # which is sequential and can't handle deep_research's variable call sequence).
            if agent.orchestrator is not None:
                mock_client = getattr(test, "mock_client", None)
                if mock_client is None:
                    from koboi.eval.t.mock import DispatchingClient, deep_research_dispatch

                    mock_client = DispatchingClient(deep_research_dispatch())
                agent.orchestrator.client = mock_client
                return agent
            if agent.core is None:
                # Orchestration configs build a KoboiAgent with _core=None
                # (facade._build_orchestration); the client swap below is then
                # impossible, so a mock eval would silently run against the live
                # orchestrator (non-deterministic, may hang). Refuse loudly.
                raise ValueError(
                    "mock mode is unsupported for orchestration configs — "
                    "agent.core is None, cannot swap client. Use live mode "
                    "or a non-orchestration CONFIG."
                )
            original = agent.core.client
            agent.core.client = scripted  # type: ignore[assignment]  # ScriptedClient is an LLMClient test-double injected for deterministic evals
            # Release the now-unused real transport (httpx) so it does not leak.
            if original is not None and original is not scripted:
                try:
                    await original.close()
                except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
                    pass
            return agent

        # Bare core: no tools, just the scripted loop (deterministic, no API key).
        from koboi.facade import KoboiAgent
        from koboi.loop import AgentCore
        from koboi.memory import ConversationMemory
        from koboi.tools.registry import ToolRegistry

        iterations = max(1, len(responses or []) + 2)
        core = AgentCore(
            client=scripted,  # type: ignore[arg-type]  # ScriptedClient is an LLMClient test-double; AgentCore.client is RetryClient for prod
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=iterations,
        )
        return KoboiAgent(core=core)

    async def _close(self, agent: KoboiAgent) -> None:
        try:
            await agent.close()
        except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
            pass

    def _fold(self, test: LoadedTest, ctx: TestContext, *, elapsed: float, error: str | None) -> EvalResult:
        assertions = ctx.collect()
        scores: list[EvalScore] = []
        gate_failed = False
        total = 0.0
        for assertion in assertions:
            # Fail-safe per assertion (mirrors EvalRunner.run_case's scorer
            # guard): a throwing check closure or an out-of-range value degrades
            # to an error score instead of crashing the whole run.
            try:
                outcome = assertion.outcome()
                value = max(0.0, min(1.0, outcome.value))
                passed_flag = outcome.passed
                reason = outcome.reason
            except Exception as exc:
                value, passed_flag, reason = 0.0, False, f"evaluation error: {exc}"
            scores.append(
                EvalScore(
                    name=assertion.name,
                    value=round(value, 3),
                    reason=f"[{assertion.severity.value}] {reason}",
                )
            )
            total += value
            if not passed_flag and assertion.severity is Severity.GATE:
                gate_failed = True

        overall = total / len(scores) if scores else (0.0 if gate_failed else 1.0)
        passed = (not gate_failed) and (overall >= self.threshold) and error is None

        last = ctx.turns[-1] if ctx.turns else None
        return EvalResult(
            case_name=test.case_name,
            output=(last.content if last and last.content else ""),
            scores=scores,
            overall_score=round(overall, 3),
            duration_seconds=round(elapsed, 2),
            token_usage=ctx.total_token_usage() if ctx.turns else None,
            tool_calls_made=ctx.all_tool_calls,
            passed=passed,
            metadata={
                "tags": list(test.tags),
                "framework": "eval-test",
                "file": str(test.file),
                "turns": len(ctx.turns),
                "gate_failed": gate_failed,
                "error": error,
            },
        )

    def _filter_by_tags(self, tests: list[LoadedTest], tags: list[str] | None) -> list[LoadedTest]:
        if not tags:
            return list(tests)
        wanted = {tag.strip() for tag in tags if tag.strip()}
        return [t for t in tests if wanted & set(t.tags)]

    def _progress(self, test: LoadedTest) -> None:
        message = f"  running: {test.case_name}..."
        if self._console is not None:
            self._console.print(message)
        else:
            print(message)
