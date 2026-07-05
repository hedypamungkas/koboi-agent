"""koboi/eval/t/context.py -- TestContext: the `t` object passed to ``async def test_*(t)`` evals."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from koboi.eval.t.assertions import (
    AssertionOutcome,
    Matcher,
    RecordedAssertion,
    Severity,
    Truth,
    binary_outcome,
    coerce_matcher,
    describe_value,
)
from koboi.exceptions import AgentError

if TYPE_CHECKING:
    from koboi.eval.scorers.base import BaseScorer
    from koboi.facade import KoboiAgent
    from koboi.types import RunResult, ToolCall


def _parse_args(arguments: str | None) -> dict:
    """Best-effort parse of a ``ToolCall.arguments`` JSON string into a dict."""
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class TestContext:
    """Per-test handle passed to ``async def test_*(t)``.

    Drives the agent (``await t.send(...)``) and records assertions
    (``t.calledTool``, ``t.check``, ``t.judge``, ...). Assertions are evaluated
    once, *after* the test function returns (record-and-collect), against the
    full transcript accumulated by ``t.send``.

    Convention: ``t.check(value, ...)`` captures ``value`` when called; tool/turn
    assertions scan every turn recorded up to evaluation time.

    Scoring: each assertion yields a value (pass=1.0, gate failure=0.0, soft
    failure=0.5, judge=its score). The test's ``overall_score`` is the mean of
    these values; a single GATE failure forces ``EvalResult.passed = False``
    regardless of ``overall_score``. ``t.check`` defaults to SOFT; tool/turn
    assertions default to GATE.
    """

    # Not a pytest test class despite the ``Test`` prefix (the `t` test context).
    __test__ = False

    def __init__(self, agent: "KoboiAgent", *, default_severity: Severity = Severity.GATE):
        self._agent = agent
        self._default = default_severity
        self._turns: list[RunResult] = []
        self._sent: list[str] = []
        self._recorded: list[RecordedAssertion] = []

    # ------------------------------------------------------------------ drive
    async def send(self, message: str | list) -> "RunResult":
        """Drive the agent one turn and record the :class:`~koboi.types.RunResult`."""
        from koboi.types import RunResult

        self._sent.append(message if isinstance(message, str) else str(message))
        try:
            result = await self._agent.run(message)
        except AgentError as exc:
            # A failed turn (max iterations, guardrail block, abort) is recorded
            # as a soft note so the report explains the empty reply; completed()
            # and downstream checks will surface the hard failure.
            result = RunResult(content="", iterations_used=0, success=False, error=exc)
            self._record(
                "send:error",
                Severity.SOFT,
                lambda e=exc: AssertionOutcome(False, 0.0, f"send raised {type(e).__name__}: {e}"),
            )
        self._turns.append(result)
        return result

    # ------------------------------------------------------------ introspection
    @property
    def turns(self) -> list["RunResult"]:
        return list(self._turns)

    @property
    def last(self) -> "RunResult":
        if not self._turns:
            raise RuntimeError("t.last / t.reply called before t.send()")
        return self._turns[-1]

    @property
    def reply(self) -> str:
        """The last turn's final answer (empty string if no turns yet)."""
        if not self._turns:
            return ""
        return self.last.content or ""

    @property
    def output(self) -> str:
        """Alias of :attr:`reply`."""
        return self.reply

    @property
    def all_tool_calls(self) -> list["ToolCall"]:
        """Every tool call across all turns (preserves order and duplicates)."""
        return [tc for result in self._turns for tc in result.tool_calls_made]

    @property
    def messages(self) -> list[dict]:
        """Full conversation trace (read-only). Empty if the agent has no core/memory."""
        core = getattr(self._agent, "core", None)
        memory = getattr(core, "memory", None) if core is not None else None
        if memory is None:
            return []
        return memory.get_messages()

    def total_token_usage(self):
        """Summed :class:`~koboi.types.TokenUsage` across all turns."""
        from koboi.types import TokenUsage

        total = TokenUsage()
        for result in self._turns:
            if result.token_usage:
                total.prompt_tokens += result.token_usage.prompt_tokens
                total.completion_tokens += result.token_usage.completion_tokens
        return total

    # --------------------------------------------------------------- assertions
    def calledTool(self, name: str, *, severity: Severity | None = None) -> None:
        """Assert a tool named ``name`` was called at least once (gate by default)."""
        sev = self._sev(severity)

        def _evaluate() -> AssertionOutcome:
            count = sum(1 for tc in self.all_tool_calls if tc.name == name)
            return binary_outcome(sev, count > 0, f"calledTool({name!r}) -> {count} call(s)")

        self._record(f"calledTool:{name}", sev, _evaluate)

    def calledToolWith(self, name: str, args: dict, *, severity: Severity | None = None) -> None:
        """Assert ``name`` was called with a superset of ``args`` (gate by default)."""
        sev = self._sev(severity)
        expected = dict(args)

        def _evaluate() -> AssertionOutcome:
            matches = 0
            for tc in self.all_tool_calls:
                if tc.name != name:
                    continue
                actual = _parse_args(tc.arguments)
                if all(actual.get(k) == v for k, v in expected.items()):
                    matches += 1
            return binary_outcome(sev, matches > 0, f"calledToolWith({name!r}, {expected!r}) -> {matches} match(es)")

        self._record(f"calledToolWith:{name}", sev, _evaluate)

    def toolWasBlocked(self, name: str, *, severity: Severity | None = None) -> None:
        """Assert a tool named ``name`` was blocked/denied/skipped by the pipeline
        at least once (gate by default).

        Outcome-aware counterpart to :meth:`calledTool`. Reads
        ``RunResult.pipeline_outcomes`` (populated at koboi/loop.py when the
        ToolExecutionPipeline returns ``skipped=True`` with a ``skip_reason`` such
        as ``mode_blocked``/``policy_denied``/``sandbox_refused``). ``calledTool``
        counts ATTEMPTED calls (preserved for back-compat); ``toolWasBlocked``
        counts DENIED outcomes.
        """
        sev = self._sev(severity)

        def _evaluate() -> AssertionOutcome:
            count = sum(
                1
                for turn in self._turns
                for o in getattr(turn, "pipeline_outcomes", []) or []
                if o.get("tool_name") == name and o.get("skipped")
            )
            return binary_outcome(sev, count > 0, f"toolWasBlocked({name!r}) -> {count} block(s)")

        self._record(f"toolWasBlocked:{name}", sev, _evaluate)

    def retrievedChunk(self, needle: str, *, severity: Severity | None = None) -> None:
        """Assert a chunk containing ``needle`` was retrieved this run (gate by default).

        Reads ``RunResult.metadata['rag_results']`` (populated by AgentCore from
        ``AugmentationStrategy.last_results``, R4). Lets mock-mode evals assert on
        RETRIEVAL (pre-LLM, deterministic) -- the answer-faithfulness NUMBER still
        needs a live judge LLM (RAGAS), but retrieval does not.
        """
        sev = self._sev(severity)

        def _evaluate() -> AssertionOutcome:
            count = 0
            for turn in self._turns:
                for chunk in (turn.metadata or {}).get("rag_results", []) or []:
                    if needle.lower() in str(chunk.get("content", "")).lower():
                        count += 1
            return binary_outcome(sev, count > 0, f"retrievedChunk({needle!r}) -> {count} match(es)")

        self._record(f"retrievedChunk:{needle}", sev, _evaluate)

    def usedNoTools(self, *, severity: Severity | None = None) -> None:
        """Assert no tools were called across the whole test (gate by default)."""
        sev = self._sev(severity)

        def _evaluate() -> AssertionOutcome:
            count = len(self.all_tool_calls)
            return binary_outcome(sev, count == 0, f"usedNoTools -> {count} call(s)")

        self._record("usedNoTools", sev, _evaluate)

    def completed(self, *, severity: Severity | None = None) -> None:
        """Assert the last turn completed successfully (``RunResult.success``)."""
        sev = self._sev(severity)

        def _evaluate() -> AssertionOutcome:
            if not self._turns:
                return binary_outcome(sev, False, "completed -> no turns recorded")
            success = bool(self._turns[-1].success)
            return binary_outcome(sev, success, f"completed -> last.success={success}")

        self._record("completed", sev, _evaluate)

    def check(
        self,
        value: Any,
        matcher: Any = None,
        *,
        name: str = "check",
        severity: Severity | None = None,
    ) -> None:
        """Assert ``value`` satisfies ``matcher`` (value captured now; matcher applied at collect time).

        ``matcher`` may be a :class:`~koboi.eval.t.assertions.Matcher`, a callable,
        or a bare value (treated as :class:`~koboi.eval.t.assertions.Equals`). If
        omitted, asserts ``value`` is truthy. Soft by default (advisory).
        """
        sev = self._sev(severity if severity is not None else Severity.SOFT)
        resolved: Matcher = coerce_matcher(matcher) if matcher is not None else Truth()
        captured = value

        def _evaluate() -> AssertionOutcome:
            ok = resolved.matches(captured)
            if ok:
                reason = f"{name}: {resolved.describe()}"
            else:
                reason = f"{name}: expected {resolved.describe()}, got {describe_value(captured)}"
            return binary_outcome(sev, ok, reason)

        self._record(name, sev, _evaluate)

    async def judge(
        self,
        scorer: "str | type | BaseScorer",
        *,
        severity: Severity = Severity.SOFT,
        min_score: float = 0.7,
        expected: list[str] | None = None,
        expected_answer: str | None = None,
        name: str | None = None,
        **scorer_kwargs: Any,
    ) -> None:
        """Run a ScorerRegistry scorer against the last turn and record its score.

        Routes through :class:`~koboi.eval.registry.ScorerRegistry` so registered
        scorers (``llm_judge``, ``ragas_*``, ``deepeval_*``, ...) are available.
        Soft by default; ``passed`` is ``score >= min_score``. Scorers that need a
        client or optional dependency are recorded as a soft skip rather than
        crashing the test.
        """
        sev = self._sev(severity)
        try:
            scorer_obj = self._resolve_scorer(scorer, **scorer_kwargs)
        except Exception as exc:  # fail-soft: missing dep / missing client kwarg
            label = name or f"judge:{scorer}"
            self._record(label, sev, lambda e=exc: AssertionOutcome(False, 0.0, f"judge unavailable: {e}"))
            return

        label = name or f"judge:{type(scorer_obj).__name__}"
        case = self._synthetic_case(expected=expected, expected_answer=expected_answer)
        context = self._build_context()
        try:
            score = await scorer_obj.score(case, self.reply, context)
        except Exception as exc:
            self._record(label, sev, lambda e=exc: AssertionOutcome(False, 0.0, f"judge error: {e}"))
            return
        passed = score.value >= min_score
        reason = f"{score.name} {score.reason} [{score.value:.2f}]"
        self._record(label, sev, lambda p=passed, v=score.value, r=reason: AssertionOutcome(p, v, r))

    # ------------------------------------------------------------------ internal
    def _sev(self, override: Severity | None) -> Severity:
        return override if override is not None else self._default

    def _record(self, name: str, severity: Severity, evaluate: Any) -> None:
        self._recorded.append(RecordedAssertion(name=name, severity=severity, evaluate=evaluate))

    def record_gate_error(self, reason: str) -> None:
        """Record a hard gate failure (used for uncaught exceptions in the test body)."""
        self._record("test:error", Severity.GATE, lambda r=reason: AssertionOutcome(False, 0.0, r))

    def collect(self) -> list[RecordedAssertion]:
        """Return the recorded assertions (evaluated later by the runner)."""
        return list(self._recorded)

    def _resolve_scorer(self, scorer: Any, **kwargs: Any) -> "BaseScorer":
        from koboi.eval.registry import ScorerRegistry
        from koboi.eval.scorers.base import BaseScorer

        if isinstance(scorer, BaseScorer):
            return scorer
        if isinstance(scorer, str):
            return ScorerRegistry.create(scorer, **kwargs)
        if isinstance(scorer, type) and issubclass(scorer, BaseScorer):
            return scorer(**kwargs)
        raise TypeError(f"Cannot resolve scorer from {scorer!r}")

    def _synthetic_case(self, *, expected: list[str] | None = None, expected_answer: str | None = None):
        from koboi.types import EvalCase

        # R4: forward retrieved chunks so RAGAS-style scorers reading
        # case.context_docs work in live mode. (RAGAS faithfulness itself still
        # needs a live judge LLM -- this only supplies the context.)
        context_docs: list[str] = []
        if self._turns:
            for chunk in (self._turns[-1].metadata or {}).get("rag_results", []) or []:
                context_docs.append(str(chunk.get("content", "")))

        return EvalCase(
            name="t.judge",
            user_message=self._sent[0] if self._sent else "",
            expected_keywords=list(expected) if expected else [],
            expected_answer=expected_answer,
            context_docs=context_docs,
        )

    def _build_context(self) -> dict:
        """Build the scorer context dict, mirroring EvalRunner.run_case."""
        context: dict[str, Any] = {}
        if hasattr(self._agent, "get_telemetry"):
            telemetry = self._agent.get_telemetry()
            if telemetry:
                context["telemetry"] = telemetry
        usage = self.total_token_usage()
        if usage.total_tokens:
            context["token_usage"] = usage
        calls = self.all_tool_calls
        if calls:
            context["tool_calls"] = calls
        return context
