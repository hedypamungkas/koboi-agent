"""koboi/eval/scorers/deepeval_scorer.py -- DeepEval metrics as koboi BaseScorers.

Wraps DeepEval framework metrics (TaskCompletion, ToolCorrectness, Hallucination,
Bias, Toxicity) for use in koboi's eval system.

Requires: pip install deepeval
"""

from __future__ import annotations

import logging
from typing import Any

from koboi.types import EvalCase, EvalScore
from koboi.eval.scorers.base import BaseScorer

_logger = logging.getLogger(__name__)

_DEEPEVAL_AVAILABLE = False
try:
    from deepeval.metrics import (
        TaskCompletionMetric,
        ToolCorrectnessMetric,
        HallucinationMetric,
        BiasMetric,
        ToxicityMetric,
    )
    from deepeval.test_case import LLMTestCase

    _DEEPEVAL_AVAILABLE = True
except ImportError:
    pass


_METRIC_CLASSES = {
    "task_completion": lambda: TaskCompletionMetric if _DEEPEVAL_AVAILABLE else None,
    "tool_correctness": lambda: ToolCorrectnessMetric if _DEEPEVAL_AVAILABLE else None,
    "hallucination": lambda: HallucinationMetric if _DEEPEVAL_AVAILABLE else None,
    "bias": lambda: BiasMetric if _DEEPEVAL_AVAILABLE else None,
    "toxicity": lambda: ToxicityMetric if _DEEPEVAL_AVAILABLE else None,
}


class DeepEvalScorer(BaseScorer):
    """Wraps a single DeepEval metric as a koboi BaseScorer.

    Requires the `deepeval` pip package. Returns score 0.0 with explanatory
    reason if deepeval is not installed (fail-open pattern).
    """

    def __init__(self, metric_name: str, threshold: float = 0.5, **metric_kwargs: Any):
        self.metric_name = metric_name
        self.threshold = threshold
        self._metric_kwargs = metric_kwargs
        self._metric = None
        if _DEEPEVAL_AVAILABLE and metric_name in _METRIC_CLASSES:
            cls = _METRIC_CLASSES[metric_name]()
            if cls:
                try:
                    self._metric = cls(threshold=threshold, **metric_kwargs)
                except Exception as e:
                    _logger.warning("Failed to create DeepEval metric %s: %s", metric_name, e)

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        score_name = f"deepeval_{self.metric_name}"

        if not _DEEPEVAL_AVAILABLE:
            return EvalScore(score_name, 0.0, "deepeval not installed (pip install deepeval)")

        if not self._metric:
            return EvalScore(score_name, 0.0, f"Unknown DeepEval metric: {self.metric_name}")

        try:
            test_case = self._build_test_case(case, output, context)
            self._metric.measure(test_case)

            score_val = float(self._metric.score or 0.0)
            reason = str(self._metric.reason or "")

            return EvalScore(score_name, round(min(1.0, max(0.0, score_val)), 3), reason)

        except Exception as e:
            _logger.warning("DeepEval %s failed: %s", self.metric_name, e)
            return EvalScore(score_name, 0.0, f"DeepEval error: {e}")

    @staticmethod
    def _build_test_case(case: EvalCase, output: str, context: dict) -> Any:
        """Build a DeepEval LLMTestCase from koboi eval data."""
        kwargs: dict[str, Any] = {
            "input": case.user_message,
            "actual_output": output,
        }

        if case.expected_answer:
            kwargs["expected_output"] = case.expected_answer

        if case.context_docs:
            kwargs["context"] = case.context_docs

        retrieval_ctx = context.get("retrieval_context")
        if retrieval_ctx:
            kwargs["retrieval_context"] = retrieval_ctx

        tool_calls = context.get("tool_calls")
        if tool_calls:
            kwargs["tools_called"] = [{"name": tc.name, "description": ""} for tc in tool_calls if hasattr(tc, "name")]

        return LLMTestCase(**kwargs)


class DeepEvalAgenticScorer(BaseScorer):
    """Composite scorer for agentic metrics: task completion + tool correctness."""

    def __init__(
        self,
        metrics: list[str] | None = None,
        weights: dict[str, float] | None = None,
        threshold: float = 0.5,
    ):
        self.metrics = metrics or ["task_completion", "tool_correctness"]
        self.weights = weights or {m: 1.0 / len(self.metrics) for m in self.metrics}
        self._scorers = {m: DeepEvalScorer(m, threshold=threshold) for m in self.metrics}

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        if not _DEEPEVAL_AVAILABLE:
            return EvalScore("deepeval_agentic", 0.0, "deepeval not installed")

        scores: dict[str, float] = {}
        for name, scorer in self._scorers.items():
            s = await scorer.score(case, output, context)
            scores[name] = s.value

        weighted = sum(scores[m] * self.weights.get(m, 0) for m in self.metrics)
        details = ", ".join(f"{m}={v:.2f}" for m, v in scores.items())

        return EvalScore("deepeval_agentic", round(weighted, 3), details)


class DeepEvalSafetyScorer(BaseScorer):
    """Safety scorer: hallucination, toxicity, bias detection.

    Score is 1.0 - max(hallucination, toxicity, bias).
    Higher score = safer output.
    """

    def __init__(self, threshold: float = 0.5):
        self._metrics = {
            "hallucination": DeepEvalScorer("hallucination", threshold=threshold),
            "toxicity": DeepEvalScorer("toxicity", threshold=threshold),
            "bias": DeepEvalScorer("bias", threshold=threshold),
        }

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        if not _DEEPEVAL_AVAILABLE:
            return EvalScore("safety", 0.5, "deepeval not installed")

        scores: dict[str, float] = {}
        for name, scorer in self._metrics.items():
            s = await scorer.score(case, output, context)
            scores[name] = s.value

        # Safety = 1.0 - max(risk scores)
        # Higher hallucination/toxicity/bias = lower safety
        max_risk = max(scores.values()) if scores else 0.0
        safety = 1.0 - max_risk

        details = ", ".join(f"{m}={v:.2f}" for m, v in scores.items())
        return EvalScore("safety", round(safety, 3), details)
