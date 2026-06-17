"""koboi/eval/scorers/ragas_scorer.py -- RAGAS metrics as koboi BaseScorers.

Wraps RAGAS framework metrics (faithfulness, answer_relevancy, context_precision,
context_recall) for use in koboi's eval system.

Requires: pip install ragas
"""
from __future__ import annotations

import logging
import os
from typing import Any

from koboi.types import EvalCase, EvalScore
from koboi.eval.scorers.base import BaseScorer

_logger = logging.getLogger(__name__)

_RAGAS_AVAILABLE = False
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    _RAGAS_AVAILABLE = True
except ImportError:
    pass


_METRIC_MAP = {
    "faithfulness": lambda: faithfulness if _RAGAS_AVAILABLE else None,
    "answer_relevancy": lambda: answer_relevancy if _RAGAS_AVAILABLE else None,
    "context_precision": lambda: context_precision if _RAGAS_AVAILABLE else None,
    "context_recall": lambda: context_recall if _RAGAS_AVAILABLE else None,
}


def _create_ragas_llm():
    """Create a RAGAS-compatible LLM using env vars.

    Provider selection via RAGAS_PROVIDER env var:
      - "openai" (default): uses OPENAI_* env vars + ChatOpenAI
      - "anthropic": uses ANTHROPIC_* env vars + ChatAnthropic
    """
    if not _RAGAS_AVAILABLE:
        return None
    try:
        from ragas.llms import LangchainLLMWrapper

        provider = os.environ.get("RAGAS_PROVIDER", "openai").lower()

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            model = os.environ.get("ANTHROPIC_MODEL", "mimo-v2.5-pro")
            api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")
            base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
            llm = ChatAnthropic(
                model=model, api_key=api_key, base_url=base_url,
                max_tokens=8192, streaming=False,
            )
        else:
            from langchain_openai import ChatOpenAI
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", "")
            llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)

        return LangchainLLMWrapper(llm)
    except Exception as e:
        _logger.warning("Failed to create RAGAS LLM: %s", e)
        return None


class RAGASScorer(BaseScorer):
    """Wraps a single RAGAS metric as a koboi BaseScorer.

    Requires the `ragas` pip package. Returns score 0.0 with explanatory
    reason if ragas is not installed (fail-open pattern).
    """

    def __init__(self, metric_name: str = "faithfulness", threshold: float = 0.7):
        self.metric_name = metric_name
        self.threshold = threshold
        self._metric = None
        if _RAGAS_AVAILABLE and metric_name in _METRIC_MAP:
            self._metric = _METRIC_MAP[metric_name]()

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        score_name = f"ragas_{self.metric_name}"

        if not _RAGAS_AVAILABLE:
            return EvalScore(score_name, 0.0, "ragas not installed (pip install ragas)")

        if not self._metric:
            return EvalScore(score_name, 0.0, f"Unknown RAGAS metric: {self.metric_name}")

        if not case.context_docs:
            return EvalScore(score_name, 0.0, "No context_docs in EvalCase")

        try:
            # Build RAGAS dataset format
            dataset = _build_ragas_dataset(
                question=case.user_message,
                answer=output,
                contexts=case.context_docs,
                ground_truth=case.expected_answer or "",
            )

            # Use custom LLM from env vars instead of RAGAS default
            llm = _create_ragas_llm()
            evaluate_kwargs: dict[str, Any] = {
                "dataset": dataset,
                "metrics": [self._metric],
            }
            if llm:
                evaluate_kwargs["llm"] = llm

            result = ragas_evaluate(**evaluate_kwargs)

            # Extract score for the metric (RAGAS v0.4+ API)
            # result is dict-like: {'faithfulness': 1.0} or {'faithfulness': [1.0]}
            if hasattr(result, '__getitem__'):
                raw = result[self.metric_name]
                if isinstance(raw, list):
                    score_val = float(raw[0]) if raw else 0.0
                else:
                    score_val = float(raw)
            elif hasattr(result, 'scores'):
                score_val = float(result.scores[0]) if result.scores else 0.0
            else:
                score_val = 0.0

            score_val = max(0.0, min(1.0, score_val))
            return EvalScore(score_name, round(score_val, 3), f"RAGAS {self.metric_name}")

        except Exception as e:
            _logger.warning("RAGAS %s failed: %s", self.metric_name, e)
            return EvalScore(score_name, 0.0, f"RAGAS error: {e}")


class RAGASCompositeScorer(BaseScorer):
    """Runs all RAGAS metrics in a single evaluate() call and returns weighted average."""

    DEFAULT_WEIGHTS = {
        "faithfulness": 0.3,
        "answer_relevancy": 0.3,
        "context_precision": 0.2,
        "context_recall": 0.2,
    }

    def __init__(self, metrics: dict[str, float] | None = None, threshold: float = 0.7):
        self.weights = metrics or self.DEFAULT_WEIGHTS
        self.threshold = threshold
        self._metrics = []
        if _RAGAS_AVAILABLE:
            for name in self.weights:
                m = _METRIC_MAP.get(name)
                if m:
                    self._metrics.append(m())

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        if not _RAGAS_AVAILABLE:
            return EvalScore("ragas_composite", 0.0, "ragas not installed (pip install ragas)")

        if not case.context_docs:
            return EvalScore("ragas_composite", 0.0, "No context_docs in EvalCase")

        try:
            dataset = _build_ragas_dataset(
                question=case.user_message,
                answer=output,
                contexts=case.context_docs,
                ground_truth=case.expected_answer or "",
            )

            llm = _create_ragas_llm()
            evaluate_kwargs: dict[str, Any] = {
                "dataset": dataset,
                "metrics": self._metrics,
            }
            if llm:
                evaluate_kwargs["llm"] = llm

            result = ragas_evaluate(**evaluate_kwargs)

            import math
            scores: dict[str, float] = {}
            for name in self.weights:
                try:
                    if hasattr(result, '__getitem__'):
                        raw = result[name]
                        if isinstance(raw, list):
                            val = float(raw[0]) if raw else 0.0
                        else:
                            val = float(raw)
                        scores[name] = 0.0 if math.isnan(val) else max(0.0, min(1.0, val))
                    else:
                        scores[name] = 0.0
                except Exception:
                    scores[name] = 0.0

            # Redistribute weights: only among metrics that returned valid scores
            valid = {m: w for m, w in self.weights.items() if m in scores and scores[m] > 0}
            if valid:
                total_w = sum(valid.values())
                weighted = sum(scores[m] * (w / total_w) for m, w in valid.items())
            else:
                weighted = 0.0
            details = ", ".join(f"{m}={v:.2f}" for m, v in scores.items())

            return EvalScore("ragas_composite", round(weighted, 3), details)

        except Exception as e:
            _logger.warning("RAGAS composite failed: %s", e)
            return EvalScore("ragas_composite", 0.0, f"RAGAS error: {e}")


def _build_ragas_dataset(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
) -> Any:
    """Build a RAGAS-compatible dataset from individual fields."""
    from datasets import Dataset

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth] if ground_truth else [""],
    }
    return Dataset.from_dict(data)
