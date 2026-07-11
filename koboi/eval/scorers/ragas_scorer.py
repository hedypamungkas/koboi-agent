"""koboi/eval/scorers/ragas_scorer.py -- RAGAS metrics as koboi BaseScorers.

Wraps RAGAS framework metrics (faithfulness, answer_relevancy, context_precision,
context_recall, factual_correctness) for use in koboi's eval system.

Requires: pip install ragas  (the ``[eval-ragas]`` extra)

ragas 0.4.x note: ``evaluate()`` validates metrics against the legacy
``ragas.metrics.base.Metric`` hierarchy. The forward-looking classes under
``ragas.metrics.collections`` extend a *new* ``BaseMetric`` that ``evaluate()`` does
NOT accept yet, so we deliberately import the (deprecated-but-working) top-level
``ragas.metrics`` classes. Metrics are constructed with no args; ``evaluate()`` injects
the judge ``llm`` / ``embeddings`` into ``MetricWithLLM`` / ``MetricWithEmbeddings``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from koboi.types import EvalCase, EvalScore
from koboi.eval.scorers.base import BaseScorer

_logger = logging.getLogger(__name__)


def _apply_langchain_community_shim() -> None:
    """Compat shim so ``import ragas`` succeeds on modern langchain-community.

    ragas imports ``langchain_community.chat_models.vertexai.ChatVertexAI`` at module
    load, but langchain-community >=0.4 (sunset release) removed that path. ragas only
    *uses* ChatVertexAI for the Vertex provider; on the OpenAI/Anthropic path it is
    never instantiated, so we stub it when the real import is missing. No-op when the
    path already exists or langchain-community is absent.
    """
    import sys
    import types

    try:
        import langchain_community.chat_models.vertexai  # type: ignore[import-not-found]  # noqa: F401
        return
    except Exception:
        pass  # nosec B110 - intentional: detecting the missing import decides whether to shim
    try:
        import langchain_community  # type: ignore[import-not-found]
    except Exception:
        return  # langchain-community not installed -> ragas import fails softly below

    class _Stub:  # minimal placeholder; never instantiated on OpenAI/Anthropic paths
        pass

    cm = types.ModuleType("langchain_community.chat_models")
    cm.ChatVertexAI = _Stub  # type: ignore[attr-defined]
    vx = types.ModuleType("langchain_community.chat_models.vertexai")
    vx.ChatVertexAI = _Stub  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models"] = cm
    sys.modules["langchain_community.chat_models.vertexai"] = vx
    langchain_community.chat_models = cm  # type: ignore[attr-defined]


_apply_langchain_community_shim()

_RAGAS_AVAILABLE = False
_METRIC_CLASSES: dict[str, Any] = {}
# True when _METRIC_CLASSES maps names to CLASSES (instantiate with no args); False for
# ragas 0.1 module-level singleton instances.
_INSTANTIATE_METRICS = False

try:
    from ragas import evaluate as ragas_evaluate

    # ragas 0.2-0.4: top-level metric CLASSES (ragas.metrics) are the legacy Metric
    # subclasses evaluate() accepts. (ragas 0.4's ragas.metrics.collections classes use
    # a new BaseMetric that evaluate() rejects -- avoid them.)
    from ragas.metrics import (  # type: ignore[import-not-found]
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    _METRIC_CLASSES = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
        "factual_correctness": FactualCorrectness,
    }
    _INSTANTIATE_METRICS = True
    _RAGAS_AVAILABLE = True
except ImportError:
    try:
        # ragas 0.1: module-level singleton metric instances.
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (  # type: ignore[import-not-found]
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        _METRIC_CLASSES = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        _INSTANTIATE_METRICS = False
        _RAGAS_AVAILABLE = True
    except ImportError:
        pass

# Alias for clarity in scorer logs.
_METRIC_MAP = _METRIC_CLASSES


def _build_metric(name: str) -> Any:
    """Build one RAGAS metric instance (no args; ``evaluate()`` injects llm/embeddings)."""
    cls = _METRIC_CLASSES.get(name)
    if cls is None:
        return None
    return cls() if _INSTANTIATE_METRICS else cls


def _extract_ragas_score(result: Any, name: str) -> float:
    """Pull one metric's score out of a ragas EvaluationResult.

    Result is dict-like keyed by metric name -> float or list[float]. Some metrics
    (e.g. FactualCorrectness) key under ``name(mode=...)``; fall back to a prefix match.
    """
    try:
        raw = result[name]
    except (KeyError, TypeError, IndexError):
        # Fallback: find a key that starts with the metric name.
        raw = None
        if hasattr(result, "__getitem__"):
            for key in list(getattr(result, "_scores_dict", {}).keys()):
                if str(key).startswith(name):
                    raw = result[key]
                    break
        if raw is None:
            return 0.0
    if isinstance(raw, list):
        return float(raw[0]) if raw else 0.0
    return float(raw)


def _create_ragas_llm():
    """Create a RAGAS judge LLM from env vars (provider via RAGAS_PROVIDER, default openai)."""
    if not _RAGAS_AVAILABLE:
        return None
    provider = os.environ.get("RAGAS_PROVIDER", "openai").lower()
    try:
        if _INSTANTIATE_METRICS:
            # ragas >=0.2: InstructorLLM via llm_factory + a raw provider client.
            from ragas.llms import llm_factory

            if provider == "anthropic":
                from anthropic import Anthropic  # type: ignore[import-not-found]

                model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
                api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
                base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
                kwargs: dict[str, Any] = {"api_key": api_key}
                if base_url:
                    kwargs["base_url"] = base_url
                return llm_factory(model, provider="anthropic", client=Anthropic(**kwargs))

            from openai import OpenAI  # type: ignore[import-not-found]

            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL") or None
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return llm_factory(model, provider="openai", client=OpenAI(**kwargs))

        # ragas 0.1: LangchainLLMWrapper around a langchain chat model.
        from ragas.llms import LangchainLLMWrapper

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                model=os.environ.get("ANTHROPIC_MODEL", "mimo-v2.5-pro"),
                api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", ""),
                base_url=os.environ.get("ANTHROPIC_BASE_URL", ""),
                max_tokens=8192,
                streaming=False,
            )
        else:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                base_url=os.environ.get("OPENAI_BASE_URL", ""),
            )
        return LangchainLLMWrapper(llm)
    except Exception as e:
        _logger.warning("Failed to create RAGAS LLM: %s", e)
        return None


def _create_ragas_embeddings():
    """Create RAGAS embeddings (LangchainEmbeddingsWrapper) for embedding-based metrics.

    Uses a dedicated embedding endpoint (EMBEDDING_*) when configured, else falls back
    to the OpenAI chat creds. Returns None when langchain_openai isn't installed -- LLM-
    only metrics (faithfulness, context_recall) still work without it.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
    except Exception as e:  # ragas/langchain-openai not installed
        _logger.debug("RAGAS embeddings unavailable: %s", e)
        return None
    try:
        model = os.environ.get("EMBEDDING_MODEL") or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None
        kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return LangchainEmbeddingsWrapper(OpenAIEmbeddings(**kwargs))
    except Exception as e:
        _logger.warning("Failed to create RAGAS embeddings: %s", e)
        return None


class RAGASScorer(BaseScorer):
    """Wraps a single RAGAS metric as a koboi BaseScorer.

    Requires the `ragas` pip package. Returns score 0.0 with explanatory reason if ragas
    is not installed or the judge LLM/embeddings aren't configured (fail-open pattern).
    """

    def __init__(self, metric_name: str = "faithfulness", threshold: float = 0.7):
        self.metric_name = metric_name
        self.threshold = threshold

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        score_name = f"ragas_{self.metric_name}"

        if not _RAGAS_AVAILABLE:
            return EvalScore(score_name, 0.0, "ragas not installed (pip install ragas)")

        if self.metric_name not in _METRIC_CLASSES:
            return EvalScore(score_name, 0.0, f"Unknown RAGAS metric: {self.metric_name}")

        if not case.context_docs:
            return EvalScore(score_name, 0.0, "No context_docs in EvalCase")

        try:
            metric = _build_metric(self.metric_name)
            if metric is None:
                return EvalScore(score_name, 0.0, f"RAGAS {self.metric_name} unavailable")
            llm = _create_ragas_llm()
            if llm is None:
                return EvalScore(score_name, 0.0, "RAGAS needs a judge LLM (set OPENAI_*/ANTHROPIC_*)")

            dataset = _build_ragas_dataset(
                question=case.user_message,
                answer=output,
                contexts=case.context_docs,
                ground_truth=case.expected_answer or "",
            )

            evaluate_kwargs: dict[str, Any] = {
                "dataset": dataset,
                "metrics": [metric],
                "llm": llm,
            }
            embeddings = _create_ragas_embeddings()
            if embeddings is not None:
                evaluate_kwargs["embeddings"] = embeddings

            result = ragas_evaluate(**evaluate_kwargs)
            score_val = max(0.0, min(1.0, _extract_ragas_score(result, self.metric_name)))
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

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        if not _RAGAS_AVAILABLE:
            return EvalScore("ragas_composite", 0.0, "ragas not installed (pip install ragas)")

        if not case.context_docs:
            return EvalScore("ragas_composite", 0.0, "No context_docs in EvalCase")

        try:
            built = [m for m in (_build_metric(name) for name in self.weights) if m is not None]
            if not built:
                return EvalScore("ragas_composite", 0.0, "RAGAS composite: no metrics available")
            llm = _create_ragas_llm()
            if llm is None:
                return EvalScore("ragas_composite", 0.0, "RAGAS needs a judge LLM (set OPENAI_*/ANTHROPIC_*)")

            dataset = _build_ragas_dataset(
                question=case.user_message,
                answer=output,
                contexts=case.context_docs,
                ground_truth=case.expected_answer or "",
            )

            evaluate_kwargs: dict[str, Any] = {
                "dataset": dataset,
                "metrics": built,
                "llm": llm,
            }
            embeddings = _create_ragas_embeddings()
            if embeddings is not None:
                evaluate_kwargs["embeddings"] = embeddings

            result = ragas_evaluate(**evaluate_kwargs)

            import math

            scores: dict[str, float] = {}
            for name in self.weights:
                try:
                    val = _extract_ragas_score(result, name)
                    scores[name] = 0.0 if math.isnan(val) else max(0.0, min(1.0, val))
                except Exception:
                    scores[name] = 0.0

            # Redistribute weights: only among metrics that returned valid (>0) scores.
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
