"""koboi/eval/scorers/deep_research_scorer.py -- W6.1 faithfulness scorer for deep research.

Reads the gathered source TEXT from ``context['research_sources']`` (populated by
``TestContext._build_context`` from ``RunResult.metadata['research_sources_with_text']``) and
scores whether the report's claims are GROUNDED in those sources, via RAGAS ``faithfulness``.
Unlike ``RAGASScorer`` (which reads static ``case.context_docs``), this scorer reads DYNAMIC
run-derived sources — the precedent is ``SkillTriggerAccuracyScorer`` (reads
``context['skills_activated']``).
"""

from __future__ import annotations

import logging
from typing import Any

from koboi.eval.scorers.base import BaseScorer
from koboi.eval.scorers.ragas_scorer import (
    _RAGAS_AVAILABLE,
    _METRIC_MAP,
    _build_ragas_dataset,
    _create_ragas_llm,
)
from koboi.types import EvalCase, EvalScore

_logger = logging.getLogger(__name__)

try:
    from ragas import evaluate as ragas_evaluate  # type: ignore[import-not-found]
except ImportError:
    ragas_evaluate = None  # type: ignore[assignment]


class DeepResearchFaithfulnessScorer(BaseScorer):
    """Scores whether a deep_research report's claims are grounded in its gathered sources.

    Reads ``context['research_sources']`` (list of ``{citation_id, node_id, text}`` dicts) +
    runs RAGAS ``faithfulness`` on ``(question, answer=report, contexts=source_texts)``. Fail-open:
    no ragas / no sources → score 0.0 + reason.
    """

    def __init__(self) -> None:
        self._metric = _METRIC_MAP.get("faithfulness", lambda: None)() if _RAGAS_AVAILABLE else None

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:  # noqa: C901
        score_name = "deep_research_faithfulness"

        if not _RAGAS_AVAILABLE or ragas_evaluate is None:
            return EvalScore(score_name, 0.0, "ragas not installed (pip install ragas)")

        if not self._metric:
            return EvalScore(score_name, 0.0, "RAGAS faithfulness metric unavailable")

        sources = context.get("research_sources") or []
        contexts = [s.get("text", "") for s in sources if isinstance(s, dict) and s.get("text")]
        if not contexts:
            return EvalScore(score_name, 0.0, "no research sources with text in context")

        try:
            dataset = _build_ragas_dataset(
                question=case.user_message,
                answer=output,
                contexts=contexts,
                ground_truth=case.expected_answer or "",
            )
            llm = _create_ragas_llm()
            evaluate_kwargs: dict[str, Any] = {"dataset": dataset, "metrics": [self._metric]}
            if llm:
                evaluate_kwargs["llm"] = llm
            result = ragas_evaluate(**evaluate_kwargs)

            if hasattr(result, "__getitem__"):
                raw = result["faithfulness"]
                score_val = float(raw[0]) if isinstance(raw, list) and raw else float(raw)
            elif hasattr(result, "scores"):
                score_val = float(result.scores[0]) if result.scores else 0.0
            else:
                score_val = 0.0

            return EvalScore(
                score_name,
                round(max(0.0, min(1.0, score_val)), 3),
                "RAGAS faithfulness (deep research)",
            )
        except Exception as e:  # noqa: BLE001 - scorer boundary: any failure -> fail-open
            _logger.warning("deep_research_faithfulness failed: %s", e)
            return EvalScore(score_name, 0.0, f"RAGAS error: {e}")
