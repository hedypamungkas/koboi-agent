"""koboi/eval/scorers/recency_scorer.py -- heuristic recency scorer for deep research.

Reads gathered source TEXT from ``context['research_sources']`` (populated by
``TestContext._build_context`` from ``RunResult.metadata['research_sources_with_text']``) and
checks what fraction mention a year within an N-year window of today. A stale-knowledge leak
(the model answering from training data instead of fetched sources) shows up as sources that
only mention old years.

This is a HEURISTIC PROXY (regex year extraction), not a perfect recency measure: it can be
fooled by a source that says "since 2019" or a historical mention. It is cheap (no LLM call),
deterministic, and catches the dominant failure mode (no recent-year signal at all). For a
stricter bar, swap in an LLM judge. Fail-open: no year extractable -> score 0.0 + reason.
"""

from __future__ import annotations

import re
from datetime import datetime

from koboi.eval.scorers.base import BaseScorer
from koboi.types import EvalCase, EvalScore

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _years_in(text: str) -> set[int]:
    return {int(m) for m in _YEAR_RE.findall(text or "")}


class RecencyScorer(BaseScorer):
    """Fraction of gathered sources that mention a year within ``recent_years`` of today.

    Args:
        recent_years: a source counts as "recent" if it mentions any year >=
            ``current_year - recent_years`` (default 1 -> this year or last year).
    """

    def __init__(self, recent_years: int = 1) -> None:
        self._recent_years = max(0, recent_years)

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        score_name = "recency"
        sources = context.get("research_sources") or []
        if not sources:
            return EvalScore(score_name, 0.0, "no research sources in context")

        current_year = datetime.now().year
        threshold = current_year - self._recent_years

        recent_sources = 0
        any_year_seen = False
        for s in sources:
            if not isinstance(s, dict):
                continue
            years = _years_in(str(s.get("text", "")))
            if years:
                any_year_seen = True
                if max(years) >= threshold:
                    recent_sources += 1

        if not any_year_seen:
            # No years in any source text -> fall back to the report itself.
            report_years = _years_in(output)
            if report_years:
                ok = max(report_years) >= threshold
                return EvalScore(
                    score_name,
                    1.0 if ok else 0.0,
                    f"no source years; report max year {max(report_years)} vs threshold {threshold}",
                )
            return EvalScore(score_name, 0.0, "no years extractable from sources or report")

        frac = recent_sources / len(sources)
        return EvalScore(
            score_name,
            round(frac, 3),
            f"{recent_sources}/{len(sources)} sources mention a year >= {threshold} (current={current_year})",
        )
