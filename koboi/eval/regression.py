"""koboi/eval/regression.py -- Regression tracking for eval results.

Store baselines, compare against new runs, detect regressions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from koboi.types import EvalResult

_logger = logging.getLogger(__name__)


@dataclass
class RegressionReport:
    """Comparison between current and baseline eval results."""

    improved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    new_cases: list[str] = field(default_factory=list)
    removed_cases: list[str] = field(default_factory=list)
    score_delta: dict[str, float] = field(default_factory=dict)

    @property
    def has_regression(self) -> bool:
        return len(self.regressed) > 0

    def summary(self) -> str:
        lines = [
            f"Improved:   {len(self.improved)}",
            f"Regressed:  {len(self.regressed)}",
            f"Unchanged:  {len(self.unchanged)}",
            f"New:        {len(self.new_cases)}",
            f"Removed:    {len(self.removed_cases)}",
        ]
        if self.regressed:
            lines.append("")
            lines.append("Regressed cases:")
            for name in self.regressed:
                delta = self.score_delta.get(name, 0)
                lines.append(f"  - {name}: {delta:+.3f}")
        return "\n".join(lines)


class RegressionTracker:
    """Compare eval results across runs, store and load baselines."""

    def __init__(self, baseline_dir: str = "eval_baselines"):
        self.baseline_dir = Path(baseline_dir)

    def save_baseline(self, suite_name: str, results: list[EvalResult]) -> Path:
        """Save eval results as a baseline for future comparison."""
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        path = self.baseline_dir / f"{suite_name}.json"

        data = {
            "suite_name": suite_name,
            "cases": [
                {
                    "case_name": r.case_name,
                    "overall_score": r.overall_score,
                    "passed": r.passed,
                    "framework": r.framework,
                    "scores": [{"name": s.name, "value": s.value, "reason": s.reason} for s in r.scores],
                    "duration_seconds": r.duration_seconds,
                }
                for r in results
            ],
        }

        path.write_text(json.dumps(data, indent=2))
        _logger.info("Saved baseline for '%s' to %s", suite_name, path)
        return path

    def load_baseline(self, suite_name: str) -> list[dict[str, Any]] | None:
        """Load baseline results for a suite. Returns None if not found."""
        path = self.baseline_dir / f"{suite_name}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data.get("cases", [])

    def compare(
        self,
        current: list[EvalResult],
        baseline: list[dict[str, Any]],
        threshold: float = 0.05,
    ) -> RegressionReport:
        """Compare current results against baseline.

        Args:
            current: Current eval results.
            baseline: Baseline case dicts (from load_baseline).
            threshold: Score drop exceeding this is considered a regression.
        """
        baseline_map = {c["case_name"]: c for c in baseline}
        current_map = {r.case_name: r for r in current}

        report = RegressionReport()

        for name, result in current_map.items():
            if name not in baseline_map:
                report.new_cases.append(name)
                continue

            base_score = baseline_map[name]["overall_score"]
            curr_score = result.overall_score
            delta = curr_score - base_score
            report.score_delta[name] = delta

            if delta > threshold:
                report.improved.append(name)
            elif delta < -threshold:
                report.regressed.append(name)
            else:
                report.unchanged.append(name)

        for name in baseline_map:
            if name not in current_map:
                report.removed_cases.append(name)

        return report
