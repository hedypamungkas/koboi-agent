"""koboi/eval/scorers/ci.py -- bootstrap confidence-interval helper + scorer.

Turns point-estimate eval scores (e.g. recall@10 over a handful of queries) into a
statistically defensible claim by gating on the 95% CI *lower bound* rather than the
mean. Required before any external "production-ready" assertion -- an auditor asking
"at what confidence?" needs a CI, not a single number.

Pure stdlib (``random``), seedable for determinism; sufficient for N ~ 100. A
prediction-powered-inference (PPI) variant that incorporates human annotations is
deferred to a later tier.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from koboi.types import EvalCase, EvalScore
from koboi.eval.scorers.base import BaseScorer


@dataclass(frozen=True)
class CIResult:
    """Bootstrap confidence interval for a list of per-query scores."""

    mean: float
    lower: float
    upper: float
    half_width: float
    n: int


def bootstrap_ci(
    scores: list[float],
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 42,
) -> CIResult:
    """Percentile bootstrap CI over ``scores``.

    Resamples with replacement ``n_boot`` times, computes the mean of each resample,
    and takes the percentile bounds. Deterministic given ``seed``.
    """
    n = len(scores)
    if n == 0:
        return CIResult(0.0, 0.0, 0.0, 0.0, 0)
    if n == 1:
        # One observation of a [0,1] variable carries ~no information about the spread:
        # the honest conservative 95% CI is full-width. This makes any CI-lower-bound
        # gate FAIL at N=1 (you cannot pass on a single sample) and forces growing N.
        return CIResult(float(scores[0]), 0.0, 1.0, 0.5, 1)

    rng = random.Random(seed)  # nosec B311 - bootstrap resampling, not cryptographic
    means: list[float] = []
    for _ in range(n_boot):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += scores[rng.randrange(n)]
        means.append(sample_sum / n)
    means.sort()

    alpha = 1.0 - confidence
    lo_idx = min(max(int((alpha / 2.0) * n_boot), 0), n_boot - 1)
    hi_idx = min(max(int((1.0 - alpha / 2.0) * n_boot) - 1, 0), n_boot - 1)
    lower = means[lo_idx]
    upper = means[hi_idx]
    mean = sum(scores) / n
    return CIResult(mean, lower, upper, (upper - lower) / 2.0, n)


class BootstrapCIScorer(BaseScorer):
    """Gate a dimension on the bootstrap 95% CI lower bound of per-query scores.

    Reads ``case.metadata['samples']`` (or ``context['samples']``) -- a list of
    per-query metric values -- and returns the CI lower bound as the score. Pair with
    ``min_score`` on the desired lower bound (not the mean).
    """

    def __init__(self, confidence: float = 0.95, n_boot: int = 2000, seed: int = 42):
        self.confidence = confidence
        self.n_boot = n_boot
        self.seed = seed

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        samples: list[float] = []
        meta = case.metadata.get("samples") if isinstance(case.metadata, dict) else None
        if isinstance(meta, list):
            samples = [float(x) for x in meta]
        if not samples and isinstance(context.get("samples"), list):
            samples = [float(x) for x in context["samples"]]
        if not samples:
            return EvalScore("bootstrap_ci", 0.0, "no samples in case.metadata/context")

        ci = bootstrap_ci(samples, self.confidence, self.n_boot, self.seed)
        pct = int(self.confidence * 100)
        note = " (n<2: uninformative full-width CI — grow N)" if ci.n < 2 else ""
        reason = f"{pct}% CI=[{ci.lower:.3f}, {ci.upper:.3f}] hw={ci.half_width:.3f} n={ci.n} mean={ci.mean:.3f}{note}"
        return EvalScore("bootstrap_ci", round(ci.lower, 3), reason)
