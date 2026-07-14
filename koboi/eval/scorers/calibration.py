"""koboi/eval/scorers/calibration.py -- conformal-style threshold selector (Wave 3 A5).

The principled ON-RAMP to calibrated confidence. Given per-sample
``(confidence_score, is_correct)`` pairs from a labeled set, derive the confidence
threshold below which the agent should abstain (A3) / hand over (B1.5) so that
answers kept (confidence >= threshold) meet a target correctness rate (CI lower bound).

This is the tool an operator runs on THEIR labeled data (e.g. a human-annotated PPI
set, or an LLM-judged set against ``expected_answer``) to set
``grounding_check.threshold`` / ``handover.detection.coverage_threshold`` PRINCIPLEDLY
instead of by guess. Reuses the existing ``bootstrap_ci`` (``ci.py``).

WHY NOT ship calibrated thresholds: there is NO human-annotated PPI dataset in the
repo (only retrieval qrels + reference answers). Shipping uncalibrated defaults as
"calibrated" is the false-confidence failure mode. This selector is the on-ramp; the
thresholds come from YOUR data. Runtime verbalized-P(IK) stamping is deferred -- it is
only valuable POST-calibration (an uncalibrated P(IK) is noise).

Usage (offline, needs a labeled set + optionally an LLM key for the judge)::

    # 1. Drive the agent on a labeled set; collect (coverage, correct) per query.
    samples = []
    for q in labeled_set:
        await t.send(q["query"])
        coverage = (t.last.metadata or {}).get("grounding_coverage")  # needs A5 stamping (deferred)
        correct = await llm_judge(t.reply, q["expected_answer"])       # 0/1
        samples.append((coverage, correct))
    # 2. Derive the threshold.
    from koboi.eval.scorers.calibration import select_conformal_threshold
    t_abstain = select_conformal_threshold(samples, target_correctness=0.9)
    # 3. Stamp to config: grounding_check.threshold = t_abstain; handover.detection.coverage_threshold = ...
"""

from __future__ import annotations

from collections.abc import Sequence

from koboi.eval.scorers.ci import bootstrap_ci


def select_conformal_threshold(
    samples: Sequence[tuple[float, bool]],
    target_correctness: float = 0.9,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 42,
) -> float | None:
    """Return the LOWEST confidence threshold whose kept-set (confidence >= threshold)
    correctness CI lower bound clears ``target_correctness`` -- i.e. the least-abstention
    threshold that still meets the target. ``None`` if unachievable on this data.

    ``samples``: ``[(confidence_score, is_correct), ...]``. The confidence_score is the
    per-turn signal you calibrate against (A3 grounding coverage, retrieval score, or a
    future verbalized P(IK)). ``is_correct`` is the ground-truth label (human or LLM judge).
    """
    valid = [(float(c), bool(ok)) for c, ok in samples if c is not None]
    if not valid:
        return None
    # Candidate thresholds = each unique confidence value; sweep to find those clearing target.
    candidates = sorted({c for c, _ in valid})
    clearing: list[float] = []
    for thresh in candidates:
        kept = [1.0 if ok else 0.0 for c, ok in valid if c >= thresh]
        if len(kept) < 2:
            # bootstrap_ci needs >=2 samples for a meaningful lower bound (N=1 -> [0,1] -> fails).
            continue
        ci = bootstrap_ci(kept, confidence=confidence, n_boot=n_boot, seed=seed)
        if ci.lower >= target_correctness:
            clearing.append(thresh)
    if not clearing:
        return None
    # Lowest clearing threshold = least abstention (keep the most) while meeting the target.
    return min(clearing)
