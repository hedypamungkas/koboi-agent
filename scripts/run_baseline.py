#!/usr/bin/env python
"""Baseline benchmark runner.

Loads the benchmark eval suite, runs all cases through the agent,
prints formatted results, and saves a baseline for regression tracking.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --config configs/benchmark_baseline.yaml
    python scripts/run_baseline.py --max-cases 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def main() -> None:
    # Load .env file before anything else
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Run baseline benchmark eval suite")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "benchmark_baseline.yaml"),
        help="Agent config YAML path",
    )
    parser.add_argument(
        "--eval-config",
        default=str(ROOT / "configs" / "benchmark_eval.yaml"),
        help="Eval suite config YAML path",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of cases to run (for quick smoke tests)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Pass/fail threshold (default: 0.6)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run cases in parallel",
    )
    args = parser.parse_args()

    # Lazy imports to avoid slow startup on --help
    import yaml
    from koboi.facade import KoboiAgent
    from koboi.eval.runner import EvalRunner
    from koboi.eval.loaders import YAMLLoader
    from koboi.eval.scorers.base import (
        ToolUsageScorer,
        KeywordPresenceScorer,
        OutputLengthScorer,
        HealthScoreScorer,
        RAGNoiseScorer,
        ContextEfficiencyScorer,
        ToolSelectionScorer,
        TokenEfficiencyScorer,
    )
    from koboi.eval.regression import RegressionTracker

    # Load eval config to get suite info
    with open(args.eval_config) as f:
        eval_cfg = yaml.safe_load(f).get("eval", {})

    suite_cfg = eval_cfg.get("suites", [{}])[0]
    suite_name = suite_cfg.get("name", "baseline_routing")
    suite_source = suite_cfg.get("source", "")

    # Resolve source path relative to project root
    source_path = ROOT / suite_source
    if not source_path.exists():
        print(f"ERROR: Eval source not found: {source_path}")
        sys.exit(1)

    # Load eval cases
    loader = YAMLLoader()
    cases = await loader.load(str(source_path))

    if args.max_cases:
        cases = cases[: args.max_cases]

    if not cases:
        print("WARNING: No eval cases found.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  BASELINE BENCHMARK: {suite_name}")
    print(f"  Config: {args.config}")
    print(f"  Cases:  {len(cases)}")
    print(f"  Threshold: {args.threshold}")
    print(f"{'=' * 70}\n")

    # Define harness factory
    config_path = args.config

    def harness_factory():
        return KoboiAgent.from_config(config_path)

    # Build scorers
    scorers = [
        ToolUsageScorer(),
        KeywordPresenceScorer(),
        OutputLengthScorer(),
        HealthScoreScorer(),
        RAGNoiseScorer(),
        ContextEfficiencyScorer(),
        ToolSelectionScorer(),
        TokenEfficiencyScorer(),
    ]

    # Create runner and execute
    runner = EvalRunner(
        harness_factory=harness_factory,
        scorers=scorers,
        threshold=args.threshold,
    )

    results = await runner.run_suite(
        cases,
        parallel=args.parallel,
        max_concurrency=eval_cfg.get("max_concurrency", 5),
        threshold=args.threshold,
    )

    # Print formatted results
    formatted = runner.format_results(results, threshold=args.threshold)
    print(formatted)

    # Save baseline for regression tracking
    regression_cfg = eval_cfg.get("regression", {})
    baseline_dir = ROOT / regression_cfg.get("baseline_dir", "eval_baselines")
    baseline_dir.mkdir(parents=True, exist_ok=True)

    tracker = RegressionTracker(baseline_dir=str(baseline_dir))
    tracker.save_baseline(suite_name, results)
    print(f"\nBaseline saved to: {baseline_dir / f'{suite_name}.json'}")

    # Print per-category summary
    _print_category_summary(results, args.threshold)


def _print_category_summary(results, threshold: float) -> None:
    """Print a summary grouped by eval case tags."""
    from collections import defaultdict

    categories: dict[str, list] = defaultdict(list)
    for r in results:
        tags = r.metadata.get("tags", [])
        for tag in tags:
            if tag in ("rag", "tools", "ambiguous", "noise"):
                categories[tag].append(r)

    if not categories:
        return

    print(f"\n{'=' * 70}")
    print("  CATEGORY SUMMARY")
    print(f"{'=' * 70}")

    for cat in ["rag", "tools", "ambiguous", "noise"]:
        if cat not in categories:
            continue
        cat_results = categories[cat]
        passed = sum(1 for r in cat_results if r.overall_score >= threshold)
        total = len(cat_results)
        avg = sum(r.overall_score for r in cat_results) / total if total else 0
        avg_tokens = 0
        token_counts = [r.token_usage.total_tokens for r in cat_results if r.token_usage]
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
        print(f"  {cat:12s}: {passed}/{total} passed | avg score: {avg:.1%} | avg tokens: {avg_tokens:.0f}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
