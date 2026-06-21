"""examples/28_rag_baseline.py -- RAG retrieval baseline + RAGAS end-to-end.

Establishes a measurable RAG baseline over the Acme Corp gold set
(examples/data/eval_cases/rag_acme_gold.yaml), mirroring examples/27_benchmark_suite.py.

Two modes:
  - offline (DEFAULT, no API key): deterministic recall@k / precision@k / hit@k over
    the retriever for factual cases, plus gate_noise for gating negatives. Fully
    reproducible -> committed as the regression baseline (eval_baselines/rag_retrieval.json).
  - live (needs API + `ragas`): end-to-end RAGAS faithfulness/precision/recall over a
    RAG-enabled agent. A model/machine-specific snapshot (eval_baselines/rag_ragas.json).

Usage:
    python examples/28_rag_baseline.py                              # offline run, print table
    python examples/28_rag_baseline.py --save-baseline              # offline + write baseline
    python examples/28_rag_baseline.py --compare                    # offline + diff vs baseline
    python examples/28_rag_baseline.py --mode live --save-baseline  # end-to-end RAGAS snapshot
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Project root on path (mirror examples/27_benchmark_suite.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

import yaml  # noqa: E402

from koboi import KoboiAgent  # noqa: E402
from koboi.eval import (  # noqa: E402
    EvalRunner,
    EvalResult,
    RegressionTracker,
    LoaderRegistry,
    ScorerRegistry,
)
from koboi.eval.scorers.retrieval_scorer import RetrievalScorer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = PROJECT_ROOT / "examples" / "data" / "eval_cases" / "rag_acme_gold.yaml"
RAG_CONFIG = PROJECT_ROOT / "configs" / "rag_eval.yaml"
THRESHOLD = 0.6


def _display(path: Path) -> str:
    """Render a path relative to PROJECT_ROOT when possible, else as-is."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_rag_conf() -> dict:
    """Load the `rag:` block from configs/rag_eval.yaml with absolute document paths."""
    rag_conf = yaml.safe_load(RAG_CONFIG.read_text())["rag"]
    for doc in rag_conf.get("documents", []):
        rel = doc.get("path", "")
        if rel and not Path(rel).is_absolute():
            doc["path"] = str(PROJECT_ROOT / rel)
    return rag_conf


def build_retriever():
    """Build the keyword retriever over the Acme corpus (no LLM, no API)."""
    from koboi.rag.registry import _load_documents
    from koboi.rag.retriever import resolve_retriever

    rag_conf = _load_rag_conf()
    _chunker, chunks = _load_documents(rag_conf)
    retriever = resolve_retriever(rag_conf, chunks)
    return retriever, len(chunks)


# ---------------------------------------------------------------------------
# Offline mode: deterministic retrieval scoring
# ---------------------------------------------------------------------------


async def run_offline(top_k: int, save: bool, compare: bool) -> list[EvalResult]:
    """Score factual cases (recall/precision/hit) + gating cases (gate_noise)."""
    print("\n=== RAG Retrieval Baseline (offline, deterministic) ===")
    retriever, chunk_count = build_retriever()
    print(f"Indexed {chunk_count} chunks from {RAG_CONFIG.relative_to(PROJECT_ROOT)}")

    cases = await LoaderRegistry.load("yaml", str(GOLD_PATH))
    factual = [c for c in cases if c.metadata.get("needs_retrieval", True)]
    gating = [c for c in cases if not c.metadata.get("needs_retrieval", True)]
    print(f"Gold set: {len(factual)} factual, {len(gating)} gating negatives")

    results: list[EvalResult] = []

    fact_scorers = [
        RetrievalScorer("recall", retriever=retriever, top_k=top_k),
        RetrievalScorer("precision", retriever=retriever, top_k=top_k),
        RetrievalScorer("hit", retriever=retriever, top_k=top_k),
    ]
    for case in factual:
        start = time.time()
        scores = [await s.score(case, "", {}) for s in fact_scorers]
        overall = sum(s.value for s in scores) / len(scores) if scores else 0.0
        results.append(
            EvalResult(
                case_name=case.name,
                output="",
                scores=scores,
                overall_score=round(overall, 3),
                duration_seconds=round(time.time() - start, 3),
                passed=overall >= THRESHOLD,
                framework="retrieval",
            )
        )

    gate_scorer = RetrievalScorer("gate_noise", retriever=retriever, top_k=top_k)
    for case in gating:
        start = time.time()
        score = await gate_scorer.score(case, "", {})
        # gate_noise (raw) is in score.value: lower = better. Invert for overall_score
        # so the higher=better convention (format_results status, regression compare)
        # holds: a future noise *increase* drops overall_score -> flagged as regression.
        overall = round(1.0 - score.value, 3)
        results.append(
            EvalResult(
                case_name=case.name,
                output="",
                scores=[score],
                overall_score=overall,
                duration_seconds=round(time.time() - start, 3),
                passed=overall >= THRESHOLD,
                framework="retrieval-gate",
            )
        )

    print(EvalRunner.format_results(results, threshold=THRESHOLD))

    _summarize(results)

    if save:
        tracker = RegressionTracker()
        path = tracker.save_baseline("rag_retrieval", results)
        print(f"\nBaseline saved to {_display(path)}")

    if compare:
        _compare(results)

    return results


def _summarize(results: list[EvalResult]) -> None:
    """Print aggregate metric averages split by framework."""
    by_fw: dict[str, list[EvalResult]] = {}
    for r in results:
        by_fw.setdefault(r.framework or "?", []).append(r)

    print(f"\n{'=' * 60}")
    for fw, group in by_fw.items():
        # average each score name across the group
        name_buckets: dict[str, list[float]] = {}
        for r in group:
            for s in r.scores:
                name_buckets.setdefault(s.name, []).append(s.value)
        label = "factual" if fw == "retrieval" else "gating (lower=better)"
        line = [f"{label} ({len(group)}):"]
        for name, vals in name_buckets.items():
            avg = sum(vals) / len(vals)
            line.append(f"{name} avg={avg:.3f}")
        print("  " + " | ".join(line))
    print(f"{'=' * 60}")


def _compare(current: list[EvalResult]) -> None:
    tracker = RegressionTracker()
    baseline = tracker.load_baseline("rag_retrieval")
    if not baseline:
        print("\nNo rag_retrieval baseline found -- run with --save-baseline first.")
        return
    report = tracker.compare(current, baseline, threshold=0.05)
    print("\n--- Regression vs rag_retrieval baseline ---")
    print(report.summary())


# ---------------------------------------------------------------------------
# Live mode: end-to-end RAGAS over a RAG-enabled agent
# ---------------------------------------------------------------------------


async def run_live(top_k: int, save: bool) -> list[EvalResult]:
    print("\n=== RAG End-to-End Baseline (live, RAGAS) ===")
    print(f"Config: {RAG_CONFIG.relative_to(PROJECT_ROOT)}")
    print("Available scorers:", ScorerRegistry.list_available())

    cases = await LoaderRegistry.load("yaml", str(GOLD_PATH))

    def harness_factory() -> KoboiAgent:
        return KoboiAgent.from_config(str(RAG_CONFIG))

    scorers = ScorerRegistry.from_config(
        [
            {"name": "ragas_faithfulness"},
            {"name": "ragas_precision"},
            {"name": "ragas_recall"},
        ]
    )
    runner = EvalRunner(harness_factory=harness_factory, scorers=scorers, threshold=THRESHOLD)
    results = await runner.run_suite(cases, parallel=False)
    print(runner.format_results(results, threshold=THRESHOLD))

    if save:
        tracker = RegressionTracker()
        path = tracker.save_baseline("rag_ragas", results)
        print(f"\nBaseline (snapshot) saved to {_display(path)}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Koboi RAG baseline runner")
    parser.add_argument("--mode", default="offline", choices=["offline", "live"])
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth k (default 5)")
    parser.add_argument("--save-baseline", action="store_true", help="Save results as baseline")
    parser.add_argument("--compare", action="store_true", help="Compare offline run vs baseline")
    args = parser.parse_args()

    print("Koboi RAG Baseline Runner")
    print(f"  gold set : {GOLD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  mode     : {args.mode}")
    print(f"  top_k    : {args.top_k}")
    print(f"  scorers  : {ScorerRegistry.list_available()}")
    print(f"  loaders  : {LoaderRegistry.list_available()}")

    if args.mode == "offline":
        asyncio.run(run_offline(args.top_k, args.save_baseline, args.compare))
    else:
        asyncio.run(run_live(args.top_k, args.save_baseline))


if __name__ == "__main__":
    main()
