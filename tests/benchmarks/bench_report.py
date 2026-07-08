"""Benchmark report generator + NFR regression gate.

Reads pytest-benchmark JSON output (``--benchmark-json=``), checks each
benchmark against an absolute NFR threshold, and prints a human-readable
summary. With ``--check`` it also exits non-zero if any threshold is breached,
so it can gate a CI job.

Gating metric: the NFR check uses ``stats["min"]`` (the outlier-resistant
floor), NOT ``mean``. Micro-benchmarks on shared CI runners have extreme
variance (e.g. ``test_config_loading`` median 0.7ms / stddev 125ms under
``mean``); ``min`` is the clean compute floor, so a threshold on ``min`` is
both stable and sensitive. ``compare_baselines`` likewise compares on ``min``.

Threshold provenance: the values below are PROVISIONAL, measured on a dev
machine (macOS / Python 3.13) as ``ceil(min_ms * 3)`` with a 1ms floor (the x3
headroom absorbs the ~1.5-3x dev->CI ``min`` gap). They are far tighter than
the old hand-set values (which had 25-150x headroom and caught nothing), but
MUST be re-measured on the CI runner (ubuntu / Python 3.12) and re-tuned per
the Wave-1 calibration procedure (see docs/performance-benchmarking.md). The
``--check`` gate is non-blocking during that calibration window.

Usage:
    python bench_report.py <benchmark_json>            # report + save baseline
    python bench_report.py <benchmark_json> --check     # also exit 1 on NFR fail
    python bench_report.py <benchmark_json> --no-save   # don't overwrite baseline
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASELINE_DIR = Path(__file__).parent / "baselines"

# NFR latency thresholds (ms), gated on `min`. PROVISIONAL dev-measured values
# (ceil(dev_min_ms * 3), 1ms floor) -- see module docstring for provenance and
# the CI recalibration caveat.
NFR_THRESHOLDS = {
    # --- core ---
    "test_config_loading": 3,
    "test_facade_creation": 12,
    "test_tool_registration": 1,
    "test_memory_add": 1,
    "test_memory_get": 1,
    "test_token_estimation_single": 1,
    "test_token_estimation_multiple": 1,
    "test_token_estimation_large": 1,
    "test_context_truncation": 1,
    "test_context_smart_truncation": 1,
    "test_context_key_facts": 1,
    "test_ensure_tool_integrity": 1,
    # --- hooks / telemetry / doom-loop ---
    "test_hook_chain_emit": 1,
    "test_hook_chain_single_hook": 1,
    "test_hook_chain_10_hooks": 1,
    "test_hook_chain_list_hooks": 1,
    "test_hook_chain_find_hook": 1,
    "test_hook_chain_emit_6_events_per_iteration": 1,
    "test_telemetry_collection": 1,
    "test_telemetry_health_score": 1,
    "test_telemetry_report_generation": 1,
    "test_doom_loop_check": 1,
    "test_doom_loop_consecutive_detection": 1,
    "test_doom_loop_pattern_detection": 1,
    "test_doom_loop_error_retry": 1,
    # --- rag ---
    "test_fixed_chunking": 1,
    "test_fixed_chunking_small_overlap": 1,
    "test_fixed_chunking_large_chunks": 1,
    "test_sentence_chunking": 2,
    "test_sentence_chunking_small_max": 2,
    "test_paragraph_chunking": 1,
    "test_keyword_retrieval": 1,
    "test_keyword_retrieval_single_chunk": 1,
    "test_keyword_retrieval_top_10": 1,
    "test_keyword_indexing": 2,
    "test_augmentation_in_memory": 1,
    "test_augmentation_in_memory_no_results": 1,
    "test_chunk_creation": 1,
    "test_document_creation": 1,
    "test_retrieval_result_creation": 1,
    # --- tui ---
    "test_slash_command_dispatch": 1,
    "test_slash_command_with_tools": 1,
    "test_slash_command_reset_call": 1,
    "test_slash_command_history_call": 1,
    "test_welcome_panel_building": 1,
    "test_export_markdown_50": 1,
    "test_export_markdown_500": 1,
    "test_export_markdown_2000": 2,
    "test_export_json_500": 2,
    "test_export_json_2000": 6,
    "test_export_html_500": 1,
    "test_export_html_2000": 4,
    "test_is_diff_content_positive": 1,
    "test_is_diff_content_negative": 1,
    "test_is_diff_content_1000_lines": 1,
    "test_count_changes": 1,
    "test_diff_parse": 1,
    "test_diff_parse_1000_lines": 1,
    "test_diff_build_rich_text": 1,
    "test_thinking_pattern_match": 1,
    "test_thinking_pattern_no_match": 1,
    "test_thinking_pattern_large_content": 1,
    "test_thinking_pattern_50kb_no_match": 1,
    "test_thinking_pattern_50kb_with_match": 1,
    "test_slash_suggester_match": 1,
    "test_slash_suggester_no_match": 1,
    "test_composite_suggester_route": 1,
    "test_bridge_message_creation": 2,
    "test_bridge_mixed_message_creation": 2,
    "test_theme_registration": 1,
    # --- token estimation at scale ---
    "test_token_estimation_100_messages_with_tool_calls": 1,
    "test_token_estimation_500_messages_with_tool_calls": 6,
    "test_token_estimation_2000_messages_with_tool_calls": 22,
    # --- memory copy / string accumulation ---
    "test_memory_get_messages_100": 1,
    "test_memory_get_messages_1000": 1,
    "test_string_concatenation_50kb": 1,
    "test_string_concatenation_200kb": 1,
}


def generate_report(benchmark_json_path: str) -> dict:
    """Generate a baseline report from pytest-benchmark JSON output."""
    with open(benchmark_json_path) as f:
        data = json.load(f)

    report = {
        "generated_at": datetime.now().isoformat(),
        "machine": data.get("machine_info", {}),
        "benchmarks": {},
        "nfr_status": {},
    }

    for bench in data.get("benchmarks", []):
        name = bench["name"]
        stats = bench["stats"]
        min_ms = stats["min"] * 1000  # ms -- the gating metric (outlier floor)
        report["benchmarks"][name] = {
            "min_ms": round(min_ms, 4),
            "mean_ms": round(stats["mean"] * 1000, 4),
            "median_ms": round(stats["median"] * 1000, 4),
            "stddev_ms": round(stats["stddev"] * 1000, 4),
            "rounds": stats["rounds"],
        }

        # NFR check against the absolute threshold, gated on `min`.
        threshold = NFR_THRESHOLDS.get(name.split("[")[0])
        if threshold:
            passed = min_ms <= threshold
            report["nfr_status"][name] = {
                "passed": passed,
                "threshold_ms": threshold,
                "actual_ms": round(min_ms, 4),
            }

    return report


def check_nfr(report: dict) -> list[dict]:
    """Return the list of NFR failures (empty if all pass)."""
    return [{"name": name, **status} for name, status in report.get("nfr_status", {}).items() if not status["passed"]]


def save_baseline(report: dict, name: str = "baseline") -> Path:
    """Save baseline report to JSON file."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def compare_baselines(current: dict, previous: dict, pct_threshold: int = 20) -> dict:
    """Compare two baseline reports for regressions (on `min_ms`).

    Prefers the outlier-resistant ``min_ms``; falls back to ``mean_ms`` when
    either side is a legacy baseline that predates the ``min_ms`` field, so a
    refreshed report can still be diffed against the old committed baseline.
    """
    regressions = {}
    for name, stats in current.get("benchmarks", {}).items():
        prev_stats = previous.get("benchmarks", {}).get(name)
        if not prev_stats:
            continue
        key = "min_ms" if ("min_ms" in stats and "min_ms" in prev_stats) else "mean_ms"
        cur_v = stats.get(key, 0.0)
        prev_v = prev_stats.get(key, 0.0)
        if prev_v <= 0:
            continue
        change_pct = ((cur_v - prev_v) / prev_v) * 100
        if change_pct > pct_threshold:  # >pct_threshold % slower = regression
            regressions[name] = {
                "previous_ms": prev_v,
                "current_ms": cur_v,
                "change_pct": round(change_pct, 1),
            }
    return regressions


def print_summary(report: dict) -> None:
    """Print a human-readable summary of the benchmark report."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Generated: {report['generated_at']}")
    print(f"Total benchmarks: {len(report['benchmarks'])}")

    # NFR Status
    nfr_total = len(report.get("nfr_status", {}))
    if nfr_total > 0:
        nfr_passed = sum(1 for s in report["nfr_status"].values() if s["passed"])
        print(f"\nNFR Compliance (gated on `min`): {nfr_passed}/{nfr_total} passed")

        failed = [(name, s) for name, s in report["nfr_status"].items() if not s["passed"]]
        if failed:
            print("\nFailed NFR thresholds:")
            for name, status in failed:
                print(f"  - {name}: {status['actual_ms']}ms > {status['threshold_ms']}ms")

    # Slowest benchmarks (by min)
    sorted_benchmarks = sorted(report["benchmarks"].items(), key=lambda x: x[1]["min_ms"], reverse=True)[:5]

    print("\nSlowest benchmarks (by min):")
    for name, stats in sorted_benchmarks:
        print(f"  - {name}: min={stats['min_ms']}ms (mean={stats['mean_ms']}ms)")

    print("=" * 60 + "\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("benchmark_json", help="Path to pytest-benchmark --benchmark-json output")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any NFR threshold is breached (CI gate mode).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not overwrite baselines/baseline.json (use in CI PR runs).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    report = generate_report(args.benchmark_json)
    if not args.no_save:
        save_baseline(report)
    print_summary(report)

    if args.check:
        failures = check_nfr(report)
        if failures:
            print(f"\nNFR GATE: FAILED ({len(failures)} regression(s) over threshold)\n")
            return 1
        print("\nNFR GATE: PASSED\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
