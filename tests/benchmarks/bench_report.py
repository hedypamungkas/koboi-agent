"""Utility to generate baseline benchmark reports."""

import json
import os
from datetime import datetime
from pathlib import Path

BASELINE_DIR = Path(__file__).parent / "baselines"

NFR_THRESHOLDS = {
    "test_config_loading": 50,  # ms
    "test_facade_creation": 100,  # ms
    "test_tool_registration": 10,  # ms
    "test_memory_add": 1,  # ms
    "test_memory_get": 1,  # ms
    "test_token_estimation_single": 1,  # ms
    "test_token_estimation_multiple": 1,  # ms
    "test_token_estimation_large": 1,  # ms
    "test_context_truncation": 5,  # ms
    "test_context_smart_truncation": 5,  # ms
    "test_context_key_facts": 10,  # ms
    "test_ensure_tool_integrity": 5,  # ms
    "test_hook_chain_emit": 5,  # ms
    "test_hook_chain_single_hook": 1,  # ms
    "test_hook_chain_10_hooks": 10,  # ms
    "test_telemetry_collection": 50,  # ms
    "test_telemetry_health_score": 1,  # ms
    "test_telemetry_report_generation": 1,  # ms
    "test_doom_loop_check": 5,  # ms
    "test_doom_loop_consecutive_detection": 5,  # ms
    "test_doom_loop_pattern_detection": 5,  # ms
    "test_doom_loop_error_retry": 5,  # ms
    "test_hook_chain_list_hooks": 1,
    "test_hook_chain_find_hook": 1,
    "test_fixed_chunking": 500,  # ms
    "test_fixed_chunking_small_overlap": 500,  # ms
    "test_fixed_chunking_large_chunks": 300,  # ms
    "test_sentence_chunking": 500,  # ms
    "test_sentence_chunking_small_max": 500,  # ms
    "test_paragraph_chunking": 400,  # ms
    "test_keyword_retrieval": 50,  # ms
    "test_keyword_retrieval_single_chunk": 10,  # ms
    "test_keyword_retrieval_top_10": 50,  # ms
    "test_keyword_indexing": 100,  # ms
    "test_augmentation_in_memory": 50,  # ms
    "test_augmentation_in_memory_no_results": 10,  # ms
    "test_chunk_creation": 10,
    "test_document_creation": 1,
    "test_retrieval_result_creation": 1,
    "test_slash_command_dispatch": 10,  # ms
    "test_slash_command_with_tools": 10,
    "test_slash_command_reset_call": 1,
    "test_slash_command_history_call": 5,
    "test_welcome_panel_building": 10,
    # Export throughput
    "test_export_markdown_50": 5,  # ms
    "test_export_markdown_500": 1,  # ms (tightened: actual 0.114ms)
    "test_export_json_500": 5,  # ms (tightened: actual 1.488ms)
    "test_export_html_500": 2,  # ms (tightened: actual 0.322ms)
    # Diff detection and rendering
    "test_is_diff_content_positive": 1,  # ms
    "test_is_diff_content_negative": 1,  # ms
    "test_count_changes": 1,  # ms
    "test_diff_parse": 1,  # ms
    "test_diff_build_rich_text": 2,  # ms (tightened: actual 0.367ms)
    # Thinking block regex
    "test_thinking_pattern_match": 1,  # ms
    "test_thinking_pattern_no_match": 1,  # ms
    "test_thinking_pattern_large_content": 1,  # ms
    # Suggestion matching
    "test_slash_suggester_match": 1,  # ms
    "test_slash_suggester_no_match": 1,  # ms
    "test_composite_suggester_route": 1,  # ms
    # Bridge message creation
    "test_bridge_message_creation": 3,  # ms (tightened: actual 0.897ms)
    "test_bridge_mixed_message_creation": 3,  # ms (tightened: actual 0.729ms)
    # Theme
    "test_theme_registration": 1,  # ms
    # Token estimation at scale
    "test_token_estimation_100_messages_with_tool_calls": 5,  # ms
    "test_token_estimation_500_messages_with_tool_calls": 25,  # ms
    "test_token_estimation_2000_messages_with_tool_calls": 100,  # ms
    # Memory copy overhead
    "test_memory_get_messages_100": 0.5,  # ms
    "test_memory_get_messages_1000": 5,  # ms
    # String accumulation
    "test_string_concatenation_50kb": 1,  # ms
    "test_string_concatenation_200kb": 10,  # ms
    # Thinking regex at scale
    "test_thinking_pattern_50kb_no_match": 2,  # ms
    "test_thinking_pattern_50kb_with_match": 2,  # ms
    # Export at scale
    "test_export_markdown_2000": 5,  # ms
    "test_export_json_2000": 15,  # ms
    "test_export_html_2000": 5,  # ms
    # Diff at scale
    "test_diff_parse_1000_lines": 2,  # ms
    "test_is_diff_content_1000_lines": 1,  # ms
    # Hook chain per-iteration
    "test_hook_chain_emit_6_events_per_iteration": 5,  # ms
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
        mean_ms = stats["mean"] * 1000  # convert to ms
        report["benchmarks"][name] = {
            "mean_ms": round(mean_ms, 3),
            "median_ms": round(stats["median"] * 1000, 3),
            "stddev_ms": round(stats["stddev"] * 1000, 3),
            "rounds": stats["rounds"],
        }

        # Check against NFR thresholds
        threshold = NFR_THRESHOLDS.get(name.split("[")[0])
        if threshold:
            passed = mean_ms <= threshold
            report["nfr_status"][name] = {
                "passed": passed,
                "threshold_ms": threshold,
                "actual_ms": round(mean_ms, 3),
            }

    return report


def save_baseline(report: dict, name: str = "baseline") -> Path:
    """Save baseline report to JSON file."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def compare_baselines(current: dict, previous: dict) -> dict:
    """Compare two baseline reports for regressions."""
    regressions = {}
    for name, stats in current.get("benchmarks", {}).items():
        prev_stats = previous.get("benchmarks", {}).get(name)
        if prev_stats:
            change_pct = ((stats["mean_ms"] - prev_stats["mean_ms"]) / prev_stats["mean_ms"]) * 100
            if change_pct > 20:  # >20% slower = regression
                regressions[name] = {
                    "previous_ms": prev_stats["mean_ms"],
                    "current_ms": stats["mean_ms"],
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
        print(f"\nNFR Compliance: {nfr_passed}/{nfr_total} passed")

        failed = [(name, s) for name, s in report["nfr_status"].items() if not s["passed"]]
        if failed:
            print("\nFailed NFR thresholds:")
            for name, status in failed:
                print(f"  - {name}: {status['actual_ms']}ms > {status['threshold_ms']}ms")

    # Slowest benchmarks
    sorted_benchmarks = sorted(report["benchmarks"].items(), key=lambda x: x[1]["mean_ms"], reverse=True)[:5]

    print("\nSlowest benchmarks:")
    for name, stats in sorted_benchmarks:
        print(f"  - {name}: {stats['mean_ms']}ms (median: {stats['median_ms']}ms)")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bench_report.py <benchmark_json_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    report = generate_report(json_path)
    save_baseline(report)
    print_summary(report)
