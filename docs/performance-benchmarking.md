# Performance Benchmarking & Regression Gate

koboi-agent runs a `pytest-benchmark` suite under `tests/benchmarks/` and gates
PRs on it via `.github/workflows/benchmark.yml`. This doc covers how to run it,
how the gate works, the noise policy, and how to maintain it.

## Run locally

```bash
# all benchmarks (the exact invocation CI uses)
.venv/bin/python -m pytest tests/benchmarks/ \
  -o python_files="bench_*.py" \
  --benchmark-only \
  --benchmark-min-rounds=50 \
  --benchmark-warmup=on --benchmark-warmup-iterations=5 \
  --benchmark-disable-gc

# one file / one bench
.venv/bin/python -m pytest tests/benchmarks/bench_loop.py -o python_files="bench_*.py" --benchmark-only
.venv/bin/python -m pytest tests/benchmarks/bench_server.py::test_server_chat_stream -o python_files="bench_*.py" --benchmark-only

# generate the NFR report + run the gate locally
.venv/bin/python -m pytest tests/benchmarks/ -o python_files="bench_*.py" --benchmark-only --benchmark-json=/tmp/b.json
.venv/bin/python tests/benchmarks/bench_report.py /tmp/b.json --check --no-save
```

> The `-o python_files="bench_*.py"` override is **required**: benchmark files are
> named `bench_*.py`, which don't match pytest's default `test_*.py` collection
> pattern. Without the override, pytest collects 0 of them — which is exactly why
> this suite sat dormant for months. The override is scoped to the benchmark
> invocation; the normal `pytest` test gate is untouched.

> `bench_server.py` needs the `[api]` extra (fastapi/httpx). CI installs
> `.[dev,tui,api]`; locally, make sure your venv has it (`pip install -e ".[dev,tui,api]"`).

## What's covered

| File | Surface | Gate |
|---|---|---|
| `bench_core.py` | config, memory, tokens, context truncation, doom-loop, telemetry | absolute NFR (`min`) |
| `bench_hooks.py` | hook chain emit/find at 1/5/10 depth | absolute NFR (`min`) |
| `bench_rag.py` | chunking (fixed/sentence/paragraph), keyword retrieval, augmentation | absolute NFR (`min`) |
| `bench_tui.py` | slash dispatch, export, diff, thinking regex, suggester, bridge, theme | absolute NFR (`min`) |
| `bench_loop.py` | `AgentCore` turn, N-turn throughput, 8-step `ToolExecutionPipeline`, hook overhead in-loop | report + relative compare (Wave 2) |
| `bench_server.py` | `/healthz`, `/v1/chat/stream` SSE, `/v1/jobs` admission, pool reuse, idempotency, Bearer auth | report + relative compare (Wave 2) |
| `bench_memory.py` | peak-bytes (tracemalloc) for a turn, RAG index, server boot | generous absolute ceiling (assertion) |

`bench_report.py` is the reporter/gate (NFR threshold check + comparator); the
thresholds live in `NFR_THRESHOLDS` at its top.

## Gating strategy

**Two layers:**

1. **Absolute NFR threshold** (`bench_report.py --check`) — for the pure-CPU
   micro-benchmarks (core/hooks/rag/tui). The check uses `stats["min"]` (the
   outlier-resistant floor), **not `mean`** — micro-benchmarks on shared CI
   runners have extreme variance under `mean` (e.g. `test_config_loading` median
   0.7ms / stddev 125ms), and `min` is the clean compute floor, so a threshold on
   `min` is both stable and sensitive. Exit non-zero on breach.
2. **Relative compare** (`--benchmark-compare-fail=min:X%`, planned Wave 2) — for
   the higher-variance server/loop benches, where no sensible absolute threshold
   exists. Compares the PR branch against the merge-base on the same runner
   (neutralizes cross-machine variance).

`bench_memory.py` self-asserts a generous peak-bytes ceiling (catches egregious
leaks); its *time* numbers are tracemalloc-inflated and meaningless — read
`benchmark.extra_info["peak_kb"]`.

**Threshold provenance:** `NFR_THRESHOLDS` values are `ceil(min_ms × 3)` with a
1ms floor, provisionally measured on a dev machine. They are far tighter than the
old hand-set values (which had 25–150× headroom and caught nothing) but **must be
re-measured on the CI runner** (see *Refresh the baseline* below).

## Noise policy

Micro-benchmarks are noisy on shared GitHub runners. We control it with:

- **Metric = `min`**, not `mean`/`median` (outlier-resistant floor).
- `--benchmark-min-rounds=50` — stabilizes the low-round benches.
- `--benchmark-warmup=on` (5 iterations) — avoids cold-start in the numbers.
- `--benchmark-disable-gc` — measures pure compute (note: hides GC-spike
  regressions by design; that axis is covered by `bench_memory.py`).
- **Single matrix cell** — `ubuntu-latest` / Python 3.12 only. Benchmarks must
  run on a fixed, controlled environment; the 6-cell test matrix is not used.

## Soft → hard gate

`benchmark.yml` runs as a **soft gate** initially: the `--check` step has
`continue-on-error: true`, so an NFR breach warns (visible in the step log +
`$GITHUB_STEP_SUMMARY`) but does **not** block the PR. This is intentional while
thresholds are being calibrated against real CI noise.

**Promote to a hard gate** once you have ~2 weeks of clean runs:

1. Re-measure thresholds on CI (see below) and update `NFR_THRESHOLDS`.
2. Remove `continue-on-error: true` from the `--check` step in `benchmark.yml`.
3. Add `Benchmark` (or the specific check name) to the repo's required status
   checks in branch protection.

## Refresh the CI baseline (W1.4)

The committed `tests/benchmarks/baselines/baseline.json` is a reference snapshot.
To refresh it from a clean `main` run on the CI runner:

1. Trigger `benchmark.yml` on `main` (push or `workflow_dispatch`).
2. Download the `bench-results` artifact from the run.
3. Generate the report (which also writes `baselines/baseline.json`):
   ```bash
   .venv/bin/python tests/benchmarks/bench_report.py bench-results.json
   ```
4. Commit the refreshed `baselines/baseline.json` and, if recalibrating,
   recompute `NFR_THRESHOLDS` as `ceil(ci_min_ms × 2.5)` (1ms floor) from the
   CI-measured `min` values and commit `bench_report.py`.

> Recalibrate from **CI** measurements, not a dev laptop: CI (`ubuntu`/3.12,
> virtualized) `min` values run ~1.5–3× higher than dev (`macOS`/3.13). A dev
> baseline would flake the moment it hits CI.

## Adding a new benchmark

1. Add a `bench_*.py` file (or a `test_*` function to an existing one) under
   `tests/benchmarks/`. Use the `benchmark` fixture; for async work, wrap
   `asyncio.run(coro)` in a sync closure (see `bench_loop.py`).
2. If it's a stable micro-benchmark, add an entry to `NFR_THRESHOLDS` in
   `bench_report.py` (`ceil(measured_min × 3)`, 1ms floor).
3. If it's high-variance (server/loop), leave it ungated (report-only) until the
   relative-compare layer lands.
4. Run locally to capture the `min`, then set the threshold.
