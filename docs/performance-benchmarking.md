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
2. **Relative compare** (`--benchmark-compare-fail=min:25%`) — for the macro
   latency benches (`bench_loop.py` + `bench_server.py`), where no sensible
   absolute threshold exists. The `benchmark.yml` "Macro relative-compare" step
   captures a base at the PR's merge-base, then compares the PR head against it
   **on the same runner** (cross-machine compare is invalid — 88/93 benches
   differ >15% between macOS/3.13 and ubuntu/3.12, so the base is captured
   in-job, never read from `baselines/baseline.json`). `min:25%` gives ~16pts
   headroom over the measured <15% same-machine run-to-run noise. PR-only; soft.

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

Both gates run **soft** for now (each step has `continue-on-error: true`, so a
breach warns in the step log + `$GITHUB_STEP_SUMMARY` but does **not** block the
PR). This is intentional while thresholds calibrate against real CI noise.

- **Micro NFR gate** (`NFR report + gate` step): `bench_report.py --check` on the
  77 absolute-threshold micro-benchmarks.
- **Macro relative-compare gate** (`Macro relative-compare` step): native
  `--benchmark-compare-fail=min:25%` of PR head vs merge-base on the 13 macro
  latency benches.

**Promote to a hard gate** once you have ~2 weeks of clean runs (no spurious
failures on either gate):

1. **Micro:** re-measure thresholds on CI (see *Refresh the CI baseline*) and
   update `NFR_THRESHOLDS` (`ceil(ci_min_ms × 2.5)`, 1ms floor).
2. **Macro:** re-measure the same-machine run-to-run delta; retune `min:25%`
   upward if a stable bench flakes, downward once noise is characterized.
3. Remove `continue-on-error: true` from **both** gate steps in `benchmark.yml`.
4. Add the `Benchmark` check to the repo's branch-protection required checks.

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
5. **Avoid per-round state accumulation.** A bench that grows a shared structure
   every round (e.g. an idempotency registry's `_seen`, a list) is O(N) per call;
   pytest-benchmark auto-runs ~10^5 rounds for a microsecond op, so total work
   becomes O(N^2) and the bench never finishes (it spins until the job's
   `timeout-minutes`). Rebuild fresh state per round, or reuse a fixed key so the
   structure stays O(1). (This exact bug hung a CI run for 70 min once.)
6. **Don't benchmark HTTP paths that spawn background work.** `/v1/jobs` admits
   (202) then kicks off an autonomous background job whose execution outlives a
   per-round `asyncio.run` loop and then blocks on `per_tenant_max`. Measure the
   admission write directly (`JobStore.insert`) instead, and leave full HTTP
   throughput to the relative-compare layer.
