---
name: issue-triage-batch-fix
description: This skill should be used when the user asks to "check the GitHub issues", "triage the open issues", "assess and rank the issues", "what issues should we fix", "fix the top issues", or otherwise wants to go from the issue tracker to fixed, merged PRs. It runs the whole loop end-to-end without further prompting — verify each issue empirically against current code, rank by severity × effort, surface the top high/critical ones (TDD is the default method, not an option), and once the user confirms scope, automatically dispatch the fixes in parallel as isolated worktree workers, each shipping its own PR, plus a reproducible harness, CI verification, and a memory note.
version: 0.1.0
---

# Issue Triage & Batch-Fix

The end-to-end loop from **issue tracker → merged PRs**: triage open issues, verify each is real against current code, rank by severity × effort, then fix the top high/critical ones in parallel, one isolated worktree worker per issue, each shipping its own PR.

## Defaults (always apply unless the user overrides)

These are built-in behavior — the user never has to ask for them:

- **High/critical only, by default.** Rank every confirmed issue, but only proceed to *fix* high/critical severity. Confirm N with the user (default **3**). Lower-severity items are listed for awareness, not fixed.
- **TDD is the method, not an opt-in.** Every fix is red-test-first (prove the bug) → fix → green → full suite (zero breakage) → lint/format → code-review. Never "just patch it."
- **Auto-dispatch once top-N is confirmed.** After the plan and top-N are approved, proceed directly to the parallel worktree dispatch (the `/batch` pattern). Do **not** wait for a separate `/batch` or "go" command — the user confirms scope; the skill runs the loop.
- **Evidence-first throughout.** Verify each issue against current code before ranking; ship a reproducible harness; record outcomes + gotchas to memory.

## Core principle: evidence-first, empirical

Never rank or fix an issue on the strength of its prose alone. **Verify every claim against the current code before it influences a decision** — issue reports go stale (line numbers drift, code moves, the bug may already be fixed). A bug is "real" only when a failing test reproduces it. A fix is "done" only when that test passes *and* the full suite stays green. Ship a reproducible harness so anyone can re-confirm.

## The pipeline

Run the stages in order. Detail for each lives in `references/`.

### 1. Map the issue landscape
- `gh issue list --state open --limit 100` for the open set; `gh issue list --state all` to see what was recently closed (already-fixed siblings inform severity and avoid rework).
- Read each open issue body. Pull `--json number,title,labels,body,createdAt` for structured triage.
- Note how many are open vs. recently-closed — recent closures often come from the same audit pass and hint at remaining latent findings.

### 2. Verify each issue empirically (before ranking)
For every open issue, confirm the bug exists in the **current** code:
- Re-locate the symbols yourself — do **not** trust the issue's line numbers (grep / `search_graph` / `trace_path` / `get_code_snippet`). Quote the real `file:line`.
- If the code contradicts the report (already fixed, misread, unreachable), say so and drop it. Rank only confirmed-real issues.
- See `references/empirical-red-tests.md` for verification techniques, including platform-gated code (e.g. Linux-only features on a macOS box → mock-injection, not real execution).

### 3. Rank severity × effort; pick top-N
For each confirmed issue, score:
- **Severity = blast radius × exploit ease** (server-wide DoS > single-feature; trivial/any-authenticated > needs-privileged).
- **Effort** (LOC, blast-radius of the change, test difficulty).
Present a ranked table with one-line evidence (`file:line` + why). Recommend an order. Confirm N with the user (default 3).

### 4. If fewer than N are open, audit for more
When the open set is smaller than N, offer to surface additional real, TDD-reproducible bugs from unaudited surfaces (sibling routes, the same bug-class elsewhere, missing admission/auth on parallel endpoints). Spawn read-only `Explore` agents over candidate surfaces; file a **new** GitHub issue for any confirmed finding so the work unit tracks against a real issue. Never fabricate a third to fill a quota — only ship what reproduces.

### 5. Plan: research → per-unit TDD design → approval
Enter plan mode (`EnterPlanMode`). Launch parallel `Explore` research agents (one per candidate unit, plus an audit agent if needed). For each unit, design:
- The **exact fix** (files + the shape of the change, reusing existing patterns).
- A **deterministic RED test** that fails today for the bug's reason (see `references/empirical-red-tests.md`).
- The **e2e/verification recipe** (the RED test IS the proof for logic bugs; for platform-gated code use mocks; for UI use a browser skill).
- **Independence**: confirm units touch disjoint files (or non-overlapping regions) so PRs don't block each other.
Surface genuine forks (fix depth, back-compat tradeoffs) via `AskUserQuestion`. Write the plan, then `ExitPlanMode` for approval. See `references/workflow-stages.md`.

### 6. Dispatch one worktree worker per unit (automatic after approval)
This is the continuation of Stage 5, not a separate command — **once the plan + top-N are approved (`ExitPlanMode`), proceed straight here** (the `/batch` pattern is internal to this skill; the user should not have to invoke it). Spawn one background `Agent` per unit: `isolation: "worktree"`, `run_in_background: true`, `subagent_type: "general-purpose"`, all in a single message so they run concurrently. Each prompt is **fully self-contained** (goal, unit spec with `file:line`, repo conventions, the verification recipe, the worker template). The worker template enforces **strict TDD** (the default method — never skip):
1. **RED first** — write the failing test; run it; confirm it fails *for the bug's reason* (not an unrelated error); capture the red output.
2. **Fix** per spec.
3. **GREEN** — test passes.
4. **Full suite** (`pytest` / repo equivalent) — zero broken changes; fix any regressions the change caused.
5. **Gates** — lint + format (e.g. `ruff check`, `ruff format --check`).
6. **Code-review** — invoke the `code-review` skill; fix findings.
7. **Commit + push + PR** — test + fix together (final green); PR body documents bug (`file:line`), the RED proof, the fix, verification. For a new finding, `gh issue create` first and `Closes #NN`.
8. **Report** — end with one line: `PR: <url>` (or `PR: none — <reason>`).

Track in a status table; update as `PR:` lines arrive. Full template + repo-fact injection: `references/worker-prompt-template.md` and `examples/worker_prompt.md`.

### 7. Ship a reproducible experiment harness
For each non-trivial bug, ship a standalone `experiment_*.py` (or repo-equivalent probe) that anyone can run to see the bug red / the fix green: drives the **real** production classes (no mocks of the system under test), no network, prints `CHECK → PASS/FAIL` with concrete evidence, exits non-zero on RED. This is the artifact that makes "evidence-based" reproducible by others. Template: `examples/experiment_template.py`.

### 8. Verify CI; diagnose + fix failures from logs
For every PR: `gh pr checks <pr>`. On failure, pull the **real** error (CI logs are full of checkout/setup noise — filter it):
- `gh run view <run> --log-failed` then grep for `##[error]`, `error:`, `Traceback`, the failing step's `#N` lines.
- Common patterns: mypy rejects annotations on non-self attrs; PEP-517 build-isolation hides preinstalled build deps (`--no-build-isolation` + preinstall `setuptools wheel`); Dockerfile/can't-run-locally features need mock-based tests, not real execution.
- When a from-source build (or any untestable-locally step) fails repeatedly, **pivot** to a robust alternative (distro package, best-effort + runtime fail-closed) rather than chasing a fragile rabbit hole — and note the tradeoff in the PR.
Fix, push, re-verify until green. See `references/ci-triage.md`.

### 9. Record outcomes + gotchas to memory
Write a memory entry: PR URLs, what each closes, non-obvious gotchas discovered (e.g. "`connectat` isn't a Linux syscall", "libseccomp Python binding needs the full autotools C build"), and any **deferred** follow-up bugs. Correct any memory the new work contradicts (e.g. "X discarded" → "X re-implemented and shipped").

## Merge-order (once PRs are approved)
PRs that touch the same file can still conflict once one merges. Before advising order, **simulate** pairwise merges against `main` (the authoritative 3-way test): `scripts/merge_order_check.sh pr1 pr2 ...` runs `git merge-tree` sequences and reports conflicts. Recommend an order (independent/lowest-risk first; group same-file PRs adjacent); note that conflict-free PRs may still benefit from a rebase to re-run CI on the combined code.

## Additional resources

### Reference files
- **`references/workflow-stages.md`** — concrete commands + decisions per stage, including plan-mode research decomposition and `AskUserQuestion` forks.
- **`references/worker-prompt-template.md`** — the self-contained TDD worker prompt + how to inject repo-specific facts.
- **`references/empirical-red-tests.md`** — writing deterministic failing tests: race-gating with `asyncio.Event`, `sys.modules` mock-injection for platform-gated code, pre-seeding registries, concurrency bursts; + the `experiment_*.py` harness pattern.
- **`references/ci-triage.md`** — reading `gh pr checks`, extracting real errors from noisy logs, common CI failure patterns + fixes, can't-test-locally caveats.

### Examples
- **`examples/experiment_template.py`** — skeleton reproducible harness (real classes, PASS/FAIL evidence, exit code = red/green).
- **`examples/worker_prompt.md`** — full worker-prompt boilerplate to adapt.

### Scripts
- **`scripts/merge_order_check.sh`** — pairwise `git merge-tree` merge-order conflict simulator.
