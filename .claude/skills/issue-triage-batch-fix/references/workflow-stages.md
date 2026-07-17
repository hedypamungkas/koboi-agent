# Workflow Stages — detailed commands and decisions

Concrete elaboration of the 9 stages in `SKILL.md`. Cross-project; adapt tooling to the repo.

## Stage 1 — Map the landscape

```
gh issue list --state open  --limit 100 --json number,title,labels,body,createdAt
gh issue list --state all   --limit 100 --json number,title,state,createdAt   # see recent closures
```
Read bodies. Cluster by theme (an audit pass often files a run of related findings — closed siblings tell you what's already addressed and what bug-classes remain).

## Stage 2 — Verify empirically

For each open issue, re-confirm against current `main`:
- Locate the symbol yourself. Prefer structural tools when available (`search_graph` / `trace_path` / `get_code_snippet` from codebase-memory-mcp; else `grep -rn`/`Grep`). **Quote the real `file:line`** — never the issue's.
- Distinguish: confirmed-real (code does what the report says) vs. already-fixed vs. misread/unreachable. Rank only the first.
- If the issue cites an exploit/repro, trace whether it's actually reachable under current config/guards.

## Stage 3 — Rank severity × effort

Score each confirmed issue:
- **Severity = blast radius × exploit ease.** Server-wide/availability > single-feature/integrity > hardening. Exploitable by any authenticated caller > needs privileged input > defense-in-depth.
- **Effort** = LOC × change-blast-radius × test difficulty.
Present a ranked table; one-line evidence each. Recommend an order. Confirm N with the user (default 3).

## Stage 4 — Audit if fewer than N open

When `open < N`, do not pad the list with weak items. Offer to surface more from unaudited surfaces:
- Spawn read-only `Explore` agents over candidates: sibling routes/endpoints missing the same guard, parallel admission/auth paths, the same bug-class elsewhere (e.g. one IDOR route fixed → audit the other owner-scoped routes).
- For a confirmed finding, file a **new** issue (`gh issue create`) with `file:line` evidence + a repro, and use its number for the work unit. Only ship what reproduces.

## Stage 5 — Plan

`EnterPlanMode`. Launch parallel foreground `Explore` agents (need their results). One per candidate unit; plus an audit agent if Stage 4 is in-line. Each must return: exact fix (files + change shape, reusing existing patterns), a **deterministic RED test** (see `empirical-red-tests.md`), the e2e/verification recipe, and a file-disjointness check (units must touch disjoint files or non-overlapping regions).

Surface genuine forks with `AskUserQuestion` (not for plan approval — that's `ExitPlanMode`). Examples worth asking: fix depth (safe+opt-in vs. flip-default), scope (lean fix vs. include adjacent hardening), back-compat tradeoffs. Then write the plan and `ExitPlanMode`.

After approval, dispatch workers (Stage 6).

## Stage 6 — Dispatch

Spawn one background `Agent` per unit in **one message** (parallel):
```
Agent(description=..., subagent_type="general-purpose",
      isolation="worktree", run_in_background=true, name=f"fix-{n}",
      prompt=<self-contained unit prompt>)
```
Self-contained = goal, unit spec (title, files, `file:line`, fix shape, RED test spec), repo conventions (test runner, fixtures, owner/api-key derivation, commit style), the worker template (see `worker-prompt-template.md`), and `PR: <url>` report requirement.

Render a status table (`# | unit | status | PR`); update as `PR:` lines arrive.

## Stage 7 — Reproducible harness

Ship `experiment_<topic>.py` per non-trivial bug (template in `examples/`). Hard rules: real production classes (no mocking the system under test; mock only external I/O like the LLM/HTTP), no network, deterministic, prints `CHECK → PASS/FAIL` + concrete evidence, `sys.exit(1)` on RED / `0` on GREEN. Match the repo's existing probe convention if one exists (e.g. a `experiment_*.py` family).

## Stage 8 — CI verify + fix

`gh pr checks <pr>` per PR. On failure, find the real error (see `ci-triage.md`). Fix on the PR's branch, push, re-verify. Heuristic for untestable-locally failures: after 2–3 rounds on a fragile from-source build / platform-specific step, **pivot** to a robust alternative and document the tradeoff — don't burn the day on a rabbit hole.

## Stage 9 — Memory

Record a project memory: PR URLs, issues closed, non-obvious gotchas, deferred follow-ups. Correct any memory the new work contradicts. Link related memories.

## Merge order (post-approval)
Run `scripts/merge_order_check.sh <pr1> <pr2> ...` (or `git merge-tree` pairwise) to detect inter-PR conflicts. Recommend independent/lowest-risk first; group same-file PRs adjacent. Conflict-free PRs merge in any order; a rebase re-runs CI on combined code if belt-and-suspenders is wanted.
