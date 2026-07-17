# CI triage — read checks, extract real errors, fix, re-verify

## See the status
```
gh pr checks <pr>                      # per-job pass/fail/skip
gh pr checks <pr> --json name,state    # structured (state ∈ SUCCESS/FAILURE/SKIPPED/…)
gh run list --branch <branch> --limit 6 --json databaseId,name,status,conclusion,headSha
```
Note: `build-and-push`/publish jobs are often SKIPPED on PR branches (tag/main-only) — that's expected, not a failure. `mergeable: UNKNOWN` means GitHub hasn't recomputed yet; the authoritative conflict test is local `git merge-tree` (see `scripts/merge_order_check.sh`).

## Extract the REAL error from noisy logs
CI logs are dominated by checkout/setup/apt noise. `gh run view <run> --log-failed` returns failed-step logs but still needs filtering:
```
gh run view <run> --log | grep -E "##\[error\]|error:|Traceback|exit code|FAIL|assert"
# for a docker build step, isolate the failing #N step:
gh run view <run> --log | grep "#9 " | grep -iE "error|cannot|not found|No module"
```
Filter aggressively (drop `Get:|Selecting|Preparing|Unpacking|Setting up|debconf|##[group]`). If the first grep returns only runner-setup lines, the failing step is elsewhere — widen or look at the raw tail. Two failures in one workflow often share a root cause (e.g. a `docker build` step inside another job hits the same Dockerfile error).

## Common CI failure patterns + fixes

| Symptom | Cause | Fix |
|---|---|---|
| mypy `Type cannot be declared in assignment to non-self attribute [misc]` | annotated `obj.attr: T = ...` | drop the annotation (put the type in a comment) |
| mypy false-reds only locally | dev venv drift (e.g. numpy stubs) | trust CI mypy; don't block on local mypy |
| build `ModuleNotFoundError: No module named 'Cython'` | PEP-517 build-isolation hid preinstalled Cython | `--no-build-isolation` |
| build `BackendUnavailable: Cannot import 'setuptools.build_meta'` | `--no-build-isolation` but base image ships no setuptools | preinstall `setuptools wheel` |
| build `KeyError: 'VERSION_RELEASE'` / metadata-gen-failed | binding's setup.py needs generated `version.h` from a full autotools C build | **pivot**: use the distro package (e.g. `apt install python3-seccomp` + `PYTHONPATH`) instead of from-source |
| test skipped locally, fails on CI (or vice-versa) | platform-gated (Linux-only etc.) | mock the platform module in unit tests; keep the real-subprocess test `skipif` on the right OS |

## Can't-test-locally caveat (the rabbit-hole rule)
Some steps can't run on the dev box (seccomp, Docker build, Linux syscalls). After **2–3 failed CI rounds** on such a step, stop guessing — **pivot** to a robust alternative:
- from-source build that needs the full C build → distro package (+ `PYTHONPATH` so the runtime python finds it).
- or make the step best-effort (`|| echo WARN`) and rely on a runtime fail-closed guard so the image never silently ships broken.
Document the tradeoff in the PR. The real value (code-level fixes, fully unit-tested) ships regardless.

## Fix → push → re-verify
Edit on the PR's branch (the worktree still has it checked out), then:
```
git -C <worktree> add <file> && git -C <worktree> commit -m "fix(#NN): ..." && git -C <worktree> push origin HEAD
```
`gh pr checks` re-runs on the new commit. Wait (`sleep ~120-160s` then re-check, or `gh run watch`), confirm green. Repeat per failure until the table is all `SUCCESS`.

## Merge-order check (inter-PR conflicts)
PRs touching the same file can conflict once one merges. `git merge-tree` is the authoritative 3-way test — run `scripts/merge_order_check.sh` over the PR set; it reports per-order conflicts so you can recommend a safe sequence (independent first; same-file PRs adjacent).
