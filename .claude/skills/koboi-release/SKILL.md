---
name: koboi-release
description: >-
  This skill should be used when the user asks to "release koboi-agent", "bump version
  and publish", "create a new release", "push to PyPI", "tag and release", "ship version
  vX.Y.Z", or mentions PyPI publishing, GitHub releases, or GHCR container image publishing
  for the koboi-agent project. Encodes the full release sequence (merge → pre-check → bump →
  tag → PyPI + GHCR auto-publish → verify → GitHub release) plus the gotchas learned from
  releases v0.3.0–v0.4.3.
---

# koboi-release — Publish a new koboi-agent release

## Overview
Execute the full release sequence for koboi-agent: ensure code is merged → pre-check → bump
version → tag → tag-push triggers auto-publish to PyPI (Trusted Publishing) + GHCR (Docker
image) → verify → GitHub release. Both publish workflows are tag-triggered: `release.yml`
(PyPI) and `docker.yml` (GHCR).

Three scripts (all under `.claude/skills/koboi-release/scripts/`, invoked from the repo root):
- **`pre-release-check.sh`** — the 6 CI gates locally (Gate 0 refreshes the `.venv` editable
  install + dev toolchain; Gates 1-5 = ruff/format/mypy/bandit/pytest).
- **`bump-and-tag.sh [--dry-run] X.Y.Z "msg"`** — mechanical bump + commit + push main + tag +
  push tag (validates `X.Y.Z`, resumable, `--dry-run` rehearses without pushing).
- **`verify-release.sh X.Y.Z`** — waits for *this tag's* PyPI+GHCR runs, watches them, verifies
  PyPI version + GHCR `:X.Y.Z`/`:latest` + a `/healthz` smoke.

## When to use
Triggered by: "release koboi-agent", "bump version and publish", "create release vX.Y.Z",
"push to PyPI", "tag and release". Determine the version bump (patch/minor/major) from the
changes since the last release. Check `git tag --sort=-creatordate | head` for the latest tag.

## Pre-conditions (one-time, already done at v0.3.0/v0.4.1)
- **PyPI Trusted Publishing** configured for `koboi-agent` (repo `hedypamungkas/koboi-agent`,
  workflow `release.yml`, environment `pypi`).
- **GHCR package** created + public (`ghcr.io/hedypamungkas/koboi-agent`).
- **Branch protection** on `main`: "PR required" rule. Release version-bump commits bypass it
  (owner override — works, prints a warning).
- A CI-faithful `.venv` (Python 3.10+) with the dev toolchain. If missing, the scripts will
  tell you: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev,tui,api]"`.

## Step 0 — Pre-release validation (BEFORE tagging)
```bash
.claude/skills/koboi-release/scripts/pre-release-check.sh
```
Gate 0 refreshes the editable install + dev toolchain (mirrors ci.yml), so mypy/bandit are
present AND `test_version_matches_pyproject` sees the current version (no stale-install
false-red). Gates 1-5: ruff check, ruff format, mypy (~=1.19.1), bandit (>=1.9), pytest
(cov ≥ 80; e2e self-skips). If any gate fails, FIX before releasing.

## Step 1 — Merge PRs + sync main
Ensure all intended code is on `main` (merge any open PRs first via `gh pr merge <n> --merge`):
```bash
git checkout main && git pull origin main
```

## Step 2 — Bump version
Rehearse first (no commit/tag/push — validates the version + bump mechanism):
```bash
.claude/skills/koboi-release/scripts/bump-and-tag.sh --dry-run X.Y.Z "vX.Y.Z — <summary>"
```
Then release (bumps `pyproject.toml`, commits `chore(release): bump version to X.Y.Z`, pushes
main — owner-override bypass warning is expected — tags `vX.Y.Z`, pushes the tag):
```bash
.claude/skills/koboi-release/scripts/bump-and-tag.sh X.Y.Z "vX.Y.Z — <summary>"
```
The tag push is **irreversible** (triggers PyPI, which is immutable) — that's why `--dry-run`
exists. The script refuses if the tag already exists, and resumes (skip-bump) if `main` was
pushed but the tag wasn't.

## Step 3 — Verify the publish (PyPI + GHCR + smoke)
```bash
.claude/skills/koboi-release/scripts/verify-release.sh X.Y.Z
```
This waits for *this tag's* `release.yml` and `docker.yml` runs (filtered by `--branch vX.Y.Z`,
which kills the "latest-run" race that returned the previous release's run), watches both to
green, retries PyPI propagation (~90s), pulls `:X.Y.Z` + `:latest`, and smoke-tests
`/healthz` with retries + cleanup. **Verification comes before the GitHub release on purpose**
— don't advertise a version that hasn't reached PyPI.

## Step 4 — GitHub release
```bash
gh release create vX.Y.Z --title "vX.Y.Z — <title>" --notes "$(cat <<'EOF'
<compact notes: reference PR number(s), bullet the key changes, keep to ~10 lines>
EOF
)"
```

## Critical gotchas (inline — see references/gotchas.md for detail)
- **Interpreters**: scripts use `.venv/bin/python` or `python3` — never bare `python` (macOS
  non-interactive bash has no `python`, only a zsh alias → `set -e` abort).
- **Bandit**: now in the `dev` extra (>=1.9). Always `.venv/bin/bandit` — system bandit
  (1.8.6) gives a false green.
- **release.yml test job**: installs `.[dev,tui,api]` — if `[api]` missing, server test
  modules error on `import fastapi`.
- **Docker default config**: `server_simple.yaml` (concrete model). NOT `e2e_full.yaml`.
- **e2e self-skip**: `tests/e2e/conftest.py` skips when no live server.
- **Tag naming**: git tag = `vX.Y.Z` (with v); PyPI version = `X.Y.Z` (no v); **GHCR image tag = `X.Y.Z` (no `v`) + `:latest`** — since #13 rewrote docker.yml (`type=semver,{{version}}`). Releases ≤ v0.7.0 used the old `:vX.Y.Z` convention.
- **PyPI is immutable**: a published version CANNOT be re-published. To ship a fix, bump to
  the next version.

## Error handling
- **A workflow job fails** (release.yml test, docker build, etc.): fix the root cause, then
  `gh run rerun <run-id> --failed` (re-runs only the failed job). Do NOT re-tag — the tag
  already exists and PyPI may already have the version.
- **PyPI publish fails** ("invalid-publisher"): Trusted Publishing mismatch. Check PyPI →
  koboi-agent → Publishing (repo=`hedypamungkas/koboi-agent`, workflow=`release.yml`,
  environment=`pypi`). Then `gh run rerun`.
- **PyPI version already exists**: cannot re-publish. Bump to the next version instead.
- **Tag pushed but version was wrong** (rare, requires re-tag): only then delete + recreate
  the tag — `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`, fix, re-bump to a NEW
  version (PyPI immutability means you can't reuse vX.Y.Z on PyPI even if GHCR can be rebuilt).

## Additional resources

### Reference files
- **`references/gotchas.md`** — Full gotcha catalog from v0.3.0–v0.4.3 (each release's lesson).
- **`references/workflows.md`** — `release.yml` + `docker.yml` structure, triggers, debugging.

### Scripts (under `.claude/skills/koboi-release/scripts/`)
- **`pre-release-check.sh`** — 6 CI gates locally (Gate 0 refreshes `.venv` + toolchain).
- **`bump-and-tag.sh`** — bump + commit + push main + tag + push tag (`--dry-run` to rehearse).
- **`verify-release.sh`** — wait/watch this tag's PyPI+GHCR runs + verify artifacts + smoke.
