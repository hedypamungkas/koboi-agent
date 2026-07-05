---
name: koboi-release
description: >-
  This skill should be used when the user asks to "release koboi-agent", "bump version
  and publish", "create a new release", "push to PyPI", "tag and release", "ship version
  vX.Y.Z", or mentions PyPI publishing, GitHub releases, or GHCR container image publishing
  for the koboi-agent project. Encodes the full release sequence (merge → bump → tag →
  PyPI + GHCR auto-publish → verify) plus the gotchas learned from releases v0.3.0–v0.4.2.
---

# koboi-release — Publish a new koboi-agent release

## Overview
Execute the full release sequence for koboi-agent: ensure code is merged → bump version →
tag → tag-push triggers auto-publish to PyPI (Trusted Publishing) + GHCR (Docker image) →
verify. Both workflows are tag-triggered: `release.yml` (PyPI) and `docker.yml` (GHCR).

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

## Step 0 — Pre-release validation (BEFORE tagging)
Run all CI gates locally with CI-faithful tooling:
```bash
scripts/pre-release-check.sh
```
All 5 must pass: ruff check, ruff format, mypy, bandit 1.9.4, pytest (cov ≥80, e2e self-skips).
If any fail, FIX before releasing.

**CRITICAL**: Use `.venv/bin/ruff` + `.venv/bin/bandit` (CI-matching versions). Do NOT use
system ruff/bandit (1.8.6 → false green). mypy: `python -m mypy`. pytest: `.venv/bin/python -m pytest`.

## Step 1 — Merge PRs + sync main
Ensure all code is on `main` (merge any open PRs first via `gh pr merge <n> --merge`):
```bash
git checkout main && git pull origin main
```

## Step 2 — Bump version
The mechanical bump + commit + push + tag:
```bash
scripts/bump-and-tag.sh X.Y.Z "vX.Y.Z — <one-line summary>"
```
This bumps `pyproject.toml`, commits `chore(release): bump version to X.Y.Z`, pushes main
(may warn "Bypassed rule violations" — expected), tags `vX.Y.Z`, and pushes the tag.

The tag push triggers BOTH `release.yml` (PyPI) and `docker.yml` (GHCR image) workflows.

## Step 3 — GitHub release
```bash
gh release create vX.Y.Z --title "vX.Y.Z — <title>" --notes "$(cat <<'EOF'
<compact notes: reference PR number(s), bullet the key changes, keep to ~10 lines>
EOF
)"
```

## Step 4 — Watch + verify PyPI (release.yml)
```bash
gh run list --workflow=release.yml --limit 1   # get the run ID
gh run watch <run-id> --exit-status
```
Verify (allow ~30s propagation after publish):
```bash
python -c "import urllib.request,json; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/koboi-agent/X.Y.Z/json'))['info']['version'])"
```

## Step 5 — Watch + verify GHCR image (docker.yml)
```bash
gh run list --workflow=docker.yml --limit 1    # get the run ID
gh run watch <run-id> --exit-status
```
Verify the image pulls:
```bash
docker pull ghcr.io/hedypamungkas/koboi-agent:vX.Y.Z
```
**Note**: the image tag is `vX.Y.Z` (WITH the `v` prefix, matching the git tag), NOT `X.Y.Z`
(the PyPI version). Both `:vX.Y.Z` and `:latest` are pushed.

## Step 6 — (Optional) Smoke-test the published image
```bash
docker run --rm -e KOBOI_API_KEYS=koboi_test -p 8080:8080 ghcr.io/hedypamungkas/koboi-agent:vX.Y.Z &
sleep 5 && curl -sf http://localhost:8080/healthz && kill %1
```

## Critical gotchas (inline — see references/gotchas.md for detail)
- **Bandit version**: CI uses 1.9.4. System ruff/bandit (1.8.6) gives false green. Always `.venv/bin/`.
- **release.yml test job**: installs `.[dev,tui,api]` — if `[api]` missing, 11 server test modules error on `import fastapi`.
- **Docker default config**: `server_simple.yaml` (concrete model). NOT `e2e_full.yaml` (env-parameterized → validation crash on bare run).
- **e2e self-skip**: `tests/e2e/conftest.py` skips when no live server. If removed, CI fails on e2e ConnectError.
- **Tag naming**: GHCR image = `vX.Y.Z` (with v). PyPI version = `X.Y.Z` (no v). Don't confuse.
- **PyPI propagation**: `/json` endpoint lags ~30s. Use `/<version>/json` or wait.
- **Re-tagging**: if a workflow fails, fix root cause first, then delete + recreate tag (`git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z && git tag -a ... && git push origin vX.Y.Z`). PyPI versions are immutable (cannot re-publish).

## Error handling
- **PyPI publish fails** ("invalid-publisher"): Trusted Publishing mismatch. Check PyPI → koboi-agent → Publishing (repo=`hedypamungkas/koboi-agent`, workflow=`release.yml`, environment=`pypi`).
- **release.yml test fails**: fix the code, re-tag (delete + recreate). The test job runs full pytest (e2e self-skips).
- **docker.yml build fails**: check Dockerfile (CMD format, config validity, COPY paths).
- **PyPI version already exists**: cannot re-publish. Bump to the next version instead.

## Additional resources

### Reference files
- **`references/gotchas.md`** — Full gotcha catalog from v0.3.0–v0.4.2 (each release's lesson).
- **`references/workflows.md`** — `release.yml` + `docker.yml` structure, triggers, debugging.

### Scripts
- **`scripts/pre-release-check.sh`** — Run all 5 CI gates locally (CI-faithful tooling).
- **`scripts/bump-and-tag.sh`** — Mechanical bump + commit + push + tag + push (args: version + message).
