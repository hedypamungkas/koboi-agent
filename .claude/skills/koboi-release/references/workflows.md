# Release workflows — release.yml + docker.yml

## release.yml (PyPI publish)

**Trigger**: `on: push: tags: ["v*"]`

**Jobs**:
1. `build` — `python -m build` (sdist + wheel), upload artifact.
2. `test` (matrix 3.10/3.11/3.12) — `pip install -e ".[dev,tui,api]"` + `pytest --tb=short -q`. e2e self-skips (no live server).
3. `publish` — `needs: [build, test]`. Downloads artifact, `pypa/gh-action-pypi-publish@release/v1` with `id-token: write` (Trusted Publishing, no API token). Environment: `pypi`.

**Key**: `publish` is GATED on `build` + `test`. If test fails, publish is SKIPPED. If build fails, nothing runs.

**Debugging**:
- If test fails: check the matrix version's log. Common causes: missing extra (`[api]`), e2e not self-skipping, or a real test regression.
- If publish fails ("invalid-publisher"): Trusted Publishing not configured or mismatched (check repo/workflow/environment on PyPI).
- If publish fails ("file already exists"): the version was already published. PyPI versions are immutable. Bump to next version.

## docker.yml (GHCR image publish)

**Trigger**: `on: push: tags: ["v*"]` (same trigger as release.yml — both fire on tag push, in parallel).

**Job**:
1. `build-and-push` — `docker/setup-qemu-action` (multi-arch) + `docker/setup-buildx-action` + `docker/login-action` (ghcr.io, `${{ github.token }}`) + `docker/metadata-action` (tags: `type=ref,event=tag` → git tag name `vX.Y.Z`; `type=raw,value=latest`) + `docker/build-push-action` (platforms: `linux/amd64,linux/arm64`, push: true, GHA cache).

**Key**: Uses `GITHUB_TOKEN` (no extra secrets needed). Multi-arch via QEMU (slower build, ~5-8 min). Tags: `vX.Y.Z` (git tag name) + `latest`.

**Debugging**:
- If build fails: check Dockerfile (CMD format, COPY paths, config validity). The Dockerfile must COPY all needed dirs (koboi/, configs/, examples/, skills/).
- If push fails: check `permissions: packages: write` is in the workflow.
- Dockerfile lint warning `JSONArgsRecommended`: shell-form CMD triggers this heuristic. Harmless (sh execs the single python command → PID 1 → signals work).

## CI (ci.yml — NOT a release workflow, but runs on main push)

**Trigger**: `on: push: branches: [main]` + `pull_request: branches: [main]`.

**Jobs**: `lint` (ruff check + format + mypy), `test` (matrix 3.10/3.11/3.12 + pytest cov≥80), `security` (pip-audit [continue-on-error] + bandit).

**Note**: CI runs on every PR + main push. The RELEASE workflows (release.yml + docker.yml) only run on tag push. So the release version-bump commit (pushed to main) triggers CI (which should pass — the bump is trivial), and the TAG push triggers the release workflows.

## Workflow run order on a release
```
git push origin vX.Y.Z
  ├─ release.yml starts (build → test → publish PyPI)
  ├─ docker.yml starts (build + push GHCR image)
  └─ ci.yml starts (on main push, lint/test/security — the version-bump commit)
```
All three run in parallel. ci.yml is not gated; release.yml + docker.yml are independent (neither gates the other). Watch all three if desired, but only release.yml + docker.yml are release-critical.
