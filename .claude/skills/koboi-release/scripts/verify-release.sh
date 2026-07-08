#!/usr/bin/env bash
# Verify a just-published release: wait for THIS tag's release.yml (PyPI) + docker.yml (GHCR)
# runs, watch them, then confirm PyPI version, GHCR image (:X.Y.Z + :latest), and a /healthz
# smoke test.
#
# Why this exists (replaces the fragile inline commands that used to live in SKILL.md):
#   - `gh run list --limit 1` RACES GitHub's run creation and can return the PREVIOUS release's
#     run → false green. We poll for the run whose --branch is the tag instead.
#   - PyPI's /json endpoint lags ~30s after publish → a single un-retried urlopen 404s. We retry.
#   - The smoke test used a single `sleep 5 && curl` that leaked the container on failure. We
#     poll /healthz and clean up via a trap.
#
# Usage: verify-release.sh X.Y.Z
set -euo pipefail

VERSION="${1:?usage: $0 X.Y.Z}"
VERSION="${VERSION#v}"
# GHCR image tag has NO "v" prefix since PR #13 rewrote docker.yml: the metadata
# action's `type=semver,pattern={{version}}` yields e.g. 0.8.0, not v0.8.0.
# (Releases <= v0.7.0 used the old v6 workflow and DO carry a v prefix — :v0.7.0.)
TAG="${VERSION}"
IMG="ghcr.io/hedypamungkas/koboi-agent"

# Wait until a run of <workflow> triggered by <tag> appears, then watch it to completion.
watch_tag_run() {
  local wf="$1" name="$2" i run_id
  echo "=== waiting for ${name} (${wf}) run on ${TAG} to appear ==="
  run_id=""
  for i in $(seq 1 30); do          # up to ~150s for GitHub to register the run
    run_id="$(gh run list --workflow="${wf}" --branch="${TAG}" --limit 1 \
                --json databaseId --jq '.[0].databaseId // empty' 2>/dev/null || true)"
    [ -n "$run_id" ] && break
    sleep 5
  done
  if [ -z "$run_id" ]; then
    echo "❌ no ${name} run found for ${TAG} after ~150s."
    echo "   check: gh run list --workflow=${wf} --branch=${TAG}"
    exit 1
  fi
  echo "  ${name} run: https://github.com/hedypamungkas/koboi-agent/actions/runs/${run_id}"
  gh run watch "$run_id" --exit-status --interval 15
}

watch_tag_run release.yml "PyPI"
watch_tag_run docker.yml "GHCR"

# Verify PyPI (retry for ~90s to absorb propagation lag)
echo "=== verify PyPI: koboi-agent ${VERSION} ==="
ok=0
for i in $(seq 1 18); do
  if python3 -c "import json,urllib.request,sys; sys.exit(0 if json.load(urllib.request.urlopen('https://pypi.org/pypi/koboi-agent/${VERSION}/json', timeout=15))['info']['version']=='${VERSION}' else 1)" 2>/dev/null; then
    ok=1; break
  fi
  sleep 5
done
[ "$ok" = 1 ] && echo "  PyPI ✅ koboi-agent ${VERSION} live" || { echo "❌ PyPI never reported ${VERSION}"; exit 1; }

# Verify GHCR :vX.Y.Z + :latest (a silent :latest failure would otherwise ship a stale latest)
echo "=== verify GHCR: ${IMG}:${TAG} + :latest ==="
docker pull "${IMG}:${TAG}" >/dev/null && echo "  ✅ ${IMG}:${TAG}" \
  || { echo "❌ docker pull ${IMG}:${TAG} failed"; exit 1; }
docker pull "${IMG}:latest" >/dev/null && echo "  ✅ ${IMG}:latest" \
  || { echo "❌ docker pull ${IMG}:latest failed"; exit 1; }

# Smoke test /healthz with retries + guaranteed cleanup
echo "=== smoke test /healthz ==="
CONTAINER="koboi-verify-${VERSION}-$$"
HOST_PORT=8088
docker run -d --rm --name "$CONTAINER" -e KOBOI_API_KEYS=koboi_test -p ${HOST_PORT}:8080 "${IMG}:${TAG}" >/dev/null
trap 'docker stop "$CONTAINER" >/dev/null 2>&1 || true' EXIT
ok=0
for i in $(seq 1 20); do            # up to ~100s (pull already done; allow slow boot)
  if curl -sf "http://localhost:${HOST_PORT}/healthz" >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
if [ "$ok" = 1 ]; then
  echo "  /healthz ✅ $(curl -sf "http://localhost:${HOST_PORT}/healthz")"
else
  echo "❌ /healthz did not respond; container logs:"; docker logs "$CONTAINER" 2>&1 | tail -15
  exit 1
fi
docker stop "$CONTAINER" >/dev/null 2>&1 || true; trap - EXIT

echo ""
echo "=========================="
echo " RELEASE ${TAG} VERIFIED ✅"
echo "=========================="
echo "PyPI ${VERSION} + GHCR ${TAG}/latest + /healthz all green."
echo "Now publish the GitHub release (release-create comes AFTER verification on purpose):"
echo "  gh release create ${TAG} --title '${TAG} — ...' --notes '...'"
