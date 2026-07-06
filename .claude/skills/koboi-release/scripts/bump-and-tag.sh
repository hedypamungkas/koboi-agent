#!/usr/bin/env bash
# Mechanical release: bump version + commit + push main + tag + push tag.
# The tag push is IRREVERSIBLE: it triggers release.yml (PyPI, immutable) + docker.yml (GHCR).
#
# Usage:
#   bump-and-tag.sh [--dry-run] X.Y.Z "<tag-message>"
#   bump-and-tag.sh 0.4.3 "v0.4.3 — security fixes"
#   bump-and-tag.sh --dry-run 0.4.3 "v0.4.3 — security fixes"   # bump+revert, no commit/tag/push
#
# Safety:
#   - validates X.Y.Z (strips a leading 'v' — prevents 'vv0.4.3' tags / bad pyproject);
#   - refuses if the tag already exists (PyPI versions are immutable — bump to a new version);
#   - resumable: if pyproject is already at X.Y.Z (main pushed but tag push failed), skips the
#     bump+commit and only creates+pushes the tag;
#   - --dry-run bumps pyproject, shows the diff, and reverts — no commit, no tag, no push.
set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi

VERSION="${1:?usage: $0 [--dry-run] X.Y.Z 'tag message'}"
MSG="${2:?usage: $0 [--dry-run] X.Y.Z 'tag message'}"

# Strip a leading 'v' and validate X.Y.Z (a leading-'v' arg would else yield 'vv0.4.3').
VERSION="${VERSION#v}"
if ! printf '%s' "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "❌ VERSION must be X.Y.Z (got '$VERSION')"; exit 1
fi

cd "$(git rev-parse --show-toplevel)"

# Prefer the venv interpreter (CI-faithful); fall back to python3 (present on macOS + Linux + CI).
# A zsh-only `python` alias is NOT visible to non-interactive bash, so never call bare `python`.
PY="python3"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

TAG="v${VERSION}"

# Resume-safety: refuse if the tag already exists.
if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
  echo "❌ tag ${TAG} already exists. PyPI versions are immutable — to ship a fix, bump to a new version."
  echo "   (local-only delete, only if you know no publish happened: git tag -d ${TAG})"
  exit 1
fi

current_version() {
  "$PY" -c "import re,sys; sys.stdout.write(re.search(r'version = \"([^\"]+)\"', open('pyproject.toml').read()).group(1))"
}

if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN ($VERSION) — bump pyproject, show diff, revert. No commit/tag/push. ==="
  BAK="/tmp/koboi-pyproject.$$.bak"
  cp pyproject.toml "$BAK"
  trap 'cp "$BAK" pyproject.toml 2>/dev/null || true; rm -f "$BAK"' EXIT
  "$PY" -c "
import re
p='pyproject.toml'; s=open(p).read()
old=re.search(r'version = \"([^\"]+)\"', s).group(1)
open(p,'w').write(s.replace(f'version = \"{old}\"', f'version = \"${VERSION}\"'))
print(f'  would bump {old} -> ${VERSION}')
"
  git --no-pager diff -- pyproject.toml || true
  cp "$BAK" pyproject.toml; rm -f "$BAK"; trap - EXIT
  echo "✅ dry-run ok: version bump is well-formed. Re-run without --dry-run to release."
  exit 0
fi

# Ensure on main + up to date
git checkout main
git pull origin main -q

# Bump (skip if pyproject is already at VERSION — resumable after a main-only push)
CURRENT="$(current_version)"
if [ "$CURRENT" = "$VERSION" ]; then
  echo "pyproject already at ${VERSION} (resuming) — skipping bump + commit"
else
  "$PY" -c "
import re
p='pyproject.toml'; s=open(p).read()
old=re.search(r'version = \"([^\"]+)\"', s).group(1)
open(p,'w').write(s.replace(f'version = \"{old}\"', f'version = \"${VERSION}\"'))
print(f'version {old} -> ${VERSION}')
"
  git add pyproject.toml
  git commit -m "chore(release): bump version to ${VERSION}"
  # Push main (owner override of the "PR required" rule — expected bypass warning)
  git push origin main
fi

# >>> IRREVERSIBLE: tag push triggers PyPI (immutable) + GHCR. Point of no return. <<<
echo ""
echo ">>> Pushing ${TAG} — triggers release.yml (PyPI, immutable) + docker.yml (GHCR)."
git tag -a "${TAG}" -m "${MSG}"
git push origin "${TAG}"

echo ""
echo "=========================="
echo " Tagged ${TAG} ✅"
echo "=========================="
echo "release.yml (PyPI) + docker.yml (GHCR) triggered."
echo ""
echo "Next: verify the publish, THEN create the GitHub release:"
echo "  .claude/skills/koboi-release/scripts/verify-release.sh ${VERSION}"
echo "  gh release create ${TAG} --title '${TAG} — ...' --notes '...'"
