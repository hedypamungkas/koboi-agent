#!/usr/bin/env bash
# Mechanical release: bump version + commit + push main + tag + push tag.
# Triggers release.yml (PyPI) + docker.yml (GHCR image) on the tag push.
#
# Usage: bump-and-tag.sh <version> "<tag-message>"
# Example: bump-and-tag.sh 0.4.3 "v0.4.3 — fix X + add Y"
set -euo pipefail

VERSION="${1:?usage: $0 X.Y.Z 'tag message (e.g. v0.4.3 — summary)'}"
MSG="${2:?usage: $0 X.Y.Z 'tag message'}"

cd "$(git rev-parse --show-toplevel)"

# Ensure on main + up to date
git checkout main
git pull origin main -q

# Bump version in pyproject.toml (single location)
python -c "
p='pyproject.toml'; s=open(p).read()
import re
m=re.search(r'version = \"([^\"]+)\"', s)
old=m.group(1)
assert old != '${VERSION}', f'version already {old}'
s=s.replace(f'version = \"{old}\"', f'version = \"${VERSION}\"')
open(p,'w').write(s)
print(f'version {old} -> ${VERSION}')
"

# Commit + push (may warn "Bypassed rule violations: Changes must be made through a PR" — expected)
git add pyproject.toml
git commit -m "chore(release): bump version to ${VERSION}"
git push origin main

# Tag + push (triggers release.yml + docker.yml)
git tag -a "v${VERSION}" -m "${MSG}"
git push origin "v${VERSION}"

echo ""
echo "=========================="
echo " Tagged v${VERSION} ✅"
echo "=========================="
echo "release.yml (PyPI) + docker.yml (GHCR) triggered."
echo ""
echo "Next steps:"
echo "  1. gh release create v${VERSION} --title 'v${VERSION} — ...' --notes '...'"
echo "  2. gh run list --limit 3    # find Release + Docker run IDs"
echo "  3. gh run watch <release-run-id> --exit-status"
echo "  4. gh run watch <docker-run-id> --exit-status"
echo "  5. python -c \"import urllib.request,json; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/koboi-agent/${VERSION}/json'))['info']['version'])\""
echo "  6. docker pull ghcr.io/hedypamungkas/koboi-agent:v${VERSION}"
