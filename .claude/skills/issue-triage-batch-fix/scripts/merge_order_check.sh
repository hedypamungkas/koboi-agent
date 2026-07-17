#!/usr/bin/env bash
# merge_order_check.sh — detect inter-PR merge conflicts before advising merge order.
#
# Usage:
#   merge_order_check.sh <pr-or-branch> [<pr-or-branch> ...]
#
# Each arg is either a PR number (resolved to its head branch via `gh`) or a git ref.
# Fetches origin, then for every ordered pair (A then B) simulates "merge A into main,
# then B into that" via `git merge-tree` and reports CLEAN / CONFLICT.
#
# Requires: git >= 2.38 (for `git merge-tree --write-tree`), gh (only if args are PR #s).
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <pr-or-branch> [<pr-or-branch> ...]" >&2
  exit 2
fi

cd "$(git rev-parse --show-toplevel)"
git fetch origin -q

refs=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    br=$(gh pr view "$arg" --json headRefName -q .headRefName 2>/dev/null \
         || { echo "ERROR: cannot resolve PR #$arg (gh failed)" >&2; exit 1; })
    # ensure the branch ref is locally known
    git fetch origin "$br" -q 2>/dev/null || true
    refs+=("origin/$br (#$arg)")
  else
    refs+=("$arg")
  fi
done

ref_of() { echo "$1" | sed 's/ (#.*//'; }

echo "Base: origin/main ($(git log -1 --oneline origin/main | cut -d' ' -f1))"
echo "Pairwise order check (does merging B after A conflict?):"
echo ""

conflict_found=0
for a in "${refs[@]}"; do
  for b in "${refs[@]}"; do
    [ "$a" = "$b" ] && continue
    ra=$(ref_of "$a"); rb=$(ref_of "$b")
    # Build the (main + A) merge tree, then merge B into it.
    if tree_a=$(git merge-tree --write-tree origin/main "$ra" 2>/dev/null); then
      # main+A merged cleanly; now test B into that tree
      if git merge-tree --write-tree "$tree_a" "$rb" >/dev/null 2>&1; then
        status="CLEAN"
      else
        status="CONFLICT"; conflict_found=1
      fi
    else
      # A itself conflicts with main — note and still test B vs main
      if git merge-tree --write-tree origin/main "$rb" >/dev/null 2>&1; then
        status="CLEAN (note: $a itself conflicts main)"
      else
        status="CONFLICT"; conflict_found=1
      fi
    fi
    printf "  %-34s then %-34s : %s\n" "$a" "$b" "$status"
  done
done

echo ""
# Also run the full sequence in argument order.
echo "Full sequence in arg order (${refs[*]}):"
SIM=$(mktemp -d)
git worktree add --detach "$SIM" origin/main -q
ok=1
for r in "${refs[@]}"; do
  rr=$(ref_of "$r")
  if (cd "$SIM" && git merge --no-edit "$rr" -q >/dev/null 2>&1); then
    printf "  merge %-34s : OK\n" "$r"
  else
    printf "  merge %-34s : CONFLICT\n" "$r"
    (cd "$SIM" && git merge --abort 2>/dev/null || true)
    ok=0
    break
  fi
done
git worktree remove "$SIM" --force 2>/dev/null || true

echo ""
if [ "$conflict_found" -eq 0 ] && [ "$ok" -eq 1 ]; then
  echo "RESULT: all orders CLEAN — merge in any order."
  exit 0
else
  echo "RESULT: conflicts possible — see above; pick an order or rebase the second."
  exit 1
fi
