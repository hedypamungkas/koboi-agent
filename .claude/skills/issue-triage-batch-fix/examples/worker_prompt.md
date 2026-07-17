# Worker prompt — full boilerplate (adapt per unit)

Copy this into each `Agent(prompt=...)` call. Replace `<…>`. Keep it self-contained — the
worker cannot see the parent conversation.

---

GOAL: Fix GitHub issue #<NN> (<one-line title>) in the <repo> repo, using STRICT TDD, in
your isolated git worktree, then open a PR to `main` that closes #<NN>.

This is work unit <N> of <total> (parallel). CONSTRAINTS: <e.g. do NOT edit <file> — stay
conflict-free with PR #X; implement <alternative> instead>.

=== STEP 0 (only if the issue was newly filed) ===
`gh issue create --title "..." --body "..."`; capture the number; use it in branch/commit/PR.

=== THE BUG (verified on `main`) ===
<2-4 sentences.>
Relevant current code (re-read to confirm exact line numbers):
- `<file>:<lines>` — <symbol>; quote what it does wrong.
- Compare the correct sibling pattern at `<file>:<lines>` (if one exists).

=== THE FIX ===
1. <file>:<lines> — <concrete change, reusing existing pattern <name>>.
2. <every failure/rollback path covered>.
3. <back-compat note: what stays unchanged and why>.

=== RED TESTS (write FIRST; assert the FIXED behavior — they fail today) ===
- `<test file>` → `class TestX`:
  - `test_<headline>`: <assertion that fails today for the bug's reason>. Determinism:
    <race-gate with asyncio.Event / pre-seed the registry to CAP / mock the platform
    module via sys.modules — see references/empirical-red-tests.md>.
  - `test_<edge>`: <adjacent assertion / regression guard>.
Run `pytest <file>::<test> -x`; CONFIRM it fails for the bug's reason; capture the red output.

=== SHARED WORKER TEMPLATE (follow exactly) ===
You are fixing ONE unit in an isolated git worktree. STRICT TDD — never write the fix
before the red test.
1. RED TEST FIRST — write the failing test(s) above using repo conventions (<test runner>;
   <asyncio_mode=auto → no decorators>; fixtures <…>; how to build <the app/client>;
   <owner/api-key derivation>; <commit style>). Run `pytest <file>::<test> -x` and CONFIRM
   it fails for the bug's reason (not an unrelated error). Capture the red output.
2. IMPLEMENT THE FIX exactly per spec.
3. GREEN — re-run the new test(s); confirm PASS.
4. FULL SUITE — run `pytest` (or repo equivalent). Everything passes. Fix regressions
   YOUR change caused.
5. GATES — `<lint>` + `<format check>`; fix issues.
6. CODE REVIEW — invoke the Skill tool with `skill="code-review"`; fix findings.
7. COMMIT + PUSH + PR — branch `fix/<NN>-<slug>`; commit `fix(#NN): <summary>` (match repo
   style); commit test+fix together (final green). `gh pr create` body documents: bug
   (file:line), the RED proof (failing assertion + captured red output), the fix,
   verification (full suite + gates green). Add `Closes #NN`. Note any platform-gated /
   can't-test-locally part and how it's guarded.
8. REPORT — end with one line: `PR: <url>` (or `PR: none — <reason>`).

REPO FACTS: working dir is <repo>; <asyncio_mode=auto>; never read <giant fixture path>;
CI gates: <lint/format/cov/build/mypy>; recent commit style: `fix(#NN): ...`; <symbols
already in scope>; <platform-gated code → mock-based tests only>.

---

Adaptation checklist before dispatch:
- [ ] `<NN>`, title, repo, unit index filled in.
- [ ] `THE BUG` quotes real current `file:line` (re-located, not the issue's stale numbers).
- [ ] `THE FIX` reuses a named existing pattern + covers every failure path.
- [ ] `RED TESTS` assert FIXED behavior, are deterministic (gate/seed/mock named).
- [ ] CONSTRAINTS prevent same-file collisions with sibling units.
- [ ] REPO FACTS accurate (runner, fixtures, owner derivation, commit style, never-read paths).
