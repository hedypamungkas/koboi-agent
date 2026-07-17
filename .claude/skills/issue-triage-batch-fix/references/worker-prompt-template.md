# Worker Prompt Template

Each worktree worker gets a **fully self-contained** prompt (it has no access to this conversation). Structure:

```
GOAL: Fix GitHub issue #NN (<one-line>) in <repo>, using STRICT TDD, in your
isolated git worktree, then open a PR to `main` that closes #NN.

This is work unit <N> of <total> (parallel). CONSTRAINTS: <do/don't — e.g. do NOT
edit jobs.py to stay conflict-free with PR #X>.

=== STEP 0 (only for a newly-filed issue) ===
gh issue create ...; capture the number NN; use it in branch/commit/PR.

=== THE BUG (verified on main) ===
<2-4 sentences>. Relevant current code (re-read to confirm exact line numbers):
- <file>:<lines> — <symbol>, quote what it does wrong.
- <compare to the correct sibling pattern if one exists, with file:line>.

=== THE FIX ===
<numbered, concrete: which file, which lines, what shape; reuse existing patterns
by name. Cover every failure/rollback path. Note back-compat decisions.>

=== RED TESTS (write FIRST; assert the FIXED behavior, fail today) ===
<test file path + class/method names; fixtures to use; the exact assertion that
fails today and why; how to make it deterministic (race-gate / mock / pre-seed).>
See empirical-red-tests.md for the techniques.

=== SHARED WORKER TEMPLATE (follow exactly) ===
You are fixing ONE unit in an isolated git worktree. STRICT TDD — never write the
fix before the red test.
1. RED TEST FIRST — write the failing test(s) above using repo conventions
   (<test runner; asyncio_mode; fixtures; how to build the app/client; owner/api-key
   derivation; commit message style>). Run `pytest <file>::<test> -x` and CONFIRM it
   fails for the bug's reason (not an unrelated error). Capture the red output.
2. IMPLEMENT THE FIX exactly per spec.
3. GREEN — re-run the new test(s); confirm PASS.
4. FULL SUITE — run `pytest` (or repo equivalent). Everything passes. Fix regressions
   YOUR change caused.
5. GATES — `ruff check <pkg> tests` (or repo linter) + format check; fix issues.
6. CODE REVIEW — invoke the Skill tool with `skill="code-review"`; fix findings.
7. COMMIT + PUSH + PR — branch `fix/<NN>-<slug>`; commit `fix(#NN): <summary>`
   (match repo style); commit test+fix together (final green). `gh pr create` with
   body documenting: bug (file:line), the RED proof (failing assertion + captured
   red output), the fix, verification (full suite + gates green). Add `Closes #NN`.
   For platform-gated/untestable-locally parts, note how they're guarded.
8. REPORT — end with one line: `PR: <url>` (or `PR: none — <reason>`).

REPO FACTS: <working dir; test mode; things never to read (e.g. giant fixtures);
CI gates; recent commit style; symbols already in scope>.
```

## Why each piece matters
- **Self-contained**: the worker can't see the parent chat — every `file:line`, convention, and constraint must be inline.
- **RED first, confirmed for the right reason**: a test that fails for an unrelated cause (import error, wrong fixture) proves nothing. Always read the red output.
- **Assert the FIXED behavior**: the red test encodes the desired post-fix state, so it flips red→green without rewriting.
- **Full suite = zero broken changes**: the regression gate; the user's hard requirement.
- **`PR: <url>`**: machine-parseable so the coordinator can update the status table.
- **Constraints block**: prevents two parallel workers editing the same file from producing unmergeable PRs (e.g. "do NOT edit X; implement media-local admission instead").

## Injecting repo facts
Before dispatching, gather once (so every worker prompt is accurate):
- Test runner + mode (e.g. `pytest`, `asyncio_mode="auto"` → no decorators).
- Shared fixtures + how to build the system under test (e.g. `create_app(...)` + `httpx ASGITransport`; `MockClient` from `conftest.py`).
- Identity derivation if auth matters (e.g. owner = `"env:"+sha256(api_key)[:12]`).
- Lint/format/mypy commands + any local-venv drift caveats (e.g. dev venv has false-red mypy stubs → trust CI mypy).
- Commit/branch conventions (e.g. `fix(#NN): ...`).
- Files/paths to never read (giant fixtures) and platform-gated code that needs mocks.
