"""Tests for koboi.tools.builtin._patch (unified-diff parser + applier)."""

from __future__ import annotations

import pytest

from koboi.tools.builtin._patch import Hunk, PatchError, apply_hunks, parse_unified_diff


class TestParseUnifiedDiff:
    def test_single_hunk_bare(self):
        patch = "@@ -1,3 +1,3 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
        hunks = parse_unified_diff(patch)
        assert len(hunks) == 1
        h = hunks[0]
        # The leading space on a context line is the diff MARKER, not file
        # content -- the parser strips it (file has no leading space there).
        assert h.old_text == "def add(a, b):\n    return a - b\n"
        assert h.new_text == "def add(a, b):\n    return a + b\n"
        assert h.old_start == 1 and h.new_start == 1

    def test_ignores_file_header_pair(self):
        patch = "--- a/code.py\n+++ b/code.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        hunks = parse_unified_diff(patch)
        assert len(hunks) == 1
        assert hunks[0].old_text == "old\n"
        assert hunks[0].new_text == "new\n"

    def test_ignores_git_prologue(self):
        patch = (
            "diff --git a/code.py b/code.py\n"
            "index abc..def 100644\n"
            "--- a/code.py\n"
            "+++ b/code.py\n"
            "@@ -1,1 +1,1 @@\n-old\n+new\n"
        )
        hunks = parse_unified_diff(patch)
        assert len(hunks) == 1

    def test_multiple_hunks(self):
        patch = "@@ -1,1 +1,1 @@\n-a\n+x\n@@ -10,1 +10,1 @@\n-b\n+y\n"
        hunks = parse_unified_diff(patch)
        assert len(hunks) == 2
        assert hunks[0].old_text == "a\n"
        assert hunks[1].old_text == "b\n"

    def test_counts_omitted_when_one(self):
        # `diff` drops the ,count when it is 1
        patch = "@@ -7 +7 @@\n def f():\n-    return 1\n+    return 2\n"
        hunks = parse_unified_diff(patch)
        assert hunks[0].old_start == 7

    def test_section_heading_after_double_at(self):
        patch = "@@ -1,1 +1,1 @@ def add\n-a\n+b\n"
        hunks = parse_unified_diff(patch)
        assert len(hunks) == 1

    def test_no_newline_at_end_removed_side(self):
        # old line has no trailing newline; new line does.
        patch = "@@ -1,1 +1,1 @@\n-old\n\\ No newline at end of file\n+new\n"
        hunks = parse_unified_diff(patch)
        assert hunks[0].old_text == "old"
        assert hunks[0].new_text == "new\n"

    def test_no_newline_at_end_added_side(self):
        patch = "@@ -1,1 +1,1 @@\n-old\n+new\n\\ No newline at end of file\n"
        hunks = parse_unified_diff(patch)
        assert hunks[0].old_text == "old\n"
        assert hunks[0].new_text == "new"

    def test_no_newline_at_end_context_side(self):
        patch = "@@ -1,1 +1,1 @@\n same\n\\ No newline at end of file\n"
        hunks = parse_unified_diff(patch)
        assert hunks[0].old_text == "same"
        assert hunks[0].new_text == "same"

    def test_blank_context_line_whitespace_stripped_recovered(self):
        # A blank context line stripped of its leading space should still parse
        # as an empty context line, not an "unexpected marker" error.
        patch = "@@ -1,3 +1,3 @@\n line one\n\n line three\n"
        hunks = parse_unified_diff(patch)
        assert hunks[0].old_text == "line one\n\nline three\n"

    def test_empty_patch_raises(self):
        with pytest.raises(PatchError, match="empty"):
            parse_unified_diff("")

    def test_whitespace_only_patch_raises(self):
        with pytest.raises(PatchError, match="empty"):
            parse_unified_diff("   \n  ")

    def test_no_hunks_raises(self):
        with pytest.raises(PatchError, match="no valid hunks"):
            parse_unified_diff("--- a/x\n+++ b/x\n")

    def test_bad_header_raises(self):
        with pytest.raises(PatchError, match="hunk header"):
            parse_unified_diff("not a diff at all\n")

    def test_dashed_header_without_plusplus_raises(self):
        with pytest.raises(PatchError, match="without '\\+\\+\\+'"):
            parse_unified_diff("--- a/x\norphan\n")

    def test_unknown_marker_raises(self):
        with pytest.raises(PatchError, match="unexpected line marker"):
            parse_unified_diff("@@ -1,1 +1,1 @@\n~weird\n")

    def test_backslash_content_line_rejected(self):
        # A context/+/- line starting with '\' is NOT the no-newline marker --
        # silently dropping it would lose content (a regex like \bword). Raise.
        with pytest.raises(PatchError, match="unexpected line marker"):
            parse_unified_diff("@@ -1,2 +1,2 @@\n ctx\n\\backslash content\n")

    def test_crlf_patch_no_stray_cr(self):
        h = parse_unified_diff(
            "@@ -1,1 +1,1 @@\r\n-old\r\n\\ No newline at end of file\r\n+new\r\n\\ No newline at end of file\r\n"
        )
        assert h[0].old_text == "old"
        assert h[0].new_text == "new"

    def test_multi_file_patch_rejected(self):
        with pytest.raises(PatchError, match="single file"):
            parse_unified_diff(
                "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n-c\n+d\n"
            )

    def test_same_file_two_hunks_accepted(self):
        h = parse_unified_diff("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n@@ -3 +3 @@\n-c\n+d\n")
        assert len(h) == 2

    def test_bare_plussplus_tolerated_as_prologue(self):
        h = parse_unified_diff("+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n")
        assert h[0].old_text == "a\n"

    def test_orphan_no_newline_marker_raises(self):
        with pytest.raises(PatchError, match="no preceding content"):
            parse_unified_diff("@@ -1,1 +1,1 @@\n\\ No newline at end of file\n+new\n")

    def test_removed_line_starting_with_dashes_is_content(self):
        # A removed line whose content is '-- stray' encodes as '--- stray'; it
        # must NOT be mistaken for a file header (count-based reading keeps it).
        h = parse_unified_diff("@@ -1,3 +1,3 @@\n line1\n--- stray\n+++ stray\n line2\n")
        assert "-- stray" in h[0].old_text
        assert "++ stray" in h[0].new_text


class TestApplyHunks:
    def test_single_hunk_applies(self):
        content = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n"
        hunk = Hunk(
            old_start=1,
            new_start=1,
            old_text="def add(a, b):\n    return a - b\n",
            new_text="def add(a, b):\n    return a + b\n",
        )
        out = apply_hunks(content, [hunk])
        assert "return a + b" in out
        assert "return a - b" not in out
        assert "def sub(a, b):" in out  # rest of file untouched

    def test_line_drift_tolerated_via_content_match(self):
        # The @@ says line 1 but the real block is now at line 3. Content match
        # still finds it.
        content = "# header\n# more\n\ndef add(a, b):\n    return a - b\n"
        hunk = Hunk(
            old_start=1,
            new_start=1,
            old_text="def add(a, b):\n    return a - b\n",
            new_text="def add(a, b):\n    return a + b\n",
        )
        out = apply_hunks(content, [hunk])
        assert "return a + b" in out

    def test_multi_hunk_applies_in_order(self):
        content = "a = 1\nb = 2\nc = 3\n"
        hunks = [
            Hunk(1, 1, "a = 1\n", "a = 10\n"),
            Hunk(3, 3, "c = 3\n", "c = 30\n"),
        ]
        out = apply_hunks(content, hunks)
        assert out == "a = 10\nb = 2\nc = 30\n"

    def test_missing_context_raises(self):
        content = "a = 1\n"
        hunk = Hunk(1, 1, "not present\n", "x\n")
        with pytest.raises(PatchError, match="context not found"):
            apply_hunks(content, [hunk])

    def test_ambiguous_context_raises_with_count(self):
        content = "dup\na = 1\ndup\n"
        hunk = Hunk(1, 1, "dup\n", "unique\n")
        with pytest.raises(PatchError, match="matched 2 times"):
            apply_hunks(content, [hunk])

    def test_pure_insertion_no_context_raises(self):
        content = "a = 1\n"
        hunk = Hunk(0, 1, "", "inserted\n")
        with pytest.raises(PatchError, match="pure insertion"):
            apply_hunks(content, [hunk])

    def test_all_or_nothing_on_later_hunk_failure(self):
        # Hunk 1 applies, hunk 2 fails -> content is NOT partially mutated
        # (apply_hunks raises before returning; the caller never sees a half edit).
        content = "a = 1\nb = 2\n"
        hunks = [
            Hunk(1, 1, "a = 1\n", "a = 10\n"),
            Hunk(2, 2, "missing\n", "x\n"),
        ]
        with pytest.raises(PatchError, match="hunk #2"):
            apply_hunks(content, hunks)

    def test_noop_hunk_skipped(self):
        content = "a = 1\n"
        hunk = Hunk(1, 1, "a = 1\n", "a = 1\n")
        assert apply_hunks(content, [hunk]) == "a = 1\n"
