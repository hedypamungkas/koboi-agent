"""Tests for DiffView and PlanView widgets -- unit-level without Textual app."""

from __future__ import annotations


from koboi.tui.widgets.diff_view import is_diff_content, count_changes, _parse_diff_lines
from koboi.tui.widgets.plan_view import PlanStep


class TestDiffViewHelpers:
    def test_is_diff_content_unified(self):
        assert is_diff_content("--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,4 @@\n+new line") is True

    def test_is_diff_content_git_diff(self):
        assert is_diff_content("diff --git a/file.py b/file.py\nindex abc..def 100644") is True

    def test_is_diff_content_not_diff(self):
        assert is_diff_content("Hello, this is normal text") is False

    def test_is_diff_content_empty(self):
        assert is_diff_content("") is False

    def test_is_diff_content_short(self):
        assert is_diff_content("+a\n-b") is False

    def test_count_changes_additions(self):
        diff = "--- a/f\n+++ b/f\n@@ -1,2 +1,3 @@\n line\n+new\n line"
        additions, deletions = count_changes(diff)
        assert additions == 1
        assert deletions == 0

    def test_count_changes_deletions(self):
        diff = "--- a/f\n+++ b/f\n@@ -1,3 +1,2 @@\n line\n-old\n line"
        additions, deletions = count_changes(diff)
        assert additions == 0
        assert deletions == 1

    def test_count_changes_mixed(self):
        diff = "--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n line\n-old\n+new\n line"
        additions, deletions = count_changes(diff)
        assert additions == 1
        assert deletions == 1

    def test_count_changes_empty(self):
        additions, deletions = count_changes("")
        assert additions == 0
        assert deletions == 0

    def test_parse_diff_lines(self):
        diff = "--- a/f\n+++ b/f\n@@ -1,2 +1,3 @@\n line\n+new\n-old"
        lines = _parse_diff_lines(diff)
        assert len(lines) > 0
        assert all(isinstance(line, tuple) for line in lines)


class TestPlanStep:
    def test_plan_step_creation(self):
        step = PlanStep(index=0, description="Do something")
        assert step.index == 0
        assert step.description == "Do something"
        assert step.completed is False
        assert step.skipped is False

    def test_plan_step_completed(self):
        step = PlanStep(index=1, description="Done", completed=True)
        assert step.completed is True

    def test_plan_step_skipped(self):
        step = PlanStep(index=2, description="Skip me", skipped=True)
        assert step.skipped is True
