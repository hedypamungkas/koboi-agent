"""tests/test_search_sandbox_root -- Bucket A2: glob_find/grep_search honor the sandbox.

Regression for the e2e inconsistency where ``glob_find``/``grep_search`` walked
the process CWD (repo root) and 'found' KB files that ``read_file`` then
couldn't open (it anchors to the per-session workdir). After the fix both
search tools resolve their root through ``sandbox.validate_path``, sharing one
root with the other fs-tools, while preserving CWD behavior for direct callers.
"""

from __future__ import annotations

from koboi.sandbox.restricted import RestrictedProcessBackend
from koboi.tools.builtin.search import glob_find, grep_search


class TestSearchSandboxRoot:
    def test_glob_find_anchored_to_workdir(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path))
        (tmp_path / "in_workdir.md").write_text("hello")

        out = glob_find("*.md", _deps={"sandbox": sb})

        assert "in_workdir.md" in out

    def test_glob_find_no_sandbox_uses_cwd(self, tmp_path, monkeypatch):
        """Back-compat: direct callers with no sandbox keep CWD behavior."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cwd_only.md").write_text("y")

        out = glob_find("cwd_only.md")

        assert "cwd_only.md" in out

    def test_glob_find_outside_workdir_rejected(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path))

        # An absolute path outside the workdir is rejected by the sandbox.
        out = glob_find("*.md", path="/etc", _deps={"sandbox": sb})

        assert out.startswith("Error:")

    def test_grep_search_anchored_to_workdir(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path))
        (tmp_path / "note.txt").write_text("foo bar baz\n")

        out = grep_search("foo", ".", _deps={"sandbox": sb})

        assert "note.txt" in out

    def test_grep_search_no_match_within_workdir_only(self, tmp_path, monkeypatch):
        """A match that exists only in CWD (not the workdir) is NOT found when a
        sandbox is wired -- proving the root is the workdir, not CWD."""
        other = tmp_path / "other_dir"
        other.mkdir()
        (other / "decoy.txt").write_text("needle here\n")
        monkeypatch.chdir(other)

        # Sandbox wired to a DIFFERENT, empty workdir -> decoy must not be found.
        empty_wd = tmp_path / "wd"
        empty_wd.mkdir()
        sb = RestrictedProcessBackend(workdir=str(empty_wd))

        out = grep_search("needle", ".", _deps={"sandbox": sb})

        assert "No match" in out

    def test_grep_search_no_sandbox_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cwd.txt").write_text("alpha\n")

        out = grep_search("alpha", ".")

        assert "cwd.txt" in out
