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


class TestGlobFindSandboxEscape:
    """Regression: glob_find's ``pattern`` arg is LLM-controlled and must not be
    able to escape the sandbox workdir -- neither via an absolute pattern
    (os.path.join discards the validated base) nor via a ".." component (resolves
    up and out), nor via a symlink inside the workdir pointing outside.
    """

    def test_absolute_pattern_rejected(self, tmp_path):
        workdir = tmp_path / "session_A"
        workdir.mkdir()
        (workdir / "in_workdir.txt").write_text("x")
        sb = RestrictedProcessBackend(workdir=str(workdir))

        out = glob_find(pattern="/etc/*", _deps={"sandbox": sb})

        assert out.startswith("Error:")
        assert "in_workdir" not in out

    def test_dotdot_pattern_rejected_against_sibling(self, tmp_path):
        workdir = tmp_path / "session_A"
        workdir.mkdir()
        sibling = tmp_path / "session_B"  # co-tenant workdir, OUTSIDE ours
        sibling.mkdir()
        (sibling / "tenant_secret.txt").write_text("x")
        (workdir / "mine.txt").write_text("x")
        sb = RestrictedProcessBackend(workdir=str(workdir))

        out = glob_find(pattern="../session_B/*", _deps={"sandbox": sb})

        assert out.startswith("Error:")
        assert "tenant_secret" not in out

    def test_symlink_escape_filtered(self, tmp_path):
        workdir = tmp_path / "session_A"
        workdir.mkdir()
        outside = tmp_path / "outside_target"  # sibling of workdir -> outside it
        outside.mkdir()
        (outside / "leaked_via_symlink.txt").write_text("x")
        (workdir / "keep.txt").write_text("x")
        (workdir / "lnk").symlink_to(outside)  # symlink inside workdir -> outside
        sb = RestrictedProcessBackend(workdir=str(workdir))

        out = glob_find(pattern="**/*", _deps={"sandbox": sb})

        assert "keep.txt" in out  # benign still found
        assert "leaked_via_symlink" not in out  # symlink target NOT disclosed

    def test_benign_relative_pattern_still_works(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path))
        (tmp_path / "keep.txt").write_text("x")

        out = glob_find(pattern="**/*.txt", _deps={"sandbox": sb})

        assert "keep.txt" in out
