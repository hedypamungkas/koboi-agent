"""Unit tests for koboi.checkpoint.WorkdirCheckpointer (Wave 2 shadow repo)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from koboi.checkpoint import CHECKPOINT_DIR, WorkdirCheckpointer

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@pytest.fixture
def workdir(tmp_path):
    ws = tmp_path / "workdir"
    ws.mkdir()
    (ws / "app.py").write_text("x = 1\n")
    return ws


def _shadow_git(ws, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "--git-dir", str(ws / CHECKPOINT_DIR / "git"), "--work-tree", str(ws), *args],
        capture_output=True,
        text=True,
    )


class TestEnsure:
    def test_creates_shadow_and_baseline(self, workdir):
        cp = WorkdirCheckpointer(str(workdir))
        assert cp.ensure() is True
        assert cp.head() is not None
        # Baseline tracked the existing file.
        ls = _shadow_git(workdir, "ls-files")
        assert "app.py" in ls.stdout

    def test_idempotent_never_rebaselines(self, workdir):
        cp = WorkdirCheckpointer(str(workdir))
        cp.ensure()
        head1 = cp.head()
        (workdir / "later.txt").write_text("x")
        assert cp.ensure() is True
        assert cp.head() == head1  # no new commit from the second ensure

    def test_empty_workdir_baseline(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        cp = WorkdirCheckpointer(str(ws))
        assert cp.ensure() is True
        assert cp.head() is not None  # --allow-empty covers the empty tree

    def test_corrupt_head_does_not_rebaseline(self, workdir):
        # P1 data-loss: if the shadow HEAD goes missing AFTER a baseline
        # succeeded, ensure() must DISABLE for the run rather than freeze the
        # current (possibly crash-partial) tree as the restore target.
        cp = WorkdirCheckpointer(str(workdir))
        assert cp.ensure() is True
        assert cp.head() is not None
        # A baseline sidecar was written -- corrupt the HEAD so rev-parse fails.
        head_file = workdir / CHECKPOINT_DIR / "git" / "HEAD"
        assert head_file.exists()
        head_file.unlink()
        assert cp.head() is None
        assert cp.ensure() is False  # refuses to re-baseline (data-loss guard)


class TestCommitRestore:
    def test_commit_returns_new_sha(self, workdir):
        cp = WorkdirCheckpointer(str(workdir))
        cp.ensure()
        base = cp.head()
        (workdir / "app.py").write_text("x = 2\n")
        sha = cp.commit("step 1")
        assert sha is not None and sha != base
        assert cp.head() == sha

    def test_no_change_commit_still_returns_sha(self, workdir):
        cp = WorkdirCheckpointer(str(workdir))
        cp.ensure()
        sha = cp.commit("no-op step")
        assert sha is not None  # --allow-empty: uniform sha per step

    def test_restore_reverts_modification_and_removes_new_files(self, workdir):
        cp = WorkdirCheckpointer(str(workdir))
        cp.ensure()
        (workdir / "app.py").write_text("x = 2\n")
        cp.commit("step 1")
        # Partial effects of an "interrupted" call:
        (workdir / "app.py").write_text("x = CORRUPT\n")
        (workdir / "junk.txt").write_text("partial")
        assert cp.restore_to_head() is True
        assert (workdir / "app.py").read_text() == "x = 2\n"
        assert not (workdir / "junk.txt").exists()

    def test_restore_spares_pre_existing_untracked_and_gitignored(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / ".gitignore").write_text("ignored_dir/\n")
        (ws / "ignored_dir").mkdir()
        (ws / "ignored_dir" / "cache.bin").write_text("cache")
        (ws / "untracked_by_user_repo.txt").write_text("keep me")
        cp = WorkdirCheckpointer(str(ws))
        cp.ensure()  # baseline tracks the non-ignored file
        (ws / "ignored_dir" / "cache.bin").write_text("mutated")
        assert cp.restore_to_head() is True
        # gitignored content untouched (out of checkpoint scope both directions)
        assert (ws / "ignored_dir" / "cache.bin").read_text() == "mutated"
        # baseline-tracked file survives clean
        assert (ws / "untracked_by_user_repo.txt").read_text() == "keep me"

    def test_restore_without_shadow_is_noop_false(self, workdir):
        cp = WorkdirCheckpointer(str(workdir))
        (workdir / "app.py").write_text("x = 99\n")
        assert cp.restore_to_head() is False
        assert (workdir / "app.py").read_text() == "x = 99\n"  # tree untouched

    def test_restore_clears_stale_index_lock(self, workdir):
        cp = WorkdirCheckpointer(str(workdir))
        cp.ensure()
        lock = workdir / CHECKPOINT_DIR / "git" / "index.lock"
        lock.write_text("")
        assert cp.restore_to_head() is True
        assert not lock.exists()


class TestRealRepoCoexistence:
    def test_real_repo_git_dir_excluded_and_untouched(self, workdir):
        subprocess.run(["git", "init", "-q"], cwd=str(workdir), check=True)
        subprocess.run(["git", "config", "user.email", "u@t"], cwd=str(workdir), check=True)
        subprocess.run(["git", "config", "user.name", "u"], cwd=str(workdir), check=True)
        subprocess.run(["git", "add", "-A"], cwd=str(workdir), check=True)
        subprocess.run(["git", "commit", "-q", "-m", "user commit"], cwd=str(workdir), check=True)

        cp = WorkdirCheckpointer(str(workdir))
        cp.ensure()
        # Shadow index has no /.git entry (excluded, not a gitlink).
        ls = _shadow_git(workdir, "ls-files")
        assert ".git" not in ls.stdout.split()
        # The user's repo log is intact and shows only their commit.
        log = subprocess.run(["git", "log", "--oneline"], cwd=str(workdir), capture_output=True, text=True)
        assert "user commit" in log.stdout
        assert "koboi-checkpoint" not in log.stdout
        # .koboi-checkpoint hidden from the user's git status.
        status = subprocess.run(["git", "status", "--porcelain"], cwd=str(workdir), capture_output=True, text=True)
        assert CHECKPOINT_DIR not in status.stdout
        # restore never rolls back the user's .git.
        assert cp.restore_to_head() is True
        log2 = subprocess.run(["git", "log", "--oneline"], cwd=str(workdir), capture_output=True, text=True)
        assert "user commit" in log2.stdout

    def test_nested_repo_becomes_gitlink_contents_untouched(self, workdir):
        nested = workdir / "vendor" / "lib"
        nested.mkdir(parents=True)
        (nested / "lib.py").write_text("v = 1\n")
        subprocess.run(["git", "init", "-q"], cwd=str(nested), check=True)
        cp = WorkdirCheckpointer(str(workdir))
        cp.ensure()
        (nested / "lib.py").write_text("v = 2\n")
        cp.restore_to_head()
        # Nested repo contents are outside checkpoint scope (gitlink) -- not rolled back.
        assert (nested / "lib.py").read_text() == "v = 2\n"


class TestFailSafe:
    def test_git_absent_soft_fails(self, workdir, monkeypatch):
        monkeypatch.setenv("PATH", str(workdir / "nonexistent-bin"))
        cp = WorkdirCheckpointer(str(workdir))
        assert cp.ensure() is False
        assert cp.commit("x") is None
        assert cp.restore_to_head() is False

    def test_unwritable_workdir_soft_fails(self, tmp_path):
        # Parent is a FILE -> makedirs raises -> soft-disable, never raise.
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        cp = WorkdirCheckpointer(str(blocker / "ws"))
        assert cp.ensure() is False
        assert cp.commit("x") is None
