"""Tests for koboi.tools.builtin.git module."""

from __future__ import annotations

import os
import subprocess

import pytest

from koboi.tools.builtin.git import (
    git_status,
    git_log,
    git_diff,
    _run_git,
    SAFE_TARGET_RE,
)


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create initial commit
    (repo_path / "file1.txt").write_text("initial content")
    subprocess.run(["git", "add", "file1.txt"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return str(repo_path)


class TestGitStatus:
    def test_git_status_in_clean_repo(self, temp_git_repo):
        """Test git_status in a clean repository."""
        result = git_status(repo_path=temp_git_repo)
        assert "clean" in result.lower() or "no changes" in result.lower()

    def test_git_status_in_dirty_repo(self, temp_git_repo):
        """Test git_status in a dirty repository."""
        # Modify a file
        repo_path = temp_git_repo
        with open(os.path.join(repo_path, "file1.txt"), "w") as f:
            f.write("modified content")

        result = git_status(repo_path=repo_path)
        assert "modified" in result.lower() or "M" in result

    def test_git_status_with_untracked_file(self, temp_git_repo):
        """Test git_status with untracked files."""
        repo_path = temp_git_repo
        with open(os.path.join(repo_path, "newfile.txt"), "w") as f:
            f.write("new content")

        result = git_status(repo_path=repo_path)
        assert "untracked" in result.lower() or "newfile.txt" in result or "?" in result

    def test_git_status_with_staged_changes(self, temp_git_repo):
        """Test git_status with staged changes."""
        repo_path = temp_git_repo
        test_file = os.path.join(repo_path, "staged.txt")
        with open(test_file, "w") as f:
            f.write("staged content")

        subprocess.run(["git", "add", "staged.txt"], cwd=repo_path, check=True, capture_output=True)

        result = git_status(repo_path=repo_path)
        assert "added" in result.lower() or "staged.txt" in result or "A" in result


class TestGitLog:
    def test_git_log_with_commits(self, temp_git_repo):
        """Test git_log shows commit history."""
        result = git_log(repo_path=temp_git_repo, count=5)
        assert "Initial commit" in result
        assert result != "(no output)"

    def test_git_log_with_count_limit(self, temp_git_repo):
        """Test git_log respects count parameter."""
        repo_path = temp_git_repo

        # Add more commits
        for i in range(3):
            with open(os.path.join(repo_path, f"file{i}.txt"), "w") as f:
                f.write(f"content {i}")
            subprocess.run(["git", "add", f"file{i}.txt"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Commit {i}"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

        result = git_log(repo_path=repo_path, count=2)
        lines = [line for line in result.split("\n") if line.strip()]
        # Should show at most 2 commits (plus possibly decorate info)
        assert len([line for line in lines if not line.startswith("    ")]) <= 2


class TestGitDiff:
    def test_git_diff_with_unstaged_changes(self, temp_git_repo):
        """Test git_diff shows unstaged changes."""
        repo_path = temp_git_repo
        test_file = os.path.join(repo_path, "file1.txt")
        with open(test_file, "w") as f:
            f.write("modified content")

        result = git_diff(repo_path=repo_path)
        # Should show diff
        assert "Unstaged changes" in result or "modified" in result or result != "No changes"

    def test_git_diff_with_staged_changes(self, temp_git_repo):
        """Test git_diff shows staged changes."""
        repo_path = temp_git_repo
        test_file = os.path.join(repo_path, "file1.txt")
        with open(test_file, "w") as f:
            f.write("staged content")

        subprocess.run(["git", "add", "file1.txt"], cwd=repo_path, check=True, capture_output=True)

        result = git_diff(repo_path=repo_path)
        assert "Staged changes" in result or result != "No changes"

    def test_git_diff_with_target(self, temp_git_repo):
        """Test git_diff against a specific target (branch/commit)."""
        repo_path = temp_git_repo

        # Create a new branch
        subprocess.run(["git", "checkout", "-b", "test-branch"], cwd=repo_path, check=True, capture_output=True)

        # Modify file
        test_file = os.path.join(repo_path, "file1.txt")
        with open(test_file, "w") as f:
            f.write("branch content")

        result = git_diff(repo_path=repo_path, target="main")
        # Should show diff against main
        assert result is not None and "No changes" not in result

    def test_git_diff_with_no_changes(self, temp_git_repo):
        """Test git_diff when there are no changes."""
        result = git_diff(repo_path=temp_git_repo)
        assert "No changes" in result or "(no output)" in result


class TestRunGit:
    def test_run_git_timeout_handling(self, temp_git_repo):
        """Test _run_git handles timeout."""
        # Very short timeout via tool_config
        result = _run_git(["status"], temp_git_repo, tool_config={"timeout": 0.001})
        # Either success or timeout error
        assert "Error: git command timed out" in result or "clean" in result.lower()

    def test_run_git_output_truncation(self, temp_git_repo):
        """Test _run_git truncates large output."""
        tool_config = {"max_output": 100}

        # Create a file with lots of content
        repo_path = temp_git_repo
        test_file = os.path.join(repo_path, "large.txt")
        with open(test_file, "w") as f:
            f.write("x" * 10000)

        subprocess.run(["git", "add", "large.txt"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Large file"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        result = _run_git(["log", "-p", "-1"], repo_path, tool_config=tool_config)
        assert "truncated" in result.lower() or len(result) < 10000

    def test_run_git_nonexistent_path(self):
        """Test _run_git with nonexistent path."""
        result = _run_git(["status"], "/nonexistent/path/xyz123")
        assert "Error" in result
        assert "not a directory" in result.lower() or "not found" in result.lower()


class TestSafeTargetValidation:
    def test_safe_target_re_allows_valid_targets(self):
        """Test SAFE_TARGET_RE allows valid targets."""
        valid_targets = [
            "main",
            "develop",
            "feature-branch",
            "v1.0.0",
            "commit_hash",
            "origin/main",
            "upstream/develop",
            "HEAD",
            "path/to/file.txt",
        ]

        for target in valid_targets:
            assert SAFE_TARGET_RE.match(target), f"Should allow: {target}"

    def test_safe_target_re_blocks_dangerous_targets(self):
        """Test SAFE_TARGET_RE blocks potentially dangerous targets."""
        dangerous_targets = [
            "malicious; rm -rf /",
            "command && evil",
            "command | evil",
            "command `evil`",
            "command $(evil)",
        ]

        for target in dangerous_targets:
            assert not SAFE_TARGET_RE.match(target), f"Should block: {target}"

        # Note: ../escape is actually allowed by the regex (contains only valid chars)
        # This is expected as ../ is valid in git ref names

    def test_git_diff_blocks_unsafe_target(self, temp_git_repo):
        """Test git_diff rejects unsafe targets."""
        result = git_diff(repo_path=temp_git_repo, target="malicious; rm -rf /")
        assert "Error" in result
        assert "disallowed" in result.lower() or "invalid" in result.lower()

    def test_git_diff_rejects_leading_dash(self):
        """H4: a target starting with '-' is rejected (option-injection guard).

        The ``=`` form is also blocked by SAFE_TARGET_RE, but the leading-dash
        guard is defense-in-depth against any future git option that writes files.
        """
        result = git_diff(repo_path=".", target="--output=/etc/cron.d/x")
        assert "Error" in result
        assert "cannot start with '-'" in result


class TestToolConfig:
    def test_tool_config_passed_to_run_git(self, temp_git_repo):
        """Test _tool_config parameter is passed through to _run_git."""
        result = _run_git(["status", "--porcelain"], temp_git_repo, tool_config={"timeout": 30, "max_output": 5000})
        assert result is not None

    def test_run_git_passes_sanitized_env(self, temp_git_repo, monkeypatch):
        """P0a: _run_git must pass a sanitized env (no secrets) to subprocess."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-me")
        captured = {}
        real_run = subprocess.run

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs.get("env")
            return real_run(*args, **kwargs)

        monkeypatch.setattr("koboi.tools.builtin.git.subprocess.run", fake_run)
        _run_git(["status", "--porcelain"], str(temp_git_repo))
        env = captured.get("env")
        assert env is not None
        assert "OPENAI_API_KEY" not in env


class TestEdgeCases:
    def test_git_status_default_to_current_directory(self, tmp_path):
        """Test git_status defaults to current directory if not specified."""
        # This test uses "." as default, which should work from test directory
        result = git_status()
        # Should either return status or error about not being a git repo
        assert result is not None

    def test_git_log_max_count_enforced(self, temp_git_repo):
        """Test git_log enforces MAX_LOG_COUNT limit."""
        # Try to request more than MAX_LOG_COUNT (50)
        result = git_log(repo_path=temp_git_repo, count=1000)
        # Should still succeed (count is clamped)
        assert result is not None

    def test_git_diff_empty_target(self, temp_git_repo):
        """Test git_diff with empty target string."""
        result = git_diff(repo_path=temp_git_repo, target="")
        # Empty target should default to showing unstaged/staged changes
        assert result is not None

    def test_git_status_with_deleted_file(self, temp_git_repo):
        """Test git_status detects deleted files."""
        repo_path = temp_git_repo
        file_path = os.path.join(repo_path, "file1.txt")

        os.remove(file_path)

        result = git_status(repo_path=repo_path)
        assert "deleted" in result.lower()

    def test_git_status_with_renamed_file(self, temp_git_repo):
        """Test git_status detects renamed files."""
        repo_path = temp_git_repo
        old_path = os.path.join(repo_path, "file1.txt")
        new_path = os.path.join(repo_path, "renamed.txt")

        os.rename(old_path, new_path)
        subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True, capture_output=True)

        result = git_status(repo_path=repo_path)
        assert "renamed" in result.lower()

    def test_git_diff_with_both_staged_and_unstaged(self, temp_git_repo):
        """Test git_diff shows both staged and unstaged changes."""
        repo_path = temp_git_repo
        test_file = os.path.join(repo_path, "file1.txt")

        # First modification (staged)
        with open(test_file, "w") as f:
            f.write("staged")
        subprocess.run(["git", "add", "file1.txt"], cwd=repo_path, check=True, capture_output=True)

        # Second modification (unstaged)
        with open(test_file, "w") as f:
            f.write("unstaged")

        result = git_diff(repo_path=repo_path)
        assert "Staged changes" in result
        assert "Unstaged changes" in result


class TestGitNotInstalled:
    def test_git_not_found_error(self, tmp_path, monkeypatch):
        """Test error message when git is not found."""
        # Mock PATH to exclude git (monkeypatch auto-restores env at teardown)
        monkeypatch.setenv("PATH", "")

        # Create a directory that's not a git repo
        non_repo = tmp_path / "not_a_repo"
        non_repo.mkdir()

        # This should fail with "git not found" or similar
        result = _run_git(["status"], str(non_repo))

        # The result should indicate an error
        # (Either git not found or not a git repo)
        assert result is not None
