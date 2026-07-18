"""Tests for koboi.eval.workspace + EvalRunner workspace lifecycle (Wave 1)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.eval.runner import EvalRunner
from koboi.eval.workspace import WorkspaceSetupError, cleanup_workspace, prepare_workspace
from koboi.types import EvalCase, EvalScore, RunResult, TokenUsage


def _git(args: list[str], cwd: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def fixture_repo(tmp_path):
    """A local git repo with two commits (so base_commit is testable)."""
    repo = tmp_path / "source_repo"
    repo.mkdir()
    _git(["init"], str(repo))
    _git(["config", "user.email", "eval@test.local"], str(repo))
    _git(["config", "user.name", "Eval Test"], str(repo))
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    _git(["add", "."], str(repo))
    _git(["commit", "-m", "buggy add"], str(repo))
    first_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.strip()
    (repo / "README.md").write_text("# fixture\n")
    _git(["add", "."], str(repo))
    _git(["commit", "-m", "add readme"], str(repo))
    return repo, first_sha


def _make_harness_mock(response: str = "done"):
    harness = MagicMock()
    run_result = MagicMock(spec=RunResult)
    run_result.content = response
    run_result.token_usage = TokenUsage()
    run_result.tool_calls_made = []
    harness.run = AsyncMock(return_value=run_result)
    harness.close = AsyncMock()
    harness.hook_chain = MagicMock()
    harness.hook_chain.add = MagicMock()
    return harness


class _CaptureScorer:
    """Records the context/case it was scored with."""

    def __init__(self, value: float = 1.0):
        self.value = value
        self.seen_context: dict | None = None
        self.seen_case: EvalCase | None = None

    async def score(self, case, output, context):
        self.seen_context = dict(context)
        self.seen_case = case
        return EvalScore("capture", self.value, "captured")


class TestPrepareWorkspace:
    def test_no_repo_returns_none(self):
        assert prepare_workspace(EvalCase(name="plain", user_message="hi")) is None

    def test_local_git_repo_cloned_and_detached_at_base_commit(self, fixture_repo, tmp_path):
        repo, first_sha = fixture_repo
        case = EvalCase(name="c", user_message="m", repo=str(repo), base_commit=first_sha)
        ws = prepare_workspace(case, root=tmp_path)
        try:
            assert ws is not None and ws.is_dir()
            assert (ws / "calc.py").exists()
            assert not (ws / "README.md").exists()  # detached at first commit
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(ws), check=True, capture_output=True, text=True
            ).stdout.strip()
            assert head == first_sha
            # Source repo untouched: still clean, still on its own HEAD
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=str(repo), check=True, capture_output=True, text=True
            ).stdout
            assert status == ""
        finally:
            cleanup_workspace(ws)

    def test_non_git_dir_copied(self, tmp_path):
        src = tmp_path / "plain_dir"
        src.mkdir()
        (src / "app.py").write_text("x = 1\n")
        case = EvalCase(name="c", user_message="m", repo=str(src))
        ws = prepare_workspace(case, root=tmp_path)
        try:
            assert ws is not None and (ws / "app.py").read_text() == "x = 1\n"
        finally:
            cleanup_workspace(ws)

    def test_non_git_dir_with_base_commit_errors(self, tmp_path):
        src = tmp_path / "plain_dir"
        src.mkdir()
        case = EvalCase(name="c", user_message="m", repo=str(src), base_commit="abc123")
        with pytest.raises(WorkspaceSetupError, match="base_commit requires a git repo"):
            prepare_workspace(case, root=tmp_path)

    def test_missing_repo_path_errors(self, tmp_path):
        case = EvalCase(name="c", user_message="m", repo=str(tmp_path / "nope"))
        with pytest.raises(WorkspaceSetupError, match="does not exist"):
            prepare_workspace(case, root=tmp_path)

    def test_bogus_base_commit_errors_and_removes_partial_workspace(self, fixture_repo, tmp_path):
        repo, _ = fixture_repo
        ws_root = tmp_path / "ws_root"
        ws_root.mkdir()
        case = EvalCase(name="c", user_message="m", repo=str(repo), base_commit="deadbeef" * 5)
        with pytest.raises(WorkspaceSetupError, match="checkout of base_commit"):
            prepare_workspace(case, root=ws_root)
        assert list(ws_root.iterdir()) == []  # no leaked partial workspace

    def test_failing_setup_command_errors_and_removes_workspace(self, fixture_repo, tmp_path):
        repo, _ = fixture_repo
        ws_root = tmp_path / "ws_root"
        ws_root.mkdir()
        case = EvalCase(name="c", user_message="m", repo=str(repo), setup_commands=["exit 3"])
        with pytest.raises(WorkspaceSetupError, match="setup command failed"):
            prepare_workspace(case, root=ws_root)
        assert list(ws_root.iterdir()) == []

    def test_setup_commands_run_in_workspace(self, fixture_repo, tmp_path):
        repo, _ = fixture_repo
        case = EvalCase(name="c", user_message="m", repo=str(repo), setup_commands=["echo ok > setup_ran.txt"])
        ws = prepare_workspace(case, root=tmp_path)
        try:
            assert ws is not None and (ws / "setup_ran.txt").exists()
        finally:
            cleanup_workspace(ws)

    def test_cleanup_workspace_tolerates_none_and_missing(self, tmp_path):
        cleanup_workspace(None)
        cleanup_workspace(tmp_path / "never_existed")


class TestRunnerWorkspaceLifecycle:
    async def test_no_repo_case_uses_zero_arg_factory_and_no_workspace_context(self):
        harness = _make_harness_mock()
        calls: list = []

        def factory():
            calls.append("zero-arg")
            return harness

        scorer = _CaptureScorer()
        runner = EvalRunner(harness_factory=factory, scorers=[scorer])
        result = await runner.run_case(EvalCase(name="plain", user_message="hi"))
        assert result.passed is True
        assert calls == ["zero-arg"]
        assert "workspace" not in (scorer.seen_context or {})

    async def test_workspace_reaches_factory_context_and_metadata(self, fixture_repo, tmp_path):
        repo, _ = fixture_repo
        harness = _make_harness_mock()
        seen_ws: list = []

        def factory(workspace):
            seen_ws.append(workspace)
            return harness

        scorer = _CaptureScorer()
        runner = EvalRunner(harness_factory=factory, scorers=[scorer], workspace_root=str(tmp_path))
        case = EvalCase(name="ws-case", user_message="fix it", repo=str(repo))
        result = await runner.run_case(case)
        assert result.passed is True
        assert len(seen_ws) == 1 and Path(seen_ws[0]).name.startswith("koboi-eval-ws-case")
        assert scorer.seen_context["workspace"] == seen_ws[0]
        assert case.metadata["workspace"] == seen_ws[0]

    async def test_zero_arg_factory_with_repo_case_warns_but_runs(self, fixture_repo, tmp_path, caplog):
        repo, _ = fixture_repo
        harness = _make_harness_mock()
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[_CaptureScorer()], workspace_root=str(tmp_path))
        with caplog.at_level("WARNING"):
            result = await runner.run_case(EvalCase(name="c", user_message="m", repo=str(repo)))
        assert result.passed is True
        assert any("takes no arguments" in r.message for r in caplog.records)

    async def test_setup_failure_returns_failed_result_not_crash(self, fixture_repo, tmp_path):
        repo, _ = fixture_repo
        factory = MagicMock()
        runner = EvalRunner(harness_factory=factory, scorers=[_CaptureScorer()], workspace_root=str(tmp_path))
        case = EvalCase(name="c", user_message="m", repo=str(repo), setup_commands=["exit 7"])
        result = await runner.run_case(case)
        assert result.passed is False
        assert result.overall_score == 0.0
        assert "workspace setup failed" in result.scores[0].reason
        factory.assert_not_called()  # harness never built on setup failure

    async def test_workspace_cleaned_up_after_success(self, fixture_repo, tmp_path):
        repo, _ = fixture_repo
        ws_root = tmp_path / "ws_root"
        ws_root.mkdir()

        def factory(workspace):
            return _make_harness_mock()

        runner = EvalRunner(harness_factory=factory, scorers=[_CaptureScorer()], workspace_root=str(ws_root))
        await runner.run_case(EvalCase(name="c", user_message="m", repo=str(repo)))
        assert list(ws_root.iterdir()) == []

    async def test_workspace_cleaned_up_when_harness_raises(self, fixture_repo, tmp_path):
        repo, _ = fixture_repo
        ws_root = tmp_path / "ws_root"
        ws_root.mkdir()
        harness = _make_harness_mock()
        harness.run = AsyncMock(side_effect=RuntimeError("agent exploded"))

        def factory(workspace):
            return harness

        runner = EvalRunner(harness_factory=factory, scorers=[_CaptureScorer()], workspace_root=str(ws_root))
        with pytest.raises(RuntimeError, match="agent exploded"):
            await runner.run_case(EvalCase(name="c", user_message="m", repo=str(repo)))
        assert list(ws_root.iterdir()) == []

    async def test_keep_failed_workspaces_retains_dir_and_surfaces_path(self, fixture_repo, tmp_path):
        repo, _ = fixture_repo
        ws_root = tmp_path / "ws_root"
        ws_root.mkdir()

        def factory(workspace):
            return _make_harness_mock()

        runner = EvalRunner(
            harness_factory=factory,
            scorers=[_CaptureScorer(value=0.0)],  # forces failure
            workspace_root=str(ws_root),
            keep_failed_workspaces=True,
        )
        result = await runner.run_case(EvalCase(name="c", user_message="m", repo=str(repo)))
        assert result.passed is False
        kept = list(ws_root.iterdir())
        assert len(kept) == 1
        assert result.metadata["workspace"] == str(kept[0])
