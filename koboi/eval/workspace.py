"""koboi/eval/workspace -- per-case repo workspace materialization for coding evals.

``prepare_workspace`` turns an ``EvalCase`` with a ``repo`` into an isolated
working copy the agent can mutate freely: clone/copy into a temp dir, detach at
``base_commit``, run ``setup_commands`` inside a restricted sandbox. The runner
anchors the agent's sandbox workdir at the returned path and cleans it up after
scoring.

Git plumbing (clone/checkout) runs via plain ``subprocess`` -- the repo source
is operator-supplied eval config, not agent output, and a network-deny sandbox
would block URL clones. Only ``setup_commands`` (which may execute arbitrary
project code, e.g. ``pip install -e .``) are routed through the sandbox.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from koboi.sandbox.registry import build_sandbox
from koboi.types import EvalCase

_logger = logging.getLogger(__name__)

_GIT_TIMEOUT = 300.0
_TAIL_CHARS = 500


class WorkspaceSetupError(RuntimeError):
    """Raised when a case workspace cannot be materialized."""


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name)[:40] or "case"


def _tail(*chunks: str) -> str:
    return "\n".join(c for c in chunks if c).strip()[-_TAIL_CHARS:]


def _git(args: list[str], *, cwd: str | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except FileNotFoundError as e:
        raise WorkspaceSetupError("git binary not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise WorkspaceSetupError(f"git {args[0]} timed out after {_GIT_TIMEOUT}s") from e


def _looks_like_url(repo: str) -> bool:
    return "://" in repo or repo.startswith("git@")


def _materialize(case: EvalCase, ws: Path) -> None:
    """Fill ``ws`` with the case's repo content (clone or copy)."""
    repo = str(case.repo)
    src = Path(repo)
    if src.is_dir():
        if (src / ".git").exists():
            # --no-hardlinks: the agent mutating the workspace .git must never
            # be able to corrupt the source fixture's object store.
            res = _git(["clone", "--no-hardlinks", str(src), str(ws)])
            if res.returncode != 0:
                raise WorkspaceSetupError(f"git clone failed: {_tail(res.stdout, res.stderr)}")
        else:
            if case.base_commit:
                raise WorkspaceSetupError(f"base_commit requires a git repo, but '{repo}' has no .git")
            shutil.copytree(src, ws, dirs_exist_ok=True)
    elif _looks_like_url(repo):
        res = _git(["clone", repo, str(ws)])
        if res.returncode != 0:
            raise WorkspaceSetupError(f"git clone failed: {_tail(res.stdout, res.stderr)}")
    else:
        raise WorkspaceSetupError(f"repo path '{repo}' does not exist and is not a git URL")


def prepare_workspace(
    case: EvalCase,
    *,
    root: str | Path | None = None,
    setup_timeout: float = 600.0,
    network: str = "deny",
) -> Path | None:
    """Materialize an isolated workspace for ``case``; None when ``case.repo`` is unset.

    Lifecycle: mkdtemp -> clone/copy -> ``git checkout --detach base_commit`` ->
    run ``setup_commands`` in a restricted sandbox anchored at the workspace.
    Any failure removes the partial workspace and raises ``WorkspaceSetupError``.
    """
    if not case.repo:
        return None
    ws = Path(tempfile.mkdtemp(prefix=f"koboi-eval-{_slug(case.name)}-", dir=root))
    try:
        _materialize(case, ws)
        if case.base_commit:
            res = _git(["-C", str(ws), "checkout", "--detach", case.base_commit])
            if res.returncode != 0:
                raise WorkspaceSetupError(
                    f"checkout of base_commit '{case.base_commit}' failed: {_tail(res.stdout, res.stderr)}"
                )
        if case.setup_commands:
            sandbox = build_sandbox({"backend": "restricted", "workdir": str(ws), "network": network})
            for cmd in case.setup_commands:
                res = sandbox.run(cmd, shell=True, cwd=str(ws), timeout=setup_timeout)
                if getattr(res, "timed_out", False):
                    raise WorkspaceSetupError(f"setup command timed out after {setup_timeout}s: {cmd!r}")
                if res.returncode != 0:
                    raise WorkspaceSetupError(
                        f"setup command failed (exit={res.returncode}): {cmd!r}\n{_tail(res.stdout, res.stderr)}"
                    )
    except BaseException:
        shutil.rmtree(ws, ignore_errors=True)
        raise
    return ws


def cleanup_workspace(path: Path | str | None) -> None:
    """Remove a workspace directory (best-effort; missing/partial dirs are fine)."""
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)
