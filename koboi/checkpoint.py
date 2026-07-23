"""koboi/checkpoint -- shadow-repo workdir checkpoints for crash resume (Wave 2).

``WorkdirCheckpointer`` commits the sandbox workdir into a SHADOW git repo
(``git --git-dir=<workdir>/.koboi-checkpoint/git --work-tree=<workdir>``) after
each mutating tool call, so:

- crash resume can roll the tree back to the last durable step (the diff
  between the tree and shadow HEAD is exactly the interrupted call's partial
  effects -- every completed mutating call committed), and
- unattended runs get a reviewable per-step diff trail
  (``git --git-dir=<workdir>/.koboi-checkpoint/git log --stat``).

The shadow never COMMITS to the user's own repo: commits go to a separate object
store (``<workdir>/.koboi-checkpoint/git``), the workdir's ``/.git/`` is excluded
from the shadow index, and ``/.koboi-checkpoint/`` is idempotently appended to
the user's ``.git/info/exclude`` (when the workdir IS a git repo) so the shadow
directory stays out of ``git status``. That ``.git/info/exclude`` append is the
only mutation of the user's repo metadata -- it is additive and reversible.

NEVER re-baselines: a baseline sidecar (``<workdir>/.koboi-checkpoint/baseline``)
records that a baseline succeeded; if the shadow HEAD later goes missing (disk
corruption, manual edit), the checkpointer DISABLES itself for the run rather
than freeze the current (possibly crash-partial) tree as the restore target.

Scope boundaries (documented, accepted): the shadow honors in-tree ``.gitignore``
files (ignored dirs like ``node_modules`` are neither committed nor cleaned -- no
rollback fidelity there), nested repos are recorded as gitlinks (contents
never rolled back), and out-of-workdir side effects are not rolled back. The
shadow dir is shared across sessions that hardcode the SAME ``sandbox.workdir``
-- use a per-session workdir (the server default) when checkpointing is on.

Fail-safe philosophy (mirrors ``pool._git_init_workdir``): every method
catches subprocess/OS errors, logs, and returns ``None``/``False`` -- a
checkpoint failure must never break the agent loop. Git runs via plain
``subprocess`` (harness durability must not be gated by the sandbox's
network/rlimit policy).
"""

from __future__ import annotations

import logging
import os
import subprocess

_logger = logging.getLogger(__name__)

CHECKPOINT_DIR = ".koboi-checkpoint"

_EXCLUDES = "/.koboi-checkpoint/\n/.git/\n"

# Workdirs with a live WorkdirCheckpointer in THIS process. Two checkpointers on
# the same workdir share one linear shadow history (last-writer-wins HEAD) -- the
# server uses per-session workdirs, but an operator who hardcodes a shared
# sandbox.workdir gets a warning rather than silent history collapse.
_ACTIVE_WORKDIRS: set[str] = set()


class WorkdirCheckpointer:
    def __init__(self, workdir: str, *, git_timeout: float = 60.0):
        self._workdir = os.path.realpath(workdir)
        self._git_dir = os.path.join(self._workdir, CHECKPOINT_DIR, "git")
        self._git_timeout = git_timeout
        self._warned_unavailable = False
        if self._workdir in _ACTIVE_WORKDIRS:
            _logger.warning(
                "two WorkdirCheckpointpers anchored to the same workdir (%s) in one "
                "process -- the shadow history is shared (last-writer-wins HEAD). Use "
                "a per-session sandbox.workdir when journal.checkpoint is on.",
                self._workdir,
            )
        _ACTIVE_WORKDIRS.add(self._workdir)

    @property
    def workdir(self) -> str:
        return self._workdir

    def _run(self, *args: str, wrapper: bool = True) -> subprocess.CompletedProcess | None:
        """Run a git command against the shadow repo; None on any failure.

        ``wrapper=False`` drops the ``--git-dir/--work-tree`` pair (needed for
        ``git init``, which rejects a work-tree without an existing git dir).
        """
        env = {
            **os.environ,
            # Neutralize global/system config (hooksPath, gpgsign, fsmonitor)
            # and any inherited git context (this process may itself run
            # inside a git hook or worktree).
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
        for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            env.pop(var, None)
        cmd = ["git"]
        if wrapper:
            cmd += ["--git-dir", self._git_dir, "--work-tree", self._workdir]
        cmd += list(args)
        try:
            return subprocess.run(
                cmd,
                cwd=self._workdir,
                capture_output=True,
                text=True,
                timeout=self._git_timeout,
                env=env,
            )
        except FileNotFoundError:
            if not self._warned_unavailable:
                self._warned_unavailable = True
                _logger.warning("git not found on PATH -- workdir checkpoints disabled")
            return None
        except (subprocess.TimeoutExpired, OSError) as exc:
            _logger.warning("checkpoint git %s failed: %s", args[0] if args else "?", exc)
            return None

    def head(self) -> str | None:
        """Shadow HEAD sha, or None when no shadow/baseline exists."""
        if not os.path.isdir(self._git_dir):
            return None
        res = self._run("rev-parse", "HEAD")
        if res is None or res.returncode != 0:
            return None
        return res.stdout.strip() or None

    @property
    def _baseline_marker(self) -> str:
        # Lives OUTSIDE the shadow git dir (so git never manages it) and is
        # excluded from the shadow index (/.koboi-checkpoint/). Survives a crash.
        return os.path.join(self._workdir, CHECKPOINT_DIR, "baseline")

    def ensure(self) -> bool:
        """Init the shadow repo + baseline commit. Idempotent; NEVER re-baselines.

        Re-baselining on resume would freeze the crashed partial tree as the
        restore target -- an existing HEAD is always kept. If HEAD is missing
        AFTER a baseline succeeded (disk corruption / a manual edit of the shadow
        git dir), the checkpointer DISABLES itself for the run rather than freeze
        the current tree as the restore target (data-loss guard).
        """
        if self.head() is not None:
            return True
        if os.path.exists(self._baseline_marker):
            _logger.error(
                "checkpoint shadow HEAD missing after baseline (%s) -- refusing to "
                "re-baseline; checkpoints disabled for this run",
                self._git_dir,
            )
            return False
        try:
            os.makedirs(self._git_dir, exist_ok=True)
        except OSError as exc:
            _logger.warning("checkpoint disabled -- cannot create %s: %s", self._git_dir, exc)
            return False
        init = self._run("init", "-q", "--bare", self._git_dir, wrapper=False)
        if init is None or init.returncode != 0:
            return False
        # bare init + --work-tree works for add/commit, but reset/clean on some
        # git versions refuse with core.bare=true; flip it (dotfiles pattern).
        for kv in (
            "core.bare false",
            "user.email agent@koboi.local",
            "user.name koboi-agent",
            "commit.gpgsign false",
            "gc.auto 0",
        ):
            key, val = kv.split(" ", 1)
            self._run("config", key, val)
        try:
            with open(os.path.join(self._git_dir, "info", "exclude"), "w") as f:
                f.write(_EXCLUDES)
        except OSError:
            pass  # excludes are hygiene; commit/restore still work
        self._exclude_from_real_repo()
        if self._run("-c", "advice.addEmbeddedRepo=false", "add", "-A") is None:
            return False
        commit = self._run("commit", "-q", "--allow-empty", "-m", "koboi-checkpoint: baseline")
        ok = commit is not None and commit.returncode == 0
        if not ok:
            _logger.warning(
                "checkpoint baseline commit failed: %s",
                (commit.stderr or commit.stdout)[-200:] if commit else "git unavailable",
            )
            return ok
        # Record that a baseline succeeded so a later-missing HEAD is treated as
        # corruption (disable) rather than a first-run (re-baseline).
        try:
            with open(self._baseline_marker, "w") as f:
                f.write(self.head() or "")
        except OSError as exc:
            # The marker IS the data-loss guard (see module docstring): if it can't
            # be written, a later corrupted HEAD would silently re-baseline and
            # freeze a crash-partial tree as the restore target. Surface it loudly
            # rather than swallow -- an operator must know the guard is compromised.
            _logger.warning(
                "checkpoint baseline marker could not be written (%s); "
                "the missing-HEAD data-loss guard is NOT enforced for this run",
                exc,
            )
        return ok

    def _exclude_from_real_repo(self) -> None:
        """Hide ``.koboi-checkpoint/`` from the workdir's OWN repo, if any."""
        info_dir = os.path.join(self._workdir, ".git", "info")
        if not os.path.isdir(os.path.dirname(info_dir)):
            return
        try:
            os.makedirs(info_dir, exist_ok=True)
            exclude_path = os.path.join(info_dir, "exclude")
            existing = ""
            if os.path.exists(exclude_path):
                with open(exclude_path) as f:
                    existing = f.read()
            if "/.koboi-checkpoint/" not in existing:
                with open(exclude_path, "a") as f:
                    f.write("\n/.koboi-checkpoint/\n")
        except OSError:
            pass

    def commit(self, label: str) -> str | None:
        """Commit the current tree state; returns the new HEAD sha or None.

        ``--allow-empty`` keeps one sha per mutating step even when the tree
        content is unchanged (delete-then-recreate-identical still audits).
        """
        if not self.ensure():
            return None
        if self._run("-c", "advice.addEmbeddedRepo=false", "add", "-A") is None:
            return None
        commit = self._run("commit", "-q", "--allow-empty", "-m", label)
        if commit is None or commit.returncode != 0:
            _logger.warning(
                "checkpoint commit failed: %s",
                (commit.stderr or commit.stdout)[-200:] if commit else "git unavailable",
            )
            return None
        return self.head()

    def restore_to_head(self) -> bool:
        """Reset the tree to shadow HEAD (tracked) + remove post-HEAD files.

        Deliberately does NOT ensure(): with no existing shadow/baseline there
        is nothing safe to restore to (e.g. checkpointing enabled only after
        the crash) -- returns False, tree untouched. ``clean -fd`` without
        ``-x`` spares gitignored files; baseline-tracked files always survive.
        """
        if self.head() is None:
            return False
        # A crash mid-commit() can leave a stale index.lock; the journal is
        # single-writer per session, so clearing it on the resume path is safe.
        lock = os.path.join(self._git_dir, "index.lock")
        try:
            if os.path.exists(lock):
                os.unlink(lock)
        except OSError:
            pass
        reset = self._run("reset", "--hard", "-q", "HEAD")
        if reset is None or reset.returncode != 0:
            _logger.warning(
                "checkpoint restore failed: %s",
                (reset.stderr or reset.stdout)[-200:] if reset else "git unavailable",
            )
            return False
        self._run("clean", "-fdq")
        return True
