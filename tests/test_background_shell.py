"""Tests for koboi.harness.background_shell / koboi.tools.builtin.background_shell (Wave 4)."""

from __future__ import annotations

import asyncio

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


class TestMaxLifetime:
    async def test_auto_kill_on_timeout(self):
        manager = BackgroundShellManager(grace_seconds=0.2)
        job = await manager.start("sleep 30", max_lifetime_seconds=0.1)
        await _wait_until(lambda: manager._jobs[job.job_id].status == "timeout", timeout=3.0)
        final = await manager.poll(job.job_id)
        assert final.status == "timeout"


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

    async def test_check_unknown_job_id(self):
        manager = BackgroundShellManager()
        result = await check_background_shell("nope", _deps={"background_shell_manager": manager})
        assert result.startswith("Error: no background job")


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
