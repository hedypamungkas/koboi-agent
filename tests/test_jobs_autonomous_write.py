"""tests/test_jobs_autonomous_write -- Bucket C: autonomous jobs can write files.

Regression for the e2e failure where job_multi_write_grep returned
``Error: Tool execution denied by user``: AutonomousApprovalHandler denied every
DESTRUCTIVE write_file/delete_file because the (shared, dormant) Trust DB had no
allow-rule. Fix: a job-scoped ``auto_approve_tools`` allowlist on the handler --
containment is still enforced by the restricted sandbox, which rejects
out-of-workdir paths at execution time.
"""

from __future__ import annotations

from koboi.guardrails.approval import AutonomousApprovalHandler
from koboi.types import RiskLevel


class TestAutoApproveAllowlist:
    """Unit: the allowlist lifts the approval gate for named tools only."""

    def test_allowlisted_write_file_approved(self):
        h = AutonomousApprovalHandler(auto_approve_tools={"write_file"})
        assert h.should_approve("write_file", '{"path":"log.txt"}', RiskLevel.DESTRUCTIVE) is True

    def test_non_allowlisted_destructive_still_denied(self):
        h = AutonomousApprovalHandler(auto_approve_tools={"write_file"})
        # delete_file is destructive and NOT in the list -> denied.
        assert h.should_approve("delete_file", '{"path":"x"}', RiskLevel.DESTRUCTIVE) is False

    def test_no_allowlist_denies_destructive(self):
        # Back-compat: a bare handler (M4 behavior) still denies destructive.
        h = AutonomousApprovalHandler()
        assert h.should_approve("write_file", '{"path":"x"}', RiskLevel.DESTRUCTIVE) is False

    def test_safe_tools_still_auto_approved(self):
        h = AutonomousApprovalHandler()
        assert h.should_approve("calculate", "{}", RiskLevel.SAFE) is True


class TestJobWritesFileUnderRestrictedSandbox:
    """Integration: an autonomous-style run writes a file into the workdir.

    Mirrors tests/test_server_jobs.py::TestJobGuardrails but asserts the write
    SUCCEEDS (file present, no 'denied') when auto_approve_tools includes
    write_file under a restricted sandbox.
    """

    async def test_write_file_succeeds_in_autonomous_run(self, tmp_path):
        from koboi.config import Config
        from koboi.events import ErrorEvent
        from koboi.facade import KoboiAgent
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        config = Config.from_dict(
            {
                "agent": {"name": "t", "system_prompt": "h", "max_iterations": 3, "mode": "act"},
                "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
                "memory": {"backend": "in_memory"},
                "sandbox": {"backend": "restricted", "workdir": str(tmp_path)},
                "tools": {"builtin": ["write_file"]},
            },
            validate=True,
        )
        agent = KoboiAgent.from_dict(config.raw)
        agent._core.client = MockClient(
            [
                make_mock_response(
                    tool_calls=[make_mock_tool_call("write_file", {"path": "log.txt", "content": "WARN high latency"})]
                ),
                make_mock_response(content="done"),
            ]
        )
        # Reproduce _execute_job's handler wiring exactly.
        if hasattr(agent._core, "_tool_pipeline"):
            del agent._core._tool_pipeline
        agent._core.approval_handler = AutonomousApprovalHandler(
            trust_db=agent.trust_db,
            audit_trail=agent._core.audit_trail,
            auto_approve_tools={"write_file", "delete_file"},
        )

        events: list = []
        try:
            async for ev in agent.run_stream("write log.txt"):
                events.append(ev)
        except Exception as exc:  # noqa: BLE001 - surface as an event for assertions
            events.append(ErrorEvent(error=exc))

        # The file was actually written into the sandbox workdir...
        assert (tmp_path / "log.txt").read_text() == "WARN high latency"
        # ...and the tool result is NOT a denial.
        tool_results = [e for e in events if type(e).__name__ == "ToolResultEvent"]
        assert tool_results, "expected a ToolResultEvent"
        assert all("denied" not in getattr(e, "result", "").lower() for e in tool_results)
