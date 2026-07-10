"""Issue #10b: orchestration-mode resume raises a clear, actionable error."""

from __future__ import annotations

import pytest

from koboi.exceptions import AgentError
from koboi.facade import KoboiAgent

_CFG = {
    "agent": {"name": "t", "system_prompt": "h", "max_iterations": 3, "mode": "chat"},
    "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
}


class TestOrchestrationResumeError:
    async def test_resume_in_orchestration_raises_clear_error(self):
        agent = KoboiAgent.from_dict(_CFG)
        agent._orchestrator = object()  # simulate orchestration mode

        with pytest.raises(AgentError) as ei:
            await agent.resume()

        msg = str(ei.value)
        # issue #10b: explains why + points to the single-agent workaround.
        assert "orchestration mode" in msg
        assert "per-agent" in msg or "multiple" in msg
        assert "single-agent-config" in msg  # actionable pointer
