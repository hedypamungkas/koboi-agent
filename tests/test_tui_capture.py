"""Tests for the TUI /capture slash command (v2 step 11)."""

import asyncio

from koboi.config import Config
from koboi.tui.commands import CommandContext, _cmd_capture, build_registry
from koboi.workflows.store import FileWorkflowStore


class _StubAgent:
    def __init__(self, config):
        self.config = config
        self.core = None


class TestCaptureSlashCommand:
    def test_registered(self):
        reg = build_registry()
        names = reg.get_all_names()
        assert any("capture" in n for n in names)

    def test_capture_bundles_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KOBOI_WORKFLOWS_DIR", str(tmp_path / "wfs"))
        cfg = Config.from_dict(
            {
                "agent": {"name": "myagent", "system_prompt": "h"},
                "llm": {"provider": "openai", "model": "m", "api_key": "test"},
            }
        )
        out: list[str] = []
        ctx = CommandContext(agent=_StubAgent(cfg), output=out.append, args="cap1")
        asyncio.run(_cmd_capture(ctx))
        assert FileWorkflowStore().exists("cap1")
        assert any("Captured workflow" in o for o in out)
