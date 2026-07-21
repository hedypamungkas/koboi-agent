"""Wave 2 item 3: exit-code failure signal + doom-loop progress fingerprint."""

from __future__ import annotations

import json

from koboi.harness.doom_loop import DoomLoopConfig, DoomLoopDetector
from koboi.harness.utils import is_tool_error, parse_exit_code
from koboi.hooks.failure_classifier_hook import FailureClassifierHook
from koboi.hooks.chain import HookContext, HookEvent
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry, tool, register_decorated
from koboi.types import ToolCall


class TestParseExitCode:
    def test_nonzero_exit_parsed(self):
        assert parse_exit_code("[exit code: 1]\nFAILED test_x") == 1
        assert parse_exit_code("[exit code: 127]\ncommand not found") == 127

    def test_no_prefix_is_none(self):
        assert parse_exit_code("all tests passed") is None
        assert parse_exit_code("") is None
        assert parse_exit_code(None) is None

    def test_prefix_must_anchor_at_start(self):
        assert parse_exit_code("output then [exit code: 1]") is None


class TestIsToolError:
    def test_failing_shell_output_is_error(self):
        assert is_tool_error("[exit code: 1]\n=== 3 failed, 2 passed ===") is True

    def test_error_prefixes_still_detected(self):
        assert is_tool_error("Error: file not found") is True
        assert is_tool_error("Rate limited: too many calls") is True
        assert is_tool_error("Denied by policy") is True
        assert is_tool_error("Blocked: sensitive path") is True

    def test_success_output_mentioning_errors_is_not_error(self):
        # The old \berror\b word-match false-positived on these.
        assert is_tool_error("ran with 0 errors") is False
        assert is_tool_error("test session finished, no error found") is False
        assert is_tool_error("=== 5 passed, 0 error ===") is False

    def test_empty_is_not_error(self):
        assert is_tool_error(None) is False
        assert is_tool_error("") is False


# Module-scope tools for the pipeline test (register_decorated scans a module).
@tool(
    name="fake_shell_fail",
    description="returns a failing-shell-style string",
    parameters={"type": "object", "properties": {}, "required": []},
)
def fake_shell_fail() -> str:
    return "[exit code: 2]\nFAILED tests/test_calc.py::test_add"


@tool(
    name="fake_shell_ok",
    description="returns a passing-shell-style string",
    parameters={"type": "object", "properties": {}, "required": []},
)
def fake_shell_ok() -> str:
    return "5 passed in 0.1s (0 errors)"


# Edge-case tools exercising the [exit code: N] prefix lift contract:
# the prefix token is authoritative regardless of which tool emitted it,
# but only when anchored at start, and never when N == 0.
@tool(
    name="fake_exit_zero",
    description="returns a zero-exit-code-prefixed string (success)",
    parameters={"type": "object", "properties": {}, "required": []},
)
def fake_exit_zero() -> str:
    return "[exit code: 0]\nall good"


@tool(
    name="fake_arbitrary_exit",
    description="a non-shell tool whose output coincidentally carries the exit-code prefix",
    parameters={"type": "object", "properties": {}, "required": []},
)
def fake_arbitrary_exit() -> str:
    return "[exit code: 7]\nimproved docs"


@tool(
    name="fake_embedded_exit",
    description="output mentions an exit code NOT anchored at the start",
    parameters={"type": "object", "properties": {}, "required": []},
)
def fake_embedded_exit() -> str:
    return "ran cmd [exit code: 3] ok"


@tool(
    name="fake_normal_result",
    description="returns a normal success string with no exit-code prefix",
    parameters={"type": "object", "properties": {}, "required": []},
)
def fake_normal_result() -> str:
    return "updated 3 files"


import sys  # noqa: E402

_this_module = sys.modules[__name__]


class TestPipelineCommandFailed:
    async def _run(self, tool_name: str):
        registry = ToolRegistry()
        register_decorated(registry, _this_module)
        pipeline = ToolExecutionPipeline(tools=registry, memory=ConversationMemory())
        tc = ToolCall(id="t1", name=tool_name, arguments=json.dumps({}))
        return await pipeline.execute_tool_call(tc, iteration=0)

    async def test_nonzero_exit_sets_command_failed(self):
        pr = await self._run("fake_shell_fail")
        assert pr.errored is True
        assert pr.error_kind == "command_failed"
        # Output preserved verbatim -- the command's output IS the diagnostic.
        assert pr.result.startswith("[exit code: 2]")
        assert "FAILED" in pr.result

    async def test_success_output_not_errored(self):
        pr = await self._run("fake_shell_ok")
        assert pr.errored is False
        assert pr.error_kind is None

    async def test_exit_code_zero_prefix_not_lifted(self):
        # A leading "[exit code: 0]" is the success token -- not a failure.
        pr = await self._run("fake_exit_zero")
        assert pr.errored is False
        assert pr.error_kind is None

    async def test_arbitrary_tool_coincidental_exit_code_prefix_is_lifted(self):
        # Pins the design: the prefix token is authoritative regardless of WHICH
        # tool emitted it -- the lift is not gated on tool name == run_shell.
        pr = await self._run("fake_arbitrary_exit")
        assert pr.errored is True
        assert pr.error_kind == "command_failed"

    async def test_embedded_exit_code_not_lifted(self):
        # Prefix NOT anchored at start -> parse_exit_code returns None -> no lift.
        pr = await self._run("fake_embedded_exit")
        assert pr.errored is False
        assert pr.error_kind is None

    async def test_error_kind_initialized_for_normal_result(self):
        # Defends against any UnboundLocalError regressions: referencing
        # pr.error_kind on a normal result must just return None.
        pr = await self._run("fake_normal_result")
        assert pr.errored is False
        assert pr.error_kind is None


class TestFailureClassifierCommandFailed:
    async def test_command_failed_maps_to_transient(self):
        hook = FailureClassifierHook()
        ctx = HookContext(event=HookEvent.POST_TOOL_USE, tool_name="run_shell")
        ctx.metadata["tool_error_kind"] = "command_failed"
        result = await hook.execute(ctx)
        assert result.metadata["failure_class"] == "transient"


class TestDoomLoopProgressFingerprint:
    def _detector(self, **kw):
        return DoomLoopDetector(DoomLoopConfig(**kw))

    def test_identical_failing_calls_with_changing_output_do_not_trigger(self):
        d = self._detector(error_retry_threshold=3, consecutive_identical_threshold=3)
        for i in range(4):
            d.record("run_shell", '{"command": "pytest"}', is_error=True, result_fingerprint=f"fp-{i}")
        assert d.check().detected is False

    def test_identical_failing_calls_with_identical_output_trigger(self):
        d = self._detector(error_retry_threshold=3, consecutive_identical_threshold=3)
        for _ in range(3):
            d.record("run_shell", '{"command": "pytest"}', is_error=True, result_fingerprint="same")
        result = d.check()
        assert result.detected is True
        assert result.loop_type == "error_retry"

    def test_consecutive_identical_with_changing_output_does_not_trigger(self):
        d = self._detector(consecutive_identical_threshold=3)
        for i in range(3):
            d.record("read_file", '{"path": "x"}', is_error=False, result_fingerprint=f"v{i}")
        assert d.check().detected is False

    def test_consecutive_identical_with_same_output_triggers(self):
        d = self._detector(consecutive_identical_threshold=3)
        for _ in range(3):
            d.record("read_file", '{"path": "x"}', is_error=False, result_fingerprint="v")
        result = d.check()
        assert result.detected is True
        assert result.loop_type == "consecutive_identical"

    def test_no_fingerprint_preserves_legacy_behavior(self):
        d = self._detector(consecutive_identical_threshold=3)
        for _ in range(3):
            d.record("read_file", '{"path": "x"}')
        assert d.check().detected is True

    def test_reset_clears_fingerprints(self):
        d = self._detector(consecutive_identical_threshold=3)
        d.record("t", "{}", result_fingerprint="a")
        d.reset()
        assert len(d._fingerprints) == 0
