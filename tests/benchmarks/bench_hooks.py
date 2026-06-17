"""Hook chain and telemetry benchmarks."""
import pytest

from koboi.hooks.chain import HookChain, Hook, HookContext, HookEvent
from koboi.harness.telemetry import TelemetryCollector
from koboi.harness.doom_loop import DoomLoopDetector, DoomLoopConfig


class MockHook(Hook):
    """Simple mock hook for benchmarking."""

    def __init__(self, event=HookEvent.PRE_TOOL_USE):
        self.event = event

    def handles(self):
        return [self.event]

    async def execute(self, ctx):
        # Simulate minimal work
        ctx.metadata["hook_ran"] = True
        return ctx


def test_hook_chain_emit(benchmark, hook_chain_with_5_hooks):
    """Benchmark emitting event through 5 hooks."""
    import asyncio

    def run_emit():
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE)
        return asyncio.run(hook_chain_with_5_hooks.emit(ctx))

    result = benchmark(run_emit)
    assert result.metadata.get("hook_ran") is True


def test_hook_chain_single_hook(benchmark, mock_hook):
    """Benchmark hook chain with single hook."""
    import asyncio

    chain = HookChain()
    chain.add(mock_hook())

    def run_emit():
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE)
        return asyncio.run(chain.emit(ctx))

    result = benchmark(run_emit)
    assert result.metadata.get("hook_ran") is True


def test_hook_chain_10_hooks(benchmark):
    """Benchmark hook chain with 10 hooks."""
    import asyncio

    chain = HookChain()
    for i in range(10):
        chain.add(MockHook(HookEvent.PRE_TOOL_USE))

    def run_emit():
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE)
        return asyncio.run(chain.emit(ctx))

    result = benchmark(run_emit)
    assert result.metadata.get("hook_ran") is True


def test_telemetry_collection(benchmark):
    """Benchmark recording 100 telemetry iterations."""
    import time

    def record_iterations():
        tc = TelemetryCollector(session_id="bench")
        tc.session_start()
        for i in range(100):
            tc.iteration_start(tokens_current=1000 + i * 10)
            tc.iteration_end(
                iteration=i,
                tool_names=["test_tool"],
                tokens_after=1000 + (i + 1) * 10,
                was_productive=True,
            )
            if i % 10 == 0:
                tc.record_tool_call("test_tool")
                tc.record_tool_success()
        tc.session_end()
        return tc

    result = benchmark(record_iterations)
    assert result.snapshot.total_iterations == 100


def test_telemetry_health_score(benchmark):
    """Benchmark computing health score."""
    collector = TelemetryCollector(session_id="test")
    collector.session_start()
    for i in range(50):
        collector.iteration_start(tokens_current=1000)
        collector.iteration_end(
            iteration=i,
            tool_names=["tool"],
            tokens_after=1100,
            was_productive=(i % 2 == 0),
        )
        collector.record_tool_call("tool")
        collector.record_tool_success()
    collector.session_end()

    result = benchmark(collector.health_score)
    assert 0 <= result <= 100


def test_telemetry_report_generation(benchmark, telemetry_collector):
    """Benchmark generating telemetry report."""
    telemetry_collector.session_start()
    for i in range(20):
        telemetry_collector.iteration_start(tokens_current=1000)
        telemetry_collector.iteration_end(iteration=i, tokens_after=1100)
    telemetry_collector.session_end()

    result = benchmark(telemetry_collector.report)
    assert "session_id" in result


def test_doom_loop_check(benchmark):
    """Benchmark DoomLoopDetector check after 20 varied calls."""
    detector = DoomLoopDetector()

    # Record 20 varied tool calls (different tools to avoid pattern detection)
    for i in range(20):
        detector.record(
            tool_name=f"tool_{i % 5}",
            arguments=f'{{"value": "{i}"}}',
            is_error=False,
        )

    result = benchmark(detector.check)
    assert result.detected is False


def test_doom_loop_consecutive_detection(benchmark):
    """Benchmark detecting consecutive identical pattern."""
    detector = DoomLoopDetector(
        DoomLoopConfig(consecutive_identical_threshold=3)
    )

    # Record consecutive identical calls
    for _ in range(3):
        detector.record("test_tool", '{"value": "same"}', is_error=False)

    result = benchmark(detector.check)
    assert result.detected is True


def test_doom_loop_pattern_detection(benchmark):
    """Benchmark detecting repeating pattern."""
    detector = DoomLoopDetector(
        DoomLoopConfig(
            repeating_pattern_window=6,
            repeating_pattern_threshold=2,
        )
    )

    # Record repeating pattern: A, B, A, B, A, B
    pattern = [("tool_a", '{"arg": "1"}'), ("tool_b", '{"arg": "2"}')]
    for _ in range(3):
        for tool, args in pattern:
            detector.record(tool, args, is_error=False)

    result = benchmark(detector.check)
    # May detect depending on pattern length


def test_doom_loop_error_retry(benchmark):
    """Benchmark detecting error retry pattern."""
    detector = DoomLoopDetector(
        DoomLoopConfig(error_retry_threshold=3)
    )

    # Record same call with errors
    for _ in range(3):
        detector.record("failing_tool", '{"arg": "bad"}', is_error=True)

    result = benchmark(detector.check)
    assert result.detected is True


def test_hook_chain_list_hooks(benchmark, hook_chain_with_5_hooks):
    """Benchmark list_hooks method."""
    result = benchmark(hook_chain_with_5_hooks.list_hooks)
    assert len(result) == 5


def test_hook_chain_find_hook(benchmark, hook_chain_with_5_hooks, mock_hook):
    """Benchmark find_hook method."""
    # Add a specific hook we'll look for
    specific_hook = mock_hook()
    hook_chain_with_5_hooks.add(specific_hook)

    def find():
        return hook_chain_with_5_hooks.find_hook(lambda h: h == specific_hook)

    result = benchmark(find)
    assert result is not None
