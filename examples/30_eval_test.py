"""Example 30: Eval `t` authoring surface -- test-shaped, CI-native evals.

Demonstrates the eve-style `t` API: write ``async def test_*(t)`` functions that
drive an agent and record assertions (``t.calledTool``, ``t.check``, ``t.judge``),
then run them with gate/soft severity. Uses scripted (mock) LLM responses, so it
runs with no API key.

Run:
    python examples/30_eval_test.py
"""

from __future__ import annotations

import asyncio
import tempfile
import textwrap
from pathlib import Path

from conftest import console, setup_example


# A complete `.eval.py` written to a temp dir and discovered at runtime.
SAMPLE_EVAL = textwrap.dedent(
    """
    from koboi.eval.t import scripted_response, scripted_tool_call, Contains

    # Scripted LLM responses (mock mode) -- deterministic, no API key.
    MOCK_RESPONSES = [
        scripted_response(None, [scripted_tool_call("get_weather", {"city": "Jakarta"})]),
        scripted_response("Weather in Jakarta: Sunny, 28C"),
    ]
    TAGS = ["smoke"]

    async def test_calls_weather_tool(t):
        await t.send("What is the weather in Jakarta?")
        t.calledTool("get_weather")                            # gate
        t.calledToolWith("get_weather", {"city": "Jakarta"})   # gate (args subset)
        t.check(t.reply, Contains("Sunny"))                    # soft
        t.completed()                                          # gate

    async def test_fails_on_wrong_city(t):
        # Gate fails on purpose so the report shows a failure.
        await t.send("What is the weather in Jakarta?")
        t.calledToolWith("get_weather", {"city": "Bandung"})   # gate -> fail
    """
)


async def _run(eval_dir: Path) -> None:
    from koboi.eval.t import run_tests
    from koboi.eval.runner import EvalRunner

    results = await run_tests(eval_dir, threshold=0.6)
    console.print(EvalRunner.format_results(results, 0.6))
    passed = sum(1 for r in results if r.passed)
    console.print(f"[bold]{passed}/{len(results)} tests passed[/bold]")


def main() -> None:
    setup_example("Eval `t` Surface", "eve-style test-shaped evals (mock, no API key)")
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)
        (eval_dir / "weather.eval.py").write_text(SAMPLE_EVAL)
        asyncio.run(_run(eval_dir))


if __name__ == "__main__":
    main()
