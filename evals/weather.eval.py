"""Sample `t` eval: a weather agent that must call the get_weather tool.

Run:  koboi eval-test evals/weather.eval.py --mock --strict
"""

from koboi.eval.t import Contains, Severity, scripted_response, scripted_tool_call

# Scripted (mock) LLM responses -- deterministic, no API key required.
MOCK_RESPONSES = [
    scripted_response(None, [scripted_tool_call("get_weather", {"city": "Jakarta"})]),
    scripted_response("Weather in Jakarta: Sunny, 28C"),
]
TAGS = ["smoke", "tools"]


async def test_calls_weather_tool(t):
    """The agent should call get_weather and report the result."""
    await t.send("What is the weather in Jakarta?")
    t.calledTool("get_weather")  # gate: tool must be called
    t.calledToolWith("get_weather", {"city": "Jakarta"})  # gate: with the right args
    t.check(t.reply, Contains("Sunny"))  # soft: answer mentions the condition
    t.completed()  # gate: run finished cleanly


async def test_soft_mismatch_dents_score_not_gate(t):
    """A SOFT check that does not match dents the score but does NOT fail the gate.

    Only a GATE failure forces ``passed=False``; a SOFT miss contributes 0.5 to the
    score. With two passing gates (1.0 each) + one soft miss (0.5), overall ~= 0.83,
    comfortably above the 0.6 threshold -- so the test stays green while still
    illustrating a non-matching assertion. Use SOFT for fuzzy/lenient checks so a
    miss lowers the score without breaking CI.
    """
    await t.send("What is the weather in Jakarta?")
    t.calledTool("get_weather")  # gate: passes
    # Intentionally non-matching SOFT check -> value 0.5, but gate_failed stays False.
    t.check(t.reply, Contains("Rainy"), severity=Severity.SOFT, name="wrong-condition-soft")
    t.completed()  # gate: passes
