"""Sample `t` eval: a weather agent that must call the get_weather tool.

Run:  koboi eval-test evals/weather.eval.py --mock --strict
"""

from koboi.eval.t import Contains, scripted_response, scripted_tool_call

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


async def test_rejects_wrong_arguments(t):
    """A gate that fails on purpose, to illustrate a failing assertion in the report."""
    await t.send("What is the weather in Jakarta?")
    t.calledToolWith("get_weather", {"city": "Bandung"})  # gate -> fail
