"""Sample `t` eval: a multi-turn conversation (turn 1 uses a tool, turn 2 doesn't).

Demonstrates the parts of the `t` API that only make sense across multiple
``t.send()`` calls: the conversation accumulates in memory, ``t.all_tool_calls``
and ``t.turns`` span every turn, and ``t.calledTool``/``t.completed`` evaluate
against the full transcript at collect time.

Run:  koboi eval-test evals/multi_turn.eval.py --mock --strict
"""

from koboi.eval.t import Contains, scripted_response, scripted_tool_call

# Three scripted LLM responses: turn 1 costs two (tool call, then answer),
# turn 2 costs one (direct answer).
MOCK_RESPONSES = [
    scripted_response(None, [scripted_tool_call("get_weather", {"city": "Jakarta"})]),
    scripted_response("Weather in Jakarta: Sunny, 28C"),
    scripted_response("You're welcome! Let me know if you need anything else."),
]
TAGS = ["smoke", "multi-turn"]


async def test_multi_turn_conversation(t):
    """Turn 1 calls a tool; turn 2 is a direct follow-up -- both stay in one conversation."""
    # Turn 1: the agent calls get_weather, then answers.
    await t.send("What is the weather in Jakarta?")
    t.check(t.reply, Contains("Sunny"))

    # Turn 2: a follow-up answered directly (no tool).
    await t.send("Thanks!")
    t.check(t.reply, Contains("welcome"))

    # Cross-turn assertions (evaluated against the whole transcript at collect time):
    t.calledTool("get_weather")  # gate: the tool was called somewhere in the conversation
    t.check(len(t.turns) == 2, name="two turns recorded")  # soft: multi-turn accumulation
    t.check(len(t.all_tool_calls) == 1, name="exactly one tool call across turns")  # soft
    t.completed()  # gate: the last turn finished cleanly
