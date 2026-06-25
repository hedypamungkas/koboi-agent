"""Sample `t` eval: an agent that should answer directly without any tools.

Run:  koboi eval-test evals/no_tools.eval.py --mock --strict
"""

from koboi.eval.t import Contains, scripted_response

MOCK_RESPONSES = [
    scripted_response("Capital of France is Paris."),
]
TAGS = ["smoke"]


async def test_answers_without_tools(t):
    """A knowledge question that needs no tool calls."""
    await t.send("What is the capital of France?")
    t.usedNoTools()  # gate: no tools should be called
    t.check(t.reply, Contains("Paris"))  # soft: answer is correct
    t.completed()  # gate: run finished cleanly
