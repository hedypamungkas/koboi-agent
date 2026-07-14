"""Sample `t` eval: deep_research under mock (CI-safe structural gate, W6.1).

Uses ``DispatchingClient`` (content-dispatching, not sequential) so the deep_research loop runs
deterministically without an API key. The runner auto-creates the dispatch client for orchestration
configs when ``MOCK_RESPONSES`` is set (empty list triggers mock mode). Asserts structural
correctness (``t.completed`` + ``t.citation``) -- NOT faithfulness (that's the live W6.1 eval).

Run:  koboi eval-test evals/deep_research_mock.eval.py --mock --strict
"""

CONFIG = {
    "agent": {
        "name": "deep-research-mock-eval",
        "description": "CI-safe mock eval for deep_research structural correctness",
        "system_prompt": "You plan and run iterative, cited web research.",
        "mode": "act",
    },
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "dummy"},
    "orchestration": {"enabled": True, "execution": {"mode": "deep_research"}},
    "research": {"max_depth": 1, "coverage_threshold": 0.7, "max_searches": 5, "max_fetches": 8},
    "web": {"search": {"provider": "mock"}},
    "sandbox": {"backend": "passthrough", "workdir": "./workspace"},
}

# USE_MOCK triggers mock mode (the runner creates a DispatchingClient for orchestration configs).
MOCK_RESPONSES = []
USE_MOCK = True
TAGS = ["smoke", "deep_research", "mock"]


async def test_mock_deep_research_completes(t):
    """A deep_research run under mock must complete + produce a cited report (structural).

    Asserts the loop runs deterministically (DispatchingClient), completes, and every ``[n]``
    marker resolves to a ``research_sources`` citation id. NOT a faithfulness measure (that's
    the live ``deep_research_faithfulness`` eval).
    """
    await t.send("Research the Python release cycle.")
    t.completed()
    t.citation(min_citations=1)
