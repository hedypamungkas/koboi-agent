"""Sample `t` eval: deep_research produces a cited report (structural citation correctness).

Unlike the mock-safe RAG retrieval eval, deep_research is an orchestration config (core=None),
so ``--mock`` is unsupported -- this eval runs LIVE (needs ``OPENAI_API_KEY``). The web side is
mock (``web.search.provider: mock``), so no search API key / $0 cost. It asserts STRUCTURAL
citation correctness via ``t.citation`` (every ``[n]`` resolves to a ``research_sources`` id) --
NOT faithfulness (needs RAGAS + source-text plumbing; deferred to W6.1).

Run:  OPENAI_API_KEY=... koboi eval-test evals/deep_research_citations.eval.py --strict
"""

import os

from koboi.eval.t import Contains

CONFIG = {
    "agent": {
        "name": "deep-research-citations-eval",
        "description": "Eval probe for deep_research cited-report structure",
        "system_prompt": "You plan and run iterative, cited web research.",
        "mode": "act",
    },
    "llm": {
        "provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
    },
    "orchestration": {"enabled": True, "execution": {"mode": "deep_research"}},
    # max_depth=1 for speed; coverage_threshold=0.7 (likely stops after 1 round on a simple topic).
    "research": {"max_depth": 1, "coverage_threshold": 0.7, "max_searches": 5, "max_fetches": 8},
    # Mock web -> offline, $0 search cost (nodes hit the hardcoded SEARCH_INDEX, not Brave).
    "web": {"search": {"provider": "mock"}},
    "sandbox": {"backend": "passthrough", "workdir": "./workspace"},
}

# Empty -> LIVE (orchestration configs can't --mock; agent.core is None). Web is mock -> $0.
MOCK_RESPONSES = []
TAGS = ["deep_research", "live"]


async def test_cited_report_completes(t):
    """A deep_research run must complete and produce a cited report.

    Asserts structural citation correctness: every ``[n]`` marker in the reply resolves to a
    ``research_sources`` citation id (``t.citation`` GATE) + the ``## Sources`` footer is present.
    """
    if not os.getenv("OPENAI_API_KEY"):
        # Live-only eval (orchestration can't --mock). Fail fast without a network attempt;
        # the mock-driven golden test excludes this case by name.
        raise RuntimeError("OPENAI_API_KEY required (live eval; web is mock -> $0 search)")
    await t.send("Research the Python programming language release cycle.")
    t.completed()  # the run finished successfully
    t.citation(min_citations=1)  # every [n] resolves + at least one citation
    t.check(t.reply, Contains("## Sources"))  # the Sources footer is present
