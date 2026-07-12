"""Sample `t` eval: deep_research faithfulness (RAGAS) -- are the report's claims grounded?

Live-only (needs ``OPENAI_API_KEY`` + ``pip install koboi-agent[eval-ragas]`` + ``RAGAS_PROVIDER``
env). Web is mock (``$0`` search). This is the QUALITY evidence (W6.1): structural correctness is
W6's ``t.citation``; faithfulness needs a live LLM judge. The scorer reads
``context['research_sources']`` (source TEXT surfaced from the run's ``research_sources_with_text``
metadata via ``TestContext._build_context``) and runs RAGAS ``faithfulness``.

Run:  OPENAI_API_KEY=... koboi eval-test evals/deep_research_faithfulness.eval.py --strict
"""

import os

CONFIG = {
    "agent": {
        "name": "deep-research-faithfulness-eval",
        "description": "Eval probe for deep_research faithfulness (RAGAS)",
        "system_prompt": "You plan and run iterative, cited web research.",
        "mode": "act",
    },
    "llm": {
        "provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
    },
    "orchestration": {"enabled": True, "execution": {"mode": "deep_research"}},
    "research": {"max_depth": 1, "coverage_threshold": 0.7, "max_searches": 5, "max_fetches": 8},
    "web": {"search": {"provider": "mock"}},
    "sandbox": {"backend": "passthrough", "workdir": "./workspace"},
}

MOCK_RESPONSES = []
TAGS = ["deep_research", "live", "faithfulness"]


async def test_report_claims_grounded_in_sources(t):
    """The report's claims should be grounded in the gathered sources (RAGAS faithfulness >= 0.7).

    Live-only: needs OPENAI_API_KEY + ragas + RAGAS_PROVIDER env. Structural citation correctness
    (t.citation) is separate (W6); this is the faithfulness NUMBER (claim-grounding quality).
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY required (live eval; web is mock -> $0 search)")
    await t.send("Research the Python programming language release cycle.")
    t.completed()
    await t.judge("deep_research_faithfulness", min_score=0.7, name="faithfulness")
