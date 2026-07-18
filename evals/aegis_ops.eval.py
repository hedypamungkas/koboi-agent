"""Sample `t` eval: Aegis Ops DAG produces a grounded resolution (structural gate).

LIVE-only, like ``deep_research_citations.eval.py`` -- ``--mock`` is unsupported here.
The `t` mock runner (``koboi/eval/t/runner.py``) only swaps the LLM client that the
*Orchestrator* itself calls (router/planner); it does NOT reach into each DAG node's own
``AgentCore.client`` (``AgentFactory.create_configured_agent`` builds a dedicated client
per node). A ``--mock`` run of a `dag`-mode config therefore still fires real HTTP calls
from every node -- they just get swallowed as "node failed" by
``Orchestrator._execute_node``'s broad ``except Exception``, and the overall run still
reports ``success=True`` (a false-green gate). This is a smaller-scale version of the
configs/aegis_ops_full.yaml DAG (2 nodes instead of 4) so a real run stays cheap.

Run:  OPENAI_API_KEY=... koboi eval-test evals/aegis_ops.eval.py --strict
"""

import os

CONFIG = {
    "agent": {
        "name": "aegis-ops-eval",
        "description": "Eval probe for the Aegis Ops support_kb DAG node (RAG-grounded answer)",
        "system_prompt": "You are Aegis, Northwind Cloud's customer-ops assistant.",
        "mode": "act",
    },
    "llm": {
        "provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY", "dummy"),
    },
    "orchestration": {
        "enabled": True,
        "router": {"type": "keyword"},
        # full_graph: true runs every configured node regardless of what the keyword
        # router selects -- without it, only the router's top match ("intake") runs
        # and support_kb never fires (verified empirically: agents_used=['intake']).
        "execution": {"mode": "dag", "full_graph": True},
        "agents": [
            {
                "name": "intake",
                "system_prompt": "Classify the request as product-question or infra-incident.",
                "keywords": ["help", "question", "refund", "policy"],
            },
            {
                "name": "support_kb",
                "system_prompt": "Answer the customer's question using the knowledge base.",
                "depends_on": ["intake"],
                "rag": {
                    "enabled": True,
                    "chunker": "sentence",
                    "retriever": "keyword",
                    "top_k": 3,
                    "documents": [{"path": "./data/aegis_kb/faq.md"}],
                },
            },
        ],
    },
    "sandbox": {"backend": "passthrough", "workdir": "./workspace"},
}

# Empty -> LIVE (orchestration configs can't --mock without a false-green risk; see
# the module docstring). No web calls -- $0 beyond the 2 real chat completions.
MOCK_RESPONSES = []
TAGS = ["aegis_ops", "orchestration", "rag", "live"]


async def test_support_kb_answers_from_faq(t):
    """The support_kb DAG node must answer the refund question, grounded in faq.md."""
    if not t.live_ready(extra=None):
        return
    await t.send("What is your refund policy?")
    t.completed()
    t.check(t.reply, lambda r: "30" in r or "30-day" in r.lower())
