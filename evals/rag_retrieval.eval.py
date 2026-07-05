"""Sample `t` eval: RAG retrieval is deterministic under mock (R4 closure).

Unlike answer-faithfulness (which needs a live judge LLM), RETRIEVAL is a pre-LLM
prompt-augmentation step. ``OnTheFlyAugmentation._retrieve_and_format`` runs before
the scripted client is called, so ``AugmentationStrategy.last_results`` is populated
and stamped onto ``RunResult.metadata['rag_results']`` (R4) regardless of what the
scripted LLM replies. ``t.retrievedChunk`` asserts on that retrieval -- mock-safe,
no API key.

Run:  koboi eval-test evals/rag_retrieval.eval.py --mock --strict
"""

from koboi.eval.t import scripted_response

# Self-contained mock-safe RAG config: keyword retriever (TF-IDF, no embeddings)
# over the shipped Acme Corp policy corpus. The llm block is required by KoboiConfig
# but never contacted (client swapped for ScriptedClient in mock mode).
CONFIG = {
    "agent": {
        "name": "rag-retrieval-eval",
        "description": "Eval probe for RAG retrieval (pre-LLM, mock-deterministic)",
        "system_prompt": "Use the provided context to answer.",
        "max_iterations": 4,
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",  # required by KoboiConfig even in mock (never contacted)
        "api_key": "dummy",
    },
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "chunk_size": 1000,
        "retriever": "keyword",
        "top_k": 3,
        "augmentation": "on_the_fly",
        "documents": [{"path": "./data/sample/company_policy.md"}],
    },
}

# Scripted reply (the retrieval is what we assert on, not the answer).
MOCK_RESPONSES = [scripted_response("Based on the policy: 12 days.")]
TAGS = ["smoke", "rag"]


async def test_retrieves_annual_leave_policy(t):
    """A question about annual leave must retrieve the policy chunk (R4).

    Asserts on the RETRIEVAL (pre-LLM, deterministic) via ``t.retrievedChunk`` --
    not on the scripted reply. The corpus states "Permanent employees: 12 days
    per year"; the keyword retriever must surface that chunk.
    """
    await t.send("How many annual leave days does a permanent employee get?")
    t.retrievedChunk("12 days")  # R4 primitive -- retrieval, mock-safe
    t.completed()


async def test_retrieves_remote_work_section(t):
    """A question about remote work must retrieve the remote-work section."""
    await t.send("What is the remote work policy?")
    t.retrievedChunk("remote")
    t.completed()
