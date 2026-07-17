"""tests/test_rag_issues.py -- Regression guards for the 6 empirically-confirmed RAG issues.

Each test asserts the FIXED behaviour (so it fails on the buggy code and passes after the
fix). One test per issue. These are the CI source of truth; ``scripts/verify_rag_issues.py`` is the
runnable demo companion.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path

import pytest

from koboi.context.manager import SmartTruncationManager
from koboi.events import CompleteEvent
from koboi.llm.base import LLMClient
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.orchestration.factory import AgentFactory
from koboi.rag.augmentation import OnTheFlyAugmentation
from koboi.rag.chunker import SemanticChunker
from koboi.rag.retriever import KeywordRetriever, SemanticRetriever, clear_embedding_cache
from koboi.rag.types import Chunk, Document
from koboi.types import AgentResponse

REPO = Path(__file__).resolve().parents[1]
CATALOG = str(REPO / "data" / "sample" / "product_catalog.md")
HANDBOOK = str(REPO / "data" / "sample" / "employee_handbook.md")


class _CompleteClient(LLMClient):
    """Chat completes instantly; embeddings return None (Anthropic-like provider)."""

    model = "mock-complete"

    async def complete(self, messages, tools=None, **kwargs):
        return AgentResponse(content="mock answer", tool_calls=[])

    async def complete_stream(self, messages, tools=None, **kwargs):
        yield CompleteEvent(response=AgentResponse(content="mock answer", tool_calls=[]))

    async def get_embeddings(self, text):
        return None


# --------------------------------------------------------------------------- #
# Issue 1: per-agent rag: block must merge with (not replace) the parent config
# --------------------------------------------------------------------------- #
async def test_issue_1_per_agent_rag_merges_not_replaces():
    parent = {
        "enabled": True,
        "retriever": "keyword",
        "top_k": 3,
        "documents": [{"path": CATALOG}],
    }

    # Agent customizes its corpus but omits enabled -> inherits parent's enabled (was: None)
    out = AgentFactory.build_rag_from_config({"documents": [{"path": HANDBOOK}]}, parent)
    assert out is not None, "partial agent rag block must inherit parent `enabled`"

    # Agent with no rag block inherits parent
    assert AgentFactory.build_rag_from_config(None, parent) is not None
    # Agent restating enabled works on its own
    assert AgentFactory.build_rag_from_config({"enabled": True, "documents": [{"path": HANDBOOK}]}, None) is not None
    # Explicit opt-out still wins
    assert AgentFactory.build_rag_from_config({"enabled": False}, parent) is None


# --------------------------------------------------------------------------- #
# Issue 4: smart_truncation must keep ALL user messages (no mid-conversation amnesia)
# --------------------------------------------------------------------------- #
async def test_issue_4_smart_truncation_keeps_all_user_messages():
    mgr = SmartTruncationManager(keep_last=6)
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(1, 13):
        msgs.append({"role": "user", "content": f"turn {i} " + "word " * 120})
        msgs.append({"role": "assistant", "content": "reply " + "z " * 120})
    fact = "ZZZ_SECRET_FACT_987654321_zzz"
    msgs[3]["content"] = f"note: the {fact} must be remembered"  # user turn 2 (middle)

    out = await mgr.manage(msgs, max_tokens=50)  # forces truncation
    joined = " ".join(m["content"] for m in out)

    assert fact in joined, "a user fact stated mid-conversation must survive truncation"
    assert len(out) < len(msgs), "truncation must still engage (drop some assistant/tool)"


# --------------------------------------------------------------------------- #
# Issue 6: run_stream's CompleteEvent must carry rag_results (parity with run())
# --------------------------------------------------------------------------- #
def _agent_with_rag() -> AgentCore:
    chunks = [
        Chunk(
            id="c0",
            doc_id="kb",
            content="The koboi refund window is exactly 30 days from purchase.",
            metadata={"source": "kb"},
        )
    ]
    aug = OnTheFlyAugmentation(retriever=KeywordRetriever(chunks), top_k=2)
    return AgentCore(
        client=_CompleteClient(),
        memory=ConversationMemory(system_prompt="s"),
        augmentation=aug,
        max_iterations=3,
    )


async def test_issue_6_streaming_surfaces_rag_results():
    result = await _agent_with_rag().run("what is the refund window?")
    assert result.metadata.get("rag_results"), "run() must stamp rag_results"
    rc = result.metadata.get("retrieval_confidence")
    assert rc and {"max_score", "method", "count"} <= set(rc), "run() must stamp retrieval_confidence"

    complete = None
    async for ev in _agent_with_rag().run_stream("what is the refund window?"):
        if isinstance(ev, CompleteEvent):
            complete = ev
    assert complete is not None
    assert complete.metadata.get("rag_results"), "run_stream CompleteEvent must carry rag_results"
    rcs = complete.metadata.get("retrieval_confidence")
    assert rcs and {"max_score", "method", "count"} <= set(rcs), (
        "run_stream CompleteEvent must carry retrieval_confidence"
    )


# --------------------------------------------------------------------------- #
# Issue 5: a no-embeddings provider (e.g. Anthropic) must WARN (not INFO) and name the fix
# --------------------------------------------------------------------------- #
async def test_issue_5_no_embeddings_provider_warns(caplog):
    clear_embedding_cache()
    chunks = [
        Chunk(id=f"c{i}", doc_id="kb", content=c, metadata={"source": "kb"})
        for i, c in enumerate(
            [
                "The refund window is 30 days from purchase.",
                "Invoices are due Net 14.",
                "Enterprise includes dedicated support.",
            ]
        )
    ]
    retriever = SemanticRetriever(chunks, client=_CompleteClient())

    with caplog.at_level(logging.WARNING, logger="koboi.rag.retriever"):
        results = await retriever.retrieve("refund policy", top_k=2)

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("embedding:" in m or "embeddings" in m for m in msgs), f"expected WARNING, got {msgs}"
    assert any("fallback" in r.retrieval_method for r in results)


# --------------------------------------------------------------------------- #
# Issue 3: SemanticChunker must WARN when it degrades (no silent sentence fallback)
# --------------------------------------------------------------------------- #
def test_issue_3_semantic_chunker_warns_on_degradation(caplog):
    chunker = SemanticChunker()
    with caplog.at_level(logging.WARNING, logger="koboi.rag.chunker"):
        chunks = chunker.chunk(
            Document(id="d", title="d", content="Cats are mammals. They purr. Dogs are loyal animals.")
        )
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("SemanticChunker" in m for m in warnings), f"expected WARNING, got {warnings}"
    assert len(chunks) >= 1


# --------------------------------------------------------------------------- #
# Issue 2: the dead RAGHook is removed and referenced nowhere in the wiring surface
# --------------------------------------------------------------------------- #
def test_issue_2_raghook_removed_and_unreferenced():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("koboi.hooks.rag_hook")

    wiring = [
        importlib.import_module(n)
        for n in (
            "koboi.hooks.registry",
            "koboi.hooks.builtin",
            "koboi.facade",
            "koboi.loop",
        )
    ]
    for mod in wiring:
        assert "RAGHook" not in inspect.getsource(mod), f"{mod.__name__} still references RAGHook"
