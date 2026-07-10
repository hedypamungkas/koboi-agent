"""tests/test_rag_rewrite.py -- Gap #9 (query rewriting / HyDE) regression guards.

Covers: rule-based normalization, LLM rewrite (+ cache + fallback), HyDE, no-client
degradation, augmentation wiring (last_rewrite + effective query used), default-off
no-op, build_rag chat-client threading, and _run_metadata stamping.
"""

from __future__ import annotations

import pytest

from koboi.llm.base import LLMClient
from koboi.memory import ConversationMemory
from koboi.loop import AgentCore
from koboi.rag.augmentation import InMemoryAugmentation, OnTheFlyAugmentation
from koboi.rag.registry import build_rag
from koboi.rag.retriever import KeywordRetriever
from koboi.rag.rewrite import QueryRewriter, rule_based_rewrite
from koboi.rag.types import Chunk
from koboi.types import AgentResponse


class _Chat:
    """Chat-capable mock: returns canned rewrites, counts complete() calls."""

    model = "mock-chat"

    def __init__(self, content: str = "refund policy window"):
        self._content = content
        self.calls = 0

    async def complete(self, messages, tools=None):
        self.calls += 1
        return AgentResponse(content=self._content, tool_calls=[])

    async def complete_stream(self, messages, tools=None):
        raise RuntimeError

    async def get_embeddings(self, text):
        return None  # NOT an embedding client


class _BoomChat(LLMClient):
    model = "mock-boom"

    async def complete(self, messages, tools=None):
        raise RuntimeError("provider down")

    async def complete_stream(self, messages, tools=None):
        raise RuntimeError

    async def get_embeddings(self, text):
        return None


def _chunks():
    return [Chunk(id="c0", doc_id="kb", content="refund window is 30 days", metadata={"source": "kb"})]


# --------------------------------------------------------------------------- #
# Rule-based
# --------------------------------------------------------------------------- #
def test_rule_based_strips_filler_keeps_content():
    out = rule_based_rewrite("hey, so what is the refund policy please?")
    assert "refund" in out and "policy" in out
    assert "hey" not in out and "please" not in out and "the" not in out


# --------------------------------------------------------------------------- #
# LLM rewrite + cache + fallback
# --------------------------------------------------------------------------- #
async def test_llm_rewrite_uses_client_and_caches():
    chat = _Chat("refund policy days")
    rw = QueryRewriter(client=chat)
    eff, meta = await rw.rewrite("what is the refund policy?", mode="llm")
    assert eff == "refund policy days"
    assert meta["method"] == "llm" and meta["original"] == "what is the refund policy?"
    await rw.rewrite("what is the refund policy?", mode="llm")  # identical -> cache
    assert chat.calls == 1


async def test_llm_failure_falls_back_to_rule_normalized():
    rw = QueryRewriter(client=_BoomChat())
    eff, meta = await rw.rewrite("what is the refund policy?", mode="llm")
    assert meta["method"] == "rule-fallback"
    assert "refund" in eff and "policy" in eff  # rule-normalized, not the raw filler


async def test_no_client_degrades_to_rule_only():
    rw = QueryRewriter(client=None)
    eff, meta = await rw.rewrite("what is the refund policy?", mode="llm")
    assert meta["method"] == "rule"
    assert "refund" in eff


async def test_hyde_mode():
    chat = _Chat("The refund window is 30 days from purchase.")
    rw = QueryRewriter(client=chat)
    eff, meta = await rw.rewrite("refund?", mode="hyde")
    assert meta["method"] == "hyde"
    assert "refund window" in eff


# --------------------------------------------------------------------------- #
# Augmentation wiring
# --------------------------------------------------------------------------- #
async def test_augmentation_query_rewrite_stamps_last_rewrite_and_uses_effective_query():
    chat = _Chat("refund policy")
    retrieved: list[str] = []
    kw = KeywordRetriever(_chunks())
    orig = kw.retrieve

    async def spy(query, top_k=3, metadata_filter=None):
        retrieved.append(query)
        return await orig(query, top_k)

    kw.retrieve = spy
    aug = OnTheFlyAugmentation(retriever=kw, top_k=2, query_rewrite=True, rewrite_client=chat)
    await aug.augment_for_llm([{"role": "user", "content": "what is the refund policy?"}])
    # The retriever saw the REWRITTEN query, not the raw one.
    assert retrieved == ["refund policy"]
    assert aug.last_rewrite == {
        "original": "what is the refund policy?",
        "rewritten": "refund policy",
        "method": "llm",
    }


async def test_augmentation_default_off_is_a_noop():
    aug = OnTheFlyAugmentation(retriever=KeywordRetriever(_chunks()), top_k=2)
    await aug.augment_for_llm([{"role": "user", "content": "what is the refund policy?"}])
    assert aug.last_rewrite is None
    assert aug._rewriter is None  # no chat client plumbed, no rewriter built


def test_build_rag_threads_chat_client_for_rewrite(tmp_path):
    doc = tmp_path / "kb.md"
    doc.write_text("refund window is 30 days")
    chat = _Chat("refund policy")
    aug = build_rag(
        {
            "enabled": True,
            "retriever": "keyword",
            "top_k": 2,
            "query_rewrite": True,
            "rewrite": {"timeout": 5.0},
            "documents": [{"path": str(doc)}],
        },
        chat_client=chat,
    )
    assert aug is not None
    assert aug._rewriter is not None  # chat client plumbed -> rewriter built
    assert aug._query_rewrite is True


# --------------------------------------------------------------------------- #
# _run_metadata stamps rag_rewrite (#9 + loop integration)
# --------------------------------------------------------------------------- #
def test_run_metadata_stamps_rag_rewrite():
    aug = InMemoryAugmentation(retriever=KeywordRetriever(_chunks()), top_k=2)
    aug.last_rewrite = {"original": "q", "rewritten": "q2", "method": "llm"}
    agent = AgentCore(client=_Chat(), memory=ConversationMemory(system_prompt="s"), augmentation=aug)
    meta = agent._run_metadata(resumed=False, last_step=0)
    assert meta.get("rag_rewrite") == {"original": "q", "rewritten": "q2", "method": "llm"}
