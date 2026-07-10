"""tests/test_rag_metadata_filter.py -- Gap #10 metadata filtering regression.

Relevance scoping (NOT ACL): equality + $gte/$lte/$gt/$lt + $in, applied as a
pre-filter in every retriever (so top_k isn't shrunk). Missing field / unknown
operator / incomparable type -> strict exclude.
"""

from __future__ import annotations

from koboi.rag.augmentation import OnTheFlyAugmentation, RerankerRetriever
from koboi.rag.filters import matches_filter
from koboi.rag.registry import build_rag
from koboi.rag.retriever import BM25Retriever, HybridRetriever, KeywordRetriever, SemanticRetriever
from koboi.rag.types import Chunk


def _chunks():
    return [
        Chunk(id="old", doc_id="d", content="ALPHA document", metadata={"year": 2021, "source": "policy"}),
        Chunk(id="new", doc_id="d", content="BETA document", metadata={"year": 2024, "source": "policy"}),
        Chunk(id="hb", doc_id="d", content="GAMMA document", metadata={"year": 2024, "source": "handbook"}),
    ]


# --------------------------------------------------------------------------- #
# matches_filter unit
# --------------------------------------------------------------------------- #
def test_equality():
    assert matches_filter({"a": 1}, {"a": 1})
    assert not matches_filter({"a": 1}, {"a": 2})


def test_comparison():
    assert matches_filter({"year": 2024}, {"year": {"$gte": 2024}})
    assert matches_filter({"year": 2025}, {"year": {"$gte": 2024}})
    assert not matches_filter({"year": 2023}, {"year": {"$gte": 2024}})
    assert matches_filter({"year": 2024}, {"year": {"$lte": 2024, "$gt": 2020}})


def test_in():
    assert matches_filter({"source": "policy"}, {"source": {"$in": ["policy", "handbook"]}})
    assert not matches_filter({"source": "x"}, {"source": {"$in": ["policy"]}})


def test_missing_field_excludes():
    assert not matches_filter({"a": 1}, {"b": 1})  # strict: missing -> no match


def test_none_or_empty_matches_all():
    assert matches_filter({"a": 1}, None)
    assert matches_filter({"a": 1}, {})


def test_unknown_operator_excludes():
    assert not matches_filter({"a": 1}, {"a": {"$weird": 1}})


def test_incomparable_type_excludes():
    assert not matches_filter({"year": "2024"}, {"year": {"$gte": 2024}})  # str vs int -> no match


def test_multiple_clauses_all_required():
    assert matches_filter({"year": 2024, "source": "policy"}, {"year": {"$gte": 2024}, "source": "policy"})
    assert not matches_filter({"year": 2024, "source": "handbook"}, {"year": {"$gte": 2024}, "source": "policy"})


# --------------------------------------------------------------------------- #
# Retrievers respect the filter (pre-scoring -> top_k unaffected)
# --------------------------------------------------------------------------- #
async def test_keyword_retriever_filters():
    res = await KeywordRetriever(_chunks()).retrieve(
        "ALPHA BETA GAMMA", top_k=5, metadata_filter={"year": {"$gte": 2024}}
    )
    assert {x.chunk.id for x in res} == {"new", "hb"}


async def test_bm25_retriever_filters():
    res = await BM25Retriever(_chunks()).retrieve(
        "ALPHA BETA GAMMA", top_k=5, metadata_filter={"source": {"$in": ["handbook"]}}
    )
    assert {x.chunk.id for x in res} == {"hb"}


async def test_semantic_fallback_respects_filter():
    # No client -> keyword fallback; filter still applies.
    res = await SemanticRetriever(_chunks(), client=None).retrieve(
        "ALPHA BETA GAMMA", top_k=5, metadata_filter={"year": {"$gte": 2024}}
    )
    assert {x.chunk.id for x in res} == {"new", "hb"}


async def test_no_filter_returns_all():
    res = await KeywordRetriever(_chunks()).retrieve("ALPHA BETA GAMMA", top_k=5)
    assert len(res) == 3  # default path unchanged


class _NoEmbed:
    model = "m"

    async def get_embeddings(self, t):
        return None  # semantic leg falls back to keyword

    async def complete(self, m, tools=None):
        raise RuntimeError

    async def complete_stream(self, m, tools=None):
        raise RuntimeError


async def test_hybrid_propagates_filter_to_both_legs():
    res = await HybridRetriever(_chunks(), client=_NoEmbed()).retrieve(
        "ALPHA BETA GAMMA", top_k=5, metadata_filter={"year": {"$gte": 2024}}
    )
    assert "old" not in {x.chunk.id for x in res}


async def test_reranker_propagates_filter():
    res = await RerankerRetriever(KeywordRetriever(_chunks())).retrieve(
        "ALPHA BETA GAMMA", top_k=2, metadata_filter={"year": {"$gte": 2024}}
    )
    assert "old" not in {x.chunk.id for x in res}


# --------------------------------------------------------------------------- #
# Augmentation + build_rag wiring
# --------------------------------------------------------------------------- #
async def test_augmentation_applies_metadata_filter():
    aug = OnTheFlyAugmentation(retriever=KeywordRetriever(_chunks()), top_k=5, metadata_filter={"year": {"$gte": 2024}})
    out = await aug.augment_for_llm([{"role": "user", "content": "ALPHA BETA GAMMA"}])
    content = out[-1]["content"]
    assert "BETA document" in content and "GAMMA document" in content
    assert "ALPHA document" not in content  # old (2021) chunk filtered out (Question echoes ALPHA)


def test_build_rag_wires_metadata_filter(tmp_path):
    doc = tmp_path / "kb.md"
    doc.write_text("alpha beta")
    aug = build_rag(
        {
            "enabled": True,
            "retriever": "keyword",
            "top_k": 3,
            "filter": {"year": {"$gte": 2024}},
            "documents": [{"path": str(doc)}],
        }
    )
    assert aug is not None
    assert aug.metadata_filter == {"year": {"$gte": 2024}}
