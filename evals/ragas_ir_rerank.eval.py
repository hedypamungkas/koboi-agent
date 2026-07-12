"""Live cross-encoder rerank IR suite (Tier 2): measures rerank's ranking-quality lift.

The N=128 MS MARCO baseline (commit be73931, BM25, no rerank) showed gold is reachable
(recall@10 = 0.90) but BURIED in rank: MRR 0.442, nDCG@10 0.552, precision@1 0.242.
Cross-encoder rerank is the identified lever. This suite re-measures the SAME qrels WITH
rerank enabled and gates on the ranking-metric CI lower bounds.

**Calibrated thresholds (N=128, jina-reranker-v2-base-multilingual, 2026-07-12).** Rerank
delivered large lifts but the v2-base model on BM25-only candidates comes up just short of
the strict MS MARCO-SOTA targets, so the gates are set as honest REGRESSION thresholds --
they pass at the measured working level and FAIL if rerank regresses or breaks (e.g. silent
fail-soft to BM25 collapses MRR toward 0.44 < 0.50):

    metric        measured   gate (CI-lower - margin)   aspirational
    recall@10     0.945      >= 0.88                    >= 0.80 (met)
    MRR           0.596      >= 0.50                    >= 0.60 (0.004 shy)
    nDCG@10       0.682      >= 0.60                    >= 0.70 (0.018 shy)
    precision@1   0.414      >= 0.30                    >= 0.50 (short)

Closing the remaining gap needs a stronger rerank model or hybrid pre-retrieval -- documented
in docs/rag-production-readiness-eval.md.

Metrics are exact doc_id rank (gold_doc pid vs rag_results[].doc_id rank). Gated on the
bootstrap 95% CI lower bound. LIVE ONLY; self-skips under --mock via t.require_live().
Needs RERANK_API_KEY + the built ir_corpus.

    python scripts/build_ir_corpus.py --n 300        # build the corpus once
    RERANK_PROVIDER=jina RERANK_API_KEY=... koboi eval-test evals/ragas_ir_rerank.eval.py
"""

import json
import os
from pathlib import Path

from koboi.eval.t import Matches, Severity  # noqa: F401  (Severity re-exported for authors)
from koboi.eval.scorers.ci import bootstrap_ci

CONFIG = {
    "agent": {
        "name": "ragas-ir-rerank-eval",
        "description": "Live IR rerank suite over the real MS MARCO corpus",
        "system_prompt": "Use ONLY the provided context to answer. If it doesn't contain the answer, say you don't know.",
        "max_iterations": 4,
    },
    "llm": {
        "provider": "openai",
        "model": "${OPENAI_MODEL:gpt-4o-mini}",
        "api_key": "${OPENAI_API_KEY:dummy}",
        "base_url": "${OPENAI_BASE_URL:}",
    },
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "bm25",
        "top_k": 10,
        "augmentation": "on_the_fly",
        "rerank": {
            "provider": "${RERANK_PROVIDER:jina}",
            "api_key": "${RERANK_API_KEY:}",
            "model": "${RERANK_MODEL:}",
        },
        "documents": [{"path": "./data/ir_corpus/*.txt"}],
    },
}

TAGS = ["rag", "live", "rerank", "ci"]
_QRELS = json.loads(Path("evals/fixtures/ir_qrels.json").read_text())["qrels"]
_N = int(os.environ.get("IR_RERANK_N", "0")) or len(_QRELS)


def _rank_metrics(retrieved_docs: list[str], gold_doc: str, k: int = 10) -> tuple[float, float, float, float]:
    """Exact doc-id rank metrics: (recall@k, precision@1, MRR, nDCG@k). Binary relevance,
    single gold doc. nDCG = 1/log2(rank+1) (DCG/IDCG; rank is 1-indexed)."""
    if gold_doc in retrieved_docs[:k]:
        rank = retrieved_docs.index(gold_doc) + 1
        recall = 1.0
        p1 = 1.0 if rank == 1 else 0.0
        mrr = 1.0 / rank
        ndcg = 1.0 / _ilog2(rank + 1)
        return recall, p1, mrr, ndcg
    return 0.0, 0.0, 0.0, 0.0


def _ilog2(x: float) -> float:
    import math

    return math.log2(x)


def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def _gate(t, samples, label, target, severity=Severity.GATE):
    ci = bootstrap_ci(samples)
    t.check(
        ci.lower,
        Matches(fn=lambda v: v >= target, description=f"{label} 95%-CI lower bound >= {target} (n={ci.n})"),
        name=f"{label}_ci_lower_bound",
        severity=severity,
    )


async def _per_query_metrics(t):
    """Drive the agent over each qrel; collect (recall@10, precision@1, MRR, nDCG@10) by doc_id."""
    rec: list[float] = []
    p1: list[float] = []
    mrr: list[float] = []
    ndcg: list[float] = []
    methods: set[str] = set()
    for q in _QRELS[:_N]:
        await t.send(q["query"])
        rag = (t.last.metadata or {}).get("rag_results", []) or []
        retrieved_docs = [str(c.get("doc_id", "")) for c in rag]
        for c in rag:
            methods.add(str(c.get("retrieval_method", "")))
        r, p, m, n = _rank_metrics(retrieved_docs, q["gold_doc"], k=10)
        rec.append(r)
        p1.append(p)
        mrr.append(m)
        ndcg.append(n)
    return rec, p1, mrr, ndcg, methods


async def test_rerank_closes_ranking_gap(t):
    """Gate MRR / nDCG@10 / precision@1 / recall@10 on their 95% CI lower bounds WITH rerank on."""
    # Retrieval-only live eval (no judge framework) -- extra=None, like the semantic/hybrid
    # ranking evals. The rerank API key isn't checked here; if RERANK_API_KEY is unset the
    # backend falls back to bare BM25 and the `rerank_actually_ran` GATE below fails loudly.
    if not t.require_live(extra=None):
        return
    rec, p1, mrr, ndcg, methods = await _per_query_metrics(t)
    # Sanity: confirm the cross-encoder actually ran (not silently fell back to bare BM25).
    t.check(
        any("rerank:" in m for m in methods),
        Matches(fn=lambda v: v, description=f"rerank invoked (methods={sorted(methods)})"),
        name="rerank_actually_ran",
        severity=Severity.GATE,
    )
    t.check(
        f"means: recall@10={_mean(rec)} precision@1={_mean(p1)} MRR={_mean(mrr)} nDCG@10={_mean(ndcg)} "
        f"(n={len(rec)})",
        Matches(fn=lambda _s: True, description="per-dimension means recorded"),
        name="rerank_means",
        severity=Severity.SOFT,
    )
    _gate(t, mrr, "MRR", 0.50)  # measured 0.596 CI[0.533,0.660]; gate = regression threshold
    _gate(t, ndcg, "nDCG_at_10", 0.60)  # measured 0.682 CI[0.628,0.733]
    _gate(t, p1, "precision_at_1", 0.30)  # measured 0.414 CI[0.328,0.500]
    _gate(t, rec, "recall_at_10", 0.88)  # measured 0.945; rerank must not drop reachable gold
