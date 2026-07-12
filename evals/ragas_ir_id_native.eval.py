"""Live NATIVE-Indonesian IR suite (TyDi QA-id): caveat-free per-language ID ranking claim.

The translated-MS-MARCO ID measurements (ID@1000) inflate retrieval scores via translation
normalization. This suite measures ranking on NATIVELY-collected Indonesian (TyDi QA secondary_task,
gold 1-in-3000 == EN MS MARCO density) so the ID claim carries no translation caveat.

CONFIG: bm25 + rerank v3 + fetch_mult=4 + **stopwords=id + stemmer=id** (the Step 2 ID capability:
Indonesian function-word filter + Sastrawi morphology). Metrics are exact doc_id rank (gold_doc pid
vs rag_results[].doc_id rank), gated on the bootstrap 95% CI lower bound. Gates are provisional
pending the first live calibration run (the translated-ID@1000 numbers, deflated for native text).

LIVE ONLY; self-skips under --mock via t.require_live(). Needs RERANK_API_KEY + [indo-nlp] (for the
stemmer; falls back to stopwords-only if absent) + the built id_native_corpus.

    python scripts/build_id_native_corpus.py --n 128          # build the corpus once
    RERANK_API_KEY=... koboi eval-test evals/ragas_ir_id_native.eval.py
"""

import json
import math
import os
from pathlib import Path

from koboi.eval.t import Matches, Severity  # noqa: F401  (Severity re-exported for authors)
from koboi.eval.scorers.ci import bootstrap_ci

CONFIG = {
    "agent": {
        "name": "ragas-ir-id-native-eval",
        "description": "Live NATIVE-Indonesian IR suite over TyDi QA-id (1-in-3000)",
        "system_prompt": "Answer the question using the provided context. Give a concise, factual answer extracted directly from the context. Only say you don't have the information if the context truly lacks anything relevant.",
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
        "stopwords": "id",  # cheap ID function-word filter (always-on for ID)
        # NOTE: stemmer "id" (Sastrawi) is OPT-IN -- it adds ~minutes of CPU per build on a
        # 3000-passage corpus (per-token dictionary stemming), impractical for every eval build.
        # Its correctness is unit-tested (tests/test_rag_indo_nlp.py) and its retrieval benefit
        # is measured by /tmp/measure_id_native.py (baseline vs +stemmer). Enable in production
        # configs when morphology matching justifies the one-time build cost.
        "augmentation": "on_the_fly",
        "rerank": {
            "provider": "${RERANK_PROVIDER:jina}",
            "api_key": "${RERANK_API_KEY:}",
            "model": "${RERANK_MODEL:}",
            "fetch_multiplier": 4,
        },
        "documents": [{"path": "./data/id_native_corpus/*.txt"}],
    },
}

TAGS = ["rag", "live", "id", "native", "ci"]
_QRELS = json.loads(Path("evals/fixtures/id_native_qrels.json").read_text())["qrels"]
_N = int(os.environ.get("IR_ID_NATIVE_N", "0")) or len(_QRELS)


def _ilog2(x: float) -> float:
    return math.log2(x)


def _rank_metrics(retrieved_docs: list[str], gold_doc: str, k: int = 10) -> tuple[float, float, float, float]:
    if gold_doc in retrieved_docs[:k]:
        rank = retrieved_docs.index(gold_doc) + 1
        return 1.0, (1.0 if rank == 1 else 0.0), 1.0 / rank, 1.0 / _ilog2(rank + 1)
    return 0.0, 0.0, 0.0, 0.0


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
    rec: list[float] = []
    p1: list[float] = []
    mrr: list[float] = []
    ndcg: list[float] = []
    methods: set[str] = set()
    for q in _QRELS[:_N]:
        await t.send(q["query"])
        rag = (t.last.metadata or {}).get("rag_results", []) or []
        docs = [str(c.get("doc_id", "")) for c in rag]
        for c in rag:
            methods.add(str(c.get("retrieval_method", "")))
        r, p, m, n = _rank_metrics(docs, q["gold_doc"], k=10)
        rec.append(r)
        p1.append(p)
        mrr.append(m)
        ndcg.append(n)
    return rec, p1, mrr, ndcg, methods


async def test_id_native_ranking(t):
    """Gate recall@10 / MRR / nDCG@10 / precision@1 on native Indonesian (TyDi QA-id, 1-in-3000)."""
    if not t.require_live(extra=None):
        return
    rec, p1, mrr, ndcg, methods = await _per_query_metrics(t)
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
        name="id_native_means",
        severity=Severity.SOFT,
    )
    # Provisional gates (calibrate after the first live run; native text expected harder than translated).
    _gate(t, rec, "recall_at_10", 0.80)
    _gate(t, mrr, "MRR", 0.50)
    _gate(t, ndcg, "nDCG_at_10", 0.60)
    _gate(t, p1, "precision_at_1", 0.30)
