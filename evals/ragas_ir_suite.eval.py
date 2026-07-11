"""Live IR suite: real corpus (MS MARCO) + decoupled judge + CI-lower-bound gate (Path B3).

This is the trustworthy replacement for the self-inflated "all 1.0" Acme calibration:
- a REAL ~3000-passage corpus (built by scripts/build_ir_corpus.py) so top_k=10 returns
  <1% of the corpus (retrieval stops saturating);
- the DECOUPLED judge (RAGAS_JUDGE_MODEL, a stronger/different model than the agent);
- N≈128 queries with bootstrap 95% CI lower-bound gating (not N=1 point estimates);
- per-dimension scores with variance.

Metrics over the qrels:
- retrieval recall@10 (does the gold passage appear in the top-10, by doc_id);
- ragas_faithfulness (decoupled judge) and ragas context_recall, each gated on the
  bootstrap 95% CI lower bound.

LIVE ONLY; self-skips under --mock via t.require_live(). Set IR_SUITE_N to cap the query
count for a quick smoke (default: all qrels).

    python scripts/build_ir_corpus.py --n 300   # build the corpus once
    RAGAS_JUDGE_MODEL=gpt-5.4 koboi eval-test evals/ragas_ir_suite.eval.py
"""

import json
import os
from pathlib import Path

from koboi.eval.t import Matches, Severity  # noqa: F401  (Severity re-exported for authors)
from koboi.eval.scorers.ci import bootstrap_ci
from koboi.eval.scorers.retrieval_metric import recall_at_k
from koboi.types import EvalCase

CONFIG = {
    "agent": {
        "name": "ragas-ir-suite-eval",
        "description": "Live IR suite over the real MS MARCO corpus (decoupled judge)",
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
        "documents": [{"path": "./data/ir_corpus/*.txt"}],
    },
}

TAGS = ["rag", "live", "ir", "ci"]
_QRELS = json.loads(Path("evals/fixtures/ir_qrels.json").read_text())["qrels"]
_N = int(os.environ.get("IR_SUITE_N", "0")) or len(_QRELS)


async def _per_query_scores(t):
    """Drive the agent over each qrel; return (faithfulness[], context_recall[], recall@10[])."""
    import koboi.eval  # noqa: F401 - registers ragas_* scorers
    from koboi.eval.registry import ScorerRegistry

    faith = ScorerRegistry.create("ragas_faithfulness")
    crec = ScorerRegistry.create("ragas_recall")
    f_samples: list[float] = []
    c_samples: list[float] = []
    r_samples: list[float] = []
    for q in _QRELS[:_N]:
        await t.send(q["query"])
        rag = (t.last.metadata or {}).get("rag_results", []) or []
        retrieved_contents = [str(c.get("content", "")) for c in rag]
        retrieved_docs = [str(c.get("doc_id", "")) for c in rag]
        # retrieval recall@10 by stable doc_id (gold passage in top-10)
        r_samples.append(1.0 if q["gold_doc"] in retrieved_docs else 0.0)
        # judge-scored faithfulness + context recall
        case = EvalCase(
            name="ir",
            user_message=q["query"],
            context_docs=retrieved_contents,
            expected_answer=q.get("expected_answer", ""),
        )
        f_samples.append((await faith.score(case, t.reply, {})).value)
        c_samples.append((await crec.score(case, t.reply, {})).value)
        # also record needle-based recall (content) for comparison
        _ = recall_at_k(retrieved_contents, q.get("gold_needles", []), 10)
    return f_samples, c_samples, r_samples


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


async def test_ir_suite_ci_lower_bounds(t):
    """Gate faithfulness / context_recall / retrieval-recall@10 on their 95% CI lower bounds."""
    if not t.require_live():
        return
    f_samples, c_samples, r_samples = await _per_query_scores(t)
    t.check(
        f"means: faithfulness={_mean(f_samples)} context_recall={_mean(c_samples)} recall@10={_mean(r_samples)} (n={len(r_samples)})",
        Matches(fn=lambda _s: True, description="per-dimension means recorded"),
        name="ir_suite_means",
        severity=Severity.SOFT,
    )
    _gate(t, f_samples, "faithfulness", 0.80)
    _gate(t, c_samples, "context_recall", 0.70)
    _gate(t, r_samples, "retrieval_recall_at_10", 0.70)
