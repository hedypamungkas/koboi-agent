"""koboi/eval/scorers/retrieval_scorer.py -- Deterministic retrieval-quality scorers.

Computes recall@k / precision@k / hit@k / gate_noise directly over a retriever,
offline and with no LLM. This is the reproducible foundation for measuring RAG
retrieval quality (Rec 2 threshold tuning, Rec 1 gating false-negatives) that the
LLM-judged RAGAS scorers cannot provide without an API key.

Hit definition (doc + key-fact, robust to chunker changes): a retrieved chunk is
*relevant* iff its ``doc_id`` matches the case's gold ``source_doc`` AND its content
contains at least one of the gold ``key_facts``. Matching on doc + content substring
-- not chunk index -- keeps the metric valid when chunk_size or chunker strategy
changes (see SentenceChunker using ``max_chunk_size`` with no ``chunk_size`` alias).

Gold labels live on ``EvalCase.metadata``:
    - ``needs_retrieval`` (bool): factual (True) vs gating-negative (False).
    - ``source_doc`` (str): gold ``Chunk.doc_id`` (== file stem).
    - ``key_facts`` (list[str]): substrings a correct passage must contain.

Mirrors the single-metric + fail-open pattern of ``ragas_scorer.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.types import EvalCase, EvalScore
from koboi.eval.scorers.base import BaseScorer
from koboi.rag.retriever import apply_min_score, normalize_scores

if TYPE_CHECKING:
    from koboi.rag.retriever import BaseRetriever
    from koboi.rag.types import RetrievalResult


def _is_factual(case: EvalCase) -> bool:
    """A case is factual (retrieval expected) unless explicitly marked otherwise."""
    return bool(case.metadata.get("needs_retrieval", True))


def compute_retrieval_metrics(
    results: list[RetrievalResult],
    case: EvalCase,
) -> dict:
    """Compute recall/precision/hit/gate_noise from already-retrieved results.

    Pure function -- reused by the offline runner and the scorer. Returns a dict
    with the applicable metric values set and others as ``None`` (not applicable
    for this case type), plus a ``retrieved`` count.

    - recall@k   = (# gold key_facts found in the union of top-k content) / len(key_facts)
    - precision@k = (# relevant chunks) / len(results)
    - hit@k      = 1.0 if any relevant chunk in top-k else 0.0
    - gate_noise = max retrieved score (gating cases only; lower = retriever fires less)
    """
    retrieved = len(results)

    if not _is_factual(case):
        # Gating negative: no gold to match. Measure over-firing instead.
        max_score = max((r.score for r in results), default=0.0)
        return {
            "recall": None,
            "precision": None,
            "hit": None,
            "gate_noise": round(max_score, 4),
            "retrieved": retrieved,
        }

    source_doc = case.metadata.get("source_doc")
    key_facts = case.metadata.get("key_facts", []) or []
    if not source_doc or not key_facts:
        # Factual case missing gold labels -- cannot score deterministically.
        return {
            "recall": None,
            "precision": None,
            "hit": None,
            "gate_noise": None,
            "retrieved": retrieved,
            "no_gold": True,
        }

    facts_lower = [f.lower() for f in key_facts]

    # recall: did the retrieved context surface each required fact?
    blob = " ".join(r.chunk.content for r in results).lower()
    facts_found = sum(1 for f in facts_lower if f in blob)
    recall = facts_found / len(facts_lower)

    # precision / hit: how many retrieved chunks are from the gold doc AND on-topic?
    relevant = 0
    for r in results:
        if r.chunk.doc_id != source_doc:
            continue
        content_lower = r.chunk.content.lower()
        if any(f in content_lower for f in facts_lower):
            relevant += 1

    precision = relevant / retrieved if retrieved else 0.0
    hit = 1.0 if relevant > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "hit": hit,
        "gate_noise": None,
        "retrieved": retrieved,
    }


class RetrievalScorer(BaseScorer):
    """A single deterministic retrieval metric over a retriever.

    metric_name is one of "recall", "precision", "hit", "gate_noise". The scorer
    retrieves ``top_k`` chunks for the case's user_message and computes the metric
    via :func:`compute_retrieval_metrics`. The retriever is taken from the
    constructor (like ``LLMJudgeScorer(client=)``) or, failing that, from
    ``context["retriever"]``.

    Fail-open: no retriever -> 0.0; factual case with no gold labels -> 1.0
    ("no gold labels"); metric not applicable to the case type -> 1.0 ("n/a").
    """

    METRICS = ("recall", "precision", "hit", "gate_noise")

    def __init__(
        self,
        metric_name: str = "recall",
        retriever: BaseRetriever | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
        normalize: bool = True,
        fetch_factor: int = 1,
    ):
        if metric_name not in self.METRICS:
            raise ValueError(f"Unknown retrieval metric '{metric_name}'. Available: {self.METRICS}")
        self.metric_name = metric_name
        self._retriever = retriever
        self.top_k = top_k
        # Rec 2 knobs. Defaults (min_score=0, fetch_factor=1) reproduce the baseline
        # exactly: keyword normalize is identity, no chunks filtered, same top_k fetched.
        self.min_score = min_score
        self.normalize = normalize
        self.fetch_factor = max(1, fetch_factor)

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        name = f"retrieval_{self.metric_name}@{self.top_k}"

        retriever = self._retriever or context.get("retriever")
        if retriever is None:
            return EvalScore(name, 0.0, "no retriever configured")

        if _is_factual(case):
            # Measure the SAME filtered set production would inject: fetch a wider
            # candidate pool, normalize per-method, drop below min_score, truncate.
            fetch_k = max(self.top_k * self.fetch_factor, self.top_k)
            results = await retriever.retrieve(case.user_message, top_k=fetch_k)
            if self.normalize:
                results = normalize_scores(results)
            results = apply_min_score(results, self.min_score, self.top_k)
        else:
            # Gating negative: measure RAW retriever over-firing (a Rec 1 concern),
            # unaffected by Rec 2 filtering.
            results = await retriever.retrieve(case.user_message, top_k=self.top_k)

        metrics = compute_retrieval_metrics(results, case)

        if metrics.get("no_gold"):
            return EvalScore(name, 1.0, "no gold labels (source_doc/key_facts)")

        value = metrics.get(self.metric_name)
        if value is None:
            # Metric does not apply to this case type (e.g. gate_noise on a
            # factual case, or recall on a gating negative). Neutral contribution.
            return EvalScore(name, 1.0, f"{self.metric_name} n/a for this case type")

        reason = f"{self.metric_name}={value} over {metrics['retrieved']} chunks"
        return EvalScore(name, float(value), reason)
