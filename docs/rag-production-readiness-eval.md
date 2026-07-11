# RAG Production-Readiness — Evaluation Method & Justification

> Status of the consolidated RAG stack ([PR #34](#), v0.11.0: BM25, embedding/query
> cache, rerank, dedup, citations, remote sources, PDF/DOCX/HTML+table parsing,
> size-cap, HyDE + query-rewrite, metadata filter, streaming `rag_results`) against a
> defensible "production-ready" bar, and the eval-suite adjustments that make such a
> statement auditable.

**Verdict (2026-07-11):** *Not yet — but the gap is now closed for the deterministic
half and staged for the rest.* The pre-existing eval method evidenced only ~2 of 9
weighted readiness dimensions and **could not** justify a production-ready claim: it
asserted RAG retrieval as binary substring presence (`t.retrievedChunk` = Hit@k=∞) and
the shipped `RAGASScorer` was never invoked by any eval. This document defines the
rubric, the gaps, and ships **Tier 0 + Tier 1** — a mock-safe, no-API-cost HARD gate
that makes the retrieval/abstention/citation/ingestion/scoping dimensions auditable on
every PR. Faithfulness, answer-correctness, and statistical confidence remain Tier 2/3
prerequisites for an unqualified external claim.

---

## 1. The production-readiness rubric (9 weighted dimensions)

Weights are relative importance to a *safe* RAG product (grounding + ranking +
correctness dominate). "Coverage" is what the eval method evidenced **before** this
change; "Tier" is where it closes.

| # | Dimension | w | Pre-change coverage | Target (industry) | Tier |
|---|---|---|---|---|---|
| 1 | Grounding / anti-hallucination (faithfulness) | 0.18 | **none** — `RAGASScorer` shipped but never invoked | Ragas Faithfulness ≥0.9 (high-stakes) / ≥0.8 | 2 |
| 2 | Retrieval ranking quality | 0.17 | binary substring only (Hit@k=∞); no Recall@k/MRR/nDCG/qrels | Recall@10 ≥0.8, MRR ≥0.6, nDCG@10 ≥0.7 | **0+1** ✅ |
| 3 | Answer correctness & relevance (end-to-end) | 0.13 | **none** — the one RAG eval explicitly skips the answer | FactualCorrectness F1 ≥0.75, Relevancy ≥0.7 | 2 |
| 4 | Ingestion fidelity (parsing/chunking/format) | 0.10 | component pytest only; no real-format extraction gate | 100% in-scope formats parse; chunk-boundary Hit@k overlap≥1 | **0+1** ✅ |
| 5 | Negative rejection / abstention | 0.09 | **none** | abstain-rate ≥0.9 (OOS); answerable acc ≥0.8 | **0+1** ✅ |
| 6 | Noise robustness | 0.09 | `RAGNoiseScorer` shipped, never used | Noise Sensitivity ≤0.2; faithfulness drop ≤5% | **1** (mock) / 2 (live) ✅ |
| 7 | Robustness & graceful degradation | 0.08 | **strong** (fallback + SSRF + size-cap + 6 regression guards) | zero ingestion crashes; every fallback warns | already ✅ |
| 8 | Performance / cost / scaling | 0.08 | cache correctness yes, SLA thresholds no | p95 latency budget; embed-once reuse verified | partial |
| 9 | Statistical confidence / eval validity | 0.08 | **none** — point estimate on 1–2 queries | 95% CI half-width ≤0.10; lower bound above bar; N≥100 | **1** (mech.) / 3 |

Pre-change: ~2 of 9 well-evidenced (robustness + ingestion-format); the 3 heaviest
(faithfulness 0.18, ranking 0.17, answer-correctness 0.13) **failed**.

## 2. Gap analysis (what the method could not evidence)

1. **Faithfulness had zero eval evidence** despite `RAGASScorer` existing → Tier 2.
2. **Answer correctness was explicitly not asserted** ("we assert on retrieval, not the
   answer") → Tier 2.
3. **Retrieval = binary presence only**; no quantified ranking metric or golden set →
   **closed Tier 0+1**.
4. **Semantic/Hybrid/HyDE ranking untested with real embeddings** (all tests use mock
   or `client=None`) → Tier 2.
5. **No abstention/OOS eval** → **closed Tier 1** (mock leg; live refusal leg Tier 2).
6. **No noise-robustness eval** though `RAGNoiseScorer` shipped → **closed Tier 1**
   (mock leg; live faithfulness-delta Tier 2).
7. **No statistical confidence** (point estimates on 1–2 queries) → Tier 3 (mechanism
   shipped Tier 1).
8. **Citations format-verified but not correctness-verified** (no ALCE-style
   precision/recall) → **mock leg closed Tier 1**; live leg Tier 2.
9. **Real-format extraction + chunk-boundary-split unverified** → **mock leg closed
   Tier 1** (PDF/DOCX self-skip without `[rag]` extra).
10. **rag_results reach to served `/v1/chat/stream` + guardrail-buffer interaction
    unverified** → Tier 2 (server integration test).

## 3. The 4-tier plan

- **Tier 0 — Scaffolding (mock-safe).** Deterministic scorers + `t` primitives + the
  rank-order context seam + a frozen golden qrels set. Unblocks both gates.
- **Tier 1 — Mock-safe HARD PR gate.** `koboi eval-test evals/ --mock --strict` on a
  bare install. Zero API cost, zero non-determinism. **Shipped in this change.**
- **Tier 2 — Live-LLM nightly gate.** RAGAS faithfulness/recall/relevancy/precision +
  real-embedding semantic/hybrid/HyDE + live legs of noise/abstention/citation. Needs
  `pip install -e ".[eval-ragas]"` + an LLM key. SOFT until calibrated.
- **Tier 3 — Pre-release statistical gate.** RAGASDataGenerator-seeded N≥100 golden
  suite + `BootstrapCIScorer` gating on the 95% CI **lower bound** per dimension.

## 4. What this change ships (Tier 0 + Tier 1)

### Scorers (`koboi/eval/scorers/`, stdlib-only, registered)
| File | Scorer(s) | Registered as |
|---|---|---|
| `retrieval_metric.py` | `RetrievalMetricScorer` + `recall_at_k`/`precision_at_k`/`mrr`/`ndcg_at_k`/`hit_rate` | `retrieval_metric`, `retrieval_{recall,precision,hit,mrr,ndcg}` |
| `citation_grounding.py` | `CitationGroundingScorer` + `citation_precision` (ALCE-style resolution) | `citation_grounding` |
| `ci.py` | `BootstrapCIScorer` + `bootstrap_ci` (seedable percentile bootstrap) | `bootstrap_ci` |

### `t` primitives (`koboi/eval/t/context.py`)
- `t.rankingMetric(gold, k=10, metric="recall", min_score=1.0)` — rank-aware counterpart
  to `t.retrievedChunk` (which is Hit@k=∞). Metrics: recall/precision/hit/mrr/ndcg.
- `t.citationResolves(n=None)` — every `[n]`/`[Source: x]` marker must resolve to a
  retrieved chunk (mock-safe format-vs-correctness).
- `t.abstains()` — empty retrieval OR a refusal marker.
- `_build_context()` now forwards `rag_results` (rank order) + `rag_augmented`, so the
  new scorers **and** the existing `RAGNoiseScorer` work via `t.judge`.

### Frozen golden set
- `evals/fixtures/acme_qrels.json` — 24 needle-verified queries over the shipped Acme
  corpus (hand-authored; no generation-time LLM).

### Mock-safe eval files (`evals/`, run under `--mock --strict`)
| File | Dimension | Tests (count) |
|---|---|---|
| `rag_ranking.eval.py` | Retrieval ranking | recall@10, ndcg@10, gold-within-window ×2 (4) |
| `rag_ranking_ci.eval.py` | Statistical confidence (retrieval leg) | bootstrap 95% CI lower bound on Recall@10 (1) |
| `rag_abstention.eval.py` | Negative rejection / abstention | zero-overlap empty, realistic-OOS abstains, in-corpus coverage, relevance-threshold sweep (4) |
| `rag_noise_robustness.eval.py` | Noise robustness / disambiguation | permanent-leave ranks above PTO noise, top-5 precision (2) |
| `rag_citations.eval.py` | Citation grounding | `[1]` resolves, all resolve, dangling `[9]` detected (3) |
| `rag_ingestion_fidelity.eval.py` | Ingestion fidelity | text/html parse, magic-byte detect, pdf/docx registered-iff-extra, chunker invariants (5) |
| `rag_metadata_filter.eval.py` | Relevance scoping | equality scope, `$in` multi, no-match-empty (3) |

**Totals:** 22 new mock-safe tests (33/33 with the existing samples), all GATE-green.

### Threshold table (Tier 1, provisional — calibrate after first real run)
| Dimension | Metric | Target |
|---|---|---|
| Retrieval ranking (keyword default) | Recall@10 / hit@10 | =1.0 (gold in window) |
| Retrieval ranking | nDCG@10 | ≥0.5 |
| Retrieval ranking | MRR | reported SOFT (real rank ~6 for weak entity queries) |
| Statistical confidence (N=24) | Recall@10 95%-CI lower bound | ≥0.80 (half-width ≤0.40 SOFT) |
| Noise robustness | permanent-leave MRR / top-5 precision | ≥0.5 / ≥0.2 |
| Citation grounding | resolution precision | =1.0 (no dangling markers) |
| Abstention | zero-overlap retrieval / refusal | empty + abstains |
| Ingestion | parse + magic-byte + registration contract | all pass (pdf/docx self-skip) |
| Metadata scoping | equality / `$in` / no-match | 100% |

## 5. Real findings the new eval surfaced (not pre-known)

The gate is not vacuous — running it produced two genuine retrieval-quality
observations worth tracking:

1. **Keyword TF-IDF ranks weak-entity queries mid-window.** "Who is the CEO?" and
   "parental leave duration" retrieve the gold chunk at **rank ~6**, not rank 1: the
   discriminative term ("CEO") has low TF in a chunk full of low-IDF shared terms
   ("Acme"/"Corp"). Surfaced as SOFT MRR (reported, not gating) so it's visible without
   blocking the deterministic gate. A BM25 or semantic retriever ranks these higher
   (Tier 2 will evidence that).
2. **The keyword retriever applies no stopword filtering.** A realistic OOS query
   ("...mating rituals of deep-sea anglerfish") retrieves ~6 spurious chunks because
   "the"/"of" match corpus text. The agent still abstains (refusal marker), so
   `t.abstains` passes — but the noise-injection is a real retrieval weakness for the
   noise-robustness dimension. Documented in `rag_abstention.eval.py`.

## 6. The honest claim ladder

| After | Defensible statement |
|---|---|
| **Tier 0+1 (this change)** | *"Retrieval ranking, abstention-retrieval, citation resolution, ingestion fidelity, and metadata scoping will not silently regress on any PR — evidenced by a mock-safe HARD gate at zero API cost, with a retrieval-side 95%-CI leg."* |
| Tier 2 | + *"Faithfulness ≥0.9 and end-to-end answer correctness are evidenced nightly over the Acme corpus (RAGAS), disclosed as non-deterministic until calibrated."* |
| Tier 3 | + *"Statistically defensible at N≥100 with 95%-CI lower bounds per dimension."* |
| (beyond) | A human-annotated PPI tier is the prerequisite for an **unqualified external** "RAG is production-ready" assertion — N=100 bootstrap half-width (~±0.10) is too wide for high-stakes ≥0.9 claims and judge-LLM determinism is unbounded. |

**Ceiling:** the method now supports a **CI-gated retrieval-safety + nightly-evidenced
grounding** claim (defensible for internal/pre-production), not yet an unqualified
external production assertion.

## 7. Verification (2026-07-11)

- `koboi eval-test evals/ --mock --strict` → **33/33 passed**.
- `pytest` → **3197 passed / 0 failed / 178 skipped**, coverage **83%**.
- `ruff check koboi/ evals/` → clean.
- `mypy koboi/` → clean (205 files).
- `bandit -r koboi/ -c pyproject.toml` → 0 issues (neutral vs main).
