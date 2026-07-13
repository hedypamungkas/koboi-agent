# Deep research production smoke

How we prove deep_research is production-grade before a release. Two tiers:

- **Tier 1 — mechanics** (CI, every PR): deterministic, `$0`, no keys. Fetch-robustness only
  (budget/resume/durability already covered by the main suite).
- **Tier 2 — real-provider quality** (pre-release/nightly, manual): live Firecrawl + `gpt-5.4`,
  real cost + ~10 min/scenario. The quantitative **passing grade**.

## The passing grade (Tier 2)

Every Tier 2 scenario asserts the rows relevant to its dimension. A scenario **passes** iff all
its asserted rows are green.

| Metric | Source | Threshold | Grounding |
|---|---|---|---|
| `plan_nodes` | `RunResult.metadata` | ≥ 4 | good runs 6-7; the shallow-report regression was 2 |
| `coverage` | `RunResult.metadata` | ≥ 0.6 **OR** `depth == max_depth` | 0.76-0.87 good; 0.16 bad; "drilled fully" also passes |
| `nodes_failed` | `RunResult.metadata` | == 0 | a node must not crash |
| `used_searches` / `used_fetches` | `RunResult.metadata` | ≤ configured cap | budget adherence |
| source count | `len(metadata['research_sources'])` | ≥ 3 | single-source dependency is a smell |
| inline citations | `t.citation` | ≥ 5 (Q1) / ≥ 4 (Q3), all resolve | structural grounding |
| report length | `len(reply)` | ≥ 8000 chars (Q1) | bad run was 6.7K (shallow) |
| faithfulness | `t.judge('deep_research_faithfulness')` | ≥ 0.7 | RAGAS claim-grounding (needs `[eval-ragas]`) |
| recency | `t.judge(RecencyScorer)` | ≥ 0.5 sources recent (Q2) | stale-knowledge leak |
| abstention | `t.abstains()` | present (Q4) | no hallucination under uncertainty |
| completion | `t.completed` | success | baseline |

`plan_nodes` / `used_searches` / `used_fetches` / `nodes_failed` are surfaced by
`Orchestrator._run_deep_research` into `OrchestrationCompleteEvent.metadata` (and thus
`RunResult.metadata`).

## Tier 1 — mechanics (`pytest`, CI-safe)

`tests/orchestration/test_deep_research_mechanics.py`:

- **M1 empty fetch** — fetch returns empty content (paywall/JS) → run still completes.
- **M2 raising fetch** — fetch provider raises → run still completes (`web_fetch` catches it).

Run: `pytest tests/orchestration/test_deep_research_mechanics.py -q` (no keys needed).

(Budget hard-stop, resume, and session-message durability are covered in
`tests/orchestration/test_deep_research.py` + `tests/test_server_pool.py` — not duplicated here.)

## Tier 2 — real-provider quality (`koboi eval-test`, pre-release)

Env-gated; self-skip without keys. Provider = Firecrawl (search + fetch).

| Scenario | File | Catches |
|---|---|---|
| Q1 multi-faceted factual | `evals/deep_research_prod_multifaceted.eval.py` | shallow-report regression (the 1/3 flake) — full bar |
| Q2 recency | `evals/deep_research_prod_recency.eval.py` | stale-knowledge leakage |
| Q3 comparative | `evals/deep_research_prod_comparative.eval.py` | one-sided research |
| Q4 adversarial / unanswerable | `evals/deep_research_prod_adversarial.eval.py` | hallucination under uncertainty |

Run (one file, real keys):
```
FIRECRAWL_API_KEY=... OPENAI_API_KEY=... OPENAI_MODEL=gpt-5.4 \
  koboi eval-test evals/deep_research_prod_multifaceted.eval.py --strict
```

**Q1 is the load-bearing regression test — re-run it 3× per release** (the shallow-report bug
was a 1/3 flake; the fix must hold every run).

## Recency scorer

`koboi/eval/scorers/recency_scorer.py` (`RecencyScorer`) is a **heuristic proxy**: regex-extracts
4-digit years from source texts + the report, scores the fraction within `recent_years` of today.
Cheap (no LLM call) + deterministic; catches the dominant failure (no recent-year signal). For a
stricter bar, swap in an LLM judge.
