# koboi/eval/ -- Evaluation framework

## What this is
Evaluation runner for benchmarking agent quality across multiple frameworks.

## Key files
```
runner.py           EvalRunner -- executes eval cases against an agent
config.py           EvalConfig -- eval suite configuration
registry.py         ScorerRegistry (LoaderRegistry lives in loaders/__init__.py)
regression.py       RegressionTracker (compare against baseline)
```

## Loaders (`loaders/`)
```
bfcl_loader.py        BFCL (Berkeley Function Calling Leaderboard)
gaia_loader.py        GAIA benchmark
swe_bench_loader.py   SWE-bench
ragas_generator.py    RAGAS test case generator
tydiqa_id_loader.py   TyDi QA Indonesian (native secondary_task; registered "tydiqa-id"; needs `datasets`)
```

## Scorers (`scorers/`)
```
base.py               BaseScorer ABC + 11 built-in scorers:
                        ToolUsageScorer, KeywordPresenceScorer, OutputLengthScorer,
                        IterationEfficiencyScorer, HealthScoreScorer, LLMJudgeScorer, CostScorer,
                        RAGNoiseScorer, ContextEfficiencyScorer, ToolSelectionScorer, TokenEfficiencyScorer
bfcl_scorer.py        BFCL function-calling accuracy
gaia_scorer.py        GAIA exact-match
swe_bench_scorer.py   SWE-bench patch-apply
ragas_scorer.py       RAGAS faithfulness/relevancy
deepeval_scorer.py    DeepEval integration
retrieval_metric.py   Mock-safe IR ranking: RetrievalMetricScorer (recall@k/precision@k/mrr/ndcg@k/hit) -- stdlib-only
citation_grounding.py Mock-safe citation resolution: CitationGroundingScorer (ALCE-style [n]/[Source:x] -> chunk)
ci.py                 Mock-safe bootstrap CI: BootstrapCIScorer + bootstrap_ci() (95% lower-bound gating)
skill_scorer.py       Skill scorer: trigger_accuracy only (routing_accuracy + token_overhead were removed)
deep_research_scorer.py RAGAS faithfulness over DYNAMIC run-derived sources (reads context['research_sources']
                      -- the report's gathered source text, surfaced from RunResult.metadata); fail-open
recency_scorer.py     Heuristic recency proxy (regex year extraction from source texts + report) for
                      stale-knowledge leak detection; free/deterministic; fail-open
```

## deep_research evals
deep_research is an orchestration config (`core=None`) so `--mock` is unsupported for it; the
`t` runner auto-creates a `DispatchingClient` (content-dispatching, not sequential) for orchestration
configs so the loop runs deterministically without an API key (`deep_research_mock.eval.py`).
Live structural + RAGAS-faithfulness evals: `deep_research_citations.eval.py` /
`deep_research_faithfulness.eval.py`. **Production smoke** (Tier 2, real Firecrawl + `gpt-5.4`,
env-gated, GATE-severity bar): `evals/deep_research_prod_{multifaceted,recency,comparative,adversarial}.eval.py`.
The passing grade + run commands are documented in `docs/deep-research-smoke.md`.

## How to run evals
See `examples/21_eval_suite.py` and `configs/eval_suite.yaml`.

## How to add a scorer
1. Create a class in `scorers/` that inherits from `base.BaseScorer`
2. Implement `async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore`
3. Register in `registry.py`

## How to add a loader
1. Create a class in `loaders/` that inherits from `DatasetLoader`
2. Implement `async def load(self, source: str, **kwargs: Any) -> list[EvalCase]` and `framework_name() -> str`
3. Register via `LoaderRegistry.register(...)` in `loaders/__init__.py` (add to `register_default_loaders()`)

## `t` authoring surface (`t/`) -- eve-style, test-shaped, CI-native evals
Write `evals/**/*.eval.py` files exporting `async def test_*(t)` functions. The
`t` object drives the agent and records assertions that fold into real
`EvalResult`s with **gate/soft** severity. `EvalRunner.format_results` and
`RegressionTracker` work on the output unchanged.
```
t/__init__.py        Public API: run_tests, run_tests_sync, TestContext, Severity, scripted_response/scripted_tool_call/ScriptedClient, matchers (Contains/Equals/Regex/Matches/Truth)
t/assertions.py      Severity(GATE/SOFT), Matcher ABC + built-ins, RecordedAssertion
t/context.py         TestContext (the `t`) -- send/calledTool/check/judge (record-and-collect);
                     RAG assertions: retrievedChunk (substring presence = Hit@k=∞),
                     rankingMetric (rank-aware recall@k/mrr/ndcg@k/precision@k/hit over
                     rag_results rank order), citationResolves ([n]/[Source:x]->chunk),
                     abstains (empty retrieval OR refusal marker). _build_context()
                     forwards rag_results + rag_augmented so retrieval/citation/noise
                     scorers work via t.judge.
t/mock.py            ScriptedClient + scripted_response/scripted_tool_call builders
t/loader.py          PythonTestLoader -- discover **/*.eval.py, import, collect test_*(t)
t/runner.py          TestRunner.run_tests -> list[EvalResult] (drives harness directly)
                     (`koboi eval-test` itself lives in cli_commands.cmd_eval_test -- no click)
```
```
# evals/weather.eval.py
from koboi.eval.t import scripted_response, scripted_tool_call, Contains
MOCK_RESPONSES = [scripted_response(None, [scripted_tool_call("get_weather", {"city": "Jakarta"})]),
                  scripted_response("Sunny, 28C")]
async def test_weather(t):
    await t.send("weather in Jakarta?")
    t.calledTool("get_weather")          # gate
    t.check(t.reply, Contains("Sunny"))  # soft
    t.completed()                        # gate
```
- Run: `koboi eval-test evals/ --mock --strict` (exit 1 on any gate failure).
- Programmatic: `await run_tests("evals/", threshold=0.6)`.
- Severity: a single **GATE** failure forces `EvalResult.passed = False`
  regardless of `overall_score`; **SOFT** assertions only lower the score.
  `t.check` defaults to SOFT; tool/turn assertions default to GATE.
- `t.judge("llm_judge"|"keyword_presence"|...)` routes through `ScorerRegistry`
  (fail-soft if the scorer/dep is unavailable).
- Binding: module-level `CONFIG` (YAML path/dict) = live; `MOCK_RESPONSES`
  (or `USE_MOCK`) = deterministic mock (no API key). `--mock`/`--config` override.

