# koboi/eval/ -- Evaluation framework

## What this is
Evaluation runner for benchmarking agent quality across multiple frameworks.

## Key files
```
runner.py           EvalRunner -- executes eval cases against an agent
config.py           EvalConfig -- eval suite configuration
registry.py         ScorerRegistry and LoaderRegistry
regression.py       RegressionTracker (compare against baseline)
```

## Loaders (`loaders/`)
```
bfcl_loader.py        BFCL (Berkeley Function Calling Leaderboard)
gaia_loader.py        GAIA benchmark
swe_bench_loader.py   SWE-bench
ragas_generator.py    RAGAS test case generator
```

## Scorers (`scorers/`)
```
base.py               BaseScorer ABC + 7 built-in scorers:
                        ToolUsageScorer, KeywordPresenceScorer, OutputLengthScorer,
                        IterationEfficiencyScorer, HealthScoreScorer, LLMJudgeScorer, CostScorer
bfcl_scorer.py        BFCL function-calling accuracy
gaia_scorer.py        GAIA exact-match
swe_bench_scorer.py   SWE-bench patch-apply
ragas_scorer.py       RAGAS faithfulness/relevancy
deepeval_scorer.py    DeepEval integration
```

## How to run evals
See `examples/21_eval_suite.py` and `configs/eval_suite.yaml`.

## How to add a scorer
1. Create a class in `scorers/` that inherits from `base.BaseScorer`
2. Implement `score(case: EvalCase, output: str) -> EvalScore`
3. Register in `registry.py`

## How to add a loader
1. Create a class in `loaders/` that inherits from `DatasetLoader`
2. Implement `load(source: str) -> list[EvalCase]` and `framework_name() -> str`
3. Register in `registry.py`
