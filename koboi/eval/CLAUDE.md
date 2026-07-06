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
base.py               BaseScorer ABC + 11 built-in scorers:
                        ToolUsageScorer, KeywordPresenceScorer, OutputLengthScorer,
                        IterationEfficiencyScorer, HealthScoreScorer, LLMJudgeScorer, CostScorer,
                        RAGNoiseScorer, ContextEfficiencyScorer, ToolSelectionScorer, TokenEfficiencyScorer
bfcl_scorer.py        BFCL function-calling accuracy
gaia_scorer.py        GAIA exact-match
swe_bench_scorer.py   SWE-bench patch-apply
ragas_scorer.py       RAGAS faithfulness/relevancy
deepeval_scorer.py    DeepEval integration
skill_scorer.py       Skill scorer: trigger_accuracy only (routing_accuracy + token_overhead were removed)
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

## `t` authoring surface (`t/`) -- eve-style, test-shaped, CI-native evals
Write `evals/**/*.eval.py` files exporting `async def test_*(t)` functions. The
`t` object drives the agent and records assertions that fold into real
`EvalResult`s with **gate/soft** severity. `EvalRunner.format_results` and
`RegressionTracker` work on the output unchanged.
```
t/__init__.py        Public API: run_tests, run_tests_sync, TestContext, Severity, matchers
t/assertions.py      Severity(GATE/SOFT), Matcher ABC + built-ins, RecordedAssertion
t/context.py         TestContext (the `t`) -- send/calledTool/check/judge (record-and-collect)
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

