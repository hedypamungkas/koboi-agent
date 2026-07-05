from koboi.types import EvalCase, EvalScore, EvalResult
from koboi.eval.runner import EvalRunner
from koboi.eval.scorers.base import (
    BaseScorer,
    ToolUsageScorer,
    KeywordPresenceScorer,
    OutputLengthScorer,
    IterationEfficiencyScorer,
    HealthScoreScorer,
    LLMJudgeScorer,
    CostScorer,
)
from koboi.eval.scorers.skill_scorer import SkillTriggerAccuracyScorer
from koboi.eval.registry import ScorerRegistry, register_default_scorers, register_framework_scorers
from koboi.eval.config import EvalConfig
from koboi.eval.regression import RegressionTracker
from koboi.eval.loaders import LoaderRegistry, DatasetLoader, register_default_loaders

__all__ = [
    "EvalCase",
    "EvalScore",
    "EvalResult",
    "EvalRunner",
    "EvalConfig",
    "BaseScorer",
    "ToolUsageScorer",
    "KeywordPresenceScorer",
    "OutputLengthScorer",
    "IterationEfficiencyScorer",
    "HealthScoreScorer",
    "LLMJudgeScorer",
    "CostScorer",
    "SkillTriggerAccuracyScorer",
    "ScorerRegistry",
    "register_default_scorers",
    "register_framework_scorers",
    "RegressionTracker",
    "LoaderRegistry",
    "DatasetLoader",
    "register_default_loaders",
]

# Auto-register all scorers and loaders on import
register_default_scorers()
register_framework_scorers()
register_default_loaders()

# eve-style `t` authoring surface (additive; see koboi/eval/t/)
from koboi.eval.t import (  # noqa: E402
    Contains,
    Equals,
    Matcher,
    RecordedAssertion,
    Regex,
    Severity,
    TestContext,
    run_tests,
    run_tests_sync,
    scripted_response,
    scripted_tool_call,
)

__all__ += [
    "Contains",
    "Equals",
    "Matcher",
    "RecordedAssertion",
    "Regex",
    "Severity",
    "TestContext",
    "run_tests",
    "run_tests_sync",
    "scripted_response",
    "scripted_tool_call",
]
