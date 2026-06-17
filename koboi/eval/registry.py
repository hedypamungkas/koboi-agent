"""koboi/eval/registry.py -- Named scorer factory registry.

Allows config-driven eval composition by registering scorer factories
by name, then creating them with keyword arguments from YAML config.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from koboi.eval.scorers.base import BaseScorer

_logger = logging.getLogger(__name__)


class ScorerRegistry:
    """Registry of named scorer factories for config-driven eval composition."""

    _factories: dict[str, Callable[..., BaseScorer]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseScorer]) -> None:
        cls._factories[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseScorer:
        if name not in cls._factories:
            raise ValueError(
                f"Unknown scorer '{name}'. Available: {cls.list_available()}"
            )
        return cls._factories[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._factories.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered factories. Useful for test isolation."""
        cls._factories.clear()

    @classmethod
    def from_config(cls, scorer_configs: list[dict[str, Any]]) -> list[BaseScorer]:
        """Build scorer list from config dicts.

        Each dict must have a 'name' key. Remaining keys are passed as kwargs:
            [{"name": "tool_usage"}, {"name": "llm_judge", "client": client}]
        """
        scorers: list[BaseScorer] = []
        for cfg in scorer_configs:
            name = cfg.get("name")
            if not name:
                _logger.warning("Scorer config missing 'name', skipping: %s", cfg)
                continue
            kwargs = {k: v for k, v in cfg.items() if k != "name"}
            try:
                scorers.append(cls.create(name, **kwargs))
            except (ValueError, TypeError) as e:
                _logger.warning("Failed to create scorer '%s': %s", name, e)
        return scorers


def register_default_scorers() -> None:
    """Register all built-in scorers. Called once at import time."""
    from koboi.eval.scorers.base import (
        ToolUsageScorer,
        KeywordPresenceScorer,
        OutputLengthScorer,
        IterationEfficiencyScorer,
        HealthScoreScorer,
        LLMJudgeScorer,
        CostScorer,
    )

    ScorerRegistry.register("tool_usage", lambda: ToolUsageScorer())
    ScorerRegistry.register("keyword_presence", lambda: KeywordPresenceScorer())
    ScorerRegistry.register("output_length", lambda **kw: OutputLengthScorer(**kw))
    ScorerRegistry.register("iteration_efficiency", lambda **kw: IterationEfficiencyScorer(**kw))
    ScorerRegistry.register("health_score", lambda: HealthScoreScorer())
    ScorerRegistry.register("llm_judge", lambda **kw: LLMJudgeScorer(**kw))
    ScorerRegistry.register("cost", lambda **kw: CostScorer(**kw))


def register_framework_scorers() -> None:
    """Register framework-specific scorers (fail-open if deps missing)."""
    # BFCL
    try:
        from koboi.eval.scorers.bfcl_scorer import ToolCallingScorer
        ScorerRegistry.register("tool_calling_accuracy", lambda **kw: ToolCallingScorer(**kw))
    except ImportError:
        pass

    # RAGAS
    try:
        from koboi.eval.scorers.ragas_scorer import RAGASScorer, RAGASCompositeScorer
        ScorerRegistry.register("ragas_faithfulness", lambda **kw: RAGASScorer("faithfulness", **kw))
        ScorerRegistry.register("ragas_relevancy", lambda **kw: RAGASScorer("answer_relevancy", **kw))
        ScorerRegistry.register("ragas_precision", lambda **kw: RAGASScorer("context_precision", **kw))
        ScorerRegistry.register("ragas_recall", lambda **kw: RAGASScorer("context_recall", **kw))
        ScorerRegistry.register("ragas_composite", lambda **kw: RAGASCompositeScorer(**kw))
    except ImportError:
        pass

    # GAIA
    try:
        from koboi.eval.scorers.gaia_scorer import GAIAVerificationScorer
        ScorerRegistry.register("gaia_verification", lambda **kw: GAIAVerificationScorer(**kw))
    except ImportError:
        pass

    # SWE-bench
    try:
        from koboi.eval.scorers.swe_bench_scorer import PatchGenerationScorer
        ScorerRegistry.register("patch_generation", lambda **kw: PatchGenerationScorer(**kw))
    except ImportError:
        pass

    # DeepEval
    try:
        from koboi.eval.scorers.deepeval_scorer import (
            DeepEvalScorer, DeepEvalAgenticScorer, DeepEvalSafetyScorer,
        )
        ScorerRegistry.register("deepeval_task_completion", lambda **kw: DeepEvalScorer("task_completion", **kw))
        ScorerRegistry.register("deepeval_tool_correctness", lambda **kw: DeepEvalScorer("tool_correctness", **kw))
        ScorerRegistry.register("deepeval_hallucination", lambda **kw: DeepEvalScorer("hallucination", **kw))
        ScorerRegistry.register("deepeval_agentic", lambda **kw: DeepEvalAgenticScorer(**kw))
        ScorerRegistry.register("deepeval_safety", lambda **kw: DeepEvalSafetyScorer(**kw))
    except ImportError:
        pass
