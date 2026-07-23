"""koboi/eval/registry.py -- Named scorer factory registry.

Allows config-driven eval composition by registering scorer factories
by name, then creating them with keyword arguments from YAML config.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

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
            raise ValueError(f"Unknown scorer '{name}'. Available: {cls.list_available()}")
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
        RAGNoiseScorer,
        ContextEfficiencyScorer,
        ToolSelectionScorer,
        TokenEfficiencyScorer,
    )

    ScorerRegistry.register("tool_usage", lambda: ToolUsageScorer())
    ScorerRegistry.register("keyword_presence", lambda: KeywordPresenceScorer())
    ScorerRegistry.register("output_length", lambda **kw: OutputLengthScorer(**kw))
    ScorerRegistry.register("iteration_efficiency", lambda **kw: IterationEfficiencyScorer(**kw))
    ScorerRegistry.register("health_score", lambda: HealthScoreScorer())
    ScorerRegistry.register("llm_judge", lambda **kw: LLMJudgeScorer(**kw))
    ScorerRegistry.register("cost", lambda **kw: CostScorer(**kw))
    ScorerRegistry.register("rag_noise", lambda: RAGNoiseScorer())
    ScorerRegistry.register("context_efficiency", lambda: ContextEfficiencyScorer())
    ScorerRegistry.register("tool_selection", lambda: ToolSelectionScorer())
    ScorerRegistry.register("token_efficiency", lambda **kw: TokenEfficiencyScorer(**kw))

    # Mock-safe RAG retrieval/citation/CI scorers (stdlib-only; no optional dep).
    from koboi.eval.scorers.retrieval_metric import RetrievalMetricScorer
    from koboi.eval.scorers.citation_grounding import CitationGroundingScorer
    from koboi.eval.scorers.ci import BootstrapCIScorer

    ScorerRegistry.register("retrieval_metric", lambda **kw: RetrievalMetricScorer(**kw))

    def _retrieval_factory(metric_name: str):
        def _factory(**kw: Any) -> RetrievalMetricScorer:
            kw.pop("metric", None)
            return RetrievalMetricScorer(metric=metric_name, **kw)

        return _factory

    for _metric in ("recall", "precision", "hit", "mrr", "ndcg"):
        ScorerRegistry.register(f"retrieval_{_metric}", _retrieval_factory(_metric))
    ScorerRegistry.register("citation_grounding", lambda **kw: CitationGroundingScorer(**kw))
    ScorerRegistry.register("bootstrap_ci", lambda **kw: BootstrapCIScorer(**kw))

    # Coding-harness ground truth: runs the case's real test suite in a
    # sandbox, gates on exit code (stdlib + sandbox only; no optional dep).
    from koboi.eval.scorers.test_suite import TestSuiteScorer

    ScorerRegistry.register("test_suite", lambda **kw: TestSuiteScorer(**kw))

    # Skill-specific scorers
    try:
        from koboi.eval.scorers.skill_scorer import SkillTriggerAccuracyScorer

        ScorerRegistry.register("skill_trigger_accuracy", lambda: SkillTriggerAccuracyScorer())
    except ImportError:
        pass


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
        ScorerRegistry.register("ragas_factual_correctness", lambda **kw: RAGASScorer("factual_correctness", **kw))
        ScorerRegistry.register("ragas_composite", lambda **kw: RAGASCompositeScorer(**kw))

        # W6.1: deep_research faithfulness (reads dynamic context['research_sources'] not static case.context_docs)
        from koboi.eval.scorers.deep_research_scorer import DeepResearchFaithfulnessScorer

        ScorerRegistry.register("deep_research_faithfulness", lambda **kw: DeepResearchFaithfulnessScorer())
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
            DeepEvalScorer,
            DeepEvalAgenticScorer,
            DeepEvalSafetyScorer,
        )

        ScorerRegistry.register("deepeval_task_completion", lambda **kw: DeepEvalScorer("task_completion", **kw))
        ScorerRegistry.register("deepeval_tool_correctness", lambda **kw: DeepEvalScorer("tool_correctness", **kw))
        ScorerRegistry.register("deepeval_hallucination", lambda **kw: DeepEvalScorer("hallucination", **kw))
        ScorerRegistry.register("deepeval_agentic", lambda **kw: DeepEvalAgenticScorer(**kw))
        ScorerRegistry.register("deepeval_safety", lambda **kw: DeepEvalSafetyScorer(**kw))
    except ImportError:
        pass
