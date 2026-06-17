"""koboi/eval/config.py -- Config-driven eval suite configuration.

Parses the `eval` section from YAML config and builds scorers/suites.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from koboi.config import Config
    from koboi.eval.scorers.base import BaseScorer
    from koboi.types import EvalCase

_logger = logging.getLogger(__name__)


@dataclass
class SuiteConfig:
    """Configuration for a single eval suite."""
    name: str
    framework: str = "yaml"
    source: str = ""
    scorers: list[str] = field(default_factory=list)
    max_cases: int | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class RegressionConfig:
    """Configuration for regression tracking."""
    baseline_dir: str = "eval_baselines"
    alert_on_regression: bool = True
    regression_threshold: float = 0.05


class EvalConfig:
    """Parsed eval configuration from YAML."""

    def __init__(self, data: dict[str, Any]):
        self.threshold: float = data.get("threshold", 0.6)
        self.parallel: bool = data.get("parallel", False)
        self.max_concurrency: int = data.get("max_concurrency", 5)
        self.output_dir: str = data.get("output_dir", "eval_results")
        self.default_scorers: list[str] = data.get("scorers", [])

        raw_suites = data.get("suites", [])
        self.suites: list[SuiteConfig] = [
            self._parse_suite(s) for s in raw_suites
        ]

        raw_regression = data.get("regression", {})
        self.regression = RegressionConfig(
            baseline_dir=raw_regression.get("baseline_dir", "eval_baselines"),
            alert_on_regression=raw_regression.get("alert_on_regression", True),
            regression_threshold=raw_regression.get("regression_threshold", 0.05),
        )

    @staticmethod
    def _parse_suite(raw: dict[str, Any]) -> SuiteConfig:
        return SuiteConfig(
            name=raw.get("name", "unnamed"),
            framework=raw.get("framework", "yaml"),
            source=raw.get("source", ""),
            scorers=raw.get("scorers", []),
            max_cases=raw.get("max_cases"),
            tags=raw.get("tags", []),
            metadata=raw.get("metadata", {}),
        )

    @classmethod
    def from_config(cls, config: Config) -> EvalConfig:
        data = config.eval if isinstance(config.eval, dict) else {}
        return cls(data)

    def get_suite(self, name: str) -> SuiteConfig | None:
        for s in self.suites:
            if s.name == name:
                return s
        return None

    def build_scorers(self, **extra_kwargs: Any) -> list[BaseScorer]:
        """Build default scorer list from config."""
        from koboi.eval.registry import ScorerRegistry

        if not self.default_scorers:
            return []
        configs = []
        for name in self.default_scorers:
            if isinstance(name, str):
                configs.append({"name": name, **extra_kwargs})
            elif isinstance(name, dict):
                configs.append({**name, **extra_kwargs})
        return ScorerRegistry.from_config(configs)

    async def build_suite(
        self,
        suite_name: str,
        **extra_kwargs: Any,
    ) -> tuple[list[EvalCase], list[BaseScorer]]:
        """Load cases and build scorers for a named suite."""
        from koboi.eval.registry import ScorerRegistry
        from koboi.eval.loaders import LoaderRegistry

        suite = self.get_suite(suite_name)
        if not suite:
            raise ValueError(
                f"Suite '{suite_name}' not found. Available: {[s.name for s in self.suites]}"
            )

        loader = LoaderRegistry.get(suite.framework)
        cases = await loader.load(suite.source, max_cases=suite.max_cases)

        scorer_names = suite.scorers or self.default_scorers
        scorer_configs = []
        for name in scorer_names:
            if isinstance(name, str):
                scorer_configs.append({"name": name, **extra_kwargs})
            elif isinstance(name, dict):
                scorer_configs.append({**name, **extra_kwargs})
        scorers = ScorerRegistry.from_config(scorer_configs)

        return cases, scorers
