"""koboi/eval/loaders/ -- Dataset loaders for benchmark frameworks.

Each loader converts an external dataset format into a list[EvalCase].
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from koboi.types import EvalCase

_logger = logging.getLogger(__name__)


class DatasetLoader(ABC):
    """Base class for loading benchmark datasets into EvalCase lists."""

    @abstractmethod
    async def load(self, source: str, **kwargs: Any) -> list[EvalCase]: ...

    @abstractmethod
    def framework_name(self) -> str: ...


class YAMLLoader(DatasetLoader):
    """Load EvalCases from YAML files."""

    async def load(self, source: str, **kwargs: Any) -> list[EvalCase]:
        path = Path(source)
        if path.is_dir():
            cases: list[EvalCase] = []
            for f in sorted(path.glob("*.yaml")):
                cases.extend(await self._load_file(f))
            return cases
        return await self._load_file(path)

    async def _load_file(self, path: Path) -> list[EvalCase]:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        if isinstance(data, dict):
            data = data.get("cases", [data])
        if not isinstance(data, list):
            return []

        cases: list[EvalCase] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            # Support both "user_message" (canonical) and "input" (legacy/short alias)
            user_message = item.get("user_message") or item.get("input", "")
            cases.append(
                EvalCase(
                    name=item.get("name", "unnamed"),
                    user_message=user_message,
                    expected_tools=item.get("expected_tools", []),
                    expected_keywords=item.get("expected_keywords", []),
                    max_iterations=item.get("max_iterations", 10),
                    tags=item.get("tags", []),
                    expected_answer=item.get("expected_answer"),
                    context_docs=item.get("context_docs", []),
                    metadata=item.get("metadata", {}),
                )
            )
        return cases

    def framework_name(self) -> str:
        return "yaml"


class LoaderRegistry:
    """Registry of dataset loaders by framework name."""

    _loaders: dict[str, DatasetLoader] = {}

    @classmethod
    def register(cls, name: str, loader: DatasetLoader) -> None:
        cls._loaders[name] = loader

    @classmethod
    def get(cls, name: str) -> DatasetLoader:
        if name not in cls._loaders:
            raise ValueError(f"Unknown loader '{name}'. Available: {cls.list_available()}")
        return cls._loaders[name]

    @classmethod
    async def load(cls, framework: str, source: str, **kwargs: Any) -> list[EvalCase]:
        loader = cls.get(framework)
        return await loader.load(source, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._loaders.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered loaders. Useful for test isolation."""
        cls._loaders.clear()


def register_default_loaders() -> None:
    """Register built-in loaders. Called once at import time."""
    LoaderRegistry.register("yaml", YAMLLoader())

    # BFCL - no extra deps
    try:
        from koboi.eval.loaders.bfcl_loader import BFCLLoader

        LoaderRegistry.register("bfcl", BFCLLoader())
    except ImportError:
        pass

    # GAIA - needs `datasets` package
    try:
        from koboi.eval.loaders.gaia_loader import GAIALoader

        LoaderRegistry.register("gaia", GAIALoader())
    except ImportError:
        pass

    # SWE-bench - needs `datasets` package
    try:
        from koboi.eval.loaders.swe_bench_loader import SWEBenchLoader

        LoaderRegistry.register("swe-bench", SWEBenchLoader())
    except ImportError:
        pass

    # TyDi QA (Indonesian) - native ID benchmark, needs `datasets` package
    try:
        from koboi.eval.loaders.tydiqa_id_loader import TyDiQAIDLoader

        LoaderRegistry.register("tydiqa-id", TyDiQAIDLoader())
    except ImportError:
        pass
