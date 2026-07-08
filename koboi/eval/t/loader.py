"""koboi/eval/t/loader.py -- Discover and import ``*.eval.py`` test files."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from collections.abc import Awaitable, Callable

if TYPE_CHECKING:
    from koboi.eval.t.context import TestContext
    from koboi.types import AgentResponse

_DEFAULT_GLOB = "**/*.eval.py"

TestFunc = Callable[["TestContext"], Awaitable[None]]


@dataclass
class LoadedTest:
    """A discovered ``async def test_*(t)`` and its module-level configuration."""

    file: Path
    func_name: str
    func: TestFunc
    config: str | dict | None = None
    mock_responses: list[AgentResponse] | None = None
    use_mock: bool = False
    tags: list[str] = field(default_factory=list)
    timeout: float | None = None

    @property
    def case_name(self) -> str:
        return f"{self.file.stem}::{self.func_name}"


class PythonTestLoader:
    """Discovers ``*.eval.py`` files, imports each in isolation, and collects ``async def test_*``.

    Recognized module-level configuration attributes:

    - ``CONFIG``: agent YAML path or dict (live mode).
    - ``MOCK_RESPONSES``: list of :class:`~koboi.types.AgentResponse` (selects mock mode).
    - ``USE_MOCK``: bool, force mock mode.
    - ``TAGS``: list[str] for filtering.
    - ``TIMEOUT``: float seconds per test.
    """

    def __init__(self, glob: str = _DEFAULT_GLOB):
        self._glob = glob

    def discover(self, path: str | Path) -> list[LoadedTest]:
        # A pointed-at file is loaded directly (glob only governs directory expansion).
        root = Path(path)
        files = [root] if root.is_file() else sorted(root.glob(self._glob))
        tests: list[LoadedTest] = []
        for file in files:
            tests.extend(self._load_file(file))
        return tests

    def _load_file(self, file: Path) -> list[LoadedTest]:
        module = self._import_module(file)
        config = getattr(module, "CONFIG", None)
        mock_responses = getattr(module, "MOCK_RESPONSES", None)
        use_mock = bool(getattr(module, "USE_MOCK", False)) or bool(mock_responses)
        tags = list(getattr(module, "TAGS", []) or [])
        timeout = getattr(module, "TIMEOUT", None)

        tests: list[LoadedTest] = []
        for name, func in inspect.getmembers(module, predicate=inspect.iscoroutinefunction):
            if not name.startswith("test"):
                continue
            signature = inspect.signature(func)
            if len(signature.parameters) < 1:
                continue
            tests.append(
                LoadedTest(
                    file=file,
                    func_name=name,
                    func=func,
                    config=config,
                    mock_responses=mock_responses,
                    use_mock=use_mock,
                    tags=list(tags),
                    timeout=timeout,
                )
            )
        return tests

    def _import_module(self, file: Path) -> Any:
        stem = file.stem.replace("-", "_").replace(".", "_")
        digest = hashlib.sha256(str(file.resolve()).encode()).hexdigest()[:8]
        module_name = f"koboi_eval_t_{stem}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load eval module from {file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def discover(path: str | Path, *, glob: str = _DEFAULT_GLOB) -> list[LoadedTest]:
    """Convenience: discover ``*.eval.py`` tests under ``path``."""
    return PythonTestLoader(glob=glob).discover(path)
