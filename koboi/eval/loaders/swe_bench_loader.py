"""koboi/eval/loaders/swe_bench_loader.py -- SWE-bench dataset loader.

Loads SWE-bench Verified dataset from HuggingFace or local Parquet
into EvalCase lists for coding agent evaluation.

Source: https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from koboi.types import EvalCase
from koboi.eval.loaders import DatasetLoader

_logger = logging.getLogger(__name__)

_SWE_BENCH_AVAILABLE = False
try:
    from datasets import load_dataset as hf_load_dataset

    _SWE_BENCH_AVAILABLE = True
except ImportError:
    pass


class SWEBenchLoader(DatasetLoader):
    """Loads SWE-bench Verified dataset from HuggingFace or local Parquet."""

    async def load(
        self,
        source: str = "SWE-bench/SWE-bench_Verified",
        split: str = "test",
        max_cases: int | None = None,
        local_only: bool = False,
        **kwargs: Any,
    ) -> list[EvalCase]:
        """Load SWE-bench dataset.

        Maps each entry to EvalCase:
          - name = "{repo}__{instance_id}"
          - user_message = problem_statement (issue text)
          - expected_answer = patch (expected diff)
          - metadata = {repo, base_commit, instance_id}
          - tags = ["swe-bench", "coding"]
          - max_iterations = 30 (coding tasks need more iterations)
        """
        if _SWE_BENCH_AVAILABLE and not local_only:
            return await self._load_from_hf(source, split, max_cases)
        return await self._load_from_local(source, max_cases)

    def framework_name(self) -> str:
        return "swe-bench"

    async def _load_from_hf(
        self,
        source: str,
        split: str,
        max_cases: int | None,
    ) -> list[EvalCase]:
        """Load from HuggingFace datasets."""
        try:
            ds = hf_load_dataset(source, split=split)  # nosec B615 - trusted public benchmark dataset
        except Exception as e:
            _logger.warning("HuggingFace load failed: %s, falling back to local", e)
            return await self._load_from_local(source, max_cases)

        cases: list[EvalCase] = []
        for i, row in enumerate(ds):
            if max_cases and len(cases) >= max_cases:
                break
            case = self._row_to_eval_case(row, i)
            if case:
                cases.append(case)

        return cases

    async def _load_from_local(
        self,
        source: str,
        max_cases: int | None,
    ) -> list[EvalCase]:
        """Load from local Parquet files."""
        path = Path(source)
        cases: list[EvalCase] = []

        if path.is_dir():
            for pq_file in sorted(path.glob("*.parquet")):
                cases.extend(await self._load_parquet(pq_file, max_cases - len(cases) if max_cases else None))
                if max_cases and len(cases) >= max_cases:
                    break
        elif path.suffix == ".parquet":
            cases = await self._load_parquet(path, max_cases)

        return cases[:max_cases] if max_cases else cases

    async def _load_parquet(
        self,
        path: Path,
        max_cases: int | None,
    ) -> list[EvalCase]:
        """Load from a single Parquet file."""
        try:
            import pandas as pd

            df = pd.read_parquet(path)
        except Exception as e:
            _logger.warning("Failed to read Parquet %s: %s", path, e)
            return []

        cases: list[EvalCase] = []
        for i, row in df.iterrows():
            if max_cases and len(cases) >= max_cases:
                break
            case = self._row_to_eval_case(row.to_dict(), i)
            if case:
                cases.append(case)

        return cases

    def _row_to_eval_case(self, row: dict, index: int) -> EvalCase | None:
        """Convert a SWE-bench dataset row to an EvalCase."""
        problem = row.get("problem_statement", row.get("text", ""))
        patch = row.get("patch", row.get("expected_patch", ""))

        if not problem:
            return None

        repo = row.get("repo", "unknown")
        instance_id = row.get("instance_id", f"swe_{index}")
        base_commit = row.get("base_commit", "")
        hints = row.get("hints_text", "")

        # Build message with context
        user_message = str(problem)
        if hints:
            user_message += f"\n\nHints:\n{hints}"

        return EvalCase(
            name=f"{repo}__{instance_id}",
            user_message=user_message,
            expected_answer=str(patch) if patch else None,
            max_iterations=30,
            tags=["swe-bench", "coding"],
            # Coding-harness fields: dataset `repo` is "owner/name"; per-instance
            # test specs (test_command) are out of scope here. metadata keys are
            # kept for back-compat with existing consumers.
            repo=f"https://github.com/{repo}.git" if repo != "unknown" else None,
            base_commit=base_commit or None,
            metadata={
                "repo": repo,
                "base_commit": base_commit,
                "instance_id": instance_id,
                "framework": "swe-bench",
            },
        )
