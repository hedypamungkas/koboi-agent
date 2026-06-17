"""koboi/eval/loaders/gaia_loader.py -- GAIA benchmark dataset loader.

Loads GAIA (General AI Assistants) benchmark from HuggingFace datasets
or local Parquet files into EvalCase lists.

Source: https://huggingface.co/datasets/gaia-benchmark/GAIA
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from koboi.types import EvalCase
from koboi.eval.loaders import DatasetLoader

_logger = logging.getLogger(__name__)

_GAIA_AVAILABLE = False
try:
    from datasets import load_dataset as hf_load_dataset

    _GAIA_AVAILABLE = True
except ImportError:
    pass


class GAIALoader(DatasetLoader):
    """Loads GAIA benchmark from HuggingFace datasets or local Parquet."""

    LEVELS = [1, 2, 3]

    async def load(
        self,
        source: str = "gaia-benchmark/GAIA",
        levels: list[int] | None = None,
        split: str = "validation",
        max_cases: int | None = None,
        local_only: bool = False,
        **kwargs: Any,
    ) -> list[EvalCase]:
        """Load GAIA dataset.

        Maps each entry to EvalCase:
          - name = task_id
          - user_message = Question
          - expected_answer = Final answer
          - metadata = {level, file_name, file_path}
          - file_attachments = [file_path] if present
          - tags = ["gaia", f"level-{level}"]
        """
        if _GAIA_AVAILABLE and not local_only:
            return await self._load_from_hf(source, levels, split, max_cases)
        return await self._load_from_local(source, levels, max_cases)

    def framework_name(self) -> str:
        return "gaia"

    async def _load_from_hf(
        self,
        source: str,
        levels: list[int] | None,
        split: str,
        max_cases: int | None,
    ) -> list[EvalCase]:
        """Load from HuggingFace datasets library."""
        cases: list[EvalCase] = []

        try:
            # Try loading all levels, then filter
            for level in levels or self.LEVELS:
                level_split = f"2023_level{level}"
                try:
                    ds = hf_load_dataset(source, level_split, split=split)
                except Exception:
                    # Try without level-specific split
                    try:
                        ds = hf_load_dataset(source, split=split)
                        ds = ds.filter(lambda x: x.get("Level") == level)
                    except Exception as e:
                        _logger.warning("Could not load GAIA level %d: %s", level, e)
                        continue

                for i, row in enumerate(ds):
                    if max_cases and len(cases) >= max_cases:
                        return cases
                    case = self._row_to_eval_case(row, i, level)
                    if case:
                        cases.append(case)

        except Exception as e:
            _logger.warning("HuggingFace load failed: %s, falling back to local", e)
            return await self._load_from_local(source, levels, max_cases)

        return cases

    async def _load_from_local(
        self,
        source: str,
        levels: list[int] | None,
        max_cases: int | None,
    ) -> list[EvalCase]:
        """Load from local Parquet or JSON files."""
        path = Path(source)
        cases: list[EvalCase] = []

        if path.is_dir():
            for pq_file in sorted(path.glob("*.parquet")):
                cases.extend(await self._load_parquet(pq_file, levels, max_cases - len(cases) if max_cases else None))
                if max_cases and len(cases) >= max_cases:
                    break
        elif path.suffix == ".parquet":
            cases = await self._load_parquet(path, levels, max_cases)

        return cases[:max_cases] if max_cases else cases

    async def _load_parquet(
        self,
        path: Path,
        levels: list[int] | None,
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
            row_dict = row.to_dict()
            level = row_dict.get("Level", 1)
            if levels and level not in levels:
                continue
            case = self._row_to_eval_case(row_dict, i, level)
            if case:
                cases.append(case)

        return cases

    def _row_to_eval_case(self, row: dict, index: int, level: int) -> EvalCase | None:
        """Convert a GAIA dataset row to an EvalCase."""
        question = row.get("Question", row.get("question", ""))
        answer = row.get("Final answer", row.get("final_answer", row.get("answer", "")))

        if not question:
            return None

        task_id = row.get("task_id", row.get("id", f"gaia_{index}"))
        file_name = row.get("file_name", "")
        file_path = row.get("file_path", "")

        attachments: list[str] = []
        if file_path and Path(file_path).exists():
            attachments.append(file_path)

        return EvalCase(
            name=str(task_id),
            user_message=str(question),
            expected_answer=str(answer) if answer else None,
            file_attachments=attachments,
            tags=["gaia", f"level-{level}"],
            metadata={
                "level": level,
                "file_name": file_name,
                "framework": "gaia",
            },
        )
