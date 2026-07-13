"""koboi/eval/loaders/tydiqa_id_loader.py -- TyDi QA (Indonesian) dataset loader.

Loads the NATIVE Indonesian subset of TyDi QA (secondary_task / goldp) from HuggingFace into
EvalCase lists. TyDi QA is natively collected by Indonesian speakers (NOT translated) -- the
property that makes it the right benchmark for a caveat-free Indonesian RAG claim.

Source: https://huggingface.co/datasets/google-research-datasets/tydiqa (Apache-2.0)
Language is encoded in the row ``id`` prefix (``indonesian-<n>-<passage>``); filter accordingly.
"""

from __future__ import annotations

import logging
from typing import Any

from koboi.eval.loaders import DatasetLoader
from koboi.types import EvalCase

_logger = logging.getLogger(__name__)

_AVAILABLE = False
try:
    from datasets import load_dataset as hf_load_dataset

    _AVAILABLE = True
except ImportError:
    pass

TYDI = "google-research-datasets/tydiqa"
ID_PREFIX = "indonesian"


class TyDiQAIDLoader(DatasetLoader):
    """Loads TyDi QA Indonesian (secondary_task) into EvalCase objects.

    Maps ``question`` -> user_message and the first ``answers["text"]`` -> expected_answer.
    (The row's ``context`` gold passage is consumed by the corpus builder
    ``scripts/build_id_native_corpus.py``, not this loader; this loader emits only the
    question + expected_answer pair.) Rows with empty answers are skipped.
    """

    async def load(
        self,
        source: str = TYDI,
        split: str = "train",
        max_cases: int | None = None,
        **kwargs: Any,
    ) -> list[EvalCase]:
        if not _AVAILABLE:
            raise ImportError(
                "The `datasets` package is required for the TyDi-QA-id loader: pip install -e '.[eval-ragas]'"
            )
        # Trusted public benchmark (Google TyDi QA, Apache-2.0); unpinned load matches the
        # existing gaia/swe loader pattern. Pin `revision=` only if supply-chain hardening is required.
        ds = hf_load_dataset(source, "secondary_task", split=split)  # nosec B615
        cases: list[EvalCase] = []
        for row in ds:
            if not str(row.get("id", "")).startswith(ID_PREFIX):
                continue
            answers = row.get("answers") or {}
            texts = answers.get("text") or []
            answer = texts[0].strip() if texts and texts[0].strip() else ""
            question = (row.get("question") or "").strip()
            if not question or not answer:
                continue
            cases.append(
                EvalCase(
                    name=str(row.get("id")),
                    user_message=question,
                    expected_answer=answer,
                    tags=["tydiqa-id", "id"],
                    metadata={
                        "framework": "tydiqa-id",
                        "language": "id",
                        "title": str(row.get("title", "")),
                    },
                )
            )
            if max_cases is not None and len(cases) >= max_cases:
                break
        _logger.info("TyDiQAIDLoader: %d indonesian cases (split=%s)", len(cases), split)
        return cases

    def framework_name(self) -> str:
        return "tydiqa-id"
