"""koboi/eval/loaders/ragas_generator.py -- Synthetic eval data generator for RAG.

Generates question-answer pairs from document files using an LLM,
producing EvalCases suitable for RAGAS evaluation.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from koboi.types import EvalCase

if TYPE_CHECKING:
    from koboi.client import Client

_logger = logging.getLogger(__name__)

_QA_GENERATION_PROMPT = """\
Given the following document context, generate {count} question-answer pairs.
Difficulty level: {difficulty}

Context:
{context}

For each pair, output a JSON array where each element has:
- "question": a clear, specific question answerable from the context
- "answer": a concise, accurate answer based on the context

Output ONLY valid JSON array, no markdown or explanation:
[{{"question": "...", "answer": "..."}}, ...]"""


class RAGASDataGenerator:
    """Generate synthetic eval cases from document files for RAGAS evaluation.

    Uses the LLM to generate question-answer pairs from context documents,
    then wraps them as EvalCases with context_docs for RAGAS scoring.
    """

    def __init__(self, client: Client, chunk_size: int = 1000):
        self.client = client
        self.chunk_size = chunk_size

    async def generate_from_docs(
        self,
        doc_paths: list[str],
        num_questions_per_doc: int = 5,
        difficulty: str = "mixed",
    ) -> list[EvalCase]:
        """Generate EvalCases with Q&A pairs from document files.

        Args:
            doc_paths: Paths to document files (markdown, text, etc.)
            num_questions_per_doc: Number of Q&A pairs per document chunk.
            difficulty: "easy", "medium", "hard", or "mixed"

        Returns:
            List of EvalCases with user_message (question), expected_answer,
            and context_docs (source chunks).
        """
        cases: list[EvalCase] = []

        for doc_path in doc_paths:
            path = Path(doc_path)
            if not path.exists():
                _logger.warning("Document not found: %s", doc_path)
                continue

            content = path.read_text(errors="replace")
            chunks = self._chunk_text(content)

            for chunk_idx, chunk in enumerate(chunks):
                try:
                    qa_pairs = await self._generate_qa_pairs(
                        chunk, num_questions_per_doc, difficulty,
                    )
                    for qa_idx, qa in enumerate(qa_pairs):
                        case = EvalCase(
                            name=f"{path.stem}_chunk{chunk_idx}_q{qa_idx}",
                            user_message=qa.get("question", ""),
                            expected_answer=qa.get("answer", ""),
                            context_docs=[chunk],
                            tags=["ragas", "synthetic", difficulty],
                            metadata={
                                "source_doc": str(path),
                                "chunk_index": chunk_idx,
                                "difficulty": difficulty,
                                "framework": "ragas",
                            },
                        )
                        cases.append(case)
                except Exception as e:
                    _logger.warning(
                        "Failed to generate Q&A for %s chunk %d: %s",
                        doc_path, chunk_idx, e,
                    )

        return cases

    async def _generate_qa_pairs(
        self, context: str, count: int, difficulty: str,
    ) -> list[dict[str, str]]:
        """Use LLM to generate question-answer pairs from context."""
        prompt = _QA_GENERATION_PROMPT.format(
            count=count,
            difficulty=difficulty,
            context=context[:self.chunk_size],
        )

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content or "[]"

        return self._parse_qa_response(text)

    @staticmethod
    def _parse_qa_response(text: str) -> list[dict[str, str]]:
        """Parse Q&A pairs from LLM response."""
        # Try direct JSON parse
        text = text.strip()

        # Remove markdown code blocks if present
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [r for r in result if isinstance(r, dict) and "question" in r]
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [r for r in result if isinstance(r, dict) and "question" in r]
            except json.JSONDecodeError:
                pass

        _logger.warning("Could not parse Q&A response: %s", text[:200])
        return []

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks by paragraphs, respecting chunk_size."""
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > self.chunk_size and current:
                chunks.append(current.strip())
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text[:self.chunk_size]]
