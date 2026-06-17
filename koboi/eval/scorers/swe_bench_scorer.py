"""koboi/eval/scorers/swe_bench_scorer.py -- SWE-bench patch evaluation scorer.

Evaluates generated patches against expected SWE-bench patches using
structural similarity and file overlap comparison.
"""
from __future__ import annotations

import logging
import re

from koboi.types import EvalCase, EvalScore
from koboi.eval.scorers.base import BaseScorer

_logger = logging.getLogger(__name__)


class PatchGenerationScorer(BaseScorer):
    """Evaluates generated patches against expected SWE-bench patches.

    Scoring dimensions:
    1. File overlap (60%): which files were modified
    2. Diff structure similarity (40%): number of hunks, lines changed
    """

    def __init__(self, file_weight: float = 0.6, struct_weight: float = 0.4):
        self.file_weight = file_weight
        self.struct_weight = struct_weight

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        if not case.expected_answer:
            return EvalScore("patch_generation", 0.0, "No expected patch")

        generated_patch = self._extract_patch(output)
        if not generated_patch:
            return EvalScore("patch_generation", 0.0, "No patch found in output")

        expected_patch = case.expected_answer

        # Score 1: File overlap
        file_score = self._file_overlap(generated_patch, expected_patch)

        # Score 2: Structural similarity
        struct_score = self._structural_similarity(generated_patch, expected_patch)

        # Composite
        composite = self.file_weight * file_score + self.struct_weight * struct_score

        reason = f"Files: {file_score:.2f}, Structure: {struct_score:.2f}"
        return EvalScore("patch_generation", round(composite, 3), reason)

    @staticmethod
    def _extract_patch(output: str) -> str | None:
        """Extract diff/patch content from agent output."""
        # Look for diff --git markers
        diff_match = re.search(r"(diff --git .+)", output, re.DOTALL)
        if diff_match:
            return diff_match.group(1)

        # Look for --- / +++ markers
        patch_match = re.search(r"(--- .+\n\+\+\+ .+)", output, re.DOTALL)
        if patch_match:
            return patch_match.group(1)

        # Look for @@ hunk headers
        hunk_match = re.search(r"(@@ .+ @@.+)", output, re.DOTALL)
        if hunk_match:
            return hunk_match.group(1)

        return None

    @staticmethod
    def _file_overlap(generated: str, expected: str) -> float:
        """Compare which files were modified."""
        gen_files = _extract_filenames(generated)
        exp_files = _extract_filenames(expected)

        if not exp_files:
            return 1.0 if not gen_files else 0.5

        if not gen_files:
            return 0.0

        matched = gen_files & exp_files
        union = gen_files | exp_files

        # Jaccard similarity
        return len(matched) / len(union) if union else 1.0

    @staticmethod
    def _structural_similarity(generated: str, expected: str) -> float:
        """Compare diff structure: number of hunks, lines changed.

        Over-generation (superset patch) is penalized less than
        under-generation, since extra lines may include context or
        a slightly different but still valid implementation.
        """
        gen_stats = _diff_stats(generated)
        exp_stats = _diff_stats(expected)

        if exp_stats["total"] == 0:
            return 1.0 if gen_stats["total"] == 0 else 0.0

        scores: list[float] = []

        # Compare hunk count — exact match preferred
        if exp_stats["hunks"] > 0:
            hunk_ratio = min(gen_stats["hunks"], exp_stats["hunks"]) / max(gen_stats["hunks"], exp_stats["hunks"])
            scores.append(hunk_ratio)

        # Compare lines changed — over-generation scores higher than under-generation
        if exp_stats["total"] > 0:
            if gen_stats["total"] >= exp_stats["total"]:
                # Superset: agent changed at least as many lines as expected
                # Scale from 1.0 (exact) down to 0.7 (2x or more)
                overshoot = gen_stats["total"] / exp_stats["total"]
                line_ratio = max(0.7, 1.0 - 0.3 * (overshoot - 1.0))
            else:
                # Undersize: agent changed fewer lines than expected
                line_ratio = gen_stats["total"] / exp_stats["total"]
            scores.append(line_ratio)

        return sum(scores) / len(scores) if scores else 0.0


class DockerTestScorer(BaseScorer):
    """Run SWE-bench test suite in Docker container.

    Heavy-weight scorer: clones repo, applies patch, runs tests.
    Only use when Docker is available and run_tests=True.

    NOTE: This scorer requires Docker and network access.
    It is designed for CI/CD environments, not local development.
    """

    def __init__(self, docker_image: str | None = None, timeout: int = 300):
        self.docker_image = docker_image
        self.timeout = timeout

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        import shutil

        if not shutil.which("docker"):
            return EvalScore("docker_test", 0.0, "Docker not available")

        if not case.metadata.get("repo"):
            return EvalScore("docker_test", 0.0, "No repo in metadata")

        # Extract patch from output
        generated_patch = PatchGenerationScorer._extract_patch(output)
        if not generated_patch:
            return EvalScore("docker_test", 0.0, "No patch found in output")

        # This is a placeholder for actual Docker-based test execution
        # In production, this would:
        # 1. Pull the SWE-bench Docker image for the repo
        # 2. Clone the repo at base_commit
        # 3. Apply the generated patch
        # 4. Run the test commands from metadata
        # 5. Parse test results
        _logger.info("DockerTestScorer: Docker test execution not yet implemented")
        return EvalScore("docker_test", 0.0, "Docker test not yet implemented")


def _extract_filenames(patch: str) -> set[str]:
        """Extract filenames from a diff/patch string."""
        files: set[str] = set()

        # diff --git a/file b/file
        for m in re.finditer(r"diff --git a/(.+?) b/(.+?)(?:\s|$)", patch):
            files.add(m.group(2))

        # --- a/file / +++ b/file
        for m in re.finditer(r"[-+]{3} [ab]/(.+?)(?:\s|$)", patch):
            files.add(m.group(1))

        return files


def _diff_stats(patch: str) -> dict[str, int]:
        """Extract statistics from a diff/patch string."""
        hunks = len(re.findall(r"^@@ ", patch, re.MULTILINE))
        additions = len(re.findall(r"^\+[^+]", patch, re.MULTILINE))
        deletions = len(re.findall(r"^-[^-]", patch, re.MULTILINE))

        return {
            "hunks": hunks,
            "additions": additions,
            "deletions": deletions,
            "total": additions + deletions,
        }
