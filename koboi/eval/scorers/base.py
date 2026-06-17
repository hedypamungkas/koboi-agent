"""koboi/eval/scorers.py -- Scorer classes for agent evaluation.

Heuristic and LLM-as-judge scorers for evaluating agent output quality.

Adapted from agent/eval.py scorer classes.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from koboi.types import EvalCase, EvalScore

if TYPE_CHECKING:
    from koboi.llm.base import LLMClient


class BaseScorer(ABC):
    @abstractmethod
    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore: ...


# ---------------------------------------------------------------------------
# Heuristic scorers
# ---------------------------------------------------------------------------


class ToolUsageScorer(BaseScorer):
    """Checks if expected tools were used based on telemetry."""

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        expected = case.expected_tools
        if not expected:
            return EvalScore("tool_usage", 1.0, "No expected tools specified")

        telemetry = context.get("telemetry")
        if not telemetry:
            return EvalScore("tool_usage", 0.0, "No telemetry available")

        used = telemetry.snapshot.unique_tools_used
        matched = [t for t in expected if t in used]
        ratio = len(matched) / len(expected)

        if ratio == 1.0:
            reason = f"All {len(expected)} expected tools used: {matched}"
        elif ratio > 0:
            reason = f"{len(matched)}/{len(expected)} expected tools used: {matched}, missing: {[t for t in expected if t not in used]}"
        else:
            reason = f"None of the expected tools used. Expected: {expected}, got: {list(used)}"

        return EvalScore("tool_usage", round(ratio, 3), reason)


class KeywordPresenceScorer(BaseScorer):
    """Checks if output contains expected keywords."""

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        expected = case.expected_keywords
        if not expected:
            return EvalScore("keyword_presence", 1.0, "No expected keywords specified")

        output_lower = output.lower()
        output_normalized = re.sub(r"[^a-z0-9]", "", output_lower)
        matched = [kw for kw in expected if kw.lower() in output_lower or kw.lower() in output_normalized]
        ratio = len(matched) / len(expected)

        missing = [kw for kw in expected if kw.lower() not in output_lower and kw.lower() not in output_normalized]
        reason = f"{len(matched)}/{len(expected)} keywords found"
        if missing:
            reason += f", missing: {missing}"

        return EvalScore("keyword_presence", round(ratio, 3), reason)


class OutputLengthScorer(BaseScorer):
    """Checks if output is within reasonable bounds (not too short, not too long)."""

    def __init__(self, min_length: int = 10, max_length: int = 5000):
        self.min_length = min_length
        self.max_length = max_length

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        length = len(output)
        if length == 0:
            return EvalScore("output_length", 0.0, "Empty output")
        if length < self.min_length:
            return EvalScore("output_length", 0.3, f"Output too short: {length} chars (min: {self.min_length})")
        if length > self.max_length:
            return EvalScore("output_length", 0.7, f"Output very long: {length} chars (max: {self.max_length})")
        return EvalScore("output_length", 1.0, f"Output length OK: {length} chars")


class IterationEfficiencyScorer(BaseScorer):
    """Scores based on iterations used vs max_iterations."""

    def __init__(self, target_ratio: float = 0.5):
        self.target_ratio = target_ratio

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        telemetry = context.get("telemetry")
        if not telemetry:
            return EvalScore("iteration_efficiency", 0.5, "No telemetry available")

        total = telemetry.snapshot.total_iterations
        max_iter = case.max_iterations

        if max_iter == 0:
            return EvalScore("iteration_efficiency", 1.0, "No iterations needed")

        ratio = total / max_iter
        if ratio <= self.target_ratio:
            score = 1.0
        elif ratio <= 0.8:
            score = 0.7
        else:
            score = 0.4

        reason = f"{total}/{max_iter} iterations used ({ratio:.0%})"
        return EvalScore("iteration_efficiency", round(score, 3), reason)


class HealthScoreScorer(BaseScorer):
    """Maps TelemetryCollector.health_score() to 0-1 range."""

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        telemetry = context.get("telemetry")
        if not telemetry:
            return EvalScore("health_score", 0.5, "No telemetry available")

        health = telemetry.health_score()
        score = round(health / 100.0, 3)
        return EvalScore("health_score", score, f"Health: {health}/100")


# ---------------------------------------------------------------------------
# LLM-as-judge scorer
# ---------------------------------------------------------------------------


class LLMJudgeScorer(BaseScorer):
    """Uses the LLM to judge output quality on a 1-5 scale."""

    JUDGE_PROMPT = """\
You are an evaluator judging the quality of an AI agent's answer.

## User Question
{query}

## Agent Answer
{output}

## Scoring Criteria
Rate 1-5 based on:
1. **Accuracy** (1-5): Does the answer address the question?
2. **Completeness** (1-5): Is the information sufficiently complete?
3. **Relevance** (1-5): Is the content relevant to the question?
4. **Clarity** (1-5): Is the answer easy to understand?

## Output Format (REQUIRED)
SCORE: <average number 1-5>
REASON: <brief 1-sentence reason>"""

    def __init__(self, client: LLMClient, judge_prompt: str | None = None):
        self.client = client
        self._prompt_template = judge_prompt or self.JUDGE_PROMPT

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        prompt = self._prompt_template.format(
            query=case.user_message[:500],
            output=output[:2000],
        )
        try:
            response = await self.client.complete(
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content or ""
            return self._parse_judge_response(text)
        except Exception as e:
            return EvalScore("llm_judge", 0.0, f"Judge failed: {e}")

    def _parse_judge_response(self, text: str) -> EvalScore:
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", text)
        reason_match = re.search(r"REASON:\s*(.+)", text)

        if score_match:
            raw = float(score_match.group(1))
            normalized = round(min(1.0, max(0.0, raw / 5.0)), 3)
            reason = reason_match.group(1).strip() if reason_match else f"Raw score: {raw}/5"
            return EvalScore("llm_judge", normalized, reason)

        return EvalScore("llm_judge", 0.3, f"Could not parse judge response: {text[:200]}")


# ---------------------------------------------------------------------------
# Cost scorer
# ---------------------------------------------------------------------------


class CostScorer(BaseScorer):
    """Tracks token usage cost per eval case."""

    def __init__(
        self,
        max_tokens: int = 10000,
        cost_per_1k_input: float = 0.005,
        cost_per_1k_output: float = 0.015,
    ):
        self.max_tokens = max_tokens
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        usage = context.get("token_usage")
        if not usage:
            return EvalScore("cost", 0.5, "No token usage data")

        total = usage.total_tokens
        score = max(0.0, 1.0 - (total / self.max_tokens))
        cost = (usage.prompt_tokens * self.cost_per_1k_input + usage.completion_tokens * self.cost_per_1k_output) / 1000
        return EvalScore("cost", round(score, 3), f"{total} tokens, ~${cost:.4f}")
