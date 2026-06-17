"""koboi/harness/doom_loop.py -- Detection of unproductive repeating tool-call patterns.

Detects three doom-loop patterns:
1. Consecutive identical -- same tool with same arguments repeated
2. Repeating pattern -- same sequence of tools repeated (A,B,C -> A,B,C)
3. Error retry -- same tool+args producing errors repeatedly (even if interleaved)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class DoomLoopConfig:
    consecutive_identical_threshold: int = 3
    repeating_pattern_window: int = 6
    repeating_pattern_threshold: int = 2
    enable_recovery: bool = True
    adaptive_threshold: bool = False
    task_complexity_hint: str = "auto"  # "auto" | "simple" | "moderate" | "complex"
    error_retry_threshold: int = 3


@dataclass
class DoomLoopResult:
    detected: bool = False
    loop_type: str = ""  # "consecutive_identical" | "repeating_pattern" | "error_retry"
    pattern_description: str = ""
    recovery_hint: str = ""
    iterations_wasted: int = 0


_MAX_HISTORY = 200


class DoomLoopDetector:
    def __init__(self, config: DoomLoopConfig | None = None):
        self.config = config or DoomLoopConfig()
        self._history: deque[tuple[str, str]] = deque(maxlen=_MAX_HISTORY)
        self._error_flags: deque[bool] = deque(maxlen=_MAX_HISTORY)

    def record(self, tool_name: str, arguments: str, is_error: bool = False) -> None:
        self._history.append((tool_name, arguments))
        self._error_flags.append(is_error)

    def check(self) -> DoomLoopResult:
        threshold = self.get_effective_threshold()
        if len(self._history) < threshold:
            return DoomLoopResult()

        result = self._check_error_retry()
        if result:
            return result

        result = self._check_consecutive_identical(threshold)
        if result:
            return result

        result = self._check_repeating_pattern()
        if result:
            return result

        return DoomLoopResult()

    def reset(self) -> None:
        self._history.clear()
        self._error_flags.clear()

    @property
    def history(self) -> list[tuple[str, str]]:
        return list(self._history)

    def estimate_complexity(self, available_tools: int, prompt_length: int) -> str:
        if available_tools <= 3 and prompt_length < 100:
            return "simple"
        elif available_tools <= 6 or prompt_length < 300:
            return "moderate"
        else:
            return "complex"

    def get_effective_threshold(self, available_tools: int = 0) -> int:
        base = self.config.consecutive_identical_threshold
        if not self.config.adaptive_threshold:
            return base
        hint = self.config.task_complexity_hint
        if hint == "complex":
            return max(base, min(base + 2, 6))
        elif hint == "moderate":
            return max(base, base + 1)
        return base

    def _check_consecutive_identical(self, threshold: int | None = None) -> DoomLoopResult | None:
        if threshold is None:
            threshold = self.config.consecutive_identical_threshold
        if len(self._history) < threshold:
            return None

        last_n = list(self._history)[-threshold:]
        first = last_n[0]
        if all(call == first for call in last_n[1:]):
            tool_name, arguments = first
            return DoomLoopResult(
                detected=True,
                loop_type="consecutive_identical",
                pattern_description=f"{tool_name}({arguments[:50]}) x{threshold}",
                recovery_hint=(
                    f"You have called {tool_name} with the same arguments "
                    f"{threshold} times consecutively without different results. "
                    f"Try a different approach: change arguments, use a different tool, "
                    f"or ask the user for clarification."
                ),
                iterations_wasted=threshold,
            )
        return None

    def _check_repeating_pattern(self) -> DoomLoopResult | None:
        window = self.config.repeating_pattern_window
        min_repeats = self.config.repeating_pattern_threshold

        if len(self._history) < window:
            return None

        tool_names = [name for name, _ in self._history]

        for pattern_len in range(1, min(4, len(tool_names) // 2) + 1):
            tail = tool_names[-window:]
            pattern = tail[:pattern_len]
            repeats = 0
            pos = 0
            while pos + pattern_len <= len(tail):
                if tail[pos:pos + pattern_len] == pattern:
                    repeats += 1
                    pos += pattern_len
                else:
                    break

            if repeats >= min_repeats + 1:
                pattern_str = " -> ".join(pattern)
                return DoomLoopResult(
                    detected=True,
                    loop_type="repeating_pattern",
                    pattern_description=f"[{pattern_str}] x{repeats}",
                    recovery_hint=(
                        f"You are stuck in a repeating pattern: {pattern_str}. "
                        f"This pattern has occurred {repeats} times. "
                        f"This indicates your approach is not working. "
                        f"Try a fundamentally different strategy."
                    ),
                    iterations_wasted=repeats * pattern_len,
                )
        return None

    def _check_error_retry(self) -> DoomLoopResult | None:
        threshold = self.config.error_retry_threshold
        if len(self._history) < threshold:
            return None

        # Count how many times each (tool, args) pair appears with errors
        error_counts: dict[tuple[str, str], int] = {}
        for i, (call, is_err) in enumerate(zip(self._history, self._error_flags)):
            if is_err:
                error_counts[call] = error_counts.get(call, 0) + 1

        for (tool_name, arguments), count in error_counts.items():
            if count >= threshold:
                return DoomLoopResult(
                    detected=True,
                    loop_type="error_retry",
                    pattern_description=f"{tool_name}({arguments[:50]}) error x{count}",
                    recovery_hint=(
                        f"You have called {tool_name} with the same arguments "
                        f"{count} times and it always fails. This tool will not succeed "
                        f"with those arguments. Try changing arguments or using a different tool."
                    ),
                    iterations_wasted=count,
                )
        return None

    @staticmethod
    def build_recovery_message(result: DoomLoopResult) -> str:
        if not result.detected:
            return ""
        return (
            f"\n\n[DOOM LOOP WARNING] {result.recovery_hint}\n"
            f"Detected pattern: {result.pattern_description}"
        )
