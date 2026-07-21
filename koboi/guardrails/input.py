"""Input validation and sanitization for user input before entering agent loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.guardrails.base import PatternGuardrail
from koboi.types import GuardrailResult

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


class InputGuardrail(PatternGuardrail):
    """Validate and sanitize user input before entering agent loop."""

    PATTERNS: list[tuple[str, str]] = [
        # --- Instruction-override commands (English) ---
        (r"(?i)ignore\s+.*instructions?", "Possible prompt injection: ignore instructions"),
        (r"(?i)forget\s+(everything|all|previous)", "Possible prompt injection: forget"),
        (
            r"(?i)disregard\s+(all|any|previous|the).{0,30}(instruction|prompt|rule)",
            "Possible prompt injection: disregard instructions",
        ),
        (r"(?i)\bnew\s+(instructions?|rules?|prompt)\s*:", "Possible prompt injection: new instructions override"),
        (
            r"(?i)override\s+(your|the|all).{0,20}(instruction|prompt|rule|directive)",
            "Possible prompt injection: override instructions",
        ),
        (r"(?i)you\s+are\s+now\s+", "Possible prompt injection: persona override"),
        # --- Instruction-override commands (Bahasa Indonesia) ---
        (
            r"(?i)abaikan\s+.*(instruksi|prompt|aturan|perintah|pesan\s+(di\s+)?atas)",
            "Possible prompt injection: abaikan instruksi (ID)",
        ),
        (r"(?i)lupakan\s+(semua|sebelumnya|instruksi|prompt|aturan)", "Possible prompt injection: lupakan (ID)"),
        (
            r"(?i)(sekarang|mulai\s+sekarang)\s+(kamu|anda|kau)\s+(adalah|jadi)",
            "Possible prompt injection: persona override (ID)",
        ),
        (r"(?i)jangan\s+ikuti\s+(instruksi|aturan|prompt)", "Possible prompt injection: jangan ikuti instruksi (ID)"),
        # --- Role / system spoofing (both languages) ---
        (r"(?i)system\s*:\s*", "Possible prompt injection: system role spoofing"),
        (r"(?i)\b(asistant|assistant|user|developer|tool)\s*:\s*", "Possible prompt injection: chat-role spoofing"),
        # --- Structural tag / delimiter injection ---
        (r"<\s*/?\s*(system|instruction|prompt)\s*>", "Possible prompt injection: tag injection"),
        (r"(?i)\[\s*/?\s*inst(ruct)?\s*\]", "Possible prompt injection: [INST] delimiter"),
        (r"(?i)<<\s*/?\s*sys\s*>>", "Possible prompt injection: <<SYS>> delimiter"),
        # Inline (?m) so '^' matches at the start of ANY line, not just the whole
        # string -- a mid-transcript "# System\n..." must trip it too.
        (r"(?im)^#{1,3}\s*(system|instructions?|prompt|rules?)\b", "Possible prompt injection: markdown role header"),
    ]
    DEFAULT_ACTION = "block"
    MAX_INPUT_LENGTH = 10000

    def __init__(
        self,
        max_length: int | None = None,
        custom_patterns: list[tuple[str, str]] | None = None,
        logger: AgentLogger | None = None,
        **kwargs: object,
    ):
        super().__init__(
            patterns=self.PATTERNS,
            default_action="block",
            custom_patterns=custom_patterns,
            logger=logger,
        )
        self.max_length = max_length or self.MAX_INPUT_LENGTH

    async def check(self, user_input: str, context: list[str] | None = None) -> GuardrailResult:
        if not user_input or not user_input.strip():
            return GuardrailResult(passed=False, reason="Input is empty", action="block")

        if len(user_input) > self.max_length:
            return GuardrailResult(
                passed=False,
                reason=f"Input is too long ({len(user_input)} chars, max {self.max_length})",
                action="block",
            )

        pattern_result = await self.check_patterns(user_input)
        if pattern_result is not None:
            return pattern_result

        sanitized = user_input.strip()
        return GuardrailResult(passed=True, sanitized_content=sanitized)
