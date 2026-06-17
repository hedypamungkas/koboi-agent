"""koboi/hooks/skill_hook.py -- Hook for skill activation detection at POST_LLM_CALL.

Detects skill activation patterns in LLM responses and records them.
"""
from __future__ import annotations

import re

from koboi.hooks.chain import Hook, HookContext, HookEvent


class SkillHook(Hook):
    """Hook for skill activation detection at POST_LLM_CALL.

    Scans LLM responses for skill activation patterns and records
    detected skill usage in metadata and carryover state.
    """

    # Pattern to detect skill invocations in LLM output
    SKILL_PATTERN = re.compile(r"\[ACTIVATE_SKILL:\s*([a-z0-9_-]+)\]", re.IGNORECASE)

    def __init__(
        self,
        available_skills: list[str] | None = None,
        auto_activate: bool = False,
    ):
        self.available_skills = set(available_skills or [])
        self.auto_activate = auto_activate
        self._activated: list[str] = []

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_LLM_CALL]

    async def execute(self, ctx: HookContext) -> HookContext:
        # Extract LLM response text
        response_text = ""
        if ctx.llm_response:
            response_text = getattr(ctx.llm_response, "content", "") or ""

        if not response_text:
            return ctx

        # Look for skill activation patterns
        activated_skills = []
        for match in self.SKILL_PATTERN.finditer(response_text):
            skill_name = match.group(1).lower()
            if not self.available_skills or skill_name in self.available_skills:
                activated_skills.append({
                    "name": skill_name,
                    "arguments": "",
                })

        if activated_skills:
            ctx.metadata["skills_detected"] = activated_skills
            self._activated.extend(s["name"] for s in activated_skills)

            # Record in carryover if available
            if ctx.carryover:
                for skill_info in activated_skills:
                    if hasattr(ctx.carryover, "record_skill"):
                        ctx.carryover.record_skill(skill_info["name"])

            # If auto_activate, flag for execution
            if self.auto_activate:
                ctx.metadata["skills_to_activate"] = activated_skills

        return ctx

    @property
    def activated_skills(self) -> list[str]:
        return list(self._activated)
