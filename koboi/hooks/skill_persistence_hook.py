"""koboi/hooks/skill_persistence_hook.py -- Re-inject activated skills after compaction.

When context compaction (truncation, summarization) removes activated skill
instructions, this hook re-injects them so the agent retains skill knowledge
across long conversations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.skills.registry import SkillRegistry


class SkillPersistenceHook(Hook):
    """Re-inject activated skills after context compaction.

    Listens for POST_COMPACT and appends skill bodies to the context
    so they survive truncation/summarization.

    Priority 45: runs after infrastructure hooks (0-19) and security
    hooks (20-39), but before post-processing (60+).
    """

    priority = 45

    def __init__(self, skills: SkillRegistry):
        self.skills = skills

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_COMPACT]

    async def execute(self, ctx: HookContext) -> HookContext:
        for skill_name in self.skills._activated:
            skill = self.skills.get(skill_name)
            if skill and skill.body:
                body = skill.body[:5000]
                ctx.inject_messages.append(f'<skill name="{skill_name}" dir="{skill.skill_dir}">\n{body}\n</skill>')
        return ctx
