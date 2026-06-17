"""plan_view.py -- Numbered plan checklist widget for PLAN mode."""
from __future__ import annotations

import re
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.containers import Vertical
from textual.widgets import Static


@dataclass
class PlanStep:
    """A single step in the plan."""
    index: int
    description: str
    completed: bool = False
    skipped: bool = False


class PlanStepWidget(Static):
    """A single plan step with toggle state."""

    def __init__(self, step: PlanStep, **kwargs) -> None:
        super().__init__(**kwargs)
        self._step = step

    def render(self) -> str:
        if self._step.completed:
            marker = "[green][x][/green]"
        elif self._step.skipped:
            marker = "[dim][-][/dim]"
        else:
            marker = "[ ][ ]"
        return f"  {marker} {self._step.index}. {self._step.description}"

    def toggle(self) -> None:
        if self._step.completed:
            self._step.completed = False
            self._step.skipped = True
        elif self._step.skipped:
            self._step.skipped = False
        else:
            self._step.completed = True
        self.refresh()


class PlanView(Vertical):
    """Displays the agent's plan as an interactive numbered checklist.

    Parses markdown numbered lists into structured steps.
    User can toggle steps and approve/reject the plan.
    """

    DEFAULT_CSS = """
    PlanView {
        height: auto;
        max-height: 50%;
        border: solid $accent;
        margin: 1 0;
        padding: 0 1;
    }

    #plan-header {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .plan-step {
        height: 1;
    }

    #plan-footer {
        color: $text-muted;
        margin-top: 1;
        height: 1;
    }
    """

    plan_content: reactive[str] = reactive("")
    approved: reactive[bool] = reactive(False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._steps: list[PlanStep] = []

    def compose(self) -> ComposeResult:
        yield Static("Plan", id="plan-header")
        yield Vertical(id="plan-steps")
        yield Static(
            "[[A]] Approve & Execute  [[E]] Edit  [[R]] Reject  [[1-N]] Toggle step",
            id="plan-footer",
        )

    def watch_plan_content(self, content: str) -> None:
        """Parse and render plan steps when content changes."""
        self._steps = self._parse_plan(content)
        self._render_steps()

    def _parse_plan(self, content: str) -> list[PlanStep]:
        """Parse markdown numbered list into PlanStep objects."""
        steps = []
        # Match lines like: "1. Description" or "1) Description" or "- [ ] Description"
        pattern = re.compile(r"^\s*(?:\d+[\.\)]\s*|\-\s*\[[ x]\]\s*)(.+)$", re.MULTILINE)
        for i, match in enumerate(pattern.finditer(content), 1):
            steps.append(PlanStep(index=i, description=match.group(1).strip()))

        # Fallback: if no numbered items found, try splitting by newlines
        if not steps:
            lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
            for i, line in enumerate(lines, 1):
                # Skip markdown headers
                if line.startswith("#"):
                    continue
                steps.append(PlanStep(index=i, description=line))

        return steps

    def _render_steps(self) -> None:
        """Render plan steps as widgets."""
        container = self.query_one("#plan-steps")
        container.remove_children()
        for step in self._steps:
            widget = PlanStepWidget(step, classes="plan-step")
            container.mount(widget)

    def toggle_step(self, index: int) -> None:
        """Toggle a step by its index."""
        for step in self._steps:
            if step.index == index:
                if step.completed:
                    step.completed = False
                    step.skipped = True
                elif step.skipped:
                    step.skipped = False
                else:
                    step.completed = True
                break
        self._render_steps()

    def approve_plan(self) -> None:
        """Mark the plan as approved."""
        self.approved = True

    def reject_plan(self) -> None:
        """Mark the plan as rejected."""
        self.approved = False

    def get_steps(self) -> list[PlanStep]:
        """Return the current plan steps."""
        return list(self._steps)

    def get_active_steps(self) -> list[PlanStep]:
        """Return steps that are not skipped."""
        return [s for s in self._steps if not s.skipped]
