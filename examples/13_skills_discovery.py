"""Example 13: Skills Discovery.

Demonstrates:
- SkillRegistry: discover, list, route, and activate skills
- KoboiAgent.from_config() with skill discovery from YAML
- Dual mode: automatic (demo routing) and interactive (free chat with skills)

Run:
    python examples/12_skills_discovery.py                  # automatic mode
    python examples/12_skills_discovery.py -m interactive   # interactive mode
"""
from __future__ import annotations

import click
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from conftest import (
    console,
    ensure_path,
    load_env,
    setup_example,
    dual_mode_options,
    create_agent,
    automatic_batch,
    interactive_loop,
)

ensure_path()
load_env()

from pathlib import Path

DEMO_QUERIES = [
    "review this code: def add(a,b): return a+b",
    "there is a security incident in production!",
    "What is the price of AcmeERP?",
]


def _run_registry_demo():
    """Part 1: Direct SkillRegistry demonstration."""
    console.print(Panel(
        "[bold]Part 1: Direct SkillRegistry[/bold]\n"
        "Discover, list, route, and activate skills directly.",
        title="Skills Discovery",
    ))

    from koboi.skills.registry import SkillRegistry

    registry = SkillRegistry()
    skills_dir = str(Path(__file__).parent / "data" / "skills")
    discovered = registry.discover([skills_dir])

    if not discovered:
        console.print("[yellow]No skills found in ./examples/data/skills[/yellow]")
        return registry

    console.print(f"\n[bold]Discovered {len(discovered)} skill(s):[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Skill Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Directory", style="dim")
    for skill in registry.list_skills():
        table.add_row(skill.name, skill.description[:80], skill.skill_dir)
    console.print(table)

    # Routing demo
    console.print("\n[bold]Routing Demo:[/bold]")
    for q in DEMO_QUERIES:
        routed = registry.route(q)
        if routed:
            names = ", ".join(s.name for s in routed)
            console.print(f"  Query: [green]'{q}'[/green] -> Skill: [cyan]{names}[/cyan]")
        else:
            console.print(f"  Query: [green]'{q}'[/green] -> Skill: [dim]None[/dim]")

    # Discovery prompt
    console.print("\n[bold]Discovery Prompt:[/bold]")
    prompt_text = registry.get_discovery_prompt()
    if prompt_text:
        console.print(Panel(prompt_text, title="System Prompt Injection"))

    return registry


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 13: Skills Discovery system."""
    setup_example(
        "Example 13: Skills Discovery",
        "Demonstrates skill discovery, routing, and activation.\n\n"
        "[dim]Run with -m interactive for chat mode with skills.[/dim]",
    )

    # Part 1 always runs
    _run_registry_demo()
    console.print()

    # Part 2: Agent
    agent = create_agent("13_skills_discovery", verbose=verbose)

    skills = agent.core.skills
    if skills:
        skill_names = [s.name for s in skills.list_skills()]
        console.print(f"[dim]Skills loaded: {skill_names}[/dim]\n")
    else:
        console.print("[yellow]Warning: No skills loaded[/yellow]\n")

    if mode == "interactive":
        def _show_routed(user_input):
            if skills:
                routed = skills.route(user_input)
                if routed:
                    routed_names = [s.name for s in routed]
                    console.print(f"[dim]Routed skills: {routed_names}[/dim]")

        interactive_loop(agent, pre_send=_show_routed)
    else:
        automatic_batch(agent, DEMO_QUERIES)


if __name__ == "__main__":
    main()
