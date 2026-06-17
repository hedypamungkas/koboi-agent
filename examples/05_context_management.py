"""Example 05: Context management comparison.

Demonstrates:
- 5 context management strategies: noop, truncation, smart_truncation, key_facts, sliding_window
- Dual mode: automatic (compare all strategies) and interactive (pick one, chat freely)
- Token estimate tracking per turn

Run:
    python examples/05_context_management.py                  # automatic mode
    python examples/05_context_management.py -m interactive   # interactive mode
"""
from __future__ import annotations

import click
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from conftest import (
    console,
    ensure_path,
    load_env,
    setup_example,
    dual_mode_options,
    run_async,
)

ensure_path()
load_env()

# 8 questions about Acme Corp policies
QUESTIONS = [
    "What are the working hours at Acme Corp?",
    "How many annual leave days do permanent employees get?",
    "Can unused leave be carried over to the next year?",
    "How many remote work days are allowed per week?",
    "What is the monthly meal allowance?",
    "Is there an internet allowance for remote work?",
    "When must the daily report be submitted?",
    "What is the WiFi speed in the office?",
]

STRATEGIES = {
    "noop": {
        "label": "Noop (no management)",
        "manager_cls": None,
        "kwargs": {},
        "max_context_tokens": 99999,
    },
    "truncation": {
        "label": "Truncation (keep_last=4)",
        "manager_cls": "TruncationManager",
        "kwargs": {"keep_last": 4},
        "max_context_tokens": 800,
    },
    "smart_truncation": {
        "label": "Smart Truncation (keep_last=4)",
        "manager_cls": "SmartTruncationManager",
        "kwargs": {"keep_last": 4},
        "max_context_tokens": 800,
    },
    "key_facts": {
        "label": "Key Facts (keep_last=4)",
        "manager_cls": "KeyFactsManager",
        "kwargs": {"keep_last": 4},
        "max_context_tokens": 800,
    },
    "sliding_window": {
        "label": "Sliding Window (keep_last=4)",
        "manager_cls": "SlidingWindowManager",
        "kwargs": {"keep_last": 4},
        "max_context_tokens": 800,
    },
}


def _build_core_for_strategy(strategy_key: str, verbose: bool):
    """Build an AgentCore with the specified context management strategy."""
    from pathlib import Path

    from koboi.client import Client
    from koboi.config import Config
    from koboi.context.manager import (
        KeyFactsManager,
        SlidingWindowManager,
        SmartTruncationManager,
        TruncationManager,
    )
    from koboi.logger import AgentLogger
    from koboi.loop import AgentCore
    from koboi.memory import ConversationMemory

    config_path = Path(__file__).parent / "05_context_management.yaml"
    base_config = Config.from_yaml(str(config_path))

    logger = AgentLogger(session_id=f"context-mgmt-{strategy_key}")
    client = Client(
        api_key=base_config.api_key or None,
        base_url=base_config.base_url or None,
        model=base_config.model,
        logger=logger,
        provider=base_config.provider,
    )

    manager_classes = {
        "TruncationManager": TruncationManager,
        "SmartTruncationManager": SmartTruncationManager,
        "KeyFactsManager": KeyFactsManager,
        "SlidingWindowManager": SlidingWindowManager,
    }

    strategy_conf = STRATEGIES[strategy_key]
    ctx_manager = None
    cls_name = strategy_conf["manager_cls"]
    if cls_name:
        cls = manager_classes[cls_name]
        kwargs = dict(strategy_conf["kwargs"])
        if cls_name == "SlidingWindowManager":
            kwargs["client"] = client
        ctx_manager = cls(**kwargs)

    max_ctx_tokens = strategy_conf["max_context_tokens"]

    memory = ConversationMemory(
        logger=logger,
        system_prompt=base_config.system_prompt or None,
    )

    return AgentCore(
        client=client,
        memory=memory,
        max_iterations=base_config.max_iterations,
        verbose=verbose,
        logger=logger,
        context_manager=ctx_manager,
        max_context_tokens=max_ctx_tokens,
    )


def run_automatic(verbose: bool):
    """Compare all 5 strategies automatically."""
    from koboi.tokens import estimate_tokens

    results = {}

    for strategy_key, strategy_conf in STRATEGIES.items():
        console.rule(f"[bold cyan]Strategy: {strategy_conf['label']}[/bold cyan]")

        core = _build_core_for_strategy(strategy_key, verbose)
        memory = core.memory

        token_log = []

        for i, question in enumerate(QUESTIONS, 1):
            console.print(f"  [yellow]Q{i}:[/yellow] {question}")

            try:
                answer = run_async(core.run(question))
                answer_text = str(answer)
                short = answer_text[:80] + "..." if len(answer_text) > 80 else answer_text
                console.print(f"  [green]A{i}:[/green] {short}")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

            messages = memory.get_messages()
            token_est = estimate_tokens(messages)
            msg_count = len(messages)

            managed = run_async(core._get_managed_messages())
            managed_count = len(managed)
            managed_tokens = estimate_tokens(managed)

            token_log.append({"turn": i, "messages": msg_count, "tokens": token_est,
                              "managed_messages": managed_count, "managed_tokens": managed_tokens})
            if managed_count < msg_count:
                console.print(
                    f"  [dim]  -> stored: {msg_count} msgs, ~{token_est} tokens | "
                    f"managed: {managed_count} msgs, ~{managed_tokens} tokens[/dim]"
                )
            else:
                console.print(f"  [dim]  -> {msg_count} messages, ~{token_est} tokens[/dim]")

        results[strategy_key] = {
            "label": strategy_conf["label"],
            "token_log": token_log,
            "final_messages": len(memory.get_messages()),
            "final_tokens": estimate_tokens(memory.get_messages()),
        }
        console.print()

    # Final comparison table
    console.rule("[bold magenta]Strategy Comparison[/bold magenta]")

    table = Table(title="Context Strategy Comparison", show_lines=True)
    table.add_column("Strategy", style="bold")
    table.add_column("Final Msgs", justify="right")
    table.add_column("Final Tokens", justify="right")
    for i in range(1, 9):
        table.add_column(f"Turn {i}\n(tokens)", justify="right", style="dim")

    for key, data in results.items():
        row = [data["label"], str(data["final_messages"]), str(data["final_tokens"])]
        for entry in data["token_log"]:
            row.append(str(entry["tokens"]))
        table.add_row(*row)

    console.print(table)
    console.print("\n[dim]Token estimates use approximation len(text)//3.[/dim]")
    console.print("[dim]Context management strategies activate when tokens exceed max_context_tokens.[/dim]")


def run_interactive(verbose: bool):
    """Pick a strategy, then chat freely."""
    console.print("[bold]Available strategies:[/bold]")
    for i, key in enumerate(STRATEGIES.keys(), 1):
        console.print(f"  {i}. {STRATEGIES[key]['label']}")

    choices = [str(i) for i in range(1, len(STRATEGIES) + 1)]
    choice = Prompt.ask("Pick a strategy number", choices=choices)
    selected_key = list(STRATEGIES.keys())[int(choice) - 1]
    selected_label = STRATEGIES[selected_key]["label"]

    console.print(f"\n[bold cyan]Using strategy: {selected_label}[/bold cyan]\n")

    core = _build_core_for_strategy(selected_key, verbose)

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        stripped = user_input.strip().lower()
        if stripped in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break
        if not user_input.strip():
            continue

        try:
            result = run_async(core.run(user_input))
            from rich.markdown import Markdown
            console.print(Panel(Markdown(str(result)), title=f"Agent ({selected_label})", border_style="green"))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 05: Context management comparison."""
    setup_example(
        "Example 05: Context Management Comparison",
        "Comparing 5 context management strategies.\n"
        "Each strategy receives 8 questions about Acme Corp policies.\n\n"
        "[dim]Strategies: noop, truncation, smart_truncation, key_facts, sliding_window[/dim]",
    )

    if mode == "interactive":
        run_interactive(verbose)
    else:
        run_automatic(verbose)


if __name__ == "__main__":
    main()
