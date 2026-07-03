"""Example 02: Config Showcase -- Load and compare all production YAML configs.

Demonstrates:
- Loading every config from configs/ directory
- Comparing agent capabilities per config (tools, context strategy, guardrails, etc.)
- Showing how config-driven architecture enables zero-code hyperparameter tuning
- No API key required (config inspection only, no LLM calls)

Run:
    python examples/02_config_showcase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from koboi.config import Config

console = Console()

CONFIGS_DIR = PROJECT_ROOT / "configs"

FEATURE_ICONS = {
    "tools": "[cyan]Tools[/cyan]",
    "rag": "[magenta]RAG[/magenta]",
    "guardrails": "[yellow]Guardrails[/yellow]",
    "context": "[blue]Context[/blue]",
    "policy": "[red]Policy[/red]",
    "skills": "[green]Skills[/green]",
    "mcp": "[bright_magenta]MCP[/bright_magenta]",
    "tracing": "[bright_blue]Tracing[/bright_blue]",
    "harness": "[bright_cyan]Harness[/bright_cyan]",
}


def load_configs() -> list[tuple[str, Config]]:
    """Load all YAML configs from configs/ directory."""
    configs = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            cfg = Config.from_yaml(path)
            configs.append((path.stem, cfg))
        except Exception as e:
            console.print(f"[dim]Skip {path.name}: {e}[/dim]")
    return configs


def extract_features(cfg: Config) -> dict:
    """Extract enabled features and key parameters from a config."""
    features = {}

    # Agent basics
    features["name"] = cfg.agent_name
    features["provider"] = cfg.provider
    features["model"] = cfg.model if not cfg.model.startswith("${") else "(from env)"
    features["max_iterations"] = cfg.max_iterations
    features["max_context_tokens"] = cfg.get("context", "max_context_tokens", default=8000)

    # Tools
    builtin = cfg.get("tools", "builtin", default=[])
    features["tools"] = ", ".join(builtin) if builtin else "-"
    features["tool_defaults"] = bool(cfg.get("tools", "defaults"))
    features["tool_overrides"] = list(cfg.get("tools", "overrides", default={}).keys())

    # Context
    strategy = cfg.get("context", "strategy", default="noop")
    features["context_strategy"] = strategy
    features["keep_last"] = cfg.get("context", "keep_last")

    # RAG
    features["rag_enabled"] = cfg.rag_enabled
    if cfg.rag_enabled:
        features["rag_chunker"] = cfg.get("rag", "chunker", default="paragraph")
        features["rag_top_k"] = cfg.get("rag", "top_k", default=3)
        features["rag_overlap"] = cfg.get("rag", "overlap")
        features["rag_max_chunk"] = cfg.get("rag", "max_chunk_size")

    # Guardrails
    input_grd = cfg.get("guardrails", "input")
    output_grd = cfg.get("guardrails", "output")
    rate_limit = cfg.get("guardrails", "rate_limit")
    features["guardrails_input"] = bool(input_grd)
    features["guardrails_output"] = bool(output_grd)
    features["rate_limit_session"] = rate_limit.get("max_calls_per_session") if rate_limit else None
    features["rate_limit_per_min"] = rate_limit.get("max_calls_per_minute") if rate_limit else None
    features["rate_window"] = rate_limit.get("rate_window_seconds") if rate_limit else None

    # Policy
    rules = cfg.get("policy", "rules", default=[])
    features["policy_rules"] = len(rules) if rules else 0

    # Skills
    features["skills"] = bool(cfg.get("skills", "search_paths"))

    # MCP
    features["mcp_servers"] = len(cfg.get("mcp", "servers", default=[]))

    # Tracing
    features["tracing"] = cfg.get("tracing", "provider", default="")

    # Harness
    features["telemetry"] = cfg.get("harness", "telemetry", default=False)
    features["carryover"] = cfg.get("harness", "carryover", default=False)
    features["doom_loop"] = bool(cfg.get("harness", "doom_loop"))
    features["carryover_limits"] = bool(cfg.get("harness", "carryover_limits"))
    features["health_weights"] = bool(cfg.get("harness", "health_weights"))

    # LLM tuning
    features["temperature"] = cfg.temperature
    features["max_retries"] = cfg.max_retries if cfg.max_retries != 3 else None
    features["retry_backoff"] = cfg.retry_backoff_base if cfg.retry_backoff_base != 2.0 else None

    return features


def print_overview_table(configs: list[tuple[str, Config]]):
    """Print a comparison table of all configs."""
    table = Table(
        title="Config Comparison: All Production YAML Files",
        show_lines=True,
        title_style="bold white",
    )
    table.add_column("Config", style="bold", min_width=18)
    table.add_column("Provider", width=10)
    table.add_column("Iterations", width=8)
    table.add_column("Ctx Tokens", width=10)
    table.add_column("Strategy", width=16)
    table.add_column("Tools", max_width=40)
    table.add_column("RAG", width=5)
    table.add_column("Guardrails", width=12)
    table.add_column("Policy", width=6)
    table.add_column("Harness", width=14)

    for name, cfg in configs:
        f = extract_features(cfg)

        tools_str = f["tools"] if len(f["tools"]) < 40 else f["tools"][:37] + "..."

        rag_str = "[green]ON[/green]" if f["rag_enabled"] else "[dim]off[/dim]"

        grd_parts = []
        if f["guardrails_input"]:
            grd_parts.append("in")
        if f["guardrails_output"]:
            grd_parts.append("out")
        if f["rate_limit_session"]:
            grd_parts.append(f"rl:{f['rate_limit_session']}")
        grd_str = "/".join(grd_parts) if grd_parts else "[dim]-[/dim]"

        policy_str = str(f["policy_rules"]) if f["policy_rules"] else "-"

        harness_parts = []
        if f["telemetry"]:
            harness_parts.append("tel")
        if f["carryover"]:
            harness_parts.append("car")
        if f["doom_loop"]:
            harness_parts.append("doom")
        harness_str = "+".join(harness_parts) if harness_parts else "-"

        table.add_row(
            name,
            f["provider"],
            str(f["max_iterations"]),
            str(f["max_context_tokens"]),
            f["context_strategy"],
            tools_str,
            rag_str,
            grd_str,
            policy_str,
            harness_str,
        )

    console.print(table)


def print_detail_panels(configs: list[tuple[str, Config]]):
    """Print detailed panel for each config showing tunable parameters."""
    for name, cfg in configs:
        f = extract_features(cfg)

        lines = [f"[bold]Agent:[/bold] {f['name']}"]
        lines.append(f"[bold]LLM:[/bold] {f['provider']} / {f['model']}")
        lines.append(f"[bold]Max iterations:[/bold] {f['max_iterations']}")

        # LLM tuning knobs
        llm_knobs = []
        if f["temperature"] is not None:
            llm_knobs.append(f"temperature={f['temperature']}")
        if f["max_retries"]:
            llm_knobs.append(f"max_retries={f['max_retries']}")
        if f["retry_backoff"]:
            llm_knobs.append(f"retry_backoff={f['retry_backoff']}")
        if llm_knobs:
            lines.append(f"[bold]LLM tuning:[/bold] {', '.join(llm_knobs)}")

        lines.append("")
        lines.append(f"[bold]Context:[/bold] {f['context_strategy']} (tokens={f['max_context_tokens']})")
        if f["keep_last"]:
            lines.append(f"  keep_last={f['keep_last']}")

        # Tools
        if f["tools"] != "-":
            lines.append(f"\n[bold]Tools:[/bold] {f['tools']}")
        if f["tool_defaults"]:
            lines.append("  [green]Tool defaults configured[/green]")
        if f["tool_overrides"]:
            lines.append(f"  [green]Per-tool overrides: {', '.join(f['tool_overrides'])}[/green]")

        # RAG
        if f["rag_enabled"]:
            lines.append(f"\n[bold]RAG:[/bold] chunker={f['rag_chunker']}, top_k={f['rag_top_k']}")
            if f["rag_overlap"]:
                lines.append(f"  overlap={f['rag_overlap']}")
            if f["rag_max_chunk"]:
                lines.append(f"  max_chunk_size={f['rag_max_chunk']}")

        # Guardrails
        grd_parts = []
        if f["guardrails_input"]:
            grd_parts.append("input detection")
        if f["guardrails_output"]:
            grd_parts.append("output detection")
        if grd_parts:
            lines.append(f"\n[bold]Guardrails:[/bold] {', '.join(grd_parts)}")
        if f["rate_limit_session"]:
            lines.append(f"  rate_limit: {f['rate_limit_session']}/session, {f['rate_limit_per_min']}/min")
            if f["rate_window"]:
                lines.append(f"  rate_window: {f['rate_window']}s")

        # Policy
        if f["policy_rules"]:
            lines.append(f"\n[bold]Policy rules:[/bold] {f['policy_rules']}")

        # Skills
        if f["skills"]:
            lines.append("\n[bold]Skills:[/bold] enabled")

        # MCP
        if f["mcp_servers"]:
            lines.append(f"\n[bold]MCP servers:[/bold] {f['mcp_servers']}")

        # Tracing
        if f["tracing"]:
            lines.append(f"\n[bold]Tracing:[/bold] {f['tracing']}")

        # Harness
        harness_parts = []
        if f["telemetry"]:
            harness_parts.append("telemetry")
        if f["carryover"]:
            harness_parts.append("carryover")
        if f["doom_loop"]:
            harness_parts.append("doom_loop")
        if f["carryover_limits"]:
            harness_parts.append("carryover_limits (custom)")
        if f["health_weights"]:
            harness_parts.append("health_weights (custom)")
        if harness_parts:
            lines.append(f"\n[bold]Harness:[/bold] {', '.join(harness_parts)}")

        console.print(Panel("\n".join(lines), title=f"[bold]{name}[/bold]", border_style="cyan"))
        console.print()


def print_tuning_guide():
    """Print a guide on how to tune configs without changing code."""
    console.rule("[bold magenta]Config-Driven Tuning Guide[/bold magenta]")

    guide = """[bold]All parameters below can be changed via YAML without modifying any Python code.[/bold]

[dim]LLM Parameters (configs/*.yaml → llm section):[/dim]
  temperature          float   Controls creativity (0.0=deterministic, 1.0=creative)
  max_retries          int     Retry attempts on retryable errors
  retry_backoff_base   float   Exponential backoff multiplier
  transport_retries    int     HTTP transport level retries
  timeout              float   HTTP request timeout in seconds
  max_tokens           int     Max generation tokens

[dim]Context Management (→ context section):[/dim]
  max_context_tokens   int     Token budget for context window
  keep_last            int     Messages to preserve in truncation
  summarization_truncation int Char limit per message in sliding window summary

[dim]RAG Pipeline (→ rag section):[/dim]
  chunk_size           int     Chunk size in characters
  overlap              int     Overlap between chunks
  max_chunk_size       int     Max chunk size for sentence/paragraph chunkers
  top_k                int     Number of chunks to retrieve

[dim]Rate Limiting (→ guardrails.rate_limit section):[/dim]
  max_calls_per_session  int   Max tool calls per session
  max_calls_per_minute   int   Max tool calls per minute
  rate_window_seconds    float Sliding window duration

[dim]Tool Execution (→ tools.defaults / tools.overrides):[/dim]
  Global defaults:      timeout, max_output
  Per-tool overrides:   shell.timeout, git.max_log_count, etc.

[dim]Doom Loop Detection (→ harness.doom_loop section):[/dim]
  consecutive_identical_threshold  int   Repetition trigger
  repeating_pattern_window         int   Pattern detection window
  error_retry_threshold            int   Error repetition trigger

[dim]Orchestration (for multi-agent setups):[/dim]
  quality_threshold     float   When to request revision (0-1)
  max_revisions         int     Max revision attempts
  agent_context_tokens  int     Token budget per agent (was hardcoded 2000!)
  top_k                 int     RAG results per agent
  confidence_threshold  float   Router confidence cutoff"""

    console.print(Panel(guide, title="Zero-Code Hyperparameter Tuning", border_style="green"))


def main():
    console.print(
        Panel(
            "[bold]Config Showcase[/bold]\n\n"
            "Loading and comparing all production YAML configs.\n"
            "No API key required -- config inspection only.",
            title="Example 02",
        )
    )

    configs = load_configs()
    if not configs:
        console.print("[red]No configs found in configs/ directory![/red]")
        return

    console.print(f"\n[bold]Found {len(configs)} configs in configs/[/bold]\n")

    # Section 1: Overview table
    console.rule("[bold cyan]Overview[/bold cyan]")
    print_overview_table(configs)
    console.print()

    # Section 2: Detailed panels
    console.rule("[bold cyan]Config Details[/bold cyan]")
    print_detail_panels(configs)

    # Section 3: Tuning guide
    print_tuning_guide()

    console.print(f"\n[dim]Config files loaded from: {CONFIGS_DIR}/[/dim]")
    console.print("[dim]Edit any YAML file to tune agent behavior without changing code.[/dim]")


if __name__ == "__main__":
    main()
