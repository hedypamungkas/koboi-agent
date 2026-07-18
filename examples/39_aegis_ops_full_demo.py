"""Example 39: Aegis Ops -- the "full sample" demo.

Loads configs/aegis_ops_full.yaml, which turns on (nearly) every top-level
koboi-agent subsystem in one coherent scenario: a DAG-orchestrated customer-ops
agent for a fictional SaaS ("Northwind Cloud") that triages support requests
(intake -> support_kb / ops_runbook -> synthesis), backed by RAG, proactive
memory, self-healing, handover, a restricted sandbox, MCP ticketing, media
(mock), and a servable REST/SSE surface with async jobs.

Two modes:
  --mock  (default): offline scripted demo (no API key needed). Patches
          koboi.facade.RetryClient with a scripted stub so every LLM call in
          the DAG is deterministic and $0.
  --live : real LLM calls. Requires OPENAI_API_KEY (and OPENAI_API_KEY_BACKUP,
          which can just be the same key -- see the config's provider-pool
          comment for why a *nested* ${VAR:${VAR2:}} default doesn't work).

This example also documents, in its printed output, several genuine
architecture gaps this scenario surfaced while being built (verified against
koboi 0.18.2's source, not assumed):

  1. `orchestration.enabled: true` NEVER calls `AgentAssembler.build_tools()`
     -- the top-level `tools:` section in the config is INERT once
     orchestration is on. Only each DAG node's own `tools:` block governs
     what that node can call. (koboi/facade.py `_build_orchestration`)
  2. `delegate_tasks` / `task_create` / `task_list` / `task_update` have no
     dependency-injection path in orchestration mode (`_setup_subagent` /
     `_setup_tasks` only run in the single-agent `AgentAssembler.build()`
     path) -- a DAG node with these tools in its `tools.builtin` list will
     get "Error: ... not initialized" if it ever calls them. `call_peer_agent`
     is the one exception: `_build_tools_from_config` injects `peer_registry`
     directly, so cross-instance A2A DOES work inside a DAG node.
  3. `transfer_to_human` (and the low-grounding HandoverDetectionHook trigger)
     raise `AgentHandoverError` from *inside* `agent.run()`, but
     `Orchestrator._execute_node`'s `except Exception` (koboi/orchestration/
     orchestrator.py) catches it as a generic node failure -- it never
     surfaces as `AgentHandoverError` to a DAG caller the way it does for a
     single-agent `KoboiAgent.run()` (see examples/38 / examples/35). A live
     run surfaces a related cascade: `_input_for` (orchestrator.py) injects each
     upstream node's natural-language answer into its downstream nodes' input
     (`"Upstream results:\n[intake]: <answer>..."`), and HandoverDetectionHook
     scans ALL input text on PRE_INPUT for `ask_patterns` -- so an upstream node
     whose answer says something like "talk to a human" trips handover on every
     downstream node, even when the original user message asked no such thing.
     (Observed on a real gpt-5.4 run: Turn 1's `intake` answer recommended
     human escalation, and `ops_runbook`/`support_kb` both then failed with
     `AgentHandoverError: user requested a human` from `_validate_input`.)
  4. `self_healing.tool_verification.enabled` (the P4 CRITIC tool-grounded
     claim check) silently never fires in orchestration mode: it's wired via
     `ReflectionHook(tools=self.tools, ...)` in `build_opt_in_hooks`, and
     `self.tools` is the same (always-None-in-orchestration) attribute as #1.
  5. Per-node `ConversationMemory` in a DAG is a fresh in-memory instance
     (`AgentFactory.create_configured_agent` never passes `memory=`) -- so
     `memory.proactive` (D extract / C recall / B core-block) has nothing to
     bind to inside a DAG node; it only extracts/recalls against the *parent*
     KoboiAgent-level memory, which nothing in this DAG ever talks to.
  6. Without `orchestration.execution.full_graph: true`, `dag` mode only runs
     the router's top-matched subset of nodes -- a node with no `depends_on`
     edge reached by the router's picks (e.g. `support_kb` when the keyword
     router only matches `intake`) silently never executes, and
     `RunResult.metadata['agents_used']` is the only way to notice (verified
     empirically while building evals/aegis_ops.eval.py -- omitting
     `full_graph: true` there made a "grounded RAG answer" eval assertion
     look reasonable while `support_kb` had, in fact, never run).
  7. The `koboi eval-test --mock` runner's client-swap
     (`koboi/eval/t/runner.py::_build_mock_agent`) only reaches
     `agent.orchestrator.client` (the router/planner/synthesis client) for an
     orchestration config -- it does NOT reach into each DAG node's own
     dedicated `AgentCore.client`. A `--mock` run of a `dag`-mode eval still
     fires real HTTP calls per node; they get swallowed as node failures (see
     gap #3's `except Exception`) while the overall eval still reports
     `success=True` -- a false-green gate. `evals/aegis_ops.eval.py` is
     therefore LIVE-only (mirrors `evals/deep_research_citations.eval.py`'s
     same constraint for a different orchestration mode).

None of this makes the config "wrong" -- every key is schema-valid and the
subsystems that DO cross into orchestration mode (RAG per-node, sandbox,
per-node LLM pools/determinism/output_schema, shared MCP/websearch/media,
mode-blocking, guardrails, journal, self_healing's *reflection* trigger and
escalation ladder) all verifiably work. This example demonstrates what
actually fires, and calls out what doesn't, so it doubles as ground truth for
orchestration-mode feature coverage.

Run:
    python examples/39_aegis_ops_full_demo.py --mock
    OPENAI_API_KEY=... OPENAI_API_KEY_BACKUP=... python examples/39_aegis_ops_full_demo.py --live
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

CONFIG_PATH = PROJECT_ROOT / "configs" / "aegis_ops_full.yaml"
console = Console()


# ---------------------------------------------------------------------------
# --mock: a scripted stub client so the whole DAG runs offline, deterministically.
# ---------------------------------------------------------------------------


class _ScriptedClient:
    """Minimal LLMClient stub: returns a fixed final-answer string for any call.

    Real DAG nodes issue their own LLM calls (intake, support_kb, ops_runbook,
    synthesis each call .complete() at least once); this stub does not try to
    emulate tool-calling loops -- it always returns plain content, so no node
    ever asks for a tool. That's deliberate for --mock: it proves the DAG
    wiring (routing, RAG augmentation being *injected into the prompt*,
    sandbox/journal/mode-block plumbing) without depending on tool-call
    parsing fidelity from a fake model.
    """

    def __init__(self, label: str = "aegis-mock") -> None:
        self._label = label
        self.call_count = 0

    @property
    def model(self) -> str:
        return "mock-model"

    async def complete(self, messages, tools=None, response_format=None):
        from koboi.types import AgentResponse, TokenUsage

        self.call_count += 1
        # Cheap heuristic: if the augmented prompt carries RAG context, echo a
        # grounded-looking answer; else a generic resolution. Good enough to
        # show every node produced *some* answer offline.
        last = messages[-1].get("content", "") if messages else ""
        text = str(last)
        if "refund" in text.lower() or "policy" in text.lower():
            content = "Northwind Cloud offers a 30-day money-back guarantee for Enterprise/Pro tiers."
        elif "500" in text or "checkout" in text.lower() or "down" in text.lower():
            content = "Diagnosed: stale DB connection pool after deploy. Restarting checkout-service should clear it."
        else:
            content = '{"resolution": "Reviewed the request and routed it to the correct specialist.", "confidence": 0.7}'
        return AgentResponse(content=content, tool_calls=[], usage=TokenUsage(prompt_tokens=20, completion_tokens=20))

    async def complete_stream(self, messages, tools=None, response_format=None):
        from koboi.events import CompleteEvent, TextDeltaEvent

        resp = await self.complete(messages, tools, response_format=response_format)
        yield TextDeltaEvent(content=resp.content or "")
        yield CompleteEvent(response=resp, content=resp.content or "")

    async def get_embeddings(self, text):
        # None -> RAG's semantic leg falls back to keyword (documented, fail-soft
        # behavior); proactive-memory recall similarly degrades to a no-op.
        return None

    async def close(self):
        pass


def _patch_mock_clients() -> None:
    """Monkeypatch every LLM-client construction site the facade uses.

    Fixing example 38's audit finding: --mock must not let self_healing's
    critic or guardrails.output's grounding_check leak a real (failing,
    fail-soft-swallowed) network call. koboi/facade.py does `from koboi.client
    import RetryClient` at MODULE level (not function-local), so patching
    koboi.client.RetryClient alone is not enough -- the name `RetryClient`
    inside koboi.facade's namespace must be repointed directly.
    koboi.llm.factory.create_client (used for critic_llm / grounding_check /
    handoff-digest side-LLM clients) IS imported function-local, so patching
    the module attribute there is sufficient.
    """
    import koboi.facade as _facade_mod
    import koboi.llm.factory as _factory_mod

    def _fake_retry_client(*args, **kwargs):
        return _ScriptedClient()

    def _fake_create_client(*args, **kwargs):
        return _ScriptedClient()

    _facade_mod.RetryClient = _fake_retry_client  # type: ignore[assignment]
    _factory_mod.create_client = _fake_create_client  # type: ignore[assignment]


def _print_coverage_table(agent) -> None:
    """Print a wired/inert checklist for every subsystem this config touches."""
    table = Table(title="Subsystem coverage (this run)", show_header=True, header_style="bold magenta")
    table.add_column("Subsystem")
    table.add_column("Status")
    table.add_column("Note")

    orch = agent._orchestrator
    rows = [
        ("orchestration.dag", "wired", f"{len(orch._agents_map)} nodes: {', '.join(orch._agents_map)}"),
        ("rag (support_kb node)", "wired", "on_the_fly augmentation, hybrid retriever + heuristic rerank"),
        ("sandbox (restricted)", "wired", "ops_runbook's run_shell/git tools are confined to ./.aegis_workdir"),
        ("mode-blocking (ModeHook)", "wired", "shared hook_chain enforces act-mode across every node"),
        ("guardrails (input/output)", "wired", "shared hook_chain; grounding_check needs a real LLM (--live only)"),
        ("journal + resume", "wired", "per-run steps recorded to aegis_ops_memory.db"),
        ("self_healing: reflection trigger", "wired", "ReflectionHook is in the shared hook_chain"),
        ("self_healing: tool_verification (CRITIC)", "INERT", "ReflectionHook._tools is None in orchestration mode"),
        ("memory.proactive (D/C/B)", "INERT for DAG nodes", "each node gets its own fresh ConversationMemory"),
        ("handover (transfer_to_human)", "degraded", "raises AgentHandoverError, but the DAG node catches it as a failure"),
        ("delegate_tasks / task_*", "INERT for DAG nodes", "no dependency-injection path in orchestration mode"),
        ("call_peer_agent (A2A)", "wired", "peer_registry IS injected per-node (verified working)"),
        ("mcp (ticketing)", "best-effort", "stdio subprocess; env-dependent (see caveats)"),
        ("media (mock image)", "wired", "shared MediaBackend forwarded into every node's registry"),
        ("providers/pools (failover)", "wired", "ops_runbook's llm: {pool: primary_pool}"),
    ]
    for name, status, note in rows:
        style = "green" if status == "wired" else ("yellow" if "best-effort" in status or "degraded" in status else "red")
        table.add_row(name, f"[{style}]{status}[/{style}]", note)
    console.print(table)


async def run_scenarios(agent, mock: bool) -> None:
    from koboi.exceptions import AgentError

    console.print(Panel("[bold]Turn 1[/bold]: infra incident -> intake -> ops_runbook -> synthesis"))
    try:
        result = await agent.run("My checkout page is throwing 500s, tier=enterprise")
        console.print(f"[green]OK[/green] execution_mode={result.metadata.get('execution_mode')} "
                      f"agents_used={result.metadata.get('agents_used')}")
        console.print(Panel(str(result.content)[:400], title="Answer"))
    except AgentError as e:
        console.print(f"[red]AgentError:[/red] {e}")
    console.print()

    console.print(Panel("[bold]Turn 2[/bold]: product question -> intake -> support_kb (RAG)"))
    try:
        result = await agent.run("What is your refund policy?")
        console.print(f"[green]OK[/green] execution_mode={result.metadata.get('execution_mode')} "
                      f"agents_used={result.metadata.get('agents_used')}")
        console.print(Panel(str(result.content)[:400], title="Answer"))
    except AgentError as e:
        console.print(f"[red]AgentError:[/red] {e}")
    console.print()

    console.print(Panel(
        "[bold]Turn 3[/bold]: 'talk to a human' -- demonstrates the documented handover gap.\n"
        "[dim]In a single-agent config (mode: act, no orchestration:), this would raise "
        "AgentHandoverError straight out of agent.run() (see examples/35, examples/38). "
        "Here, whichever DAG node's LLM decides to call transfer_to_human (or trips the "
        "low-grounding hook) has that error caught by Orchestrator._execute_node's "
        "generic except-Exception, so it surfaces as a normal (failed) node answer "
        "instead.[/dim]"
    ))
    try:
        result = await agent.run("I want to talk to a human about my bill")
        console.print(f"[yellow]No AgentHandoverError raised (expected, see note above)[/yellow]")
        console.print(Panel(str(result.content)[:400], title="Answer"))
    except AgentError as e:
        console.print(f"[red]AgentError:[/red] {e}")
    console.print()


def _print_companion_commands() -> None:
    console.print(Panel(
        "[bold]Companion CLI flows (printed only -- separate offline steps, not run inline)[/bold]\n\n"
        "Deterministic workflow export + offline replay (see examples/36, 37):\n"
        f"  koboi export {CONFIG_PATH.relative_to(PROJECT_ROOT)} --name aegis-ops-v1\n"
        f"  koboi capture --with-cache {CONFIG_PATH.relative_to(PROJECT_ROOT)} --name aegis-ops-v1 "
        '-m "My checkout page is throwing 500s, tier=enterprise"\n'
        f"  koboi run {CONFIG_PATH.relative_to(PROJECT_ROOT)} --workflow aegis-ops-v1 --replay-mode replay "
        '-m "My checkout page is throwing 500s, tier=enterprise"\n\n'
        "CI-gated eval (grounding/tool-use quality gate -- LIVE only, needs\n"
        "OPENAI_API_KEY: --mock is unsupported for orchestration DAG configs,\n"
        "since the t-eval mock runner only swaps the Orchestrator's own client,\n"
        "not each DAG node's dedicated per-node client -- see evals/aegis_ops.eval.py):\n"
        "  OPENAI_API_KEY=... koboi eval-test evals/aegis_ops.eval.py --strict\n\n"
        "Serve over REST/SSE (needs the \\[api] extra):\n"
        f"  koboi serve {CONFIG_PATH.relative_to(PROJECT_ROOT)}\n",
        title="Companion commands",
        border_style="blue",
    ))


@click.command()
@click.option("--live", is_flag=True, help="Use real LLM calls (needs OPENAI_API_KEY).")
def main(live: bool):
    """Aegis Ops -- the full-sample demo."""
    mock = not live
    # Load .env from the project root (mirrors koboi/cli.py + examples/conftest.py).
    # This script calls KoboiAgent.from_config directly (not the `koboi` CLI), so
    # without this the YAML's ${OPENAI_API_KEY:}/${OPENAI_BASE_URL:}/... placeholders
    # would resolve against an empty environment.
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass

    console.print(
        Panel.fit(
            "[bold]Aegis Ops -- Full Sample Demo[/bold]\n"
            "Mode: " + ("MOCK (offline, $0)" if mock else "LIVE LLM") + "\n"
            f"Config: {CONFIG_PATH}",
            border_style="blue",
        )
    )

    if mock:
        os.environ.setdefault("OPENAI_API_KEY", "mock-key")
        os.environ.setdefault("OPENAI_API_KEY_BACKUP", "mock-key")
        _patch_mock_clients()
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            console.print("[red]--live requires OPENAI_API_KEY (and ideally OPENAI_API_KEY_BACKUP).[/red]")
            sys.exit(1)
        # The pool's openai_fallback member defaults base_url to api.openai.com;
        # point it at the same gateway as the primary so both pool members are
        # reachable (the failover policy only calls fallback if primary fails, but
        # a 401-on-fallback would still trip the circuit breaker after 3 errors).
        os.environ.setdefault("OPENAI_API_KEY_BACKUP", os.environ["OPENAI_API_KEY"])
        os.environ.setdefault("OPENAI_FALLBACK_BASE_URL", os.environ.get("OPENAI_BASE_URL", ""))


    # The config's `hooks.on_event[].command` needs an absolute path to the
    # forwarder script -- sandbox.run() resolves a relative one against
    # sandbox.workdir (./.aegis_workdir), not the repo root. Set the env vars
    # the YAML's ${AEGIS_PROJECT_ROOT}/${AEGIS_PYTHON} placeholders read.
    os.environ.setdefault("AEGIS_PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("AEGIS_PYTHON", sys.executable)

    from koboi.config import Config
    from koboi.facade import KoboiAgent

    # The restricted sandbox does not create its own workdir (see
    # examples/32_sandbox_and_resume.py's same pattern) -- ensure it exists.
    cfg = Config.from_yaml(str(CONFIG_PATH))
    os.makedirs(cfg.get("sandbox", "workdir", default="."), exist_ok=True)

    agent = KoboiAgent.from_config(str(CONFIG_PATH))
    _print_coverage_table(agent)
    console.print()

    asyncio.run(run_scenarios(agent, mock))
    _print_companion_commands()


if __name__ == "__main__":
    main()
