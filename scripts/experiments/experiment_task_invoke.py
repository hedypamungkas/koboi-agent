#!/usr/bin/env python3
"""Behavioral experiment: does gpt-5.4-mini invoke the built-in `task` tool
on IMPLICIT multi-step prompts, with vs without a system-prompt nudge?

Controlled variables (held constant):
  - model: gpt-5.4-mini  (gateway https://api.surplusintelligence.ai/v1)
  - prompts: genuinely multi-step but NEVER mention "task/todo/track/steps"
  - mode: act, max_iterations: 15, in-memory memory backend

Independent variable: system_prompt (generic vs gold-standard task nudge)
                    + task-tool enablement.

Dependent variable: does RunResult.tools_used contain any task_* / task_create?

Reads .env (OPENAI_*). Never prints the API key.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

from koboi.facade import KoboiAgent  # noqa: E402

GATEWAY = os.environ.get("OPENAI_BASE_URL")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
print(f"[setup] gateway={GATEWAY} model={MODEL}")
assert GATEWAY and os.environ.get("OPENAI_API_KEY"), "OPENAI env not loaded"

COMMON = """\
llm:
  provider: openai
  model: ${OPENAI_MODEL}
  api_key: ${OPENAI_API_KEY}
  base_url: ${OPENAI_BASE_URL}
context:
  strategy: noop
memory:
  backend: memory
"""

TASK_TOOLS = "[task_create, task_list, task_get, task_update, task_add_dependency]"

# Verbatim gold-standard nudge from examples/26_task_management.yaml:5-25
NUDGE = """\
You are a helpful assistant that tracks work using tasks.

When given a multi-step request, follow this workflow:
1. CREATE tasks first -- call task_create for each step before doing any work
2. Use blocked_by to set up dependencies between tasks
3. START a task -- call task_update with status="in_progress" before working on it
4. DO the work -- use available tools to complete the step
5. COMPLETE the task -- call task_update with status="completed" when done
6. REPEAT for each remaining task

IMPORTANT RULES:
- Always create ALL tasks at the start so the user can see the full plan
- Always update task status as you work -- this is how progress is tracked
- Never skip task_create or task_update -- they are required for tracking
"""

GENERIC = (
    "You are a helpful AI assistant. Answer the user's request thoroughly "
    "using the available tools when useful."
)


def cfg(name: str, system_prompt: str, tools: str) -> str:
    indented = system_prompt.replace("\n", "\n    ")
    return (
        f"agent:\n"
        f"  name: {name}\n"
        f"  mode: act\n"
        f"  max_iterations: 15\n"
        f"  system_prompt: |\n"
        f"    {indented}\n"
        f"{COMMON}\n"
        f"tools:\n"
        f"  builtin: {tools}\n"
    )


CONDS = {
    "C1_task_NO_nudge": cfg("exp-c1", GENERIC, TASK_TOOLS),
    "C2_task_WITH_nudge": cfg("exp-c2", NUDGE, TASK_TOOLS),
    "C3_NO_task_tools": cfg("exp-c3", GENERIC, "[calculate]"),
}

# IMPLICIT multi-step prompts: real multi-step work, zero task-vocabulary.
PROMPTS = {
    "P1_conference": (
        "I'm organizing a 3-day tech conference for 200 attendees. I need to "
        "compare three venue options within a $50k budget, design a catering "
        "plan that covers vegan and gluten-free needs, and outline a run-sheet "
        "for the opening keynote day. Help me think through this."
    ),
    "P2_onboarding": (
        "I'm onboarding a new backend engineer next week. Help me prepare: a "
        "laptop setup checklist, a 30-day learning curriculum for our codebase, "
        "and a schedule of introductory meetings with the three team leads."
    ),
}

REPS = 4  # per (condition, prompt)


async def one_run(cond_name: str, yaml_str: str, prompt: str, run_idx: int) -> dict:
    """Fresh agent per run => clean TaskManager. Retry on empty/timeout (gateway flake)."""
    for attempt in range(3):
        try:
            agent = KoboiAgent.from_config_string(yaml_str, verbose=False)
            t0 = time.time()
            result = await asyncio.wait_for(agent.run(prompt), timeout=180)
            dt = time.time() - t0
            tools = list(result.tools_used)
            any_task = any(t.startswith("task_") for t in tools)
            has_create = "task_create" in tools
            empty = (not result.content) and (not tools)
            # retry on the known gateway empty-completion flake
            if empty and attempt < 2:
                print(f"  [{cond_name}#{run_idx}] empty completion, retry {attempt+1}")
                await asyncio.sleep(1)
                continue
            return dict(
                cond=cond_name, run=run_idx, tools=tools, any_task=any_task,
                has_create=has_create, iters=result.iterations_used,
                success=result.success, empty=empty, dt=round(dt, 1),
                err=str(result.error) if result.error else "",
            )
        except asyncio.TimeoutError:
            print(f"  [{cond_name}#{run_idx}] timeout, retry {attempt+1}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{cond_name}#{run_idx}] {type(e).__name__}: {e}; retry {attempt+1}")
        await asyncio.sleep(1)
    return dict(cond=cond_name, run=run_idx, tools=[], any_task=False, has_create=False,
                iters=0, success=False, empty=True, dt=0, err="FAILED_AFTER_RETRIES")


async def main() -> None:
    results: list[dict] = []
    for cond_name, yaml_str in CONDS.items():
        for pname, prompt in PROMPTS.items():
            reps = REPS if cond_name != "C3_NO_task_tools" else 1
            for r in range(reps):
                rec = await one_run(cond_name, yaml_str, prompt, r)
                rec["prompt"] = pname
                results.append(rec)
                tag = "TASK" if rec["any_task"] else ("EMPTY" if rec["empty"] else "no-task")
                print(f"[{cond_name}/{pname} #{r}] {tag} iters={rec['iters']} "
                      f"tools={rec['tools']} ({rec['dt']}s)")
                await asyncio.sleep(0.4)

    print("\n" + "=" * 72)
    print("SUMMARY — task-tool invocation rate (gpt-5.4-mini, implicit multi-step)")
    print("=" * 72)
    agg = defaultdict(lambda: dict(n=0, any_task=0, create=0, empty=0))
    for rec in results:
        a = agg[rec["cond"]]
        a["n"] += 1
        if rec["any_task"]:
            a["any_task"] += 1
        if rec["has_create"]:
            a["create"] += 1
        if rec["empty"]:
            a["empty"] += 1
    print(f"{'condition':<22}{'N':>4}{'any task_*':>14}{'task_create':>14}{'empty-flake':>13}")
    for cond, a in agg.items():
        print(f"{cond:<22}{a['n']:>4}{a['any_task']:>7}/{a['n']:<6}"
              f"{a['create']:>7}/{a['n']:<6}{a['empty']:>7}/{a['n']:<5}")

    print("\nRAW per-run:")
    for rec in results:
        print(f"  {rec['cond']:<22}{rec['prompt']:<16}#{rec['run']} "
              f"any_task={rec['any_task']!s:<5} create={rec['has_create']!s:<5} "
              f"iters={rec['iters']:<3} tools={rec['tools']}")


if __name__ == "__main__":
    asyncio.run(main())
