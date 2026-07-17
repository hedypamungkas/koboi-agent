#!/usr/bin/env python3
"""experiment_multi_provider.py -- provider A/B governance harness (#15, W1c).

QUESTION IT ANSWERS: is provider B a quality-acceptable fallback/alternative for
provider A? (e.g. failover surplusintelligence gpt-5.4 -> z.ai glm-5.1/claude).

METHOD: run the SAME workload on each provider as a single agent, then BLIND
pairwise-judge the two answers per query (randomized labels, same judge+prompt).
If B loses head-to-head, it's a poor fallback -- don't trust unequal-provider
failover/routing to it without this check. (Equivalent-provider failover -- same
model, 2 endpoints/keys -- is quality-neutral and skips this.)

Metering comes from RunResult.token_usage + elapsed_seconds, so no pool or
RecordingRetryClient is needed for the core quality question. Mirrors the blind
pairwise judge + empty-completion retry from experiment_multi_model.py.

Credentials: load_dotenv() + ${VAR} interpolation. NEVER hardcoded.

Usage:
  # Arm A (default = the OpenAI-compatible gateway already in .env)
  export ARM_A_MODEL=gpt-5.4                       # + ARM_A_PROVIDER/BASE_URL/API_KEY override
  # Arm B (the fallback candidate you want to vet)
  export ARM_B_MODEL=glm-5.1 ARM_B_PROVIDER=anthropic
  export ARM_B_BASE_URL=https://api.z.ai/api/anthropic/v1 ARM_B_API_KEY=$ANTHROPIC_API_KEY
  export ARM_REPS=3
  python3 experiment_multi_provider.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time

from dotenv import load_dotenv

load_dotenv()

RUN_TIMEOUT = int(os.environ.get("ARM_TIMEOUT", "120"))
REPS = int(os.environ.get("ARM_REPS", "3"))
SEED = int(os.environ.get("ARM_SEED", "42"))
random.seed(SEED)

# Price map {model: (in_per_1M, out_per_1M)} USD. Override via ARM_PRICES JSON.
_DEFAULT_PRICES = {
    "gpt-5.4": (1.25, 10.0),
    "gpt-5.4-nano": (0.10, 0.40),
    "gpt-5.4-mini": (0.15, 0.60),
    "glm-5.1": (0.50, 1.50),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4.5": (1.0, 5.0),
    "deepseek-v4-flash": (0.20, 0.80),
}
PRICES = json.loads(os.environ.get("ARM_PRICES", "null")) or _DEFAULT_PRICES


def _arm(prefix: str, fallback_provider: str, fallback_model: str) -> dict:
    """Read an arm's provider spec from env (prefix=ARM_A / ARM_B)."""
    return {
        "provider": os.environ.get(f"{prefix}_PROVIDER", fallback_provider),
        "model": os.environ.get(f"{prefix}_MODEL", fallback_model),
        "api_key": os.environ.get(f"{prefix}_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        "base_url": os.environ.get(f"{prefix}_BASE_URL", os.environ.get("OPENAI_BASE_URL", "")),
    }


def _assert_arms(arm_a: dict, arm_b: dict) -> None:
    for name, arm in (("ARM_A", arm_a), ("ARM_B", arm_b)):
        if not arm["model"]:
            raise SystemExit(f"{name}_MODEL not set")
        if not arm["api_key"]:
            raise SystemExit(f"{name}_API_KEY not set (and no OPENAI_API_KEY fallback)")
        if not arm["base_url"]:
            raise SystemExit(f"{name}_BASE_URL not set (and no OPENAI_BASE_URL fallback)")
    if arm_a["model"] == arm_b["model"] and arm_a["base_url"] == arm_b["base_url"]:
        raise SystemExit("ARM_A and ARM_B are identical -- nothing to compare.")


# Workload: reasoning / instruction-following / recall / arithmetic / code.
# Model-agnostic, stresses quality differences between providers.
QUERIES = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "I have 3 boxes. Each box holds 4 packs, and each pack holds 5 pens. How many pens total? Show the steps.",
    "List exactly 3 fruits, one per line, in UPPERCASE, in alphabetical order. No other text.",
    "What is the capital of Australia? Answer with just the city name.",
    "Explain why the sky is blue in exactly two sentences.",
    "What is 17 multiplied by 23?",
    "Write a Python one-liner that returns the reverse of a string s.",
    "Translate 'good morning' to Japanese. Give only the translation.",
    "If today is Wednesday, what day is it 100 days from now?",
    "Name one primary color and one secondary color, separated by a comma.",
]


def _config_yaml(arm: dict) -> str:
    return (
        "agent:\n"
        "  name: provider-ab\n"
        "  system_prompt: |\n"
        "    You are a concise, accurate assistant. Answer the question directly.\n"
        "  max_iterations: 2\n"
        "llm:\n"
        f"  provider: {arm['provider']}\n"
        f"  model: {arm['model']}\n"
        f"  api_key: {arm['api_key']}\n"
        f"  base_url: {arm['base_url']}\n"
        "  timeout: 90.0\n"
        "  max_tokens: 512\n"
        "context: { strategy: noop }\n"
        "memory: { backend: memory }\n"
    )


async def _one_run(arm: dict, query: str, rep: int) -> dict:
    """Run one query on one arm. Retry the empty-completion flake up to 3x."""
    from koboi.facade import KoboiAgent

    yaml_str = _config_yaml(arm)
    for attempt in range(3):
        try:
            agent = KoboiAgent.from_config_string(yaml_str, verbose=False)
            t0 = time.perf_counter()
            result = await asyncio.wait_for(agent.run(query), timeout=RUN_TIMEOUT)
            dt = time.perf_counter() - t0
            answer = (result.content or "").strip()
            if not answer and attempt < 2:
                await asyncio.sleep(1)
                continue
            usage = result.token_usage
            p_in = getattr(usage, "prompt_tokens", 0) if usage else 0
            p_out = getattr(usage, "completion_tokens", 0) if usage else 0
            in_p, out_p = PRICES.get(arm["model"], (0.0, 0.0))
            cost = (p_in * in_p + p_out * out_p) / 1_000_000.0
            return {"model": arm["model"], "query": query, "rep": rep, "answer": answer,
                    "empty": not answer, "elapsed": round(dt, 2),
                    "prompt_tokens": p_in, "completion_tokens": p_out, "cost": cost,
                    "success": result.success, "err": ""}
        except asyncio.TimeoutError:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            return _err(arm, query, rep, "TIMEOUT")
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            return _err(arm, query, rep, f"{type(e).__name__}: {e}")
    return _err(arm, query, rep, "FAILED_AFTER_RETRIES")


def _err(arm, query, rep, msg) -> dict:
    return {"model": arm["model"], "query": query, "rep": rep, "answer": "", "empty": True,
            "elapsed": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0,
            "success": False, "err": msg}


# ---------------------------------------------------------------------------
# Blind pairwise judge (randomized A/B labels) -- same as the MMR harness
# ---------------------------------------------------------------------------
_JUDGE_PROMPT = """\
You are comparing two answers to the SAME question. Ignore which model produced them.

Question: {q}

Answer 1:
{a1}

Answer 2:
{a2}

Which answer is BETTER (correctness, completeness, clarity)? If both are equally
good/bad, say Tie.
Reply on the FIRST line with EXACTLY one of: Answer 1, Answer 2, Tie
Then one short line with the reason.
"""


async def _pairwise(judge_client, query: str, a1: str, a2: str) -> str:
    if not a1 and not a2:
        return "Tie"
    if not a1:
        return "Answer 2"
    if not a2:
        return "Answer 1"
    try:
        resp = await asyncio.wait_for(
            judge_client.complete(messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                q=query, a1=a1, a2=a2)}]), timeout=60)
        first = (resp.content or "").strip().splitlines()[0].lower() if resp.content else ""
        if first.startswith("answer 1"):
            return "Answer 1"
        if first.startswith("answer 2"):
            return "Answer 2"
        return "Tie"
    except Exception:
        return "Tie"


async def main():
    from koboi.client import RetryClient

    arm_a = _arm("ARM_A", fallback_provider="openai", fallback_model="gpt-5.4")
    arm_b = _arm("ARM_B", fallback_provider="openai", fallback_model="")
    _assert_arms(arm_a, arm_b)
    judge_model = os.environ.get("ARM_JUDGE_MODEL", arm_a["model"])
    judge_provider = os.environ.get("ARM_JUDGE_PROVIDER", arm_a["provider"])
    judge_url = os.environ.get("ARM_JUDGE_BASE_URL", arm_a["base_url"])
    judge_key = os.environ.get("ARM_JUDGE_API_KEY", arm_a["api_key"])

    print("=== provider A/B governance harness (#15) ===")
    print(f"A: {arm_a['provider']}/{arm_a['model']}  @ {arm_a['base_url']}")
    print(f"B: {arm_b['provider']}/{arm_b['model']}  @ {arm_b['base_url']}")
    print(f"judge: {judge_provider}/{judge_model}  reps={REPS}  queries={len(QUERIES)}\n")

    judge = RetryClient(provider=judge_provider, model=judge_model, api_key=judge_key,
                        base_url=judge_url, max_tokens=64)

    runs_a: list[dict] = []
    runs_b: list[dict] = []
    pairs = []  # (query, rep, winner)
    for q_idx, query in enumerate(QUERIES):
        for rep in range(REPS):
            ra = await _one_run(arm_a, query, rep)
            rb = await _one_run(arm_b, query, rep)
            runs_a.append(ra)
            runs_b.append(rb)
            # Blind pairwise: randomize which arm is Answer 1 vs 2.
            b_first = random.random() < 0.5
            if b_first:
                verdict = await _pairwise(judge, query, rb["answer"], ra["answer"])
                b_label = "Answer 1"
            else:
                verdict = await _pairwise(judge, query, ra["answer"], rb["answer"])
                b_label = "Answer 2"
            winner = ("B" if verdict == b_label else ("A" if verdict in ("Answer 1", "Answer 2") else "tie"))
            pairs.append((query, rep, winner))
            ta = "EMPTY" if ra["empty"] else f"{ra['elapsed']:.1f}s/${ra['cost']:.5f}"
            tb = "EMPTY" if rb["empty"] else f"{rb['elapsed']:.1f}s/${rb['cost']:.5f}"
            print(f"  [q{q_idx} r{rep}] A={ta:<18} B={tb:<18} judge={winner}  | {query[:40]}")

    await judge.close()

    # --- aggregation ---
    def _stats(runs):
        ok = [r for r in runs if not r["err"]]
        return {
            "n_ok": len(ok),
            "cost": sum(r["cost"] for r in ok),
            "prompt": sum(r["prompt_tokens"] for r in ok),
            "completion": sum(r["completion_tokens"] for r in ok),
            "lat": sum(r["elapsed"] for r in ok),
            "empties": sum(1 for r in runs if r["empty"]),
            "errors": sum(1 for r in runs if r["err"]),
        }

    sa, sb = _stats(runs_a), _stats(runs_b)
    n = len(pairs)
    a_wins = sum(1 for _, _, w in pairs if w == "A")
    b_wins = sum(1 for _, _, w in pairs if w == "B")
    ties = sum(1 for _, _, w in pairs if w == "tie")

    print("\n=== SUMMARY ===")
    print(f"{'metric':<16} {'A':>14} {'B':>14} {'delta':>14}")
    for label, ak, bk in [
        ("cost ($)", sa["cost"], sb["cost"]),
        ("prompt tok", sa["prompt"], sb["prompt"]),
        ("completion tok", sa["completion"], sb["completion"]),
        ("latency (s)", sa["lat"], sb["lat"]),
    ]:
        d = f"{(bk - ak) / ak * 100:+.1f}%" if ak else "n/a"

        def _fmt(x):
            if label.startswith("cost"):
                return f"{x:.5f}"
            return f"{x:.1f}" if isinstance(x, float) else str(x)

        print(f"{label:<16} {_fmt(ak):>14} {_fmt(bk):>14} {d:>14}")
    print(f"\nempty/errors:  A={sa['empties']}/{sa['errors']}  B={sb['empties']}/{sb['errors']}")
    print(f"pairwise judge (n={n}):  A_wins={a_wins}  B_wins={b_wins}  ties={ties}")
    b_favorable = (b_wins + ties) / n if n else 0
    print(f"B favorable rate (B_wins+ties)/n = {b_favorable:.0%}")

    print("\n=== VERDICT ===")
    if not sb["n_ok"]:
        print("INCONCLUSIVE: provider B produced no successful runs (check creds/base_url).")
    elif b_favorable >= 0.5:
        print(f"B is a QUALITY-ACCEPTABLE fallback/alternative (favorable {b_favorable:.0%} >= 50%).")
        print("Safe to use in unequal-provider failover/routing (cost/latency above inform the tradeoff).")
    else:
        print(f"B LOSES head-to-head (favorable only {b_favorable:.0%} < 50%).")
        print("B is a POOR quality fallback -- unequal-provider failover to B will degrade answers.")
        print("Either use an equivalent provider (same model) as fallback, or accept the quality hit.")


if __name__ == "__main__":
    asyncio.run(main())
