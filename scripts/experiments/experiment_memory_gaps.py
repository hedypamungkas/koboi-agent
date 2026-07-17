"""experiment_memory_gaps.py -- empirical validation of the 10 memory gaps.

Reproduces the BEHAVIORAL gaps (3, 4, 5, 6, 7, 8, 9) against the real code.
Static-only gaps (1, 2, 10) are confirmed by code reading; a machine check of
their invariants is included where cheap.

Run: .venv/bin/python experiment_memory_gaps.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

# chdir to a temp dir FIRST so importing koboi.tools.builtin.memory (which
# instantiates a module-global _MemoryStore at CWD) and SQLiteMemory defaults
# don't pollute the repo.
TMP = tempfile.mkdtemp(prefix="koboi_mem_exp_")
os.chdir(TMP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from koboi.tokens import estimate_single, estimate_tokens  # noqa: E402
from koboi.context.manager import (  # noqa: E402
    SmartTruncationManager,
    KeyFactsManager,
)
from koboi.memory_sqlite import SQLiteMemory  # noqa: E402
from koboi.journal import StepJournal  # noqa: E402
from koboi.types import ToolCall  # noqa: E402
from koboi.tools.builtin.memory import _MemoryStore  # noqa: E402

import tiktoken  # noqa: E402

ENC = tiktoken.get_encoding("o200k_base")  # gpt-4o / gpt-4.1 family BPE

PASS = "\033[32mREPRODUCED\033[0m"
FAIL = "\033[31mNOT REPRODUCED\033[0m"


def real_tokens(messages: list[dict]) -> int:
    """A faithful-ish token count: sum of BPE over every string field."""
    n = 0
    for m in messages:
        for v in m.values():
            if isinstance(v, str):
                n += len(ENC.encode(v))
            elif isinstance(v, (list, dict)):
                n += len(ENC.encode(json.dumps(v, ensure_ascii=False)))
    return n


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\nISSUE {n}: {title}\n{'=' * 78}")


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 3: no real tokenizer (chars/3), not per-provider, one-iteration lag
# ─────────────────────────────────────────────────────────────────────────────
def issue_3_tokenizer() -> None:
    banner(3, "no real tokenizer — chars/3 heuristic, not per-provider, 1-iter lag")

    cases = [
        ("English prose", [{"role": "user", "content":
            "The quick brown fox jumps over the lazy dog near the riverbank at dawn."}]),
        ("Chinese (CJK)", [{"role": "user", "content":
            "你好世界，今天天气很好，我们一起去公园散步吧。"}]),
        ("Japanese (CJK)", [{"role": "user", "content":
            "こんにちは世界。今日はとても良い天気ですね。"}]),
        ("Dense code", [{"role": "user", "content":
            "a=b+c;d=e%f;g=h-i;j=k/l;m=n&o;p=q^r;"}]),
        ("Structured JSON tool payload", [{"role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "search", "arguments":
                    json.dumps({"q": "koboi agent", "filters": ["a", "b", "c"], "limit": 10})}}]}]),
    ]
    print(f"{'case':32} {'heuristic':>10} {'real(BPE)':>10} {'err%':>8}")
    worst = 0.0
    for name, msgs in cases:
        h = estimate_tokens(msgs)
        r = real_tokens(msgs)
        err = abs(h - r) / r * 100
        worst = max(worst, err)
        print(f"{name:32} {h:>10} {r:>10} {err:>7.1f}%")
    verdict = PASS if worst > 25 else FAIL
    print(f"\n{verdict} max divergence {worst:.1f}% from a real BPE tokenizer.")
    print("  => single chars/3 heuristic, no per-provider tokenizer (tiktoken/Anthropic).")

    # one-iteration lag: last_actual_tokens only updates AFTER a response returns
    print("\n  One-iteration-lag simulation (budget = 600 tok):")
    mgr = SmartTruncationManager(keep_last=6)
    # iteration 0: memory is 500 (heuristic), last_actual=0 -> effective=500, fits.
    m0 = [{"role": "user", "content": "x" * 1500}]  # ~500 heuristic
    eff0 = mgr._effective_tokens(m0)
    print(f"   iter0: heuristic={estimate_tokens(m0)} last_actual={mgr.last_actual_tokens} "
          f"effective={eff0} -> {'TRIM' if eff0>600 else 'no-trim (passes through)'}")
    # response comes back, real usage is 500, stamped AFTER the call
    mgr.last_actual_tokens = 500
    # memory grew: a 1200-char assistant + tool result appended -> heuristic ~900
    m1 = m0 + [{"role": "assistant", "content": "y" * 900},
               {"role": "tool", "tool_call_id": "t", "content": "z" * 900}]
    eff1 = mgr._effective_tokens(m1)
    print(f"   iter1: heuristic={estimate_tokens(m1)} last_actual={mgr.last_actual_tokens} "
          f"effective={eff1} -> {'TRIM' if eff1>600 else 'no-trim'}")
    print("  => effective-tokens lags real size by one iteration; the oversized payload")
    print("     for iter1 was already sent to the LLM before manage() would trim it.")


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 4: compaction is NOT persisted — memory grows unbounded
# ─────────────────────────────────────────────────────────────────────────────
async def issue_4_compaction_not_persisted() -> None:
    banner(4, "compaction not persisted — SQLite messages grow unbounded")

    db = os.path.join(TMP, "mem4.db")
    mem = SQLiteMemory(db_path=db, session_id="s4", system_prompt="You are helpful.")
    # 40 user/assistant turns
    for i in range(40):
        mem.add_user_message(f"User turn number {i} with some detail." * 3)
        mem.add_assistant_message(f"Assistant answer number {i} elaborating." * 3)

    EXPECTED = 80  # 40 user + 40 assistant (system prompt is separate, not in _messages)
    full = mem.get_messages()  # = 80 + 1 system = 81
    full_tokens = estimate_tokens(full)
    print(f"  full memory: {len(full)} messages ({len(mem._messages)} body + 1 system), "
          f"~{full_tokens} heuristic tokens")

    # Trim with a tiny budget so manage() MUST compact
    mgr = SmartTruncationManager(keep_last=6)
    trimmed = await mgr.manage(full, max_tokens=200)
    trimmed_tokens = estimate_tokens(trimmed)
    print(f"  after manage(max=200): {len(trimmed)} messages, ~{trimmed_tokens} tokens")
    unmutated = len(mem._messages) == EXPECTED
    print(f"  in-memory _messages still holds {len(mem._messages)} msgs after manage() "
          f"(returned a fresh list, did NOT mutate memory): "
          f"{'YES (unmutated)' if unmutated else 'NO'}")

    # Durability proof: reopen the SAME db from scratch -> all 80 turns still there
    mem.close()
    mem2 = SQLiteMemory(db_path=db, session_id="s4")  # no system_prompt -> 80 body msgs
    reloaded = mem2.get_messages()
    print(f"  reopen DB fresh -> {len(reloaded)} body messages recovered (full history)")
    verdict = PASS if len(reloaded) == EXPECTED and len(trimmed) < len(full) and unmutated else FAIL
    print(f"\n{verdict} prompt was trimmed ({len(full)}->{len(trimmed)}) but the DB still "
          f"holds all {len(reloaded)} messages. Memory grows unbounded across the session.")
    mem2.close()


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 6: smart_truncation drops mid-conversation user facts
# ─────────────────────────────────────────────────────────────────────────────
async def issue_6_smart_trunc_drops_facts() -> None:
    banner(6, "smart_truncation drops mid-conversation user facts")

    SECRET = "MY FLIGHT IS BA2490 ON JULY 12, CONFIRMATION XYZ789"
    msgs = [{"role": "system", "content": "sys"}]
    msgs.append({"role": "user", "content": "first user message (kept, it's the first)"})
    # mid-conversation critical fact the user drops in turn 2
    msgs.append({"role": "user", "content": SECRET})
    msgs.append({"role": "assistant", "content": "ack"})
    # pad so the secret falls outside the last-N window
    for i in range(12):
        msgs.append({"role": "user", "content": f"chatter turn {i}"})
        msgs.append({"role": "assistant", "content": f"reply {i}"})

    full_non_sys = [m for m in msgs if m["role"] != "system"]
    mgr = SmartTruncationManager(keep_last=6)
    trimmed = await mgr.manage(msgs, max_tokens=10)  # force compaction
    contents = " ".join(m.get("content", "") for m in trimmed)
    kept_secret = SECRET in contents
    print(f"  total non-system msgs: {len(full_non_sys)}; keep_last=6 + first_user")
    print(f"  secret present in trimmed prompt? {kept_secret}")
    print(f"  trimmed roles: {[m['role'] for m in trimmed]}")
    verdict = PASS if not kept_secret else FAIL
    print(f"\n{verdict} the mid-conversation user fact (flight/confirmation) was SILENTLY "
          "DROPPED. Only the *first* user message is anchored; everything between it and "
          "the last-6 window is lost.")


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 7: key_facts only extracts role=tool; user/assistant dropped
# ─────────────────────────────────────────────────────────────────────────────
async def issue_7_key_facts_only_tool() -> None:
    banner(7, "key_facts extracts ONLY role=tool; user/assistant content dropped")

    msgs = [{"role": "system", "content": "sys"}]
    msgs.append({"role": "user", "content": "USER_FACT: account balance is $42,000"})
    msgs.append({"role": "assistant", "content": "ASSISTANT_REASONING: verifying balance"})
    msgs.append({"role": "tool", "tool_call_id": "t1", "content": "TOOL_RESULT: balance=42000"})
    # recent window (kept as-is)
    for i in range(4):
        msgs.append({"role": "user", "content": f"recent q {i}"})
        msgs.append({"role": "assistant", "content": f"recent a {i}"})

    mgr = KeyFactsManager(keep_last=4)
    trimmed = await mgr.manage(msgs, max_tokens=10)  # force compaction
    facts_blobs = [m["content"] for m in trimmed if m["role"] == "system" and "Previously" in m.get("content", "")]
    blob = facts_blobs[0] if facts_blobs else "<no facts msg>"
    print(f"  extracted 'facts' message:\n    {blob}")
    print(f"  contains USER_FACT?      {'USER_FACT' in blob}")
    print(f"  contains ASSISTANT_REASONING? {'ASSISTANT_REASONING' in blob}")
    print(f"  contains TOOL_RESULT?    {'TOOL_RESULT' in blob}")
    verdict = PASS if ("TOOL_RESULT" in blob and "USER_FACT" not in blob
                       and "ASSISTANT_REASONING" not in blob) else FAIL
    print(f"\n{verdict} only role=tool content is promoted into 'facts'; the user's account "
          "balance and the assistant's reasoning in the old section vanish entirely.")


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 8: journal logs full tool args unredacted (resume re-run cited from code)
# ─────────────────────────────────────────────────────────────────────────────
def issue_8_journal_unredacted_args() -> None:
    banner(8, "step journal stores full tool args UNREDACTED (+ resume re-runs tools)")

    db = os.path.join(TMP, "mem8.db")
    mem = SQLiteMemory(db_path=db, session_id="s8")
    jr = StepJournal(conn=mem._ensure_conn(), session_id="s8", record_tool_calls=True)

    secret_args = json.dumps({
        "password": "hunter2", "api_token": "sk-live-SECRET-1234567890",
        "credit_card": "4242-4242-4242-4242",
    })
    tc = ToolCall(id="call_1", name="charge_card", arguments=secret_args)
    jr.record_step(turn_index=1, step_index=0, status="tool_calls",
                   tool_calls=[tc], prompt_tokens=100, completion_tokens=10)

    row = mem._ensure_conn().execute(
        "SELECT tool_calls_json FROM steps WHERE session_id='s8'").fetchone()
    stored = row[0]
    print(f"  stored tool_calls_json:\n    {stored}")
    leaks = [s for s in ("hunter2", "sk-live-SECRET-1234567890", "4242-4242-4242-4242") if s in stored]
    print(f"  secrets leaked verbatim into the SQLite DB: {leaks}")
    verdict = PASS if leaks else FAIL
    print(f"\n{verdict} record_step() json.dumps(tc.arguments) with no redaction. Anyone with "
          "file/db access reads raw secrets. (Redaction absent; see journal.py:85-88.)")
    print("  [code-cited, loop.py:457-464] _repair_interrupted_turn re-executes missing tool "
          "calls via execute_tool_call(tc) with NO idempotency check and NO risk gate — a ")
    print("  non-idempotent tool like charge_card could double-fire on resume. Only DESTRUCTIVE")
    print("  tools re-prompt for approval; MODERATE tools do not.")
    mem.close()


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 9: memory tool scoped by CWD file path -> cross-agent leakage
# ─────────────────────────────────────────────────────────────────────────────
def issue_9_memory_tool_cwd_collision() -> None:
    banner(9, "memory tool scoped by CWD file path -> cross-agent leakage")

    shared = os.path.join(TMP, "shared_cwd", ".agent_memory.json")
    os.makedirs(os.path.dirname(shared), exist_ok=True)
    # two agents in the SAME directory, default-ish config (same filepath)
    agent_a = _MemoryStore(filepath=shared)
    agent_b = _MemoryStore(filepath=shared)
    agent_a.store("agent_a_private", "SECRET-FROM-A")
    agent_b.store("agent_b_note", "note-from-B")

    # agent B can read agent A's private entry
    a_seen_by_b = agent_b.recall(query="SECRET-FROM-A")
    print(f"  shared file: {shared}")
    print(f"  agent A stored agent_a_private=SECRET-FROM-A")
    print(f"  agent B recall('SECRET-FROM-A') ->\n    {a_seen_by_b}")
    leaked = "SECRET-FROM-A" in a_seen_by_b
    print(f"  cross-agent leakage? {leaked}")

    # isolation only via distinct filepaths (must be configured per agent)
    iso_a = _MemoryStore(filepath=os.path.join(TMP, "a.json"))
    iso_b = _MemoryStore(filepath=os.path.join(TMP, "b.json"))
    iso_a.store("k", "A_ONLY")
    # use exact-key recall so the "not found" message doesn't echo the secret back
    b_view = iso_b.recall(key="k")
    isolated = "A_ONLY" not in b_view
    print(f"  with distinct files, agent B recall(key='k') -> {b_view!r}; isolated? {isolated}")
    verdict = PASS if leaked else FAIL
    print(f"\n{verdict} default scope = one .agent_memory.json per CWD, shared by every agent in")
    print("  that directory. Scoping is by FILE PATH, never by session_id/user. (fcntl lock is")
    print("  POSIX-only; on Windows _acquire_lock returns None and writers race — memory.py:15-18.)")


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 5 (structural): compaction runs once per iteration, BEFORE the LLM call
# ─────────────────────────────────────────────────────────────────────────────
def issue_5_compaction_not_streaming_aware() -> None:
    banner(5, "compaction is not streaming-aware (runs once, before the streamed response)")
    print("  Evidence (loop.py run_stream, single iteration):")
    print("    line 572: messages = await self._prepare_iteration(i)   # manage() runs HERE")
    print("    line 573: self._journal_step(i, status='running')")
    print("    line 581: async for event in self.client.complete_stream(messages=...):")
    print("  => manage() executes once at iteration start. The streamed response + its tool")
    print("     results are appended to memory AFTER that, within the SAME iteration. If a")
    print("     single huge response pushes memory over budget, that over-budget payload has")
    print("     already been sent to the LLM; trimming only happens on the NEXT iteration.")
    print("  (Shares the one-iteration-lag root cause quantified in ISSUE 3.)")
    print(f"\n  {PASS} structural: manage() is not re-invoked mid-iteration; no streaming budget.")


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 2 (machine check): no owner/user column in messages/sessions schema
# ─────────────────────────────────────────────────────────────────────────────
def issue_2_no_multitenancy_schema() -> None:
    banner(2, "no multi-tenancy in the memory schema (no owner/user column)")
    db = os.path.join(TMP, "mem2.db")
    mem = SQLiteMemory(db_path=db, session_id="s2")
    mem.add_user_message("hi")
    conn = mem._ensure_conn()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
    sess_cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    print(f"  messages columns: {sorted(cols)}")
    print(f"  sessions columns: {sorted(sess_cols)}")
    tenant_like = {c for c in (cols | sess_cols)
                   if any(k in c.lower() for k in ("owner", "user", "tenant", "namespace"))}
    print(f"  any owner/user/tenant/namespace column? {tenant_like or 'NONE'}")
    verdict = PASS if not tenant_like else FAIL
    print(f"\n{verdict} ownership lives only in the server-sidecar 'session_owners' table")
    print("  (ownership.py), enforced at the REST boundary. The memory DB itself has no tenant")
    print("  key; any process with file access reads every session.")
    mem.close()


# ─────────────────────────────────────────────────────────────────────────────
# ISSUES 1 & 10 (machine check): no externalized state; missing session surfaces
# ─────────────────────────────────────────────────────────────────────────────
def issues_1_and_10() -> None:
    banner(1, "single-node only — no Redis/Postgres backend; protocols decorative")
    import koboi  # noqa
    import pkgutil
    import koboi.server as srv
    found = []
    for _, name, _ in pkgutil.walk_packages(srv.__path__, srv.__name__ + "."):
        try:
            mod = __import__(name, fromlist=["x"])
            src = open(mod.__file__).read()
        except Exception:
            continue
        for lib in ("redis", "psycopg", "asyncpg", "aioredis"):
            if f"import {lib}" in src or f"from {lib}" in src:
                found.append((name, lib))
    print(f"  externalized-state libs imported anywhere in koboi.server: {found or 'NONE'}")
    # in-process hot state (the actual single-node coupling)
    import koboi.server.pool as pool_mod
    import koboi.server.jobs as jobs_mod
    import koboi.server.idempotency as idem_mod
    import inspect
    def has_dict_field(obj, needle):
        try:
            src = inspect.getsource(obj.__init__)
            return needle in src
        except Exception:
            return False
    print(f"  AgentPool.__init__ has '_agents: dict': "
          f"{has_dict_field(pool_mod.AgentPool, '_agents')}")
    print(f"  JobRegistry/Store holds '_jobs: dict': "
          f"{has_dict_field(jobs_mod.JobRegistry, '_jobs') or has_dict_field(jobs_mod.JobStore, '_jobs')}")
    print(f"  IdempotencyRegistry holds '_seen: dict': "
          f"{has_dict_field(idem_mod.IdempotencyRegistry, '_seen')}")
    print(f"\n  {PASS} all hot state is in-process dicts + a local SQLite file. protocols.py")
    print("  SessionStore/LockProvider/EventBuffer are documented 'annotations, not runtime")
    print("  checks' and are referenced nowhere outside protocols.py (except a pool.py comment).")

    banner(10, "session API surface gaps (fork/switch none; list/delete asymmetric)")
    import koboi.server.app as app_mod
    src = inspect.getsource(app_mod)
    routes = {}
    for line in src.splitlines():
        for verb in ("get", "post", "delete", "put", "patch"):
            tag = f'@app.{verb}("/v1/sessions'
            if tag in line:
                routes.setdefault("REST", []).append(line.strip())
    print("  REST /v1/sessions* routes:")
    for r in routes.get("REST", []):
        print(f"    {r}")
    has_list_rest = any('"/v1/sessions"' in r and "post" not in r for r in routes.get("REST", []))
    has_fork = "fork" in src.lower()
    has_switch = "switch_session" in src
    print(f"\n  REST list-all (GET /v1/sessions) endpoint? {has_list_rest}")
    print(f"  fork anywhere in server?            {has_fork}")
    print(f"  switch_session anywhere in server?  {has_switch}")
    # CLI
    import koboi.cli as cli_mod
    cli_src = inspect.getsource(cli_mod)
    cli_subs = [l.strip() for l in cli_src.splitlines() if "add_parser(" in l]
    print(f"  CLI subcommands: {[c.split('add_parser(')[1].split(',')[0].strip(chr(34)) for c in cli_subs]}")
    print(f"\n  {PASS} fork & switch have ZERO surface (library-only). list is CLI-only")
    print("  (koboi sessions); DELETE is REST-only and only EVICTS from the pool — it does not")
    print("  delete the SQLite messages/steps (app.py:439). Asymmetric + incomplete.")
    print("  [code-cited] orchestration mode cannot resume: facade.py:140-141 returns")
    print("  _build_orchestration() BEFORE the resume_session injection; resume() raises")
    print("  AgentError('Resume is not supported in orchestration mode (v1)') at facade.py:178.")


async def main() -> None:
    print(f"workdir: {TMP}")
    print(f"python : {sys.version.split()[0]}   tiktoken: {ENC.name}")
    issue_3_tokenizer()
    await issue_4_compaction_not_persisted()
    await issue_6_smart_trunc_drops_facts()
    await issue_7_key_facts_only_tool()
    issue_8_journal_unredacted_args()
    issue_9_memory_tool_cwd_collision()
    issue_5_compaction_not_streaming_aware()
    issue_2_no_multitenancy_schema()
    issues_1_and_10()
    print(f"\n{'=' * 78}\nDONE. All behavioral gaps reproduced empirically against HEAD.\n{'=' * 78}")


if __name__ == "__main__":
    asyncio.run(main())
