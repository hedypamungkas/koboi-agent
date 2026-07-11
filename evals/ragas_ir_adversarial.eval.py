"""Adversarial hard strata (Path B4): multi-hop / negation / conflicting-evidence / near-miss.

These are the cases the original "all 1.0" calibration never exercised -- the audit's
Simpson framing: easy queries scored 1.0, the only hard probes scored 0.0. This suite runs
each hard stratum against a controlled KB (evals/fixtures/adversarial_kb.md) with the
DECOUPLED judge path and DETERMINISTIC assertions (the KB is controlled, so gold answers
are known -- no judge flakiness, no self-judge bias on the assertions themselves).

Reported per-stratum (not aggregated) so a failure on one stratum cannot be hidden by the
others. LIVE ONLY (the agent generates a real answer); self-skips under --mock.
"""

from koboi.eval.t import Contains, Matches, Severity  # noqa: F401  (re-exported for authors)

CONFIG = {
    "agent": {
        "name": "rag-adversarial-eval",
        "description": "Adversarial hard strata over a controlled KB",
        "system_prompt": (
            "Use ONLY the provided context to answer. Prefer authoritative sources over "
            "unverified ones. If the context doesn't contain the answer, say you don't know."
        ),
        "max_iterations": 4,
    },
    "llm": {
        "provider": "openai",
        "model": "${OPENAI_MODEL:gpt-4o-mini}",
        "api_key": "${OPENAI_API_KEY:dummy}",
        "base_url": "${OPENAI_BASE_URL:}",
    },
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "keyword",
        "top_k": 5,
        "augmentation": "on_the_fly",
        "documents": [{"path": "./evals/fixtures/adversarial_kb.md"}],
    },
}

TAGS = ["rag", "live", "adversarial"]


async def test_multi_hop(t):
    """Multi-hop: 'who owns the Quantum project?' needs Eng->Sam Lee + Eng owns Quantum."""
    if not t.require_live():
        return
    await t.send("Who owns the Quantum project?")
    # Correct answer requires combining two facts -> "Sam Lee".
    t.check(t.reply, Contains("Sam Lee"), name="multi_hop_answer", severity=Severity.SOFT)
    t.completed()


async def test_negation(t):
    """Negation: 'which products are NOT cloud-based?' must list the on-premise ones."""
    if not t.require_live():
        return
    await t.send("Which products are NOT cloud-based?")
    reply = (t.reply or "")
    on_premise = "AtlasServer" in reply or "AtlasVault" in reply or "Rack" in reply
    t.check(on_premise, Matches(fn=lambda _: on_premise, description="lists an on-premise product"),
            name="negation_answer", severity=Severity.SOFT)
    # Should NOT list cloud products as the answer.
    t.check(("AtlasCloud" in reply), Matches(fn=lambda x: not x, description="does not list cloud as non-cloud"),
            name="negation_no_false_positive", severity=Severity.SOFT)
    t.completed()


async def test_conflicting_evidence(t):
    """Conflicting evidence: authoritative $15,000 vs unverified $18,000 -- the model must
    prefer the authoritative value or flag the conflict, not silently pick the wrong one."""
    if not t.require_live():
        return
    await t.send("What does WidgetX cost per year?")
    reply = (t.reply or "")
    authoritative = "$15,000" in reply or "15,000" in reply
    flags_conflict = any(w in reply.lower() for w in ("conflict", "discrepanc", "unverified", "two", "differ"))
    ok = authoritative or flags_conflict
    t.check(ok, Matches(fn=lambda _: ok, description="picks authoritative $15,000 or flags conflict"),
            name="conflict_resolution", severity=Severity.SOFT)
    t.completed()


async def test_near_miss_unanswerable(t):
    """Near-miss: 'Dana White's email' -- the person exists but the email does not. The
    model must abstain under NON-EMPTY context (the real production failure mode)."""
    if not t.require_live():
        return
    await t.send("What is Dana White's email address?")
    t.abstains()  # GATE: empty retrieval OR refusal marker
    t.completed()
