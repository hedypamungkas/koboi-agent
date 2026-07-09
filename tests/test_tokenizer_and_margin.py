"""Tests for the optional real tokenizer (#3) and context safety_margin (#5)."""

from __future__ import annotations

import sys

import pytest

from koboi.context.manager import TruncationManager, estimate_tokens
from koboi.tokens import make_tokenizer


class TestMakeTokenizer:
    def test_non_openai_returns_none(self):
        # Anthropic/Cloudflare/unknown keep the heuristic (no offline encoding).
        assert make_tokenizer("anthropic", "claude-3") is None
        assert make_tokenizer("cloudflare", None) is None
        assert make_tokenizer(None, None) is None

    def test_openai_returns_callable(self):
        # tiktoken is an optional extra (CI installs .[dev,tui,api], not tokenizer).
        pytest.importorskip("tiktoken")
        tok = make_tokenizer("openai", "gpt-4o")
        assert callable(tok)
        n = tok([{"role": "user", "content": "hello world"}])
        assert n > 0

    def test_openai_cjk_more_accurate_than_heuristic(self):
        # tiktoken is an optional extra; skip when absent (CI without tokenizer extra).
        pytest.importorskip("tiktoken")
        # chars/3 badly undercounts CJK; the BPE counter should be larger.
        msgs = [{"role": "user", "content": "你好世界，今天天气很好"}]
        heur = estimate_tokens(msgs)
        tok = make_tokenizer("openai", "gpt-4o")
        assert tok(msgs) > heur

    def test_returns_none_when_tiktoken_absent(self, monkeypatch):
        # Simulate tiktoken not installed.
        monkeypatch.setitem(sys.modules, "tiktoken", None)
        assert make_tokenizer("openai", "gpt-4o") is None


class TestEffectiveTokensUsesTokenizer:
    def test_tokenizer_overrides_heuristic(self):
        mgr = TruncationManager(keep_last=2)
        msgs = [{"role": "user", "content": "x"}]
        mgr.tokenizer = lambda m: 4242  # fake real tokenizer
        mgr.last_actual_tokens = 0
        assert mgr._effective_tokens(msgs) == 4242

    def test_last_actual_still_floors(self):
        mgr = TruncationManager(keep_last=2)
        mgr.tokenizer = lambda m: 10
        mgr.last_actual_tokens = 9000
        assert mgr._effective_tokens([{"role": "user", "content": "x"}]) == 9000


@pytest.mark.asyncio
class TestSafetyMargin:
    async def test_margin_triggers_compaction_earlier(self):
        # tokens between (max - margin) and max -> with margin, compact; without, passthrough.
        msgs = [{"role": "system", "content": "s"}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"turn {i} " * 10})

        # No margin: budget 100000 -> no compaction (well under)
        mgr0 = TruncationManager(keep_last=2)
        mgr0.last_actual_tokens = 0
        # force a known token count via tokenizer stub
        mgr0.tokenizer = lambda m: 500
        out0 = await mgr0.manage(msgs, max_tokens=1000)
        assert len(out0) == len(msgs)  # no trim (500 <= 1000)

        # With margin 600: budget = 1000 - 600 = 400; 500 > 400 -> compaction
        mgr1 = TruncationManager(keep_last=2)
        mgr1.tokenizer = lambda m: 500
        mgr1.safety_margin = 600
        out1 = await mgr1.manage(msgs, max_tokens=1000)
        assert len(out1) < len(msgs)  # trimmed

    async def test_zero_margin_preserves_old_behavior(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"}]
        mgr = TruncationManager(keep_last=2)
        mgr.last_actual_tokens = 100000  # force over budget
        out = await mgr.manage(msgs, max_tokens=10)
        # default margin 0 -> budget 10, compacts as before
        assert len(out) <= len(msgs)

    async def test_force_compact_max_tokens_zero_still_compacts(self):
        # /compact path passes max_tokens=0; margin must not break it.
        msgs = [{"role": "system", "content": "s"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        mgr = TruncationManager(keep_last=2)
        mgr.last_actual_tokens = 100000
        mgr.safety_margin = 500
        out = await mgr.manage(msgs, max_tokens=0)
        assert len(out) < len(msgs)
