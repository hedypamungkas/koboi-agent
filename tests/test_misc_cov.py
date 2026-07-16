"""Branch-coverage tests for five koboi modules whose existing tests leave gaps.

Targets:
  * koboi/sandbox/restricted.py        -- rlimit preexec, list/unbalanced-token scan,
                                         stdin feed, timeout+killpg paths
  * koboi/proactive_memory.py          -- extract/recall/core-block edge branches
  * koboi/context/manager.py           -- _flatten_text, Noop props, KeyFacts skip,
                                         SlidingWindow hydrate/persist/summarize
  * koboi/tools/builtin/web.py         -- duckduckgo search, SSRF no-addrs, retry loop,
                                         redirects, truncation, error statuses
  * koboi/tools/builtin/memory.py      -- _save/_acquire_lock/_release_lock failure paths

Lines guarded by absent deps (seccomp filter installer, fcntl-missing import branch,
latin-1 decode fallback) are intentionally skipped and noted in the final report.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import koboi.sandbox.restricted as restricted_mod
import koboi.tools.builtin.web as web_mod
from koboi.context.manager import (
    KeyFactsManager,
    NoopContextManager,
    SlidingWindowManager,
    _flatten_text,
    ensure_tool_integrity,
)
from koboi.proactive_memory import ProactiveMemory
from koboi.sandbox.restricted import RestrictedProcessBackend, _make_preexec_fn
from koboi.tools.builtin.memory import _MemoryStore, memory_recall, memory_store
from koboi.tools.builtin.web import (
    _DDGResultParser,
    _resolve_and_check,
    web_fetch,
    web_search,
)
from koboi.types import AgentResponse


# ===========================================================================
# koboi/sandbox/restricted.py
# ===========================================================================


class TestRestrictedPreexec:
    def test_preexec_none_when_no_rlimits_and_no_seccomp(self):
        # No rlimits + no seccomp callable -> returns None (clean Popen kwargs).
        assert _make_preexec_fn(None, None) is None
        assert _make_preexec_fn({}, None) is None

    def test_preexec_applies_all_rlimit_branches(self, monkeypatch):
        # Exercise the _apply closure body (setrlimit for cpu/as/fsize/nofile)
        # without actually capping the test process: patch setrlimit.
        resource = restricted_mod._resource
        if resource is None:
            pytest.skip("rlimit support absent on this platform")
        calls: list = []
        monkeypatch.setattr(resource, "setrlimit", lambda which, lim: calls.append((which, lim)))

        preexec = _make_preexec_fn({"cpu": 5, "as_mb": 10, "fsize_mb": 20, "nofile": 30}, seccomp_preexec=None)
        assert preexec is not None
        preexec()  # runs the closure

        # Four setrlimit calls, one per key, each with a (limit, limit) tuple.
        assert len(calls) == 4
        # as_mb / fsize_mb are scaled to bytes.
        as_call = [c for c in calls if c[0] == resource.RLIMIT_AS][0]
        assert as_call[1] == (10 * 1024 * 1024, 10 * 1024 * 1024)
        fsize_call = [c for c in calls if c[0] == resource.RLIMIT_FSIZE][0]
        assert fsize_call[1] == (20 * 1024 * 1024, 20 * 1024 * 1024)
        cpu_call = [c for c in calls if c[0] == resource.RLIMIT_CPU][0]
        assert cpu_call[1] == (5, 5)

    def test_preexec_invokes_seccomp_callable(self):
        invoked: list = []
        preexec = _make_preexec_fn({}, seccomp_preexec=lambda: invoked.append(True))
        assert preexec is not None
        preexec()
        assert invoked == [True]


class TestRestrictedNetworkScan:
    def test_validate_path_anchors_relative_to_workdir(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path))
        resolved = sb.validate_path("relative.txt")
        # A relative path is anchored under the workdir before realpath.
        assert resolved.startswith(os.path.realpath(str(tmp_path)))

    def test_list_command_tokens_scanned(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path))
        # A list command hits the else branch (tokens = list(command)).
        assert sb.network_allowed(["wget", "http://x"]) is False
        assert sb.network_allowed(["echo", "hello"]) is True

    def test_unbalanced_quote_falls_back_to_naive_split(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path))
        # shlex.split raises ValueError on the dangling quote -> naive split.
        found = sb._first_network_binary("wget 'http://unbalanced")
        assert found == "wget"
        assert sb.network_allowed("wget 'http://unbalanced") is False

    def test_soft_scan_allows_interpreter(self, tmp_path):
        # SOFT boundary: python3 is not in the network-binary denylist, so an
        # interpreter egress attempt is NOT token-blocked (documented gap).
        sb = RestrictedProcessBackend(workdir=str(tmp_path))
        assert sb.network_allowed("python3 -c 'import urllib'") is True
        assert sb.network_allowed("bash -c 'echo >/dev/tcp/127.0.0.1/1'") is True


class TestRestrictedRunPaths:
    def test_run_feeds_stdin_when_input_given(self, tmp_path):
        sb = RestrictedProcessBackend(workdir=str(tmp_path))
        r = sb.run("cat", shell=True, input="hello-stdin")
        assert r.returncode == 0
        assert "hello-stdin" in r.stdout

    def test_run_timeout_kills_process_group(self, tmp_path):
        # Triggers TimeoutExpired + _kill_group (os.killpg path on POSIX).
        sb = RestrictedProcessBackend(workdir=str(tmp_path), timeout=0.4)
        r = sb.run("sleep 5", shell=True)
        assert r.timed_out is True
        assert r.returncode != 0


# ===========================================================================
# koboi/proactive_memory.py
# ===========================================================================


def _mem_mock(get_meta=None, set_meta=None, get_messages=None, has_get=True, has_set=True) -> MagicMock:
    """Build a memory mock with controllable get_meta/set_meta presence."""
    spec = ["get_messages"]
    if has_get:
        spec.append("get_meta")
    if has_set:
        spec.append("set_meta")
    m = MagicMock(spec=spec)
    m.get_messages = MagicMock(side_effect=get_messages) if get_messages else MagicMock(return_value=[])
    if has_get:
        m.get_meta = MagicMock(side_effect=get_meta) if get_meta else MagicMock(return_value=None)
    if has_set:
        m.set_meta = MagicMock(side_effect=set_meta) if set_meta else MagicMock(return_value=None)
    return m


class TestProactiveExtractEdges:
    async def test_none_client_returns_zero(self, tmp_path):
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        pm = ProactiveMemory(
            client=None,
            embedding_client=None,
            memory=_mem_mock(get_messages=lambda: []),
            store=store,
            config={"extract": True},
        )
        assert await pm.extract_and_store() == 0

    async def test_empty_convo_after_format_returns_zero(self, tmp_path):
        # >= 2 messages but all content empty -> convo.strip() is empty.
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        msgs = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=_mem_mock(get_messages=lambda: msgs),
            store=store,
            config={"extract": True},
        )
        assert await pm.extract_and_store() == 0

    async def test_store_returns_error_prefix_is_not_counted(self, tmp_path):
        store = MagicMock()
        store.store = MagicMock(return_value="Error: disk full")  # Error prefix -> skipped
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content='{"preferred_language": "python"}'))
        pm = ProactiveMemory(
            client=client,
            embedding_client=None,
            memory=_mem_mock(get_messages=lambda: msgs),
            store=store,
            config={"extract": True, "core_block": True},
        )
        # All facts rejected by the store -> 0 persisted, core block not touched.
        assert await pm.extract_and_store() == 0
        store.store.assert_called()

    async def test_store_loop_exception_swallowed(self, tmp_path):
        store = MagicMock()
        store.store = MagicMock(side_effect=RuntimeError("kv down"))
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content='{"preferred_language": "python"}'))
        pm = ProactiveMemory(
            client=client,
            embedding_client=None,
            memory=_mem_mock(get_messages=lambda: msgs),
            store=store,
            config={"extract": True},
        )
        # Never raises at SESSION_END.
        assert await pm.extract_and_store() == 0


class TestProactiveFormatAndParse:
    def test_format_conversation_list_content_and_skip_and_truncate(self):
        msgs = [
            {"role": "user", "content": ""},  # empty -> skipped
            {"role": "assistant", "content": [{"text": "part1"}, "part2"]},  # list content
            {"role": "user", "content": "X" * 100},
        ]
        out = ProactiveMemory._format_conversation(msgs, max_chars=30)
        # list content flattened; the long message is truncated to fit max_chars.
        assert "part1" in out
        assert "part2" in out
        assert len(out) <= 30 + 5  # bounded by the truncation slice

    def test_parse_facts_non_dict_returns_empty(self):
        # A JSON list (not object) -> not a dict -> empty.
        assert ProactiveMemory._parse_facts('["a", "b"]') == {}
        assert ProactiveMemory._parse_facts("null") == {}
        assert ProactiveMemory._parse_facts("123") == {}


class TestProactiveRecallEdges:
    async def test_empty_query_returns_none(self, tmp_path):
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=MagicMock(),
            memory=_mem_mock(),
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"recall": True},
        )
        assert await pm.recall("") is None

    async def test_empty_query_vector_returns_none(self, tmp_path):
        ec = MagicMock()
        ec.get_embeddings = AsyncMock(return_value=[])  # falsy vector
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        store._data = {"k": "v"}
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=ec,
            memory=_mem_mock(),
            store=store,
            config={"recall": True},
        )
        assert await pm.recall("anything") is None

    async def test_ensure_embeddings_noop_without_client(self, tmp_path):
        # Constructor sets self._embedding_client = embedding_client or client,
        # so BOTH must be None to exercise the None-guarded early return.
        pm = ProactiveMemory(
            client=None,
            embedding_client=None,
            memory=_mem_mock(),
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"recall": True},
        )
        await pm._ensure_embeddings()  # early return, no error


class TestProactiveCoreBlock:
    def test_get_core_block_no_get_meta(self, tmp_path):
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=_mem_mock(has_get=False, has_set=False),
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"core_block": True},
        )
        assert pm.get_core_block() is None

    def test_get_core_block_meta_raises_returns_none(self, tmp_path):
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=_mem_mock(get_meta=RuntimeError("db locked")),
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"core_block": True},
        )
        assert pm.get_core_block() is None

    def test_get_core_block_empty_and_corrupt(self, tmp_path):
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        # empty raw
        pm_empty = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=_mem_mock(get_meta=lambda _k: None),
            store=store,
            config={"core_block": True},
        )
        assert pm_empty.get_core_block() is None
        # corrupt JSON
        pm_bad = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=_mem_mock(get_meta=lambda _k: "not json{"),
            store=store,
            config={"core_block": True},
        )
        assert pm_bad.get_core_block() is None
        # non-dict JSON
        pm_list = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=_mem_mock(get_meta=lambda _k: "[1, 2]"),
            store=store,
            config={"core_block": True},
        )
        assert pm_list.get_core_block() is None
        # valid dict -> rendered
        pm_ok = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=_mem_mock(get_meta=lambda _k: '{"name": "alice"}'),
            store=store,
            config={"core_block": True},
        )
        cb = pm_ok.get_core_block()
        assert cb is not None and "alice" in cb

    def test_merge_core_block_no_set_meta(self, tmp_path):
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=_mem_mock(get_meta=lambda _k: None, has_set=False),
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"core_block": True},
        )
        # No set_meta -> early return, no raise.
        pm._merge_core_block({"a": "b"})

    def test_merge_core_block_read_error_skips(self, tmp_path):
        mem = _mem_mock(get_meta=RuntimeError("read fail"))
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=mem,
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"core_block": True},
        )
        pm._merge_core_block({"a": "b"})
        mem.set_meta.assert_not_called()  # merge skipped on read error

    def test_merge_core_block_corrupt_skips(self, tmp_path):
        mem = _mem_mock(get_meta=lambda _k: "not json")
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=mem,
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"core_block": True},
        )
        pm._merge_core_block({"a": "b"})
        mem.set_meta.assert_not_called()

    def test_merge_core_block_trims_oversized_block(self, tmp_path):
        big_value = "Z" * 2500
        mem = _mem_mock(get_meta=lambda _k: json.dumps({"big": big_value}))
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=mem,
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"core_block": True},
        )
        pm._merge_core_block({"new": "fact"})
        # set_meta called with a rendered block trimmed under _CORE_MAX_CHARS.
        mem.set_meta.assert_called_once()
        rendered = mem.set_meta.call_args.args[1]
        assert len(rendered) <= 2000

    def test_merge_core_block_persist_error_swallowed(self, tmp_path):
        mem = _mem_mock(get_meta=lambda _k: None, set_meta=RuntimeError("write fail"))
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=mem,
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"core_block": True},
        )
        pm._merge_core_block({"a": "b"})  # no raise


# ===========================================================================
# koboi/context/manager.py
# ===========================================================================


class TestFlattenText:
    def test_none(self):
        assert _flatten_text(None) == ""

    def test_plain_string(self):
        assert _flatten_text("hello") == "hello"

    def test_list_of_dicts_and_strs(self):
        out = _flatten_text([{"text": "a"}, {"content": "b"}, "c"])
        assert "a" in out and "b" in out and "c" in out

    def test_list_of_other(self):
        assert _flatten_text([1, 2]) == "1 2"

    def test_dict_content(self):
        assert _flatten_text({"text": "from-text"}) == "from-text"
        assert _flatten_text({"content": "from-content"}) == "from-content"
        # dict with neither key -> str() fallback
        assert _flatten_text({"x": 1}) == str({"x": 1})

    def test_other_type(self):
        assert _flatten_text(42) == "42"


class TestNoopInternals:
    def test_strategy_name(self):
        assert NoopContextManager()._strategy_name == "NOOP"

    async def test_build_result_passthrough(self):
        mgr = NoopContextManager()
        res, detail = await mgr._build_result([{"role": "system", "content": "s"}], [{"role": "user", "content": "u"}])
        assert len(res) == 2
        assert "passthrough" in detail


class TestKeyFactsSkipEmpty:
    async def test_empty_content_in_old_section_skipped(self):
        mgr = KeyFactsManager(keep_last=1)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": ""},  # empty -> skipped in fact extraction
            {"role": "assistant", "content": ""},  # empty -> skipped
            {"role": "user", "content": "recent"},
        ]
        res = await mgr.manage(msgs, max_tokens=1)
        # No "Previously collected data" facts msg because all old content was empty.
        assert not any("Previously collected data" in m.get("content", "") for m in res)


class TestEnsureToolIntegrityBranches:
    def test_missing_results_with_content_keeps_partial_calls(self):
        # Assistant has content + two tool_calls; only tc_kept has a tool result,
        # so tc_gone is "missing" and dropped while tc_kept survives (kept_calls
        # branch at context/manager.py:58).
        msgs = [
            {
                "role": "assistant",
                "content": "partial answer",
                "tool_calls": [
                    {"id": "tc_kept", "function": {"name": "a"}},
                    {"id": "tc_gone", "function": {"name": "b"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_kept", "content": "result-a"},
        ]
        out = ensure_tool_integrity(msgs)
        asst = [m for m in out if m.get("role") == "assistant"][0]
        assert asst["content"] == "partial answer"  # content preserved
        # tc_gone had no result -> dropped; tc_kept had a result -> retained
        assert [tc["id"] for tc in asst["tool_calls"]] == ["tc_kept"]


class TestSlidingWindowHydratePersist:
    def test_summary_hydrated_from_meta_store(self):
        store = MagicMock()
        store.get_meta = MagicMock(return_value="hydrated summary")
        mgr = SlidingWindowManager()
        mgr.meta_store = store
        mgr._ensure_summary_loaded()
        assert mgr._summary == "hydrated summary"
        # Idempotent: second call does not re-read.
        store.get_meta.reset_mock()
        mgr._ensure_summary_loaded()
        store.get_meta.assert_not_called()

    def test_summary_hydrate_failure_falls_back_empty(self):
        store = MagicMock()
        store.get_meta = MagicMock(side_effect=RuntimeError("db"))
        mgr = SlidingWindowManager()
        mgr.meta_store = store
        mgr._ensure_summary_loaded()  # no raise
        assert mgr._summary == ""

    def test_persist_summary_writes_to_store(self):
        store = MagicMock()
        mgr = SlidingWindowManager()
        mgr.meta_store = store
        mgr._summary = "current summary"
        mgr._persist_summary()
        store.set_meta.assert_called_once_with("sliding_window_summary", "current summary")

    def test_persist_summary_failure_swallowed(self):
        store = MagicMock()
        store.set_meta = MagicMock(side_effect=RuntimeError("write"))
        mgr = SlidingWindowManager()
        mgr.meta_store = store
        mgr._summary = "x"
        mgr._persist_summary()  # no raise

    async def test_summarize_with_prev_summary_and_tool_calls(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content="new summary"))
        mgr = SlidingWindowManager(client=client)
        mgr._summary = "old summary"  # prev_summary present
        old = [
            {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}],
            },
        ]
        out = await mgr._summarize(old, mgr._summary)
        assert out == "new summary"
        # The tool_calls line + previous-summary line were both appended to the prompt.
        sent = client.complete.call_args.kwargs.get("messages") or client.complete.call_args.args[0]
        sent_text = json.dumps(sent)
        assert "Previous summary" in sent_text
        assert "read" in sent_text


# ===========================================================================
# koboi/tools/builtin/memory.py
# ===========================================================================


class TestMemoryStoreInternals:
    def test_save_returns_false_on_oserror(self, tmp_path, monkeypatch):
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        monkeypatch.setattr("koboi.tools.builtin.memory.os.replace", MagicMock(side_effect=OSError("no")))
        assert store._save() is False

    def test_acquire_lock_retries_then_fails(self, tmp_path, monkeypatch):
        # fcntl present on POSIX; make flock always raise OSError so all retries fail.
        fcntl_mod = __import__("koboi.tools.builtin.memory", fromlist=["fcntl"]).fcntl
        if fcntl_mod is None:
            pytest.skip("fcntl absent on this platform")
        monkeypatch.setattr(fcntl_mod, "flock", MagicMock(side_effect=OSError("locked")))
        monkeypatch.setattr("koboi.tools.builtin.memory.time.sleep", lambda *_a: None)
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        fd = store._acquire_lock()
        assert fd is None  # exhausted LOCK_RETRIES

    def test_store_returns_lock_error_when_acquire_fails(self, tmp_path, monkeypatch):
        fcntl_mod = __import__("koboi.tools.builtin.memory", fromlist=["fcntl"]).fcntl
        if fcntl_mod is None:
            pytest.skip("fcntl absent on this platform")
        monkeypatch.setattr(fcntl_mod, "flock", MagicMock(side_effect=OSError("locked")))
        monkeypatch.setattr("koboi.tools.builtin.memory.time.sleep", lambda *_a: None)
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        out = store.store("k", "v")
        assert out.startswith("Error: could not acquire lock")

    def test_store_returns_persist_error_when_save_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr("koboi.tools.builtin.memory.os.replace", MagicMock(side_effect=OSError("disk")))
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        out = store.store("k", "v")
        assert "failed to persist" in out

    def test_release_lock_swallows_error(self, tmp_path, monkeypatch):
        fcntl_mod = __import__("koboi.tools.builtin.memory", fromlist=["fcntl"]).fcntl
        if fcntl_mod is None:
            pytest.skip("fcntl absent on this platform")
        fd = MagicMock()
        monkeypatch.setattr(fcntl_mod, "flock", MagicMock(side_effect=OSError("bad fd")))
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        store._release_lock(fd)  # no raise

    def test_recall_query_and_key_and_empty_paths(self, tmp_path):
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        store._data = {"name": "alice", "city": "Jakarta"}
        # query match
        assert "alice" in store.recall(query="alic")
        # key miss
        assert "not found" in store.recall(key="nope")
        # key hit
        assert "alice" in store.recall(key="name")
        # query no match
        assert "No entry" in store.recall(query="zzz")

    def test_memory_tool_store_recall_roundtrip(self, tmp_path):
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        deps = {"memory_store_ref": store}
        assert "Successfully" in memory_store("k", "v", _deps=deps)
        assert "v" in memory_recall("k", _deps=deps)


# ===========================================================================
# koboi/tools/builtin/web.py
# ===========================================================================


class _FakeResponse:
    """httpx.Response double -- supports BOTH the eager ``.content`` path and the
    streaming ``.aiter_bytes()`` path, and counts total bytes the consumer pulled
    (``bytes_consumed``) so a test can assert a bound (CWE-400 #56)."""

    def __init__(self, status_code=200, content=b"ok", headers=None, reason="OK", text=None):
        self.status_code = status_code
        self._content = content
        self.headers = headers or {}
        self.reason_phrase = reason
        self._text = text
        self.bytes_consumed = 0

    @property
    def content(self):
        # Eager path: the full body is materialized at once.
        self.bytes_consumed += len(self._content)
        return self._content

    @property
    def text(self):
        if self._text is not None:
            return self._text
        return self._content.decode("utf-8", errors="replace")

    async def aiter_bytes(self, chunk_size=1024):
        # Lazy streaming path: yields chunks; the consumer is expected to stop early
        # once it has enough (bounded read). Only the chunks actually pulled count.
        for i in range(0, len(self._content), chunk_size):
            chunk = self._content[i : i + chunk_size]
            self.bytes_consumed += len(chunk)
            yield chunk


class _FakeStreamCtx:
    """Async context manager backing ``AsyncClient.stream("GET", url)``."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *a):
        return False


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient double: async context manager + get()/stream()."""

    def __init__(self, response=None, sequence=None, exc=None):
        self._response = response
        self._sequence = sequence
        self._exc = exc
        self._idx = 0
        self.last_response = None  # most recent response handed to a consumer

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def _next(self):
        if self._exc is not None:
            raise self._exc
        if self._sequence is not None:
            item = self._sequence[self._idx]
            self._idx += 1
            if isinstance(item, Exception):
                raise item
            self.last_response = item
            return item
        self.last_response = self._response
        return self._response

    async def get(self, url):
        # Eager path (kept for back-compat). Pre-fix web_fetch used this + .content.
        return self._next()

    def stream(self, method, url):
        # Streaming path (post-fix web_fetch). Returns an async context manager.
        return _FakeStreamCtx(self._next())


def _patch_httpx(monkeypatch, fake):
    monkeypatch.setattr(web_mod.httpx, "AsyncClient", lambda **kw: fake)


def _no_ssrf(monkeypatch):
    monkeypatch.setattr(web_mod, "_check_url_ssrf", lambda _url: None)


class TestWebSearchDuckDuckGo:
    async def test_duckduckgo_parse_success(self, monkeypatch):
        html = (
            '<a class="result__a" href="//duck.com/r1">Result One</a>'
            '<a class="result__snippet" href="//duck.com/r1">Snippet one</a>'
        )
        fake = _FakeAsyncClient(response=_FakeResponse(text=html))
        _patch_httpx(monkeypatch, fake)
        monkeypatch.setattr(web_mod, "WEB_SEARCH_PROVIDER", "duckduckgo")
        out = await web_search("python")
        assert "Result One" in out

    async def test_duckduckgo_http_error_returns_message(self, monkeypatch):
        fake = _FakeAsyncClient(exc=RuntimeError("network down"))
        _patch_httpx(monkeypatch, fake)
        monkeypatch.setattr(web_mod, "WEB_SEARCH_PROVIDER", "duckduckgo")
        out = await web_search("python")
        assert "Error: search failed" in out

    async def test_duckduckgo_no_results(self, monkeypatch):
        fake = _FakeAsyncClient(response=_FakeResponse(text="<html>nothing</html>"))
        _patch_httpx(monkeypatch, fake)
        monkeypatch.setattr(web_mod, "WEB_SEARCH_PROVIDER", "duckduckgo")
        out = await web_search("python")
        assert "No results" in out


class TestDDGParser:
    def test_parses_result_block(self):
        p = _DDGResultParser()
        p.feed('<a class="result__a" href="//x/r">T</a><a class="result__snippet" href="//x/r">S</a>')
        assert len(p.results) == 1
        assert p.results[0]["title"] == "T"

    def test_snippet_fallback_when_empty(self):
        p = _DDGResultParser()
        p.feed('<a class="result__a" href="//x/r">T</a><a class="result__snippet" href="//x/r"></a>')
        assert p.results[0]["snippet"] == "(no description)"


class TestResolveAndCheck:
    def test_no_addresses_raises(self, monkeypatch):
        monkeypatch.setattr(web_mod.socket, "getaddrinfo", lambda *_a, **_k: [])
        with pytest.raises(ValueError, match="no addresses"):
            _resolve_and_check("example.com")

    def test_private_ip_raises(self, monkeypatch):
        def fake_getaddrinfo(host, *a, **k):
            import socket as _s

            return [(_s.AF_INET, _s.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

        monkeypatch.setattr(web_mod.socket, "getaddrinfo", fake_getaddrinfo)
        with pytest.raises(ValueError, match="internal IP"):
            _resolve_and_check("localhost.example")


class TestWebFetchErrors:
    async def test_bad_scheme(self):
        assert "must start with" in await web_fetch("ftp://x")

    async def test_connect_error(self, monkeypatch):
        _no_ssrf(monkeypatch)
        _patch_httpx(monkeypatch, _FakeAsyncClient(exc=httpx.ConnectError("refused")))
        out = await web_fetch("https://example.com")
        assert "connection failed" in out

    async def test_timeout_error(self, monkeypatch):
        _no_ssrf(monkeypatch)
        _patch_httpx(monkeypatch, _FakeAsyncClient(exc=httpx.TimeoutException("slow")))
        out = await web_fetch("https://example.com")
        assert "timed out" in out

    async def test_gaierror_unresolvable(self, monkeypatch):
        # _check_url_ssrf raises socket.gaierror -> friendly DNS message.
        def raise_gaierror(_url):
            import socket as _s

            raise _s.gaierror("no dns")

        monkeypatch.setattr(web_mod, "_check_url_ssrf", raise_gaierror)
        out = await web_fetch("https://example.invalid")
        assert "failed to resolve hostname" in out

    async def test_ssrf_value_error(self, monkeypatch):
        monkeypatch.setattr(web_mod, "_check_url_ssrf", lambda _u: (_ for _ in ()).throw(ValueError("internal IP")))
        out = await web_fetch("https://example.com")
        assert "Error:" in out


class TestWebFetchStatuses:
    async def test_non_retryable_status(self, monkeypatch):
        _no_ssrf(monkeypatch)
        _patch_httpx(monkeypatch, _FakeAsyncClient(response=_FakeResponse(status_code=404, reason="Not Found")))
        out = await web_fetch("https://example.com")
        assert "HTTP 404" in out

    async def test_retryable_then_success(self, monkeypatch):
        _no_ssrf(monkeypatch)
        monkeypatch.setattr(web_mod.asyncio, "sleep", AsyncMock())
        seq = [_FakeResponse(status_code=503), _FakeResponse(status_code=200, content=b"hello")]
        _patch_httpx(monkeypatch, _FakeAsyncClient(sequence=seq))
        out = await web_fetch("https://example.com")
        assert "hello" in out

    async def test_retryable_exhausts_max_retries(self, monkeypatch):
        # 429 is retryable; on the final attempt attempt<MAX_RETRIES is False, so
        # the loop returns the HTTP error (the for-else "Max retries exceeded"
        # branch is dead code: the last retryable iteration always returns).
        _no_ssrf(monkeypatch)
        monkeypatch.setattr(web_mod.asyncio, "sleep", AsyncMock())
        seq = [_FakeResponse(status_code=429, reason="Too Many")] * 3
        _patch_httpx(monkeypatch, _FakeAsyncClient(sequence=seq))
        out = await web_fetch("https://example.com")
        assert "HTTP 429" in out


class TestWebFetchRedirects:
    async def test_redirect_no_location_breaks(self, monkeypatch):
        _no_ssrf(monkeypatch)
        resp = _FakeResponse(status_code=302, headers={}, reason="Found")
        _patch_httpx(monkeypatch, _FakeAsyncClient(response=resp))
        out = await web_fetch("https://example.com")
        # No Location -> outer break -> processes the (empty) 302 body.
        assert isinstance(out, str)

    async def test_too_many_redirects(self, monkeypatch):
        _no_ssrf(monkeypatch)
        resp = _FakeResponse(status_code=302, headers={"location": "/loop"}, reason="Found")
        _patch_httpx(monkeypatch, _FakeAsyncClient(response=resp))
        out = await web_fetch("https://example.com")
        assert "too many redirects" in out


class TestWebFetchTruncation:
    async def test_body_over_max_response_size_truncated(self, monkeypatch):
        _no_ssrf(monkeypatch)
        big = b"x" * 60000
        _patch_httpx(monkeypatch, _FakeAsyncClient(response=_FakeResponse(content=big)))
        out = await web_fetch("https://example.com")
        assert "response truncated" in out

    async def test_body_over_max_output_truncated(self, monkeypatch):
        _no_ssrf(monkeypatch)
        medium = b"y" * 30000  # < MAX_RESPONSE_SIZE but > MAX_OUTPUT
        _patch_httpx(monkeypatch, _FakeAsyncClient(response=_FakeResponse(content=medium)))
        out = await web_fetch("https://example.com")
        assert "truncated, total" in out


class TestWebFetchStreamingBound:
    """CWE-400 / GHSA-qf8c-xp5r-p869 (#56): web_fetch must not buffer an oversized
    response body before applying the size limit. The counting fake records how many
    bytes the consumer actually pulled; a bounded stream stops near MAX_RESPONSE_SIZE
    instead of materializing the whole body."""

    async def test_oversized_body_not_fully_buffered(self, monkeypatch):
        _no_ssrf(monkeypatch)
        oversized = b"x" * 5_000_000  # 5 MB >> MAX_RESPONSE_SIZE (50_000)
        fake = _FakeAsyncClient(response=_FakeResponse(content=oversized))
        _patch_httpx(monkeypatch, fake)
        out = await web_fetch("https://example.com")
        consumed = fake.last_response.bytes_consumed
        # A bounded reader must stop near the cap, NOT pull all 5 MB.
        assert consumed <= 60_000, f"expected bounded read, consumed {consumed} bytes"
        # Still truncates (partial content), not a hard failure for the no-Content-Length path.
        assert "response truncated" in out

    async def test_content_length_precheck_rejects_without_consuming_body(self, monkeypatch):
        _no_ssrf(monkeypatch)
        oversized = b"x" * 5_000_000
        fake = _FakeAsyncClient(response=_FakeResponse(content=oversized, headers={"Content-Length": "5000000"}))
        _patch_httpx(monkeypatch, fake)
        out = await web_fetch("https://example.com")
        # Content-Length > MAX -> rejected BEFORE any body byte is pulled.
        assert fake.last_response.bytes_consumed == 0
        assert out.startswith("Error:")
