"""Wave 3: CodingContextManager -- tool-result body eviction."""

from __future__ import annotations

import copy
import json

from koboi.context.manager import CodingContextManager

BIG = "x" * 1000  # well above the default evict_min_chars=200


def _pair(tc_id: str, tool: str, args: dict, result: str) -> list[dict]:
    """An assistant(tool_calls) + tool-result message pair."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": tc_id, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
            ],
        },
        {"role": "tool", "tool_call_id": tc_id, "content": result},
    ]


def _mgr(**kw) -> CodingContextManager:
    kw.setdefault("keep_last", 2)
    return CodingContextManager(**kw)


def _msgs(*pairs: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "go"}]
    for p in pairs:
        out += p
    return out


def _tight_budget(msgs: list[dict]) -> int:
    """A budget just BELOW the current estimate: forces the strategy to run,
    while the ~300-token savings from one stub keeps the result under budget
    (no truncation fallback)."""
    from koboi.tokens import estimate_tokens

    return estimate_tokens(msgs) - 10


def _tool_contents(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "tool"]


class TestEviction:
    async def test_under_budget_untouched(self):
        mgr = _mgr()
        msgs = _msgs(_pair("t1", "read_file", {"path": "a.py"}, BIG))
        result = await mgr.manage(list(msgs), max_tokens=10**9)
        assert result == msgs
        assert mgr.last_modified is False

    async def test_same_file_read_twice_older_stubbed(self):
        mgr = _mgr()
        msgs = _msgs(
            _pair("t1", "read_file", {"path": "a.py"}, BIG + "-OLD"),
            _pair("t2", "read_file", {"path": "a.py"}, BIG + "-NEW"),
            _pair("t3", "read_file", {"path": "b.py"}, BIG + "-B"),  # last 2 = protected window
        )
        result = await mgr.manage(msgs, max_tokens=_tight_budget(msgs))
        contents = _tool_contents(result)
        assert any("[evicted read_file('a.py')" in c and "re-read" in c for c in contents)
        # newest a.py read verbatim (it's outside the window but newest per key)
        assert any(c.endswith("-NEW") for c in contents)
        assert mgr.last_modified is True

    async def test_window_read_counts_toward_seen(self):
        # newest read of a.py is INSIDE the window -> the older one outside is stubbed.
        mgr = _mgr(keep_last=2)
        msgs = _msgs(
            _pair("t1", "read_file", {"path": "a.py"}, BIG + "-OLD"),
            _pair("t2", "read_file", {"path": "a.py"}, BIG + "-WINDOW"),
        )
        result = await mgr.manage(msgs, max_tokens=_tight_budget(msgs))
        contents = _tool_contents(result)
        assert any("[evicted read_file('a.py')" in c for c in contents)
        assert any(c.endswith("-WINDOW") for c in contents)

    async def test_edit_result_does_not_evict_newest_read(self):
        # (tool, path) keying: an edit_file confirmation for a.py must not
        # count against read_file(a.py)'s newest-per-key budget. Pure-eviction
        # semantics -> exercise _build_result directly (no budget fallback).
        mgr = _mgr(keep_last=2)
        msgs = _msgs(
            _pair("t1", "read_file", {"path": "a.py"}, BIG + "-READ"),
            _pair("t2", "edit_file", {"path": "a.py"}, BIG + "-EDIT"),
            _pair("t3", "read_file", {"path": "z.py"}, BIG + "-Z"),  # pushes t1/t2 out of window
        )
        result, _ = await mgr._build_result([msgs[0]], msgs[1:])
        contents = _tool_contents(result)
        # read and edit are DIFFERENT keys: each is the newest of its key -> both verbatim.
        assert any(c.endswith("-READ") for c in contents)
        assert any(c.endswith("-EDIT") for c in contents)

    async def test_shell_outputs_keep_newest_only(self):
        mgr = _mgr(keep_last=2)
        msgs = _msgs(
            _pair("t1", "run_shell", {"command": "pytest"}, BIG + "-RUN1"),
            _pair("t2", "run_shell", {"command": "pytest"}, BIG + "-RUN2"),
            _pair("t3", "read_file", {"path": "z.py"}, BIG + "-Z"),
        )
        result = await mgr.manage(msgs, max_tokens=_tight_budget(msgs))
        contents = _tool_contents(result)
        assert any("[evicted run_shell result" in c and "re-run" in c for c in contents)
        assert any(c.endswith("-RUN2") for c in contents)

    async def test_keep_newest_per_key_two(self):
        # keep_newest_per_key=2 keeps BOTH shell runs verbatim -- nothing is
        # evictable, so exercise _build_result directly (manage() would take
        # the budget fallback with zero savings).
        mgr = _mgr(keep_last=2, keep_newest_per_key=2)
        msgs = _msgs(
            _pair("t1", "run_shell", {"command": "pytest"}, BIG + "-RUN1"),
            _pair("t2", "run_shell", {"command": "pytest"}, BIG + "-RUN2"),
            _pair("t3", "read_file", {"path": "z.py"}, BIG + "-Z"),
        )
        result, _ = await mgr._build_result([msgs[0]], msgs[1:])
        contents = _tool_contents(result)
        assert any(c.endswith("-RUN1") for c in contents)  # 2nd-newest kept too
        assert any(c.endswith("-RUN2") for c in contents)

    async def test_small_bodies_never_stubbed(self):
        mgr = _mgr(keep_last=1)
        small = "ok"
        msgs = _msgs(
            _pair("t1", "read_file", {"path": "a.py"}, small),
            _pair("t2", "read_file", {"path": "a.py"}, small),
            _pair("t3", "read_file", {"path": "b.py"}, BIG),
        )
        result = await mgr.manage(msgs, max_tokens=_tight_budget(msgs))
        assert not any("[evicted" in c and "'a.py'" in c for c in _tool_contents(result))

    async def test_malformed_arguments_no_crash(self):
        # TWO malformed-args reads share the pathless (read_file, None) key ->
        # the older one is evictable; parsing must never crash.
        mgr = _mgr(keep_last=1)
        pair1 = _pair("t1", "read_file", {}, BIG + "-M1")
        pair1[0]["tool_calls"][0]["function"]["arguments"] = "{not json"
        pair2 = _pair("t2", "read_file", {}, BIG + "-M2")
        pair2[0]["tool_calls"][0]["function"]["arguments"] = "{not json"
        msgs = _msgs(pair1, pair2, _pair("t3", "read_file", {"path": "b.py"}, BIG))
        result, _ = await mgr._build_result([msgs[0]], msgs[1:])
        contents = _tool_contents(result)
        assert any("[evicted read_file result" in c for c in contents)  # older M1 stubbed
        assert any(c.endswith("-M2") for c in contents)  # newest pathless kept


class TestInvariants:
    async def test_input_dicts_not_mutated(self):
        mgr = _mgr()
        msgs = _msgs(
            _pair("t1", "read_file", {"path": "a.py"}, BIG + "-OLD"),
            _pair("t2", "read_file", {"path": "a.py"}, BIG + "-NEW"),
            _pair("t3", "read_file", {"path": "b.py"}, BIG),
        )
        snapshot = copy.deepcopy(msgs)
        await mgr.manage(msgs, max_tokens=_tight_budget(msgs))
        assert msgs == snapshot  # memory-owned dicts untouched

    async def test_integrity_round_trip(self):
        mgr = _mgr()
        msgs = _msgs(
            _pair("t1", "read_file", {"path": "a.py"}, BIG + "-OLD"),
            _pair("t2", "read_file", {"path": "a.py"}, BIG + "-NEW"),
            _pair("t3", "read_file", {"path": "b.py"}, BIG),
        )
        result = await mgr.manage(msgs, max_tokens=_tight_budget(msgs))
        # every assistant tool_call still has its (possibly stubbed) result
        call_ids = {tc["id"] for m in result if m.get("tool_calls") for tc in m["tool_calls"]}
        result_ids = {m.get("tool_call_id") for m in result if m.get("role") == "tool"}
        assert call_ids == result_ids
        # first non-system is user
        non_system = [m for m in result if m.get("role") != "system"]
        assert non_system[0]["role"] == "user"

    async def test_system_messages_preserved(self):
        mgr = _mgr()
        msgs = _msgs(_pair("t1", "read_file", {"path": "a.py"}, BIG))
        msgs.append({"role": "system", "content": "[Active Task State] hook inject"})
        msgs += _pair("t2", "read_file", {"path": "a.py"}, BIG)
        msgs += _pair("t3", "read_file", {"path": "b.py"}, BIG)
        result = await mgr.manage(msgs, max_tokens=_tight_budget(msgs))
        sys_contents = [m["content"] for m in result if m.get("role") == "system"]
        assert "system prompt" in sys_contents
        assert "[Active Task State] hook inject" in sys_contents

    async def test_fallback_truncates_when_stubbing_insufficient(self):
        # keep_last window itself is huge -> stubbing can't shrink under a tiny
        # budget -> fallback to system + last keep_last.
        mgr = _mgr(keep_last=2)
        pairs = [_pair(f"t{i}", "read_file", {"path": f"f{i}.py"}, BIG) for i in range(6)]
        msgs = _msgs(*pairs)
        result = await mgr.manage(msgs, max_tokens=1)
        non_system = [m for m in result if m.get("role") != "system"]
        # integrity may add a synthetic user; the tail is at most keep_last + 1
        assert len(non_system) <= mgr.keep_last + 1


class TestRegistryAndWiring:
    def test_coding_strategy_registered(self):
        from koboi.context.registry import build_context

        mgr = build_context("coding", keep_last=5, evict_min_chars=100, keep_newest_per_key=2)
        assert isinstance(mgr, CodingContextManager)
        assert mgr.keep_last == 5
        assert mgr.evict_min_chars == 100
        assert mgr.keep_newest_per_key == 2

    def test_summarization_truncation_accepted(self):
        # The facade forwards summarization_truncation unconditionally when set;
        # the coding strategy must accept-and-ignore it (no TypeError).
        from koboi.context.registry import build_context

        mgr = build_context("coding", summarization_truncation=50)
        assert isinstance(mgr, CodingContextManager)


class TestLoopCompactionSignal:
    async def test_body_only_eviction_sets_last_compacted(self):
        """Stub-only compaction keeps the message COUNT constant; the loop's
        _last_compacted must still read True via ContextManager.last_modified."""
        from koboi.loop import AgentCore
        from koboi.memory import ConversationMemory
        from koboi.tokens import estimate_tokens
        from tests.conftest import MockClient, make_mock_response

        mgr = _mgr(keep_last=2)
        memory = ConversationMemory()
        memory.add_user_message("go")
        for pair in [
            _pair("t1", "read_file", {"path": "a.py"}, BIG + "-OLD"),
            _pair("t2", "read_file", {"path": "a.py"}, BIG + "-NEW"),
            _pair("t3", "read_file", {"path": "b.py"}, BIG),
        ]:
            memory.add_assistant_message(pair[0]["content"], pair[0]["tool_calls"])
            memory.add_tool_result(pair[1]["tool_call_id"], pair[1]["content"])

        budget = estimate_tokens(memory.get_messages()) - 10
        core = AgentCore(
            client=MockClient([make_mock_response("done")]),
            memory=memory,
            context_manager=mgr,
            max_context_tokens=budget,
            max_iterations=2,
        )
        before = len(memory.get_messages())
        managed = await core._get_managed_messages()
        assert len(managed) == before  # body-only: count unchanged
        assert core._last_compacted is True  # honest signal via last_modified
