"""Tests for koboi.llm.cache (v2 step 1): ResponseCache + CachedClient."""

import asyncio

from koboi.llm.cache import (
    CacheEntry,
    CacheMissError,
    CacheMissPolicy,
    CachedClient,
    ResponseCache,
    compute_cache_key,
    _deserialize_response,
    _serialize_response,
)
from koboi.types import AgentResponse, TokenUsage, ToolCall
from tests.conftest import MockClient, make_mock_response


def _resp(content="hello", tool_calls=None, model="mock-model"):
    return AgentResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, reasoning_tokens=0),
        model=model,
        base_url="http://x/v1",
    )


class TestCacheKey:
    def test_content_addressed_perturbations(self):
        msgs = [{"role": "user", "content": "hi"}]
        base = compute_cache_key("m", msgs, None, None)
        assert compute_cache_key("m2", msgs, None, None) != base  # model
        assert compute_cache_key("m", [{"role": "user", "content": "bye"}], None, None) != base  # messages
        assert compute_cache_key("m", msgs, [{"type": "function"}], None) != base  # tools
        assert compute_cache_key("m", msgs, None, {"type": "json_object"}) != base  # response_format

    def test_none_and_empty_tools_collapse(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert compute_cache_key("m", msgs, None, None) == compute_cache_key("m", msgs, [], {})

    def test_insertion_order_independent(self):
        a = compute_cache_key("m", [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], None, None)
        # same messages, same set -> same key (sort_keys handles nested dict order)
        b = compute_cache_key("m", [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], None, None)
        assert a == b


class TestSerialization:
    def test_round_trip_tool_calls_and_usage(self):
        resp = _resp(
            content="c",
            tool_calls=[ToolCall(id="1", name="calc", arguments='{"x": 1}')],
        )
        back = _deserialize_response(_serialize_response(resp))
        assert back.content == "c"
        assert back.tool_calls[0].name == "calc"
        assert back.tool_calls[0].arguments == '{"x": 1}'
        assert back.usage.prompt_tokens == 10
        assert back.usage.reasoning_tokens == 0
        assert back.model == "mock-model"


class TestResponseCache:
    def test_get_miss_returns_none(self, tmp_path):
        assert ResponseCache(tmp_path / "c").get("deadbeef") is None

    def test_put_then_get_round_trips(self, tmp_path):
        cache = ResponseCache(tmp_path / "c")
        key = "ab" * 32
        cache.put(key, _resp("hi"), model="m")
        got = cache.get(key)
        assert got is not None and got.content == "hi"
        assert cache.has(key)

    def test_sharded_layout(self, tmp_path):
        cache = ResponseCache(tmp_path / "c")
        key = "c0ffee" + "0" * 58
        cache.put(key, _resp("hi"), model="m")
        assert (tmp_path / "c" / "c0" / f"{key}.json").exists()

    def test_corrupt_file_fail_soft(self, tmp_path):
        cache = ResponseCache(tmp_path / "c")
        key = "ab" * 32
        path = cache._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert cache.get(key) is None  # never raises

    def test_iter_entries_count_clear(self, tmp_path):
        cache = ResponseCache(tmp_path / "c")
        for i in range(3):
            cache.put(f"{i:064d}", _resp(str(i)), model="m")
        assert cache.count() == 3
        entries = list(cache.iter_entries())
        assert len(entries) == 3
        assert cache.clear() == 3
        assert cache.count() == 0

    def test_load_entries_hydrates_sidecar(self, tmp_path):
        src = ResponseCache(tmp_path / "src")
        key = "cd" * 32
        src.put(key, _resp("hi"), model="m")
        entries = [(k, p) for k, p in src.iter_entries()]
        dst = ResponseCache(tmp_path / "dst")
        assert dst.load_entries(entries) == 1
        assert dst.get(key).content == "hi"


class TestCachedClient:
    def test_miss_then_hit_stores_and_replays(self, tmp_path):
        inner = MockClient([_resp("hello")])
        client = CachedClient(inner, ResponseCache(tmp_path / "c"))
        msgs = [{"role": "user", "content": "hi"}]
        r1 = asyncio.run(client.complete(msgs))
        r2 = asyncio.run(client.complete(msgs))
        assert inner.call_count == 1  # second was a cache hit
        assert r1.content == r2.content == "hello"
        assert r2.model == "mock-model"

    def test_embeddings_not_cached(self, tmp_path):
        inner = MockClient([])
        client = CachedClient(inner, ResponseCache(tmp_path / "c"))
        asyncio.run(client.get_embeddings("x"))
        asyncio.run(client.get_embeddings("x"))
        # no cache files written for embeddings
        assert ResponseCache(tmp_path / "c").count() == 0

    def test_disabled_passthrough(self, tmp_path):
        inner = MockClient([_resp("a"), _resp("b")])
        client = CachedClient(inner, ResponseCache(tmp_path / "c"), enabled=False)
        msgs = [{"role": "user", "content": "hi"}]
        asyncio.run(client.complete(msgs))
        asyncio.run(client.complete(msgs))
        assert inner.call_count == 2  # no caching
        assert ResponseCache(tmp_path / "c").count() == 0

    def test_on_miss_raise(self, tmp_path):
        inner = MockClient([_resp("a")])
        client = CachedClient(inner, ResponseCache(tmp_path / "c"), on_miss=CacheMissPolicy.RAISE)
        try:
            asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
            assert False, "expected CacheMissError"
        except CacheMissError:
            pass
        assert inner.call_count == 0  # never called the inner

    def test_concurrent_identical_calls_single_upstream(self, tmp_path):
        inner = MockClient([_resp("a")])
        client = CachedClient(inner, ResponseCache(tmp_path / "c"))
        msgs = [{"role": "user", "content": "hi"}]

        async def run():
            await asyncio.gather(client.complete(msgs), client.complete(msgs))

        asyncio.run(run())
        assert inner.call_count == 1  # coalesced

    def test_stream_hit_yields_delta_plus_complete(self, tmp_path):
        from koboi.events import CompleteEvent, TextDeltaEvent

        inner = MockClient([_resp("streamed")])
        client = CachedClient(inner, ResponseCache(tmp_path / "c"))
        msgs = [{"role": "user", "content": "hi"}]

        async def collect(first):
            events = []
            async for ev in client.complete_stream(msgs):
                events.append(ev)
            return events

        # first call: live stream (miss) -> stores
        live = asyncio.run(collect(True))
        assert inner.call_count == 1
        # second call: cache hit -> single TextDelta + CompleteEvent
        hit = asyncio.run(collect(False))
        assert inner.call_count == 1  # still 1
        assert isinstance(hit[0], TextDeltaEvent) and hit[0].content == "streamed"
        assert isinstance(hit[-1], CompleteEvent)
        assert hit[-1].response.content == "streamed"

    def test_cache_dir_created_lazily(self, tmp_path):
        client = CachedClient(MockClient([_resp("a")]), ResponseCache(tmp_path / "nested" / "c"))
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
        assert (tmp_path / "nested" / "c").exists()

    def test_model_and_provider_delegate(self, tmp_path):
        inner = MockClient([_resp("a")])
        inner.model = "gpt-x"
        client = CachedClient(inner, ResponseCache(tmp_path / "c"))
        assert client.model == "gpt-x"
        assert client.provider == "?"  # MockClient has no provider attr
