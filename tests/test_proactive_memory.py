"""Tests for the proactive long-term-memory subsystem (D extract / C recall / B core block)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.memory_sqlite import SQLiteMemory
from koboi.proactive_memory import ProactiveMemory
from koboi.tools.builtin.memory import _MemoryStore
from koboi.types import AgentResponse


def _two_turn_memory(tmp_path) -> SQLiteMemory:
    mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S")
    mem.add_user_message("Please always reply in Python and remember I'm in UTC.")
    mem.add_assistant_message("Sure — I'll use Python and note your timezone is UTC.")
    return mem


def _proactive(tmp_path, client, cfg=None) -> ProactiveMemory:
    mem = _two_turn_memory(tmp_path)
    store = _MemoryStore(filepath=str(tmp_path / "m.json"))
    return ProactiveMemory(
        client=client,
        embedding_client=client,
        memory=mem,
        store=store,
        config=cfg or {"extract": True},
    )


class TestExtractionD:
    async def test_extracts_and_stores_facts(self, tmp_path):
        client = MagicMock()
        client.complete = AsyncMock(
            return_value=AgentResponse(content='{"preferred_language": "python", "timezone": "UTC"}')
        )
        pm = _proactive(tmp_path, client)
        n = await pm.extract_and_store()
        assert n == 2
        data = pm._store._data
        assert data.get("preferred_language") == "python"
        assert data.get("timezone") == "UTC"

    async def test_redacts_secrets_and_drops_them(self, tmp_path):
        # A password fact (sensitive key) and an OpenAI key (value shape) must NOT be stored.
        secret = "sk-live-" + "a" * 24
        client = MagicMock()
        client.complete = AsyncMock(
            return_value=AgentResponse(content='{"password": "hunter2", "api_key": "' + secret + '", "normal": "ok"}')
        )
        pm = _proactive(tmp_path, client)
        n = await pm.extract_and_store()
        assert "normal" in pm._store._data  # benign fact stored
        assert pm._store._data["normal"] == "ok"
        assert "password" not in pm._store._data  # sensitive key dropped
        assert "api_key" not in pm._store._data  # sensitive key dropped
        assert secret not in json_of(pm._store)
        assert n == 1  # only the benign fact counted

    async def test_handles_empty_and_garbage(self, tmp_path):
        for content in (None, "", "no json here", "{}", "```json\n{}\n```"):
            client = MagicMock()
            client.complete = AsyncMock(return_value=AgentResponse(content=content))
            pm = _proactive(tmp_path, client)
            assert await pm.extract_and_store() == 0

    async def test_failure_does_not_raise(self, tmp_path):
        client = MagicMock()
        client.complete = AsyncMock(side_effect=RuntimeError("network"))
        pm = _proactive(tmp_path, client)
        assert await pm.extract_and_store() == 0  # swallows, returns 0

    async def test_short_conversation_skips(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "p2.db"), session_id="S")
        mem.add_user_message("hi")  # only 1 message
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        pm = ProactiveMemory(
            client=MagicMock(), embedding_client=None, memory=mem, store=store, config={"extract": True}
        )
        assert await pm.extract_and_store() == 0


def json_of(store: _MemoryStore) -> str:
    import json

    return json.dumps(store._data)


def _embed_client_for_recall():
    """Deterministic embedding: keyword -> unit vector (python/dog/utc axes)."""

    def fake_embed(text):
        t = text.lower()
        if "python" in t:
            return [1.0, 0.0, 0.0]
        if "dog" in t:
            return [0.0, 1.0, 0.0]
        if "utc" in t:
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0]

    client = MagicMock()
    client.get_embeddings = AsyncMock(side_effect=fake_embed)
    return client


def _seeded_store(tmp_path) -> _MemoryStore:
    store = _MemoryStore(filepath=str(tmp_path / "m.json"))
    store._data = {
        "preferred_language": "user prefers python",
        "pet": "user has a dog",
        "timezone": "user is in UTC",
    }
    return store


class TestRecallC:
    async def test_ranks_relevant_fact_first(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S")
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=_embed_client_for_recall(),
            memory=mem,
            store=_seeded_store(tmp_path),
            config={"recall": True, "top_k": 4, "min_score": 0.0},
        )
        block = await pm.recall("does the user prefer python")
        assert block is not None
        assert "preferred_language" in block
        # python fact ranks first (cosine 1.0)
        assert block.split("\n")[1].startswith("- preferred_language")

    async def test_query_cache_prevents_re_embed(self, tmp_path):
        ec = _embed_client_for_recall()
        mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S")
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=ec,
            memory=mem,
            store=_seeded_store(tmp_path),
            config={"recall": True},
        )
        await pm.recall("does the user prefer python")
        calls_after_first = ec.get_embeddings.call_count
        await pm.recall("does the user prefer python")
        assert ec.get_embeddings.call_count == calls_after_first  # cache hit

    async def test_min_score_filters(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S")
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=_embed_client_for_recall(),
            memory=mem,
            store=_seeded_store(tmp_path),
            config={"recall": True, "min_score": 0.9},
        )
        block = await pm.recall("does the user prefer python")
        assert block is not None
        assert block.count("\n- ") == 1  # only the python fact (score 1.0)

    async def test_empty_store_returns_none(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S")
        store = _MemoryStore(filepath=str(tmp_path / "empty.json"))
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=_embed_client_for_recall(),
            memory=mem,
            store=store,
            config={"recall": True},
        )
        assert await pm.recall("anything") is None

    async def test_no_embedding_client_returns_none(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S")
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=mem,
            store=_seeded_store(tmp_path),
            config={"recall": True},
        )
        assert await pm.recall("anything") is None


class TestCoreBlockB:
    async def test_extraction_maintains_core_block(self, tmp_path):
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content='{"preferred_language": "python"}'))
        mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S")
        mem.add_user_message("I love python.")
        mem.add_assistant_message("Noted.")
        pm = ProactiveMemory(
            client=client,
            embedding_client=None,
            memory=mem,
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"extract": True, "core_block": True},
        )
        await pm.extract_and_store()
        cb = pm.get_core_block()
        assert cb is not None
        assert "preferred_language" in cb and "python" in cb

    def test_core_block_persists_across_instances(self, tmp_path):
        db = str(tmp_path / "p.db")
        mem1 = SQLiteMemory(db_path=db, session_id="S")
        mem1.set_meta("core_memory", '{"tz": "UTC"}')
        mem1.close()
        mem2 = SQLiteMemory(db_path=db, session_id="S")
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=mem2,
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"core_block": True},
        )
        cb = pm.get_core_block()
        assert cb is not None and "tz" in cb and "UTC" in cb

    async def test_proactive_block_combines_core_and_recall(self, tmp_path):
        from koboi.loop import AgentCore

        mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S")
        mem.set_meta("core_memory", '{"name": "alice"}')
        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=_embed_client_for_recall(),
            memory=mem,
            store=_seeded_store(tmp_path),
            config={"recall": True, "core_block": True},
        )
        core = AgentCore(client=MagicMock(), proactive_memory=pm)
        block = await core._proactive_block("does the user prefer python")
        assert "Core memory" in block and "name" in block
        assert "Relevant long-term memory" in block  # recall ran too


class TestConfigGatingAndHook:
    def test_default_off_no_proactive_memory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from koboi.facade import KoboiAgent

        a = KoboiAgent.from_dict(
            {
                "agent": {"name": "t", "system_prompt": "h", "max_iterations": 3, "mode": "chat"},
                "llm": {"provider": "openai", "model": "m", "api_key": "x", "base_url": "http://x"},
                "memory": {"backend": "sqlite", "db_path": str(tmp_path / "p.db")},
            }
        )
        assert a._core.proactive_memory is None  # opt-in: inert by default

    def test_enabled_constructs_coordinator(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from koboi.facade import KoboiAgent

        a = KoboiAgent.from_dict(
            {
                "agent": {"name": "t", "system_prompt": "h", "max_iterations": 3, "mode": "chat"},
                "llm": {"provider": "openai", "model": "m", "api_key": "x", "base_url": "http://x"},
                "memory": {
                    "backend": "sqlite",
                    "db_path": str(tmp_path / "p.db"),
                    "proactive": {"enabled": True, "extract": True, "recall": True, "core_block": True},
                },
            }
        )
        assert a._core.proactive_memory is not None
        assert a._core.proactive_memory.extract_enabled is True
        assert a._core.proactive_memory.recall_enabled is True

    def test_extraction_hook_only_session_end(self, tmp_path):
        from koboi.hooks.chain import HookEvent
        from koboi.hooks.proactive_extraction_hook import ProactiveExtractionHook

        pm = ProactiveMemory(
            client=MagicMock(),
            embedding_client=None,
            memory=SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S"),
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"extract": True},
        )
        h = ProactiveExtractionHook(pm)
        assert h.handles() == [HookEvent.SESSION_END]
        assert h.priority == 65


@pytest.mark.asyncio
class TestCriticalRegressions:
    async def test_recall_sees_newly_extracted_fact(self, tmp_path):
        # TG1: extract -> cache invalidation -> recall finds the new fact (no
        # manual seeding). Guards the extract->recall boundary.
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content='{"preferred_language": "python"}'))

        async def emb(text):
            return [1.0, 0.0, 0.0] if "python" in text.lower() else [0.0, 0.0, 0.0]

        client.get_embeddings = emb
        mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id="S")
        mem.add_user_message("I love python.")
        mem.add_assistant_message("Noted.")
        pm = ProactiveMemory(
            client=client,
            embedding_client=client,
            memory=mem,
            store=_MemoryStore(filepath=str(tmp_path / "m.json")),
            config={"extract": True, "recall": True},
        )
        assert await pm.recall("does the user prefer python") is None  # empty store
        await pm.extract_and_store()  # stores fact + clears caches
        block = await pm.recall("does the user prefer python")
        assert block is not None
        assert "preferred_language" in block

    async def test_real_turn_injection_is_ephemeral(self, tmp_path, monkeypatch):
        # TG2: a real turn injects the recall block into the LLM prompt AND does
        # not persist it as a conversation row (ephemerality).
        monkeypatch.chdir(tmp_path)
        from koboi.facade import KoboiAgent
        from tests.conftest import make_mock_response

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t", "system_prompt": "sys", "max_iterations": 3, "mode": "chat"},
                "llm": {"provider": "openai", "model": "m", "api_key": "x", "base_url": "http://x"},
                "memory": {
                    "backend": "sqlite",
                    "db_path": str(tmp_path / "p.db"),
                    "proactive": {"enabled": True, "recall": True},
                },
            }
        )
        pm = agent._core.proactive_memory
        assert pm is not None
        pm._store._data = {"preferred_language": "python"}  # seed a fact
        captured: dict = {}
        mock_client = MagicMock()

        async def fake_complete(messages, tools=None, **kwargs):
            captured["messages"] = messages
            return make_mock_response(content="ok")

        async def fake_embed(text):
            return [1.0, 0.0, 0.0] if "python" in text.lower() else [0.0, 0.0, 0.0]

        mock_client.complete = fake_complete
        mock_client.get_embeddings = fake_embed
        agent._core.client = mock_client
        pm._embedding_client = mock_client

        await agent.run("does the user prefer python")
        sent = captured.get("messages", [])
        sys_blob = " ".join(m.get("content", "") for m in sent if m.get("role") == "system")
        assert "Relevant long-term memory" in sys_blob  # injected into the prompt
        # ephemerality: the block is NOT a persisted conversation row
        persisted = " ".join(str(m.get("content", "")) for m in agent._core.memory.get_messages())
        assert "Relevant long-term memory" not in persisted
