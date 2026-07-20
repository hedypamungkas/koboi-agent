"""Tests for repo-scoped conventions memory (memory.proactive.repo_scoped, Wave 4)."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

from koboi.memory_sqlite import SQLiteMemory
from koboi.proactive_memory import ProactiveMemory
from koboi.tools.builtin.memory import _MemoryStore
from koboi.types import AgentResponse


def _client(facts: dict) -> MagicMock:
    client = MagicMock()
    client.complete = AsyncMock(return_value=AgentResponse(content=json.dumps(facts)))
    client.get_embeddings = AsyncMock(return_value=[1.0, 0.0])
    return client


def _memory(tmp_path, session_id="S") -> SQLiteMemory:
    mem = SQLiteMemory(db_path=str(tmp_path / "p.db"), session_id=session_id)
    mem.add_user_message("Always run ruff before committing.")
    mem.add_assistant_message("Noted -- I'll run ruff before every commit.")
    return mem


class TestFacadeWorkdirAnchoring:
    def test_kv_file_anchored_to_workdir(self, tmp_path):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {
                    "backend": "in_memory",
                    "proactive": {"enabled": True, "core_block": True, "repo_scoped": True},
                },
                "sandbox": {"backend": "passthrough", "workdir": str(tmp_path)},
            }
        )
        pm = agent._core.proactive_memory
        assert pm is not None
        expected = os.path.join(os.path.realpath(str(tmp_path)), ".koboi", "memory.json")
        assert pm._store.filepath == expected

    def test_tool_dep_shares_same_store_instance(self, tmp_path):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory", "proactive": {"enabled": True, "repo_scoped": True}},
                "sandbox": {"backend": "passthrough", "workdir": str(tmp_path)},
            }
        )
        pm = agent._core.proactive_memory
        dep_store = agent._core.tools.get_dep("memory_store_ref")
        assert dep_store is pm._store

    def test_unresolvable_workdir_disables_not_falls_back(self, tmp_path):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory", "proactive": {"enabled": True, "repo_scoped": True}},
                # no sandbox.workdir configured -- passthrough's own workdir is None
            }
        )
        assert agent._core.proactive_memory is None

    def test_repo_scoped_false_keeps_default_file(self, tmp_path):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory", "proactive": {"enabled": True}},
                "sandbox": {"backend": "passthrough", "workdir": str(tmp_path)},
            }
        )
        pm = agent._core.proactive_memory
        assert pm is not None
        assert ".koboi" not in pm._store.filepath


class TestCoreBlockCrossSessionPersistence:
    async def test_core_block_survives_across_separate_instances(self, tmp_path):
        store_path = str(tmp_path / "memory.json")
        client = _client({"testing_convention": "use pytest, not unittest"})

        # "Session 1": extract + merge into the repo-scoped core block.
        mem1 = _memory(tmp_path, session_id="S1")
        store1 = _MemoryStore(filepath=store_path)
        pm1 = ProactiveMemory(
            client=client,
            embedding_client=client,
            memory=mem1,
            store=store1,
            config={"extract": True, "core_block": True},
            repo_scoped=True,
        )
        n = await pm1.extract_and_store()
        assert n == 1
        assert pm1.get_core_block() is not None

        # "Session 2": a fresh ProactiveMemory + fresh _MemoryStore re-reading the
        # SAME file -- the core block must still be there (the whole point of
        # repo-scoping: conventions persist across sessions on the same repo).
        mem2 = _memory(tmp_path, session_id="S2")
        store2 = _MemoryStore(filepath=store_path)
        pm2 = ProactiveMemory(
            client=client,
            embedding_client=client,
            memory=mem2,
            store=store2,
            config={"extract": True, "core_block": True},
            repo_scoped=True,
        )
        block = pm2.get_core_block()
        assert block is not None
        assert "testing_convention" in block
        assert "use pytest, not unittest" in block

    async def test_non_repo_scoped_core_block_is_per_session(self, tmp_path):
        store_path = str(tmp_path / "memory.json")
        client = _client({"fact": "value"})

        mem1 = _memory(tmp_path, session_id="S1")
        pm1 = ProactiveMemory(
            client=client,
            embedding_client=client,
            memory=mem1,
            store=_MemoryStore(filepath=store_path),
            config={"extract": True, "core_block": True},
            repo_scoped=False,
        )
        await pm1.extract_and_store()
        assert pm1.get_core_block() is not None

        # A different session_id's session_meta never sees session 1's core block.
        mem2 = _memory(tmp_path, session_id="S2")
        pm2 = ProactiveMemory(
            client=client,
            embedding_client=client,
            memory=mem2,
            store=_MemoryStore(filepath=store_path),
            config={"extract": True, "core_block": True},
            repo_scoped=False,
        )
        assert pm2.get_core_block() is None


class TestCoreBlockNeverLeaksAsAFact:
    async def test_core_block_excluded_from_recall(self, tmp_path):
        client = _client({"convention_one": "value one"})
        mem = _memory(tmp_path)
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        pm = ProactiveMemory(
            client=client,
            embedding_client=client,
            memory=mem,
            store=store,
            config={"extract": True, "recall": True, "core_block": True},
            repo_scoped=True,
        )
        await pm.extract_and_store()
        assert pm.get_core_block() is not None
        result = await pm.recall("anything")
        assert result is None or "__core_memory__" not in result
        assert "convention_one" in store._data  # the real fact IS stored
        assert "__core_memory__" in store._data  # the core block IS stored (reserved key)

    def test_core_block_excluded_from_memory_recall_default_listing(self, tmp_path):
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        store.store("__core_memory__", '{"x": "y"}')
        store.store("visible_fact", "hello")
        listing = store.recall()
        assert "visible_fact" in listing
        assert "__core_memory__" not in listing

    def test_explicit_key_lookup_of_reserved_key_still_works(self, tmp_path):
        store = _MemoryStore(filepath=str(tmp_path / "m.json"))
        store.store("__core_memory__", '{"x": "y"}')
        result = store.recall(key="__core_memory__")
        assert "x" in result
