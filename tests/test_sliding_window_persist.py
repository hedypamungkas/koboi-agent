"""Issue #4a: sliding_window summary persists across restart/resume."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from koboi.context.manager import SlidingWindowManager
from koboi.memory_sqlite import SQLiteMemory
from koboi.types import AgentResponse


def _seed_messages() -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"turn {i} " * 20})
        msgs.append({"role": "assistant", "content": f"reply {i} " * 20})
    return msgs


class TestSlidingWindowSummaryPersist:
    async def test_summary_persisted_and_hydrated(self, tmp_path):
        db = str(tmp_path / "sw.db")
        mem = SQLiteMemory(db_path=db, session_id="SW1")
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content="SUMMARY: key facts here"))

        mgr1 = SlidingWindowManager(client=client, keep_last=4)
        mgr1.last_actual_tokens = 100000  # force over budget -> compaction
        mgr1.meta_store = mem
        await mgr1.manage(_seed_messages(), max_tokens=100)

        assert mgr1._summary == "SUMMARY: key facts here"
        # Persisted to the session_meta table.
        assert mem.get_meta("sliding_window_summary") == "SUMMARY: key facts here"

        # Fresh manager on the SAME session hydrates the summary from DB.
        mem2 = SQLiteMemory(db_path=db, session_id="SW1")
        mgr2 = SlidingWindowManager(client=None, keep_last=4)  # no client -> no re-summarize
        mgr2.last_actual_tokens = 100000
        mgr2.meta_store = mem2
        out = await mgr2.manage(_seed_messages(), max_tokens=100)
        assert mgr2._summary == "SUMMARY: key facts here"  # loaded from DB
        blob = " ".join(m.get("content", "") for m in out)
        assert "SUMMARY: key facts here" in blob

    async def test_no_meta_store_stays_in_memory(self, tmp_path):
        # Without a meta_store, behavior is unchanged (summary in-memory only).
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content="ephemeral"))
        mgr = SlidingWindowManager(client=client, keep_last=4)
        mgr.last_actual_tokens = 100000
        # meta_store stays None
        await mgr.manage(_seed_messages(), max_tokens=100)
        assert mgr._summary == "ephemeral"

    async def test_get_set_meta_roundtrip(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "m.db"), session_id="S")
        assert mem.get_meta("absent") is None
        mem.set_meta("k", "v1")
        assert mem.get_meta("k") == "v1"
        mem.set_meta("k", "v2")  # upsert
        assert mem.get_meta("k") == "v2"
