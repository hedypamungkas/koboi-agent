"""Issue #4b: opt-in per-session message retention cap (oldest pruned)."""

from __future__ import annotations

from koboi.memory_sqlite import SQLiteMemory


class TestRetentionCap:
    def test_cap_prunes_oldest_incrementally(self, tmp_path):
        db = str(tmp_path / "r.db")
        mem = SQLiteMemory(db_path=db, session_id="S", retention_cap=5)
        for i in range(10):
            mem.add_user_message(f"msg {i}")

        # in-memory and DB both capped at 5, in lockstep
        assert len(mem._messages) == 5
        rows = mem._ensure_conn().execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='S'"
        ).fetchone()[0]
        assert rows == 5
        contents = [m["content"] for m in mem.get_messages()]
        assert "msg 0" not in contents  # oldest dropped
        assert "msg 9" in contents  # newest kept
        assert "msg 5" in contents  # boundary kept

    def test_no_cap_keeps_all(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "r2.db"), session_id="S")
        for i in range(10):
            mem.add_user_message(f"msg {i}")
        assert len(mem._messages) == 10

    def test_cap_applies_on_load_of_existing_session(self, tmp_path):
        # Pre-populate without a cap, then reopen with a cap -> prunes to cap.
        db = str(tmp_path / "r3.db")
        mem1 = SQLiteMemory(db_path=db, session_id="S")
        for i in range(10):
            mem1.add_user_message(f"msg {i}")
        mem1.close()

        mem2 = SQLiteMemory(db_path=db, session_id="S", retention_cap=4)
        assert len(mem2._messages) == 4
        contents = [m["content"] for m in mem2.get_messages()]
        assert "msg 9" in contents and "msg 0" not in contents

    def test_in_memory_and_db_stay_in_lockstep(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "r4.db"), session_id="S", retention_cap=3)
        for i in range(6):
            mem.add_assistant_message(f"a{i}")
        # get_messages() should reflect exactly the retained set
        assert len(mem.get_messages()) == 3
