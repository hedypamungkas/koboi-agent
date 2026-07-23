"""Tests for the suspend/resume primitives: wal_checkpoint + consistent_backup.

``consistent_backup`` uses the SQLite Online Backup API, which yields a self-consistent
file even while other connections write the live WAL DB. The concurrent-writer test is
load-bearing: it proves atomicity-independence as a unit test (no Cloudflare needed).
"""

from __future__ import annotations

import sqlite3
import threading
import time

from koboi.memory_sqlite import SQLiteMemory


class TestWalCheckpoint:
    def test_returns_structured_result_after_writes(self, tmp_path):
        db = str(tmp_path / "t.db")
        mem = SQLiteMemory(db_path=db, session_id="s1")
        for i in range(200):
            mem.add_user_message(f"msg {i}")  # grows the WAL
        mem.close()
        res = SQLiteMemory.wal_checkpoint(db)
        assert set(res) == {"ok", "busy", "log", "checkpointed"}
        assert isinstance(res["busy"], int)
        assert isinstance(res["ok"], bool)

    def test_invalid_mode_falls_back_to_truncate(self, tmp_path):
        db = str(tmp_path / "t.db")
        SQLiteMemory(db_path=db, session_id="s1").close()
        res = SQLiteMemory.wal_checkpoint(db, mode="BOGUS")  # not a valid mode -> TRUNCATE
        assert res["ok"] is True


class TestConsistentBackup:
    def test_backup_is_standalone_and_consistent(self, tmp_path):
        db = str(tmp_path / "t.db")
        mem = SQLiteMemory(db_path=db, session_id="s1")
        for i in range(500):
            mem.add_user_message(f"m{i}")
        mem.close()
        dest = str(tmp_path / "snap.db")
        size = SQLiteMemory.consistent_backup(db, dest)
        assert size > 0
        assert not (tmp_path / "snap.db-wal").exists()  # standalone: no -wal sidecar
        v = sqlite3.connect(dest)
        assert v.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert v.execute("SELECT count(*) FROM messages").fetchone()[0] == 500
        v.close()

    def test_backup_consistent_under_concurrent_writer(self, tmp_path):
        """LOAD-BEARING: backup() yields a consistent file even while another connection
        writes the live WAL DB (the atomicity-independence guarantee the suspend route
        relies on). A torn snapshot would fail integrity_check or capture a partial batch."""
        db = str(tmp_path / "t.db")
        setup = sqlite3.connect(db)
        setup.execute("PRAGMA journal_mode=WAL")
        setup.execute("CREATE TABLE counter(n INTEGER PRIMARY KEY AUTOINCREMENT, b INTEGER)")
        setup.commit()
        setup.close()

        stop = threading.Event()

        def writer():
            c = sqlite3.connect(db, timeout=30)
            c.execute("PRAGMA journal_mode=WAL")
            n = 0
            while not stop.is_set():
                n += 1
                c.execute("BEGIN")
                c.executemany("INSERT INTO counter(b) VALUES(?)", [(k,) for k in range(50)])
                c.execute("COMMIT")
                if n % 3 == 0:
                    try:
                        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # force WAL->db page moves
                    except Exception:
                        pass
            c.close()

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        try:
            time.sleep(0.2)  # let writes accumulate + cross checkpoints
            dest = str(tmp_path / "snap.db")
            SQLiteMemory.consistent_backup(db, dest)
        finally:
            stop.set()
            t.join(timeout=5)

        v = sqlite3.connect(dest)
        integ = v.execute("PRAGMA integrity_check").fetchone()[0]
        rows = v.execute("SELECT count(*) FROM counter").fetchone()[0]
        v.close()
        assert integ == "ok", f"torn snapshot! integrity_check={integ!r}"
        assert rows > 0
        # crash-consistent recovery -> only fully-committed 50-row batches present
        assert rows % 50 == 0, f"partial transaction captured: {rows} not a multiple of 50"

    def test_instance_wrappers(self, tmp_path):
        db = str(tmp_path / "t.db")
        mem = SQLiteMemory(db_path=db, session_id="s1")
        mem.add_user_message("hi")
        assert "ok" in mem.quiesce()
        dest = str(tmp_path / "s.db")
        assert mem.backup_to(dest) > 0
        mem.close()
