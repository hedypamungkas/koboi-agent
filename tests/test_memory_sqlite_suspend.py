"""Tests for the suspend/resume primitives: wal_checkpoint + consistent_backup.

``consistent_backup`` uses the SQLite Online Backup API, which yields a self-consistent
file even while other connections write the live WAL DB. The concurrent-writer test is
load-bearing: it proves atomicity-independence as a unit test (no Cloudflare needed).
The failed-backup test pins C-1: a failed backup must leave no 0-byte destination (which
would pass integrity_check and silently wipe history on resume).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from koboi.memory_sqlite import SQLiteMemory, WalCheckpointResult


class TestWalCheckpoint:
    def test_returns_structured_result_after_writes(self, tmp_path):
        db = str(tmp_path / "t.db")
        mem = SQLiteMemory(db_path=db, session_id="s1")
        for i in range(200):
            mem.add_user_message(f"msg {i}")  # grows the WAL
        mem.close()
        res = SQLiteMemory.wal_checkpoint(db)
        assert isinstance(res, WalCheckpointResult)
        assert set(res.as_dict()) == {"ok", "busy", "log", "checkpointed"}
        assert isinstance(res.busy, int)
        assert isinstance(res.ok, bool)
        assert res.ok is True  # no concurrent reader -> clean checkpoint

    def test_invalid_mode_raises(self, tmp_path):
        # Fail-closed (S-1): an unknown mode raises instead of silently degrading to
        # TRUNCATE -- fail-soft belongs on the busy result, not on mode parsing.
        db = str(tmp_path / "t.db")
        SQLiteMemory(db_path=db, session_id="s1").close()
        with pytest.raises(ValueError):
            SQLiteMemory.wal_checkpoint(db, mode="BOGUS")  # not a valid mode

    def test_busy_surfaced_when_reader_blocks_truncate(self, tmp_path):
        # Pins the "surface, don't swallow" contract: a held read transaction blocks
        # TRUNCATE -> busy must be surfaced (!= 0) and ok must be False. Deterministic
        # (a held read snapshot always blocks TRUNCATE in WAL mode -- not timing-based).
        db = str(tmp_path / "t.db")
        mem = SQLiteMemory(db_path=db, session_id="s1")
        for i in range(50):
            mem.add_user_message(f"m{i}")
        reader = sqlite3.connect(db)
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM messages").fetchone()  # open a read txn
        try:
            res = SQLiteMemory.wal_checkpoint(db, mode="TRUNCATE")
            assert res.busy != 0, "expected busy!=0 with an open reader"
            assert res.ok is False
        finally:
            reader.execute("ROLLBACK")
            reader.close()
            mem.close()


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
        assert not (tmp_path / "snap.db.tmp").exists()  # temp published away
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
        first_commit = threading.Event()  # deterministic start gate (no sleep-flake)

        def writer():
            c = sqlite3.connect(db, timeout=30)
            c.execute("PRAGMA journal_mode=WAL")
            n = 0
            while not stop.is_set():
                n += 1
                c.execute("BEGIN")
                c.executemany("INSERT INTO counter(b) VALUES(?)", [(k,) for k in range(50)])
                c.execute("COMMIT")
                first_commit.set()  # >=1 batch committed before we snapshot
                if n % 3 == 0:
                    try:
                        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # force WAL->db page moves
                    except Exception:
                        pass
            c.close()

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        try:
            assert first_commit.wait(timeout=5), "writer never committed a batch"
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

    def test_failed_backup_cleans_up_temp_and_leaves_no_dest(self, tmp_path, monkeypatch):
        """C-1: a failed backup must leave NEITHER a 0-byte/partial destination NOR the
        temp file. A 0-byte SQLite file passes PRAGMA integrity_check (valid empty DB,
        no schema), and the resume-side coordinator restores *.suspend.db by file
        existence -- so a leftover would silently wipe history on resume."""
        db = str(tmp_path / "t.db")
        mem = SQLiteMemory(db_path=db, session_id="s1")
        mem.add_user_message("hi")
        mem.close()
        dest = str(tmp_path / "snap.db")
        tmp = dest + ".tmp"

        class _BoomSrc:
            # sqlite3.Connection is an immutable type and can't be monkeypatched, so we
            # inject a source whose backup() raises -- this is exactly what
            # consistent_backup calls next.
            def backup(self, target, **kwargs):
                raise sqlite3.OperationalError("simulated backup failure")

            def close(self):
                pass

        monkeypatch.setattr(SQLiteMemory, "_open_conn", lambda _p: _BoomSrc())
        with pytest.raises(sqlite3.OperationalError):
            SQLiteMemory.consistent_backup(db, dest)

        assert not Path(dest).exists(), "failed backup left a destination file"
        assert not Path(tmp).exists(), "failed backup left a .tmp file"

    def test_instance_wrappers(self, tmp_path):
        db = str(tmp_path / "t.db")
        mem = SQLiteMemory(db_path=db, session_id="s1")
        mem.add_user_message("hi")
        assert isinstance(mem.quiesce(), WalCheckpointResult)
        dest = str(tmp_path / "s.db")
        assert mem.backup_to(dest) > 0
        mem.close()
