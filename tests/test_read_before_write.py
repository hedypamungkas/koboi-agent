"""Tests for read-before-write tracking + reset hook (P3b)."""

from __future__ import annotations

import os

import pytest

from koboi.tools.builtin.filesystem import (
    delete_file,
    get_read_paths,
    read_file,
    reset_read_before_write,
    write_file,
)
from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.read_before_write_reset_hook import ReadBeforeWriteResetHook


@pytest.fixture(autouse=True)
def _clean_tracker():
    """Isolation: the module-global _read_paths persists across tests in a process."""
    reset_read_before_write()
    yield
    reset_read_before_write()


class TestReadBeforeWriteTracker:
    def test_read_file_records_path(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hi")
        read_file(str(f))
        assert os.path.realpath(str(f)) in get_read_paths()

    def test_write_after_read_has_no_note(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hi")
        read_file(str(f))
        result = write_file(str(f), "bye")
        assert "Note:" not in result
        assert "Successfully wrote" in result

    def test_write_without_read_warns(self, tmp_path):
        f = tmp_path / "b.txt"
        result = write_file(str(f), "bye")
        assert "without having read it first" in result
        assert "Successfully wrote" in result  # advisory only, never blocks

    def test_delete_without_read_warns(self, tmp_path):
        f = tmp_path / "c.txt"
        f.write_text("x")
        result = delete_file(str(f))
        assert "without having read it first" in result

    def test_delete_after_read_no_note(self, tmp_path):
        f = tmp_path / "d.txt"
        f.write_text("x")
        read_file(str(f))
        result = delete_file(str(f))
        assert "Note:" not in result

    def test_warning_never_blocks_write(self, tmp_path):
        f = tmp_path / "e.txt"
        result = write_file(str(f), "data")
        assert result.startswith("Successfully wrote")
        assert f.read_text() == "data"

    def test_reset_clears_tracker(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hi")
        read_file(str(f))
        assert get_read_paths()
        reset_read_before_write()
        assert get_read_paths() == set()


class TestPerSessionReadPaths:
    """M6: per-session read-before-write tracking (isolated via ToolState)."""

    def test_read_populates_only_own_session(self, tmp_path):
        from koboi.tools.state import ToolState

        ts1, ts2 = ToolState(), ToolState()
        f = tmp_path / "a.txt"
        f.write_text("hi")
        read_file(str(f), _deps={"tool_state": ts1})
        assert os.path.realpath(str(f)) in ts1.read_paths
        assert ts2.read_paths == set()  # isolated

    def test_write_uses_per_session_state(self, tmp_path):
        # session 1 reads; session 2 writes the same path -> session 2 still gets
        # the "not read first" note (no cross-session leakage).
        from koboi.tools.state import ToolState

        f = tmp_path / "a.txt"
        f.write_text("hi")
        read_file(str(f), _deps={"tool_state": ToolState()})  # session 1
        result = write_file(str(f), "bye", _deps={"tool_state": ToolState()})  # session 2
        assert "without having read it first" in result

    def test_write_after_read_same_session_no_note(self, tmp_path):
        from koboi.tools.state import ToolState

        ts = ToolState()
        f = tmp_path / "a.txt"
        f.write_text("hi")
        read_file(str(f), _deps={"tool_state": ts})
        result = write_file(str(f), "bye", _deps={"tool_state": ts})
        assert "Note:" not in result

    def test_reset_clears_only_given_session(self):
        from koboi.tools.state import ToolState

        ts1, ts2 = ToolState(), ToolState()
        ts1.read_paths.add("/a")
        ts2.read_paths.add("/b")
        reset_read_before_write(ts1)
        assert ts1.read_paths == set()
        assert ts2.read_paths == {"/b"}

    def test_no_deps_falls_back_to_global(self, tmp_path):
        # back-compat: no _deps -> module global (existing behavior preserved).
        f = tmp_path / "a.txt"
        f.write_text("hi")
        read_file(str(f))
        assert os.path.realpath(str(f)) in get_read_paths()


class TestReadBeforeWriteResetHook:
    def test_priority_is_44(self):
        assert ReadBeforeWriteResetHook().priority == 44

    def test_handles_session_start_and_post_compact(self):
        events = set(ReadBeforeWriteResetHook().handles())
        assert events == {HookEvent.SESSION_START, HookEvent.POST_COMPACT}

    def _populate(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hi")
        read_file(str(f))
        assert get_read_paths(), "precondition: tracker populated"

    async def test_resets_on_session_start(self, tmp_path):
        self._populate(tmp_path)
        await ReadBeforeWriteResetHook().execute(HookContext(event=HookEvent.SESSION_START))
        assert get_read_paths() == set()

    async def test_resets_on_real_compaction(self, tmp_path):
        self._populate(tmp_path)
        ctx = HookContext(event=HookEvent.POST_COMPACT, metadata={"compacted": True})
        await ReadBeforeWriteResetHook().execute(ctx)
        assert get_read_paths() == set()

    async def test_does_not_reset_when_no_compaction(self, tmp_path):
        """Key regression: POST_COMPACT with no real trim must preserve tracking."""
        self._populate(tmp_path)
        before = get_read_paths()
        ctx = HookContext(event=HookEvent.POST_COMPACT, metadata={"compacted": False})
        await ReadBeforeWriteResetHook().execute(ctx)
        assert get_read_paths() == before

    async def test_does_not_reset_when_metadata_absent(self, tmp_path):
        """Defensive: an older loop / no ContextManager -> no metadata -> keep tracking."""
        self._populate(tmp_path)
        before = get_read_paths()
        await ReadBeforeWriteResetHook().execute(HookContext(event=HookEvent.POST_COMPACT))
        assert get_read_paths() == before
