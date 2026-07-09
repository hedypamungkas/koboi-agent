"""tests/test_jobs_resume.py -- #5 server resume hole (rehydrate-and-continue).

resume_on_startup previously marked running jobs as failed (InterruptedByRestart).
Now it rehydrates-and-continues them via run_job(resume=True); pending jobs are
still requeued fresh. These tests verify the dispatch logic (resume flag + no
mark-failed) by faking run_job.
"""

from __future__ import annotations

import asyncio

from koboi.server import jobs


class _FakeStore:
    def __init__(self, by_status):
        self._jobs = by_status
        self.updates: list[tuple] = []

    def list_by_status(self, status):
        return list(self._jobs.get(status, []))

    def update_status(self, job_id, status, **kw):
        self.updates.append((job_id, status, kw))


class _FakeRegistry:
    def __init__(self):
        self.registered: list[str] = []
        self.tasks: list[asyncio.Task] = []

    def register(self, job_id, session_id, owner):
        self.registered.append(job_id)

    def set_running(self, job_id, task):
        self.tasks.append(task)


def _job(jid, sid):
    return {"job_id": jid, "session_id": sid, "owner": "o", "message": "m", "mode": None, "max_iterations": None}


async def test_resume_on_startup_resumes_running_and_requeues_pending(monkeypatch):
    store = _FakeStore({"running": [_job("j1", "s1")], "pending": [_job("j2", "s2")]})
    registry = _FakeRegistry()
    calls: list[tuple] = []

    async def fake_run_job(job_id, pool, reg, st, message, timeout=1800, mode=None, max_iterations=None, resume=False):
        calls.append((job_id, resume))
        return None

    monkeypatch.setattr(jobs, "run_job", fake_run_job)

    count = await jobs.resume_on_startup(store, object(), registry, timeout=10)
    await asyncio.gather(*registry.tasks)  # let the created tasks execute

    assert count == 2
    assert ("j1", True) in calls  # running -> resume=True
    assert ("j2", False) in calls  # pending -> fresh run
    # No job was marked failed (the old InterruptedByRestart behavior is gone).
    assert all(status != "failed" for _, status, _ in store.updates)


async def test_resume_on_startup_no_jobs_is_noop():
    store = _FakeStore({})
    registry = _FakeRegistry()
    count = await jobs.resume_on_startup(store, object(), registry, timeout=10)
    assert count == 0
    assert registry.tasks == []
