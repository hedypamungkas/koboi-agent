"""Tests for POST /v1/jobs/{id}/capture (v2 step 10): server capture-from-run."""

import httpx

from koboi.config import Config
from koboi.llm.cache import ResponseCache
from koboi.server import create_app
from koboi.server.jobs import JobStore
from koboi.server.workflow_store import WorkflowStore
from koboi.types import AgentResponse, TokenUsage
from tests.conftest import MockClient, make_mock_response

BUNDLE = (
    "workflow:\n  name: w1\n  schema_version: '1.0'\n"
    "agent:\n  name: srv\n  system_prompt: hi\n"
    "llm:\n  provider: openai\n  model: m\n  api_key: ${OPENAI_API_KEY:}\n"
    "sandbox:\n  backend: restricted\n"
)


def _config(**ov):
    cfg = {
        "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
        "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x/v1"},
        "memory": {"backend": "in_memory"},
        "sandbox": {"backend": "restricted"},
        "server": {"auth_required": False},
    }
    cfg.update(ov)
    return Config.from_dict(cfg, validate=True)


def _client(app):
    return httpx.AsyncClient(base_url="http://testserver", transport=httpx.ASGITransport(app=app))


def _app_with_stores(tmp_path):
    db = str(tmp_path / "t.db")
    js = JobStore(db_path=db)
    ws = WorkflowStore(db_path=db)
    app = create_app(
        _config(),
        client_factory=lambda: MockClient([make_mock_response(content="x")]),
        enable_cors=False,
        job_store=js,
        workflow_store=ws,
    )
    return app, js, ws


class TestServerCapture:
    async def test_capture_non_completed_409(self, tmp_path):
        app, js, ws = _app_with_stores(tmp_path)
        async with _client(app) as c:
            await c.post("/v1/workflows", json={"name": "w1", "bundle": BUNDLE})
        js.insert("job1", "sess1", "dev", "hi", workflow_ref="w1")
        async with _client(app) as c:
            r = await c.post("/v1/jobs/job1/capture", json={"name": "cap"})
            assert r.status_code == 409  # not completed

    async def test_capture_plain_job_400(self, tmp_path):
        app, js, ws = _app_with_stores(tmp_path)
        js.insert("job2", "sess2", "dev", "hi")  # no workflow_ref
        js.update_status("job2", "completed")
        async with _client(app) as c:
            r = await c.post("/v1/jobs/job2/capture", json={"name": "cap"})
            assert r.status_code == 400  # plain job

    async def test_capture_no_cache_to_freeze_400(self, tmp_path):
        app, js, ws = _app_with_stores(tmp_path)
        async with _client(app) as c:
            await c.post("/v1/workflows", json={"name": "w1", "bundle": BUNDLE})
        js.insert("job3", "sess3", "dev", "hi", workflow_ref="w1")
        js.update_status("job3", "completed")  # completed but no cache_dir
        async with _client(app) as c:
            r = await c.post("/v1/jobs/job3/capture", json={"name": "cap", "with_cache": True})
            assert r.status_code == 400  # no cache to freeze

    async def test_capture_workflow_ref_with_cache_201(self, tmp_path):
        app, js, ws = _app_with_stores(tmp_path)
        async with _client(app) as c:
            await c.post("/v1/workflows", json={"name": "w1", "bundle": BUNDLE})
        js.insert("job4", "sess4", "dev", "hi", workflow_ref="w1", replay_mode="cache")
        js.update_status("job4", "completed")
        cache_dir = str(tmp_path / "jobcache")
        js.set_cache_dir("job4", cache_dir)
        ResponseCache(cache_dir).put("ab" * 32, AgentResponse(content="ans", usage=TokenUsage()), model="m")
        async with _client(app) as c:
            r = await c.post("/v1/jobs/job4/capture", json={"name": "cap", "with_cache": True})
            assert r.status_code == 201, r.text
            assert r.json()["cache_entries"] == 1
        # the sidecar landed in workflows_cache
        sc = ws.get_sidecar("dev", "cap")
        assert sc is not None and len(sc) == 1

    async def test_capture_other_owner_404(self, tmp_path):
        app = create_app(
            _config(server={"auth_required": True}),
            client_factory=lambda: MockClient([make_mock_response(content="x")]),
            api_keys=["keyA", "keyB"],
            enable_cors=False,
        )
        # owner A submits + completes a workflow_ref job (via direct store is hard with auth;
        # instead insert via the app's job_store on app.state)
        async with _client(app) as c:
            await c.post(
                "/v1/workflows", json={"name": "w1", "bundle": BUNDLE}, headers={"Authorization": "Bearer keyA"}
            )
        js = app.state.job_store
        js.insert("jobA", "sessA", "<ownerA>", "hi", workflow_ref="w1")
        js.update_status("jobA", "completed")
        # owner B tries to capture owner A's job -> 403 (not the owner)
        async with _client(app) as c:
            r = await c.post("/v1/jobs/jobA/capture", json={"name": "cap"}, headers={"Authorization": "Bearer keyB"})
            assert r.status_code == 403

    async def test_capture_plain_job_with_source_201(self, tmp_path):
        app, js, ws = _app_with_stores(tmp_path)
        app.state.config_source_text = (
            "agent:\n  name: srv\nllm:\n  provider: openai\n  model: m\n  api_key: ${OPENAI_API_KEY:}\n"
        )
        js.insert("job5", "sess5", "dev", "hi")  # plain job (no workflow_ref)
        js.update_status("job5", "completed")
        async with _client(app) as c:
            r = await c.post("/v1/jobs/job5/capture", json={"name": "pca"})
            assert r.status_code == 201, r.text
        # the bundle preserves the ${VAR} template (not resolved secrets)
        bundle = ws.get("pca", "dev")["bundle_yaml"]
        assert "${OPENAI_API_KEY:}" in bundle

    async def test_capture_plain_job_with_cache_400(self, tmp_path):
        app, js, ws = _app_with_stores(tmp_path)
        app.state.config_source_text = "agent:\n  name: srv\nllm:\n  model: m\n"
        js.insert("job6", "sess6", "dev", "hi")
        js.update_status("job6", "completed")
        async with _client(app) as c:
            r = await c.post("/v1/jobs/job6/capture", json={"name": "x", "with_cache": True})
            assert r.status_code == 400  # plain jobs can't isolate a run cache

    async def test_submit_plain_cache_job_accepted(self, tmp_path):
        # v3 #4-a: a plain (non-workflow_ref) job may request replay_mode=cache
        # (was 400 replay_mode_requires_workflow_ref; now accepted -- runs via the
        # fresh per-job build path). The async run needs a real LLM (not asserted
        # here); this verifies the submit gate lift + replay_mode persistence.
        app, js, ws = _app_with_stores(tmp_path)
        async with _client(app) as c:
            r = await c.post("/v1/jobs", json={"message": "hi", "replay_mode": "cache"})
            assert r.status_code == 202
            job_id = r.json()["job_id"]
        row = js.get(job_id)
        assert row["replay_mode"] == "cache"
        assert row["workflow_ref"] is None
