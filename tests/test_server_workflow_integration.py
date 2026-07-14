"""E2e: server integration upload→cache→capture→replay (Scenario B, mock).

Chains the full server lifecycle through the REST API:
  POST /v1/workflows (upload) → POST /v1/jobs (cache run) → POST /v1/jobs/{id}/capture
  → POST /v1/jobs (replay) → byte-identical result.

Uses a monkeypatched KoboiAgent.from_config_string to inject a MockClient (no API key).
The global-counter trick: job 1 gets "call-1" (cached), job 2 gets "call-2" (would be
live if replay failed). Asserting result2 == "call-1" (NOT "call-2") PROVES replay.
"""

import asyncio
import time

import httpx

from koboi.config import Config
from koboi.llm.cache import CachedClient
from koboi.server import create_app
from koboi.server.jobs import JobStore
from koboi.server.workflow_store import WorkflowStore
from tests.conftest import MockClient, make_mock_response

BUNDLE = (
    "workflow:\n  name: geo\n  schema_version: '1.0'\n"
    "agent:\n  name: geo\n  system_prompt: 'You answer concisely.'\n"
    "llm:\n  provider: openai\n  model: mock-model\n  api_key: test\n"
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


async def _poll_job(c, job_id, expected="completed", timeout=10.0):
    """Poll GET /v1/jobs/{id} until the status matches (or the job fails)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await c.get(f"/v1/jobs/{job_id}")
        body = r.json()
        status = body.get("status")
        if status == expected:
            return body
        if status in ("failed", "timed_out", "cancelled"):
            raise AssertionError(f"job {job_id} ended as {status}: {body.get('error')}")
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} timed out polling for {expected}")


def _patch_from_config_string(monkeypatch):
    """Monkeypatch KoboiAgent.from_config_string to inject a MockClient inner.

    Uses a global counter: first call → MockClient("call-1"), second → "call-2", etc.
    If replay works, the second job's result is "call-1" (from cache), NOT "call-2".
    """
    from koboi.facade import KoboiAgent

    _orig = KoboiAgent.from_config_string
    _counter = [0]

    @classmethod
    def _patched(cls, yaml_string, verbose=False, **kwargs):
        agent = _orig.__func__(cls, yaml_string, verbose=verbose, **kwargs)
        _counter[0] += 1
        mock = MockClient([make_mock_response(f"call-{_counter[0]}")])
        client = agent._core.client if agent._core else getattr(agent._orchestrator, "client", None)
        if isinstance(client, CachedClient):
            client._inner = mock  # swap inner, keep the cache wrapper + cache_dir
        elif agent._core:
            agent._core.client = mock
        return agent

    monkeypatch.setattr(KoboiAgent, "from_config_string", _patched)


class TestServerWorkflowLifecycle:
    """B1: upload → cache run → capture → replay → byte-identical."""

    async def test_upload_cache_run_capture_replay_byte_identical(self, tmp_path, monkeypatch):
        _patch_from_config_string(monkeypatch)
        app, js, ws = _app_with_stores(tmp_path)

        async with _client(app) as c:
            # 1. UPLOAD workflow bundle
            r = await c.post("/v1/workflows", json={"name": "geo", "bundle": BUNDLE})
            assert r.status_code == 201

            # 2. SUBMIT cache-mode job
            r = await c.post(
                "/v1/jobs",
                json={"workflow_ref": "geo", "replay_mode": "cache", "message": "capital of France?"},
            )
            assert r.status_code == 202
            job1_id = r.json()["job_id"]

            # 3. WAIT for completion
            job1 = await _poll_job(c, job1_id, "completed")
            result1 = job1["result"]["content"]
            assert result1 == "call-1"  # the mock's first response

            # 4. CAPTURE the completed job (freeze cache → new bundle + sidecar)
            r = await c.post(f"/v1/jobs/{job1_id}/capture", json={"name": "geo-cap", "with_cache": True})
            assert r.status_code == 201
            assert r.json()["cache_entries"] > 0

            # 5. RE-RUN the captured bundle (replay → hydrate sidecar → all hits)
            r = await c.post(
                "/v1/jobs",
                json={
                    "workflow_ref": "geo-cap",
                    "replay_mode": "cache",
                    "message": "capital of France?",
                },
            )
            assert r.status_code == 202
            job2_id = r.json()["job_id"]

            # 6. WAIT + assert byte-identical
            job2 = await _poll_job(c, job2_id, "completed")
            result2 = job2["result"]["content"]
            assert result2 == "call-1"  # from cache, NOT "call-2" (the live mock would return)

    async def test_capture_plain_job_with_source(self, tmp_path):
        app, js, ws = _app_with_stores(tmp_path)
        app.state.config_source_text = "agent:\n  name: srv\nllm:\n  provider: openai\n  model: m\n  api_key: ${K:}\n"

        async with _client(app) as c:
            # submit + complete a plain (non-workflow_ref) job
            r = await c.post("/v1/jobs", json={"message": "hi"})
            assert r.status_code == 202
            job_id = r.json()["job_id"]
            await _poll_job(c, job_id, "completed")

            # capture the plain job (bundle from config_source_text)
            r = await c.post(f"/v1/jobs/{job_id}/capture", json={"name": "plain-cap"})
            assert r.status_code == 201

            # verify the bundle preserves ${VAR} templates
            bundle = ws.get("plain-cap", "dev")["bundle_yaml"]
            assert "${K:}" in bundle
