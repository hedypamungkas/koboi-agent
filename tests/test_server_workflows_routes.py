"""Tests for the /v1/workflows REST surface + workflow_ref job validation (S5)."""

import httpx

from koboi.config import Config  # noqa: E402
from koboi.server import create_app  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402

BUNDLE = (
    "workflow:\n  name: w1\n  schema_version: '1.0'\n  description: test\n"
    "agent:\n  name: srv\n  system_prompt: hi\n"
    "llm:\n  provider: openai\n  model: gpt-4o-mini\n  api_key: ${OPENAI_API_KEY:}\n"
)
BAD_BUNDLE = "this: is: not [a valid workflow envelope"


def _config(**overrides) -> Config:
    cfg = {
        "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "base_url": "http://localhost:8080/v1",
        },
        "memory": {"backend": "in_memory"},
        "sandbox": {"backend": "restricted"},
        "server": {"auth_required": False},
    }
    cfg.update(overrides)
    return Config.from_dict(cfg, validate=True)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://testserver", transport=httpx.ASGITransport(app=app))


class TestWorkflowRoutes:
    async def test_create_list_get_delete(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hello")]),
            enable_cors=False,
        )
        async with _client(app) as c:
            r = await c.post("/v1/workflows", json={"name": "w1", "description": "d", "bundle": BUNDLE})
            assert r.status_code == 201, r.text
            assert r.json()["name"] == "w1"

            r = await c.get("/v1/workflows")
            assert r.status_code == 200
            assert "w1" in [w["name"] for w in r.json()["workflows"]]

            r = await c.get("/v1/workflows/w1")
            assert r.status_code == 200
            assert r.json()["description"] == "d"

            r = await c.delete("/v1/workflows/w1")
            assert r.status_code == 200

            r = await c.get("/v1/workflows/w1")
            assert r.status_code == 404

    async def test_invalid_bundle_rejected_400(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hello")]),
            enable_cors=False,
        )
        async with _client(app) as c:
            r = await c.post("/v1/workflows", json={"name": "bad", "bundle": BAD_BUNDLE})
            assert r.status_code == 400
            assert "invalid_workflow" in r.text

    async def test_job_unknown_workflow_ref_400(self):
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="hello")]),
            enable_cors=False,
        )
        async with _client(app) as c:
            r = await c.post("/v1/jobs", json={"message": "hi", "workflow_ref": "nope"})
            assert r.status_code == 400
            assert "unknown_workflow" in r.text

    async def test_auth_fail_closed_401(self):
        app = create_app(
            _config(server={"auth_required": True}),
            client_factory=lambda: MockClient([make_mock_response(content="x")]),
            api_keys=["secret"],
            enable_cors=False,
        )
        async with _client(app) as c:
            r = await c.get("/v1/workflows")  # no Authorization header
            assert r.status_code == 401

    async def test_owner_isolation(self):
        app = create_app(
            _config(server={"auth_required": True}),
            client_factory=lambda: MockClient([make_mock_response(content="x")]),
            api_keys=["keyA", "keyB"],
            enable_cors=False,
        )
        async with _client(app) as c:
            r = await c.post(
                "/v1/workflows",
                json={"name": "secret", "bundle": BUNDLE},
                headers={"Authorization": "Bearer keyA"},
            )
            assert r.status_code == 201
            # Owner B lists workflows -> does NOT see owner A's "secret".
            r = await c.get("/v1/workflows", headers={"Authorization": "Bearer keyB"})
            assert r.status_code == 200
            assert "secret" not in [w["name"] for w in r.json()["workflows"]]
            # Owner B GET the name -> 404 (no existence leak across tenants).
            r = await c.get("/v1/workflows/secret", headers={"Authorization": "Bearer keyB"})
            assert r.status_code == 404

    async def test_invalid_config_body_rejected_400(self):
        # Valid YAML + envelope, but an invalid llm.model (empty) -> Config.from_string
        # fails -> 400 at POST (not deferred to the first job run).
        app = create_app(
            _config(),
            client_factory=lambda: MockClient([make_mock_response(content="x")]),
            enable_cors=False,
        )
        bundle = (
            "workflow:\n  name: bad\n  schema_version: '1.0'\n"
            "agent:\n  name: x\n"
            "llm:\n  provider: openai\n  model: ''\n"
        )
        async with _client(app) as c:
            r = await c.post("/v1/workflows", json={"name": "bad", "bundle": bundle})
            assert r.status_code == 400
            assert "invalid_workflow" in r.text
