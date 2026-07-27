"""tests/test_server_mcp_attach_rce.py -- issue #91: runtime MCP stdio attach must not be RCE.

``POST /v1/sessions/{id}/mcp/servers`` used to spawn any caller-supplied stdio
``command`` + ``args`` as long as the command's basename was one of the
general-purpose interpreters in ``facade._MCP_DEFAULT_RUNNERS`` -- every one of
which executes arbitrary code handed to it in ``args``. These tests pin the
default-deny, operator-gated posture that replaced it.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.server import create_app  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402

KEY = "tenant-key"


def _config(attach: dict | None = None) -> Config:
    server: dict = {"auth_required": True}
    if attach is not None:
        server["mcp_runtime_attach"] = attach
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test",
                "base_url": "http://localhost:8080/v1",
            },
            "memory": {"backend": "in_memory"},
            "sandbox": {"backend": "restricted"},
            "server": server,
        },
        validate=True,
    )


def _app(attach: dict | None = None):
    factory = lambda: MockClient([make_mock_response(content="hello")])  # noqa: E731
    return create_app(_config(attach), client_factory=factory, enable_cors=False, api_keys=[KEY])


def _client(app):
    return httpx.AsyncClient(
        base_url="http://t",
        transport=ASGITransport(app=app),
        headers={"Authorization": f"Bearer {KEY}"},
    )


async def _new_session(c) -> str:
    r = await c.post("/v1/sessions")
    assert r.status_code in (200, 201), r.text
    return r.json()["session_id"]


class TestRuntimeStdioAttachRce:
    """Issue #91: an authenticated tenant must not get code execution on the host."""

    async def test_stdio_attach_does_not_execute_arbitrary_code(self, tmp_path):
        """RED-first proof: the payload must never run, not merely 4xx.

        The 4xx on the vulnerable code was cosmetic -- the payload executed at
        ``subprocess.Popen`` time, long before the MCP handshake failed.
        """
        proof = tmp_path / "pwned.txt"
        payload = f"import pathlib; pathlib.Path({str(proof)!r}).write_text('pwned')"
        app = _app()
        async with _client(app) as c:
            sid = await _new_session(c)
            r = await c.post(
                f"/v1/sessions/{sid}/mcp/servers",
                json={"transport": "stdio", "command": sys.executable, "args": ["-c", payload]},
            )
        assert not proof.exists(), f"RCE: attach payload executed and wrote {proof} (status {r.status_code})"
        assert r.status_code == 403, r.text
        assert r.json()["error"]["code"] == "stdio_attach_disabled"

    async def test_stdio_attach_denied_by_default(self):
        """No operator opt-in -> 403, and no process is ever spawned."""
        spawned: list[list[str]] = []

        app = _app()
        async with _client(app) as c:
            sid = await _new_session(c)
            with pytest.MonkeyPatch.context() as mp:
                import subprocess

                real_popen = subprocess.Popen

                def _record(cmd, *a, **k):
                    spawned.append(list(cmd) if isinstance(cmd, (list, tuple)) else [cmd])
                    return real_popen(cmd, *a, **k)

                mp.setattr(subprocess, "Popen", _record)
                r = await c.post(
                    f"/v1/sessions/{sid}/mcp/servers",
                    json={"transport": "stdio", "command": "python3", "args": []},
                )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "stdio_attach_disabled"
        assert spawned == []

    async def test_default_runners_are_not_an_implicit_allowlist(self):
        """Opting in with an EMPTY allowed_commands attaches nothing (fail-closed)."""
        app = _app({"allow_stdio": True, "allowed_commands": []})
        async with _client(app) as c:
            sid = await _new_session(c)
            r = await c.post(
                f"/v1/sessions/{sid}/mcp/servers",
                json={"transport": "stdio", "command": "npx", "args": ["some-server"]},
            )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "stdio_command_not_allowed"

    async def test_command_outside_operator_allowlist_refused(self):
        app = _app({"allow_stdio": True, "allowed_commands": ["/opt/mcp/weather-server"]})
        async with _client(app) as c:
            sid = await _new_session(c)
            r = await c.post(
                f"/v1/sessions/{sid}/mcp/servers",
                json={"transport": "stdio", "command": "python3", "args": []},
            )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "stdio_command_not_allowed"

    async def test_inline_code_arg_refused_even_for_allowlisted_command(self, tmp_path):
        """Defense in depth: an operator who allow-lists an interpreter is still safe."""
        proof = tmp_path / "pwned2.txt"
        payload = f"import pathlib; pathlib.Path({str(proof)!r}).write_text('pwned')"
        app = _app({"allow_stdio": True, "allowed_commands": [sys.executable]})
        async with _client(app) as c:
            sid = await _new_session(c)
            r = await c.post(
                f"/v1/sessions/{sid}/mcp/servers",
                json={"transport": "stdio", "command": sys.executable, "args": ["-c", payload]},
            )
        assert not proof.exists()
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "stdio_args_rejected"

    async def test_allowlisted_command_gets_past_the_gate(self):
        """Opt-in + allow-listed command -> the gate passes it through.

        ``mcp-weather-server`` is not a real binary, so the attach still fails --
        but with a *connect* error, proving the 403 gate was cleared.
        """
        app = _app({"allow_stdio": True, "allowed_commands": ["mcp-weather-server"]})
        async with _client(app) as c:
            sid = await _new_session(c)
            r = await c.post(
                f"/v1/sessions/{sid}/mcp/servers",
                json={"transport": "stdio", "command": "mcp-weather-server", "args": ["--port", "1"]},
            )
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "mcp_connect_failed"

    async def test_streamable_http_attach_is_unaffected(self):
        """Regression guard: the HTTP transport spawns no process and is never gated."""
        app = _app()
        async with _client(app) as c:
            sid = await _new_session(c)
            r = await c.post(
                f"/v1/sessions/{sid}/mcp/servers",
                json={"transport": "streamable-http", "url": "http://127.0.0.1:1/mcp"},
            )
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "mcp_connect_failed"


class TestCheckStdioAttach:
    """Unit tests for the validator itself (no HTTP)."""

    def _check(self, command="srv", args=(), allow_stdio=True, allowed=("srv",)):
        from koboi.server.mcp_registry import check_stdio_attach

        return check_stdio_attach(command, list(args), allow_stdio=allow_stdio, allowed_commands=list(allowed))

    def test_disabled_by_default(self):
        assert self._check(allow_stdio=False)[0] == "stdio_attach_disabled"

    def test_empty_command_rejected(self):
        assert self._check(command="", allowed=[""])[0] == "stdio_command_not_allowed"

    def test_exact_match_required(self):
        assert self._check(command="/usr/bin/srv", allowed=("srv",))[0] == "stdio_command_not_allowed"
        assert self._check(command="srv", allowed=("srv",)) is None

    @pytest.mark.parametrize(
        "arg",
        ["-c", "-e", "--eval", "--command", "--eval=1+1", "-", "-Sc", "-ic", "-We", "-EC", "--EVAL"],
    )
    def test_inline_code_args_rejected(self, arg):
        assert self._check(args=[arg])[0] == "stdio_args_rejected"

    @pytest.mark.parametrize("arg", ["--port", "8080", "server.js", "--config=x.json", "-p", "--verbose"])
    def test_benign_args_allowed(self, arg):
        assert self._check(args=[arg]) is None
