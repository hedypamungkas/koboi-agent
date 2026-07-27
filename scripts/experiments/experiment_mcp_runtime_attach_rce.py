"""experiment_mcp_runtime_attach_rce.py -- issue #91 harness: runtime MCP attach RCE.

Drives the REAL FastAPI app (``koboi.server.create_app``) over an in-process ASGI
transport as an ordinary authenticated tenant and tries to get code execution via
``POST /v1/sessions/{id}/mcp/servers``, which spawns a caller-supplied stdio
``command`` + ``args``.

Before the fix the only guard was ``facade._MCP_DEFAULT_RUNNERS`` -- a BASENAME
allow-list of general-purpose interpreters, every one of which executes whatever
is handed to it in ``args``. The route's 400 was cosmetic: the payload already ran
at ``subprocess.Popen`` time, before the MCP handshake failed. So CHECK 1 proves
execution by looking for a file the payload writes, not by reading a status code.

No network. Exits non-zero if the bug is present (or a gate regressed).
Run:  PYTHONPATH=. .venv/bin/python scripts/experiments/experiment_mcp_runtime_attach_rce.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import httpx
    from httpx import ASGITransport
except ImportError:  # pragma: no cover - harness only
    print("SKIP: needs the [api] extra (fastapi/httpx). pip install -e '.[api]'")
    sys.exit(0)

from koboi.config import Config
from koboi.server import create_app

KEY = "tenant-key"

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, evidence: str) -> None:
    results.append((name, passed, evidence))
    print(f"CHECK {name} -> {'PASS' if passed else 'FAIL'}: {evidence}")


class _StubClient:
    """Minimal LLM client -- create_app needs a factory, no call is ever made."""

    async def complete(self, *a, **k):  # pragma: no cover - never invoked
        raise RuntimeError("no LLM calls in this harness")

    async def close(self):
        return None


def build_app(attach: dict | None):
    server: dict = {"auth_required": True}
    if attach is not None:
        server["mcp_runtime_attach"] = attach
    config = Config.from_dict(
        {
            "agent": {"name": "rce-harness", "system_prompt": "h", "max_iterations": 1},
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test",
                "base_url": "http://127.0.0.1:1/v1",
            },
            "memory": {"backend": "in_memory"},
            "sandbox": {"backend": "restricted"},
            "server": server,
        },
        validate=True,
    )
    return create_app(config, client_factory=_StubClient, enable_cors=False, api_keys=[KEY])


class _Crashed:
    """Stand-in for a route that RAISED instead of returning a response.

    Pre-fix this happens on a spawned interpreter: its junk stdout breaks the MCP
    handshake with an uncaught error. A crash is not a gate, so every check treats
    it as a failure -- but the harness keeps running and prints every CHECK line.
    """

    status_code = 0

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def json(self) -> dict:
        return {"error": {"code": f"unhandled:{type(self.exc).__name__}"}}


async def _attach(app, body: dict):
    """Create a session as an authenticated tenant, then POST an attach body."""
    transport = ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(base_url="http://harness", transport=transport, headers=headers) as c:
        sid = (await c.post("/v1/sessions")).json()["session_id"]
        try:
            return await c.post(f"/v1/sessions/{sid}/mcp/servers", json=body)
        except Exception as exc:  # noqa: BLE001 - a raising route is itself a finding
            return _Crashed(exc)


async def attach_stdio(app, command: str, args: list[str]):
    return await _attach(app, {"transport": "stdio", "command": command, "args": args})


async def attach_http(app, url: str):
    return await _attach(app, {"transport": "streamable-http", "url": url})


def _code(resp) -> str:
    try:
        return str(resp.json().get("error", {}).get("code", ""))
    except Exception:  # noqa: BLE001
        return ""


async def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="koboi-rce-"))
    try:
        # --- CHECK 1: the exploit itself. Proof of EXECUTION, not a status code. ---
        proof = workdir / "pwned.txt"
        payload = f"import pathlib; pathlib.Path({str(proof)!r}).write_text('pwned')"
        resp = await attach_stdio(build_app(None), sys.executable, ["-c", payload])
        executed = proof.exists()
        record(
            "1-no-rce-by-default",
            not executed,
            (
                f"payload EXECUTED (wrote {proof}); HTTP {resp.status_code} was cosmetic"
                if executed
                else f"payload never ran; HTTP {resp.status_code} code={_code(resp)!r}"
            ),
        )
        record(
            "2-default-deny-403",
            resp.status_code == 403 and _code(resp) == "stdio_attach_disabled",
            f"HTTP {resp.status_code} code={_code(resp)!r} (want 403/stdio_attach_disabled)",
        )

        # --- CHECK 3: opting in must NOT resurrect the interpreter runner list. ---
        resp = await attach_stdio(
            build_app({"allow_stdio": True, "allowed_commands": []}), "python3", ["-c", "print(1)"]
        )
        record(
            "3-no-default-runner-fallback",
            resp.status_code == 403 and _code(resp) == "stdio_command_not_allowed",
            f"empty allowed_commands -> HTTP {resp.status_code} code={_code(resp)!r}",
        )

        # --- CHECK 4: allow-listing an interpreter still refuses inline code. ---
        proof2 = workdir / "pwned2.txt"
        payload2 = f"import pathlib; pathlib.Path({str(proof2)!r}).write_text('pwned')"
        resp = await attach_stdio(
            build_app({"allow_stdio": True, "allowed_commands": [sys.executable]}),
            sys.executable,
            ["-c", payload2],
        )
        record(
            "4-inline-code-args-refused",
            not proof2.exists() and resp.status_code == 403 and _code(resp) == "stdio_args_rejected",
            f"executed={proof2.exists()} HTTP {resp.status_code} code={_code(resp)!r}",
        )

        # --- CHECK 5: a legitimate operator-allow-listed attach clears the gate. ---
        resp = await attach_stdio(
            build_app({"allow_stdio": True, "allowed_commands": ["mcp-weather-server"]}),
            "mcp-weather-server",
            ["--port", "1"],
        )
        record(
            "5-allowlisted-command-passes-gate",
            resp.status_code not in (0, 403),
            f"HTTP {resp.status_code} code={_code(resp)!r} (a connect error, not the 403 gate)",
        )

        # --- CHECK 6: streamable-http spawns no process and is never gated. ---
        resp = await attach_http(build_app(None), "http://127.0.0.1:1/mcp")
        record(
            "6-http-transport-unaffected",
            resp.status_code not in (0, 403),
            f"HTTP {resp.status_code} code={_code(resp)!r} (want a connect error, not 403)",
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    failed = [name for name, ok, _ in results if not ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks PASS")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("Issue #91 is fixed: runtime MCP stdio attach is default-deny + operator-gated.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
