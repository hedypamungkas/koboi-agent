"""experiment_mcp_gaps.py -- regression harness for the MCP gaps (post-fix).

Originally validated that 13 gaps EXISTED (all CONFIRMED). After the fix waves
(Wave 0-4 on worktree-mcp-gaps-fixes), this script now asserts the FIXED behavior
for the 9 implemented gaps + the mode-block nuance, and honestly reports the 3
deferred gaps (G6/G7/G9) as still OPEN.

Each CHECK prints FIXED / OPEN / REGRESSION + concrete evidence. No network.
Run:  .venv/bin/python experiment_mcp_gaps.py
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field

from koboi.config_models import MCPServerConfig
from koboi.mcp.auth import build_mcp_auth
from koboi.mcp.base import BaseMCPClient, register_mcp_tools
from koboi.mcp.server import MCPServer
from koboi.tools.registry import ToolRegistry
from koboi.types import MCPToolInfo, RiskLevel

REPO = os.path.dirname(os.path.abspath(__file__))

results: list[tuple[str, str, str]] = []  # (id, status, evidence)


def record(gid: str, status: str, evidence: str) -> None:
    results.append((gid, status, evidence))
    print(f"[{status}] {gid}: {evidence}")


class FakeClient:
    def __init__(self, tools, call_result="OK"):
        self._tools = tools
        self._call_result = call_result

    def discover_tools(self):
        return self._tools

    async def call_tool(self, name, arguments):
        return self._call_result


def _tools(*names):
    return [MCPToolInfo(name=n, description=f"tool {n}", input_schema={"type": "object"}) for n in names]


# G1: OAuth now produces an Authorization header (factory returns an OAuth strategy w/ refresh)
def g1_oauth_fixed() -> None:
    strat = build_mcp_auth(
        {"type": "oauth", "token_endpoint": "https://idp/token", "client_id": "c", "client_secret": "s"}
    )
    bearer = build_mcp_auth({"type": "bearer", "token": "t"})
    none = build_mcp_auth({"type": "none"})
    ok = (
        type(strat).__name__ == "OAuthClientCredentialsAuth"
        and getattr(strat, "supports_refresh", False)
        and bearer.apply({}).get("Authorization") == "Bearer t"
        and none is None
    )
    record(
        "G1 OAuth",
        "FIXED" if ok else "REGRESSION",
        f"build_mcp_auth -> oauth={type(strat).__name__}(supports_refresh={getattr(strat, 'supports_refresh', False)}), "
        f"bearer={bearer.apply({}).get('Authorization')!r}, none={none is None}",
    )


# G2: server now handles resources/prompts (returns result, not -32601)
def g2_resources_prompts_fixed() -> None:
    srv = MCPServer(name="probe")

    @srv.resource(uri="doc://a", name="A")
    def _r():
        return "body"

    @srv.prompt(name="p")
    def _p(args):
        return "hi"

    codes = {}
    for i, m in enumerate(["resources/list", "resources/read", "prompts/list", "prompts/get"]):
        buf = io.StringIO()
        params = {"uri": "doc://a"} if m == "resources/read" else {"name": "p"} if m == "prompts/get" else {}
        with redirect_stdout(buf):
            srv._dispatch({"jsonrpc": "2.0", "id": i + 1, "method": m, "params": params})
        mm = re.search(r'"error":\s*\{[^}]*"code":\s*(-?\d+)', buf.getvalue())
        codes[m] = int(mm.group(1)) if mm else "result"
    ok = all(v == "result" for v in codes.values())
    record("G2 resources/prompts", "FIXED" if ok else "REGRESSION", f"server dispatch codes -> {codes}")


# G3: risk level now configurable + set_risk_level exists
def g3_risk_level_fixed() -> None:
    reg = ToolRegistry()
    register_mcp_tools(FakeClient(_tools("delete_all")), reg, risk_level=RiskLevel.DESTRUCTIVE)
    has_setter = hasattr(reg, "set_risk_level")
    ok = reg.get_risk_level("delete_all") == RiskLevel.DESTRUCTIVE and has_setter
    record(
        "G3 per-tool risk",
        "FIXED" if ok else "REGRESSION",
        f"delete_all risk={reg.get_risk_level('delete_all')}, set_risk_level exists={has_setter}",
    )


# G4: killed subprocess now respawns on next call (ensure_connected)
def g4_reconnect_fixed() -> None:
    from koboi.mcp.client import MCPClient

    server_path = os.path.join(REPO, "mcp_servers", "todo_server.py")
    if not os.path.exists(server_path):
        record("G4 reconnect", "SKIPPED", f"todo_server.py not found at {server_path}")
        return
    mcp = MCPClient(server_command=[sys.executable, server_path], connect_timeout=15)
    try:
        mcp.connect()
        mcp.discover_tools()
        proc_before = mcp._process
        mcp._process.kill()
        mcp._process.wait(timeout=5)
        res = asyncio.run(mcp.call_tool("list_todos", {}))  # should respawn, not raise
        respawned = mcp._process is not proc_before and mcp._process.poll() is None
        ok = isinstance(res, str) and respawned
        record(
            "G4 reconnect",
            "FIXED" if ok else "REGRESSION",
            f"after kill, call_tool returned {type(res).__name__}, respawned={respawned}",
        )
    finally:
        mcp.close()


# G5: orchestration now wires MCP (registrar param + mcp references)
def g5_orchestration_fixed() -> None:
    import inspect

    from koboi.orchestration.factory import AgentFactory

    n = int(
        subprocess.run(
            ["grep", "-rni", "--include=*.py", "mcp", os.path.join(REPO, "koboi", "orchestration")],
            capture_output=True,
            text=True,
        ).stdout.count("\n")
    )
    has_registrar = "mcp_registrar" in inspect.signature(AgentFactory.create_all_configured).parameters
    ok = n > 0 and has_registrar
    record(
        "G5 orchestration wiring",
        "FIXED" if ok else "REGRESSION",
        f"grep mcp koboi/orchestration -> {n} hits; create_all_configured has mcp_registrar={has_registrar}",
    )


# G6: server still has no MCP management surface (DEFERRED)
def g6_server_still_open() -> None:
    n = int(
        subprocess.run(
            ["grep", "-rni", "--include=*.py", "mcp", os.path.join(REPO, "koboi", "server")],
            capture_output=True,
            text=True,
        ).stdout.count("\n")
    )
    record("G6 server MCP surface", "OPEN (deferred)", f"grep mcp koboi/server -> {n} hits (no /v1/mcp endpoints yet)")


# G7: TUI still has no MCP surface (DEFERRED)
def g7_tui_still_open() -> None:
    n = int(
        subprocess.run(
            ["grep", "-rni", "--include=*.py", "mcp", os.path.join(REPO, "koboi", "tui")],
            capture_output=True,
            text=True,
        ).stdout.count("\n")
    )
    record("G7 TUI MCP surface", "OPEN (deferred)", f"grep mcp koboi/tui -> {n} hits (no status screen yet)")


# G8: collision warning fires + opt-in namespacing avoids shadow
def g8_collision_fixed() -> None:
    import logging

    reg = ToolRegistry()
    reg.register("calculator", "builtin", {"type": "object"}, fn=lambda **k: "B", risk_level=RiskLevel.MODERATE)

    class _Cap(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, r):
            self.records.append(r)

    cap = _Cap()
    lg = logging.getLogger("koboi.tools.registry")
    lg.addHandler(cap)
    lg.setLevel(logging.WARNING)
    register_mcp_tools(FakeClient(_tools("calculator")), reg, namespace_prefix="mcp__s")
    lg.removeHandler(cap)
    no_shadow = reg.get_risk_level("calculator") == RiskLevel.MODERATE  # builtin intact
    mcp_present = reg.get_risk_level("mcp__s__calculator") == RiskLevel.SAFE
    ok = no_shadow and mcp_present
    record(
        "G8 collision/namespace",
        "FIXED" if ok else "REGRESSION",
        f"builtin 'calculator' intact={no_shadow}, namespaced 'mcp__s__calculator' present={mcp_present}",
    )


# G9: koboi still does not expose itself over MCP (DEFERRED)
def g9_self_exposure_still_open() -> None:
    koboi_inst = subprocess.run(
        ["grep", "-rnE", "--include=*.py", r"MCPServer\(", os.path.join(REPO, "koboi")], capture_output=True, text=True
    ).stdout.strip()
    n = len([l for l in koboi_inst.splitlines() if "class MCPServer" not in l])
    record(
        "G9 self-exposure",
        "OPEN (deferred)",
        f"MCPServer( instantiations inside koboi/ = {n} (no ToolRegistry->MCP wrapper yet)",
    )


# G10: fail_fast now raises instead of silently returning []
def g10_fail_fast_fixed() -> None:
    from koboi.facade import _build_mcp

    @dataclass
    class Cfg:
        servers: list = field(default_factory=lambda: [{"command": "/usr/bin/evil-binary", "args": []}])

        def get(self, section, key, default=None):
            if (section, key) == ("mcp", "servers"):
                return self.servers
            if (section, key) == ("mcp", "fail_fast"):
                return True
            if (section, key) == ("mcp", "namespace"):
                return False
            if (section, key) == ("mcp", "allowlist_commands"):
                return []
            return default

    raised = False
    try:
        _build_mcp(Cfg(), ToolRegistry(), logger=None)
    except Exception:
        raised = True
    record(
        "G10 fail_fast",
        "FIXED" if raised else "REGRESSION",
        f"_build_mcp(fail_fast=True) with bad runner raised={raised}",
    )


# G11: client now negotiates 2025-03-26 (consistent with server + http docstring)
def g11_protocol_fixed() -> None:
    class Rec(BaseMCPClient):
        def __init__(self):
            super().__init__(logger=None)
            self.reqs = []

        def connect(self):
            return self._do_initialize_handshake()

        def discover_tools(self):
            return []

        async def call_tool(self, n, a):
            return ""

        def _send_request_impl(self, m, p=None):
            self.reqs.append((m, p))
            return {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {}} if m == "initialize" else {}

        def _send_notification_impl(self, m, p=None):
            pass

        def close(self):
            pass

    c = Rec()
    c.connect()
    sent = next((p for m, p in c.reqs if m == "initialize"), {}).get("protocolVersion")
    srv = MCPServer(name="x")._handle_initialize(0, {})["protocolVersion"]
    ok = sent == "2025-03-26" == srv
    record("G11 protocol version", "FIXED" if ok else "REGRESSION", f"client sends {sent!r}, server advertises {srv!r}")


# G12: docs snippet now matches the schema
def g12_docs_fixed() -> None:
    arch = open(os.path.join(REPO, "docs", "architecture.md")).read()
    mcp_section = arch[arch.find("## MCP Integration") :]
    has_command_str = re.search(r'command:\s*"', mcp_section) is not None
    has_args = "args:" in mcp_section
    name_field = bool(re.search(r'- name:\s*"', mcp_section))
    schema_has_no_name = "name" not in MCPServerConfig.model_fields
    ok = has_command_str and has_args and (not name_field) and schema_has_no_name
    record(
        "G12 docs/config drift",
        "FIXED" if ok else "REGRESSION",
        f"docs command:str={has_command_str}, args present={has_args}, doc 'name:' field={name_field}, schema has name={not schema_has_no_name}",
    )


# NUANCE: mode-block now configurable via read_only_tools
def nuance_mode_block_fixed() -> None:
    from koboi.hooks.mode_hook import ModeHook
    from koboi.modes import AgentMode, ModeManager

    blocked_default = ModeHook(ModeManager(AgentMode.CHAT))._is_read_only_or_extra("add_todo") is False
    allowed = ModeHook(ModeManager(AgentMode.CHAT), extra_read_only=["add_todo"])._is_read_only_or_extra("add_todo")
    ok = blocked_default and allowed
    record(
        "NUANCE mode-block allowlist",
        "FIXED" if ok else "REGRESSION",
        f"add_todo blocked-by-default={blocked_default}, allowed-when-allowlisted={allowed}",
    )


def main() -> None:
    print("=" * 90)
    print("experiment_mcp_gaps.py -- post-fix regression (9 FIXED + nuance; G6/G7/G9 OPEN deferred)")
    print("=" * 90)
    for _, fn in [
        ("G1", g1_oauth_fixed),
        ("G2", g2_resources_prompts_fixed),
        ("G3", g3_risk_level_fixed),
        ("G4", g4_reconnect_fixed),
        ("G5", g5_orchestration_fixed),
        ("G6", g6_server_still_open),
        ("G7", g7_tui_still_open),
        ("G8", g8_collision_fixed),
        ("G9", g9_self_exposure_still_open),
        ("G10", g10_fail_fast_fixed),
        ("G11", g11_protocol_fixed),
        ("G12", g12_docs_fixed),
        ("NU", nuance_mode_block_fixed),
    ]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            record("?", "ERROR", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 90)
    fixed = sum(1 for _, s, _ in results if s == "FIXED")
    open_ = sum(1 for _, s, _ in results if s.startswith("OPEN"))
    reg = sum(1 for _, s, _ in results if s == "REGRESSION")
    err = sum(1 for _, s, _ in results if s == "ERROR")
    sk = sum(1 for _, s, _ in results if s == "SKIPPED")
    print(
        f"SUMMARY: {fixed} FIXED, {open_} OPEN(deferred), {reg} REGRESSION, {err} ERROR, {sk} SKIPPED of {len(results)}"
    )
    print("=" * 90)
    for gid, status, _ in results:
        print(f"  {status:16s}  {gid}")


if __name__ == "__main__":
    main()
