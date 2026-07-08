"""tests/test_mcp_risk_gate.py -- #5 MCP tool risk gating.

Pre-#5 every MCP tool was unconditionally RiskLevel.SAFE, so destructive MCP
tools (stripe.refund, github.merge_pr, ...) bypassed the approval gate entirely.
Now register_mcp_tools accepts risk_level (per-server override) and an optional
risk_resolver (per-tool; default_risk_heuristic infers from the tool name).
"""

from __future__ import annotations

from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.mcp.base import default_risk_heuristic, register_mcp_tools
from koboi.tools.registry import ToolRegistry
from koboi.types import MCPToolInfo, RiskLevel, ToolCall


class _FakeMCPClient:
    """Duck-typed MCP client: register_mcp_tools only needs discover_tools + call_tool."""

    def __init__(self, tools):
        self._tools = tools

    def discover_tools(self):
        return self._tools

    async def call_tool(self, name, arguments):
        return f"called {name}"


def _info(name: str) -> MCPToolInfo:
    return MCPToolInfo(name=name, description=name, input_schema={"type": "object", "properties": {}})


class _RecordingApproval:
    def __init__(self):
        self.called = False

    def should_approve(self, tool_name, arguments, risk_level):  # sync handler
        self.called = True
        return True


async def test_heuristic_assigns_risk_by_name():
    client = _FakeMCPClient([_info("delete_record"), _info("create_user"), _info("get_status")])
    registry = ToolRegistry()
    register_mcp_tools(client, registry, risk_resolver=default_risk_heuristic)

    assert registry.get_risk_level("delete_record") == RiskLevel.DESTRUCTIVE
    assert registry.get_risk_level("create_user") == RiskLevel.MODERATE
    assert registry.get_risk_level("get_status") == RiskLevel.SAFE


async def test_default_is_safe_without_resolver():
    """No resolver -> every MCP tool is SAFE (pre-#5 behavior)."""
    client = _FakeMCPClient([_info("delete_record")])
    registry = ToolRegistry()
    register_mcp_tools(client, registry)
    assert registry.get_risk_level("delete_record") == RiskLevel.SAFE


async def test_explicit_risk_level_applies_to_all_tools():
    client = _FakeMCPClient([_info("get_status")])
    registry = ToolRegistry()
    register_mcp_tools(client, registry, risk_level=RiskLevel.DESTRUCTIVE)
    assert registry.get_risk_level("get_status") == RiskLevel.DESTRUCTIVE


async def test_destructive_mcp_tool_hits_approval_gate():
    """A heuristic-flagged DESTRUCTIVE MCP tool prompts approval (was a silent bypass pre-#5)."""
    client = _FakeMCPClient([_info("delete_record")])
    registry = ToolRegistry()
    register_mcp_tools(client, registry, risk_resolver=default_risk_heuristic)

    handler = _RecordingApproval()
    pipeline = ToolExecutionPipeline(tools=registry, memory=ConversationMemory(), approval_handler=handler)
    result = await pipeline.execute_tool_call(ToolCall(id="1", name="delete_record", arguments="{}"), iteration=0)

    assert handler.called is True  # DESTRUCTIVE -> approval prompted
    assert not result.skipped  # approved -> executed


def test_heuristic_no_false_positive_on_substring_match():
    """#1 fix: token-boundary matching prevents false positives.

    'get_deleted_items' should be SAFE (the token is 'deleted', not 'delete'),
    not DESTRUCTIVE (which the old substring match incorrectly returned).
    """
    client = _FakeMCPClient([_info("get_deleted_items")])
    registry = ToolRegistry()
    register_mcp_tools(client, registry, risk_resolver=default_risk_heuristic)
    assert registry.get_risk_level("get_deleted_items") == RiskLevel.SAFE

    # 'undelete_file' — token 'undelete' is NOT 'delete' -> SAFE.
    client2 = _FakeMCPClient([_info("undelete_file")])
    registry2 = ToolRegistry()
    register_mcp_tools(client2, registry2, risk_resolver=default_risk_heuristic)
    assert registry2.get_risk_level("undelete_file") == RiskLevel.SAFE

    # Correct positive still works: 'delete_record' -> token 'delete' IS destructive.
    client3 = _FakeMCPClient([_info("delete_record")])
    registry3 = ToolRegistry()
    register_mcp_tools(client3, registry3, risk_resolver=default_risk_heuristic)
    assert registry3.get_risk_level("delete_record") == RiskLevel.DESTRUCTIVE
