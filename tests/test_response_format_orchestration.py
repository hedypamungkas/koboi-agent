"""Tests for orchestration node response_format (S3): Gap A (output_schema
threaded to per-node AgentCore) + Gap B (force_response_format_with_tools)."""

from koboi.loop import AgentCore
from koboi.orchestration.factory import AgentFactory
from koboi.types import AgentDef

SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string"}},
    "required": ["sentiment"],
}
TOOLS = [{"type": "function", "function": {"name": "f"}}]


class _FakeClient:
    def __init__(self, provider="openai"):
        self.provider = provider


def _core(response_schema=None, force=False, provider="openai") -> AgentCore:
    """Build a minimal AgentCore (bypass __init__) for unit-testing the resolver."""
    core = AgentCore.__new__(AgentCore)
    core.response_schema = response_schema
    core.force_response_format_with_tools = force
    core.client = _FakeClient(provider)
    return core


class TestResolveResponseFormat:
    def test_no_schema_returns_none(self):
        assert _core(response_schema=None)._resolve_response_format(TOOLS) is None
        assert _core(response_schema=None)._resolve_response_format(None) is None

    def test_schema_no_tools_returns_schema(self):
        assert _core(response_schema=SCHEMA)._resolve_response_format(None) is SCHEMA
        assert _core(response_schema=SCHEMA)._resolve_response_format([]) is SCHEMA

    def test_schema_tools_no_force_returns_none(self):
        # Backward-compatible behavior preserved on OpenAI when the flag is off.
        assert _core(response_schema=SCHEMA, force=False)._resolve_response_format(TOOLS) is None

    def test_schema_tools_force_openai_returns_schema(self):
        assert _core(response_schema=SCHEMA, force=True, provider="openai")._resolve_response_format(TOOLS) is SCHEMA

    def test_schema_tools_force_cloudflare_returns_schema(self):
        assert _core(response_schema=SCHEMA, force=True, provider="cloudflare")._resolve_response_format(TOOLS) is SCHEMA

    def test_schema_tools_force_anthropic_still_suppressed(self):
        # Anthropic emulates RF via forced tool_use -> incompatible with real tools.
        assert _core(response_schema=SCHEMA, force=True, provider="anthropic")._resolve_response_format(TOOLS) is None


class TestCreateConfiguredAgentThreadsSchema:
    def test_output_schema_threaded_to_core(self, mock_client):
        ad = AgentDef(
            name="x",
            system_prompt="s",
            output_schema=SCHEMA,
            force_response_format_with_tools=True,
        )
        agent = AgentFactory.create_configured_agent(ad, mock_client)
        assert agent.response_schema == SCHEMA
        assert agent.force_response_format_with_tools is True

    def test_defaults_when_unset(self, mock_client):
        ad = AgentDef(name="x", system_prompt="s")
        agent = AgentFactory.create_configured_agent(ad, mock_client)
        assert agent.response_schema is None
        assert agent.force_response_format_with_tools is False
