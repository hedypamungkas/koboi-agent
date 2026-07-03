"""TUI responsiveness benchmarks.

Benchmarks measure real TUI operations to catch performance regressions:
- Token estimation at scale (hot path, called every iteration)
- Memory copy overhead (get_messages list copy)
- String accumulation (streaming content += delta pattern)
- Export throughput (MD/JSON/HTML formatters at scale)
- Diff detection and rendering at scale
- Thinking block regex at scale
- Bridge message creation throughput
- Hook chain per-iteration cost
- Slash command building
- Welcome panel construction
"""

from unittest.mock import MagicMock


from koboi.tui.loop import build_slash_commands


# -- Helpers ------------------------------------------------------------------


def _make_messages(n: int) -> list[dict]:
    """Generate n realistic conversation messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 3 == 0 else "assistant" if i % 3 == 1 else "tool"
        content = f"Message {i}: " + "word " * 50
        msg: dict = {"role": role, "content": content}
        if role == "assistant" and i % 6 == 1:
            msg["tool_calls"] = [{"id": f"tc_{i}", "function": {"name": "read", "arguments": '{"path": "f.py"}'}}]
        msgs.append(msg)
    return msgs


def _make_messages_with_tool_calls(n: int) -> list[dict]:
    """Generate n message cycles with realistic tool_calls (4 msgs per cycle)."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"Question {i}: " + "word " * 30})
        msgs.append(
            {
                "role": "assistant",
                "content": f"Let me use a tool for question {i}.",
                "tool_calls": [
                    {
                        "id": f"tc_{i}_0",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path": "f.py", "offset": 0}'},
                    },
                    {
                        "id": f"tc_{i}_1",
                        "type": "function",
                        "function": {"name": "write", "arguments": '{"path": "f.py", "content": "' + "x" * 200 + '"}'},
                    },
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"tc_{i}_0", "content": "file content " * 20})
        msgs.append({"role": "tool", "tool_call_id": f"tc_{i}_1", "content": "ok"})
    return msgs


def _make_large_diff(n_lines: int) -> str:
    """Generate a diff with n_lines of changes."""
    lines = ["diff --git a/big.py b/big.py", "index 000..111 100644"]
    for i in range(n_lines):
        if i % 3 == 0:
            lines.append(f"+added line {i}")
        elif i % 3 == 1:
            lines.append(f"-removed line {i}")
        else:
            lines.append(f" context line {i}")
    return "\n".join(lines)


def test_export_markdown_50(benchmark):
    """Export 50 messages to markdown."""
    from koboi.tui.export import export_markdown

    msgs = _make_messages(50)
    result = benchmark(export_markdown, msgs)
    assert len(result) > 0


def test_export_markdown_500(benchmark):
    """Export 500 messages to markdown -- large conversation."""
    from koboi.tui.export import export_markdown

    msgs = _make_messages(500)
    result = benchmark(export_markdown, msgs)
    assert len(result) > 0


def test_export_json_500(benchmark):
    """Export 500 messages to JSON."""
    from koboi.tui.export import export_json

    msgs = _make_messages(500)
    result = benchmark(export_json, msgs)
    assert len(result) > 0


def test_export_html_500(benchmark):
    """Export 500 messages to HTML."""
    from koboi.tui.export import export_html

    msgs = _make_messages(500)
    result = benchmark(export_html, msgs)
    assert "<!DOCTYPE html>" in result


# -- Diff detection and rendering ---------------------------------------------

_SMALL_DIFF = """\
diff --git a/foo.py b/foo.py
index abc123..def456 100644
--- a/foo.py
+++ b/foo.py
@@ -1,5 +1,6 @@
 import os
+import sys

 def main():
-    print("hello")
+    print("hello world")
+    return 0
"""

_LARGE_DIFF = _make_large_diff(200)


def test_is_diff_content_positive(benchmark):
    """Detect diff content (positive case)."""
    from koboi.tui.widgets.diff_view import is_diff_content

    result = benchmark(is_diff_content, _LARGE_DIFF)
    assert result is True


def test_is_diff_content_negative(benchmark):
    """Detect non-diff content."""
    from koboi.tui.widgets.diff_view import is_diff_content

    plain = "This is just a regular message " * 20
    result = benchmark(is_diff_content, plain)
    assert result is False


def test_count_changes(benchmark):
    """Count additions/deletions in large diff."""
    from koboi.tui.widgets.diff_view import count_changes

    result = benchmark(count_changes, _LARGE_DIFF)
    assert result[0] > 0
    assert result[1] > 0


def test_diff_parse(benchmark):
    """Parse large diff into (line, style) pairs."""
    from koboi.tui.widgets.diff_view import DiffViewWidget

    result = benchmark(DiffViewWidget._parse_diff, _LARGE_DIFF)
    assert len(result) > 200


def test_diff_build_rich_text(benchmark):
    """Build Rich Text object from large diff."""
    from koboi.tui.widgets.diff_view import DiffViewWidget

    result = benchmark(DiffViewWidget._build_rich_text, _LARGE_DIFF)
    assert len(result.plain) > 0


# -- Thinking block regex extraction ------------------------------------------

_THINKING_TEXT = (
    "Before answer\n<thinking>\nLet me think about this carefully.\n"
    + "Step 1: analyze\nStep 2: conclude\n</thinking>\nHere is my answer."
)
_NO_THINKING_TEXT = "This is a normal response without any thinking blocks. " * 50


def test_thinking_pattern_match(benchmark):
    """Extract thinking block from content."""
    from koboi.tui.widgets.thinking_block import THINKING_PATTERNS

    def extract():
        for pattern in THINKING_PATTERNS:
            match = pattern.search(_THINKING_TEXT)
            if match:
                return match.group(1).strip()
        return None

    result = benchmark(extract)
    assert "Step 1" in result


def test_thinking_pattern_no_match(benchmark):
    """Search for thinking block in content without one."""
    from koboi.tui.widgets.thinking_block import THINKING_PATTERNS

    def extract():
        for pattern in THINKING_PATTERNS:
            match = pattern.search(_NO_THINKING_TEXT)
            if match:
                return match.group(1)
        return None

    result = benchmark(extract)
    assert result is None


def test_thinking_pattern_large_content(benchmark):
    """Search for thinking block in large content (10KB)."""
    from koboi.tui.widgets.thinking_block import THINKING_PATTERNS

    large = "Some text. " * 1000 + "<thinking>found it</thinking>" + " more text " * 100

    def extract():
        for pattern in THINKING_PATTERNS:
            match = pattern.search(large)
            if match:
                return match.group(1)
        return None

    result = benchmark(extract)
    assert result == "found it"


# -- Suggestion matching ------------------------------------------------------


def test_slash_suggester_match(benchmark):
    """Find slash command suggestion -- sync linear scan."""
    cmds = sorted(["/reset", "/history", "/help", "/info", "/tools", "/export", "/clear", "/undo"])
    value = "/he"

    def find():
        lower = value.lower()
        for cmd in cmds:
            if cmd.startswith(lower) and cmd != lower:
                return cmd
        return None

    result = benchmark(find)
    assert result == "/help"


def test_slash_suggester_no_match(benchmark):
    """No match for slash suggestion -- full scan."""
    cmds = sorted(["/reset", "/history", "/help", "/info", "/tools"])
    value = "/xyz"

    def find():
        lower = value.lower()
        for cmd in cmds:
            if cmd.startswith(lower) and cmd != lower:
                return cmd
        return None

    result = benchmark(find)
    assert result is None


def test_composite_suggester_route(benchmark):
    """Route to correct suggester -- prefix check."""
    value = "/re"

    def route():
        stripped = value.lstrip()
        if stripped.startswith("/"):
            # would delegate to slash suggester
            cmds = sorted(["/reset", "/help"])
            lower = value.lower()
            for cmd in cmds:
                if cmd.startswith(lower) and cmd != lower:
                    return cmd
        return None

    result = benchmark(route)
    assert result == "/reset"


# -- Bridge message creation throughput ---------------------------------------


def test_bridge_message_creation(benchmark):
    """Create 1000 StreamDelta messages (bridge throughput)."""
    from koboi.tui.bridge import StreamDelta

    def create_messages():
        msgs = []
        for i in range(1000):
            msgs.append(StreamDelta(content=f"chunk {i} "))
        return msgs

    result = benchmark(create_messages)
    assert len(result) == 1000


def test_bridge_mixed_message_creation(benchmark):
    """Create mixed bridge message types."""
    from koboi.tui.bridge import (
        StreamDelta,
        StreamToolCall,
        StreamToolResult,
        StreamIteration,
        StreamComplete,
    )

    def create_messages():
        msgs = []
        for i in range(200):
            msgs.append(StreamDelta(content=f"text {i} "))
            msgs.append(StreamToolCall(tool_name=f"tool_{i}", tool_call_id=f"tc_{i}", arguments="{}"))
            msgs.append(StreamToolResult(tool_name=f"tool_{i}", tool_call_id=f"tc_{i}", result="ok"))
            msgs.append(StreamIteration(iteration=i, messages_count=10, tokens_estimated=100))
        msgs.append(StreamComplete(content="Done"))
        return msgs

    result = benchmark(create_messages)
    assert len(result) == 801


# -- Slash command building (kept from original) ------------------------------


def test_slash_command_dispatch(benchmark):
    """Benchmark building slash commands."""
    mock_agent = MagicMock()
    mock_agent.config.agent_name = "test"
    mock_agent.config.provider = "openai"
    mock_agent.config.model = "gpt-4o"
    mock_agent.config.max_iterations = 10
    mock_agent.config.rag_enabled = False
    mock_agent.core.tools._tools = {}
    mock_agent.core.hooks.list_hooks.return_value = []
    mock_agent.core.input_guardrail = None
    mock_agent.core.output_guardrail = None
    mock_agent.core.rate_limiter = None
    mock_agent.core.approval_handler = None

    result = benchmark(build_slash_commands, mock_agent)
    assert "/reset" in result
    assert "/info" in result
    assert "/history" in result
    assert "/tools" in result
    assert "/help" in result


def test_slash_command_with_tools(benchmark):
    """Benchmark building slash commands with many tools."""
    mock_agent = MagicMock()
    mock_agent.config.agent_name = "test"
    mock_agent.config.provider = "openai"
    mock_agent.config.model = "gpt-4o"
    mock_agent.config.max_iterations = 10
    mock_agent.config.rag_enabled = False
    mock_agent.core.hooks.list_hooks.return_value = []
    mock_agent.core.input_guardrail = None
    mock_agent.core.output_guardrail = None
    mock_agent.core.rate_limiter = None
    mock_agent.core.approval_handler = None

    tools = {}
    for i in range(20):
        mock_tool = MagicMock()
        mock_tool.risk_level.value = "safe"
        tools[f"tool_{i}"] = mock_tool
    mock_agent.core.tools._tools = tools

    result = benchmark(build_slash_commands, mock_agent)
    assert "/tools" in result


def test_slash_command_reset_call(benchmark):
    """Benchmark calling /reset command."""
    mock_agent = MagicMock()
    mock_agent.reset = MagicMock()
    mock_console = MagicMock()

    commands = build_slash_commands(mock_agent)

    def call_reset():
        commands["/reset"](mock_agent, mock_console)

    benchmark(call_reset)
    assert mock_agent.reset.called


def test_slash_command_history_call(benchmark):
    """Benchmark calling /history command with messages."""
    mock_agent = MagicMock()
    mock_agent.core.memory.get_messages = MagicMock(
        return_value=[{"role": "user", "content": f"Message {i}: " + "test " * 10} for i in range(50)]
    )
    mock_console = MagicMock()

    commands = build_slash_commands(mock_agent)

    def call_history():
        commands["/history"](mock_agent, mock_console)

    benchmark(call_history)


# -- Welcome panel (kept from original) ---------------------------------------


def test_welcome_panel_building(benchmark):
    """Benchmark building welcome panel."""
    from koboi.tui.app import _build_welcome_panel

    mock_agent = MagicMock()
    mock_agent.config.agent_name = "test-agent"
    mock_agent.config.provider = "openai"
    mock_agent.config.model = "gpt-4o"
    mock_agent.config.max_iterations = 10
    mock_agent.config.rag_enabled = True

    mock_agent.core.hooks.list_hooks.return_value = [{"name": f"Hook{i}", "events": []} for i in range(5)]

    mock_agent.core.input_guardrail = MagicMock()
    mock_agent.core.output_guardrail = MagicMock()
    mock_agent.core.rate_limiter = None
    mock_agent.core.approval_handler = MagicMock()

    tools = {f"tool_{i}": MagicMock() for i in range(10)}
    mock_agent.core.tools._tools = tools

    result = benchmark(_build_welcome_panel, mock_agent)
    from rich.panel import Panel

    assert isinstance(result, Panel)


# -- Theme registration -------------------------------------------------------


def test_theme_registration(benchmark):
    """Benchmark registering themes."""
    from koboi.tui.themes import register_themes

    mock_app = MagicMock()

    def do_register():
        register_themes(mock_app)

    benchmark(do_register)
    assert mock_app.theme == "koboi-dark"


# -- Token estimation at scale (hot path) ------------------------------------


def test_token_estimation_100_messages_with_tool_calls(benchmark):
    """estimate_tokens on 100 message cycles with tool_calls."""
    from koboi.tokens import estimate_tokens

    msgs = _make_messages_with_tool_calls(100)
    result = benchmark(estimate_tokens, msgs)
    assert result > 0


def test_token_estimation_500_messages_with_tool_calls(benchmark):
    """estimate_tokens on 500 message cycles with tool_calls."""
    from koboi.tokens import estimate_tokens

    msgs = _make_messages_with_tool_calls(500)
    result = benchmark(estimate_tokens, msgs)
    assert result > 0


def test_token_estimation_2000_messages_with_tool_calls(benchmark):
    """estimate_tokens on 2000 message cycles with tool_calls -- stress test."""
    from koboi.tokens import estimate_tokens

    msgs = _make_messages_with_tool_calls(2000)
    result = benchmark(estimate_tokens, msgs)
    assert result > 0


# -- Memory copy overhead -----------------------------------------------------


def test_memory_get_messages_100(benchmark):
    """get_messages with 100 pre-loaded messages."""
    from koboi.memory import ConversationMemory

    mem = ConversationMemory()
    for msg in _make_messages_with_tool_calls(100):
        if msg["role"] == "user":
            mem.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            mem.add_assistant_message(msg["content"], msg.get("tool_calls"))
        elif msg["role"] == "tool":
            mem.add_tool_result(msg["tool_call_id"], msg["content"])
    result = benchmark(mem.get_messages)
    assert len(result) > 0


def test_memory_get_messages_1000(benchmark):
    """get_messages with 1000 pre-loaded messages."""
    from koboi.memory import ConversationMemory

    mem = ConversationMemory()
    for msg in _make_messages_with_tool_calls(1000):
        if msg["role"] == "user":
            mem.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            mem.add_assistant_message(msg["content"], msg.get("tool_calls"))
        elif msg["role"] == "tool":
            mem.add_tool_result(msg["tool_call_id"], msg["content"])
    result = benchmark(mem.get_messages)
    assert len(result) > 0


# -- String accumulation (streaming pattern) ----------------------------------


def test_string_concatenation_50kb(benchmark):
    """Simulate content += delta for 50KB (250 x 200-char deltas)."""
    delta = "x" * 200

    def accumulate():
        content = ""
        for _ in range(250):
            content += delta
        return content

    result = benchmark(accumulate)
    assert len(result) == 50000


def test_string_concatenation_200kb(benchmark):
    """Simulate content += delta for 200KB (1000 x 200-char deltas)."""
    delta = "x" * 200

    def accumulate():
        content = ""
        for _ in range(1000):
            content += delta
        return content

    result = benchmark(accumulate)
    assert len(result) == 200000


# -- Thinking block regex at scale -------------------------------------------


def test_thinking_pattern_50kb_no_match(benchmark):
    """Run THINKING_PATTERNS on 50KB with no thinking block."""
    from koboi.tui.widgets.thinking_block import THINKING_PATTERNS

    large = "Normal content. " * 3500  # ~50KB

    def extract():
        for pattern in THINKING_PATTERNS:
            match = pattern.search(large)
            if match:
                return match.group(1)
        return None

    result = benchmark(extract)
    assert result is None


def test_thinking_pattern_50kb_with_match(benchmark):
    """Run THINKING_PATTERNS on 50KB with thinking block in the middle."""
    from koboi.tui.widgets.thinking_block import THINKING_PATTERNS

    large = "Normal content. " * 1750 + "<thinking>found it</thinking>" + " more text " * 1750

    def extract():
        for pattern in THINKING_PATTERNS:
            match = pattern.search(large)
            if match:
                return match.group(1)
        return None

    result = benchmark(extract)
    assert result == "found it"


# -- Export at scale ----------------------------------------------------------


def test_export_markdown_2000(benchmark):
    """Export 2000 messages with tool_calls to markdown."""
    from koboi.tui.export import export_markdown

    msgs = _make_messages(2000)
    result = benchmark(export_markdown, msgs)
    assert len(result) > 0


def test_export_json_2000(benchmark):
    """Export 2000 messages with tool_calls to JSON."""
    from koboi.tui.export import export_json

    msgs = _make_messages(2000)
    result = benchmark(export_json, msgs)
    assert len(result) > 0


def test_export_html_2000(benchmark):
    """Export 2000 messages with tool_calls to HTML."""
    from koboi.tui.export import export_html

    msgs = _make_messages(2000)
    result = benchmark(export_html, msgs)
    assert "<!DOCTYPE html>" in result


# -- Diff at scale ------------------------------------------------------------

_DIFF_1000 = _make_large_diff(1000)


def test_diff_parse_1000_lines(benchmark):
    """Parse 1000-line diff into (line, style) pairs."""
    from koboi.tui.widgets.diff_view import DiffViewWidget

    result = benchmark(DiffViewWidget._parse_diff, _DIFF_1000)
    assert len(result) > 500


def test_is_diff_content_1000_lines(benchmark):
    """Detect diff content in 1000-line diff."""
    from koboi.tui.widgets.diff_view import is_diff_content

    result = benchmark(is_diff_content, _DIFF_1000)
    assert result is True


# -- Hook chain per-iteration cost -------------------------------------------


def test_hook_chain_emit_6_events_per_iteration(benchmark):
    """Emit 6 hook events through 5 hooks (simulates one agent iteration)."""
    from koboi.hooks.chain import HookChain, Hook, HookContext, HookEvent

    class RecordingHook(Hook):
        def handles(self):
            return list(HookEvent)

        async def execute(self, ctx):
            ctx.metadata["hook_ran"] = True
            return ctx

    chain = HookChain()
    for _ in range(5):
        chain.add(RecordingHook())

    events = [
        HookEvent.PRE_COMPACT,
        HookEvent.POST_COMPACT,
        HookEvent.PRE_LLM_CALL,
        HookEvent.POST_LLM_CALL,
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
    ]

    import asyncio

    async def emit_all():
        for event in events:
            ctx = HookContext(event=event, messages=[], metadata={})
            await chain.emit(ctx)

    benchmark(lambda: asyncio.get_event_loop().run_until_complete(emit_all()))
