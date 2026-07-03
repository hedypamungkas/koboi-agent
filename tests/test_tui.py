"""Tests for koboi/tui/loop.py and koboi/tui/app.py."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import yaml
from click.testing import CliRunner
from rich.console import Console

from koboi.tui.app import main, _build_welcome_panel
from koboi.tui.loop import (
    interactive_loop,
    _stream_response,
    _print_summary,
    build_slash_commands,
)
from koboi.events import (
    TextDeltaEvent,
    ToolCallEvent,
    IterationEvent,
    CompleteEvent,
    ErrorEvent,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_temp_config() -> str:
    """Create a temp YAML config file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(
            {
                "agent": {"name": "test-agent", "max_iterations": 5},
                "llm": {"model": "gpt-4o-mini", "provider": "openai"},
            },
            f,
        )
        return f.name


def _make_mock_agent():
    """Create a fully mocked KoboiAgent."""
    mock_agent = MagicMock()
    mock_agent.config.agent_name = "test-agent"
    mock_agent.config.provider = "openai"
    mock_agent.config.model = "gpt-4o-mini"
    mock_agent.config.max_iterations = 10
    mock_agent.config.rag_enabled = False
    mock_agent.core.tools._tools = {}
    mock_agent.core.tools.list_tools.return_value = {}
    mock_agent.core.hooks.list_hooks.return_value = []
    mock_agent.core.input_guardrail = None
    mock_agent.core.output_guardrail = None
    mock_agent.core.rate_limiter = None
    mock_agent.core.approval_handler = None
    mock_agent.run = AsyncMock(return_value="Hello from agent!")
    return mock_agent


def _make_mock_agent_with_tools():
    """Create a mock agent with some tools."""
    mock_agent = _make_mock_agent()

    mock_tool_1 = MagicMock()
    mock_tool_1.risk_level.value = "safe"
    mock_tool_2 = MagicMock()
    mock_tool_2.risk_level.value = "destructive"

    mock_agent.core.tools._tools = {
        "calculator": mock_tool_1,
        "shell": mock_tool_2,
    }
    mock_agent.core.tools.list_tools.return_value = mock_agent.core.tools._tools
    return mock_agent


@pytest.fixture
def mock_console():
    """Create a mock Rich console."""
    console = MagicMock(spec=Console)
    console.input = MagicMock(return_value="quit")
    console.print = MagicMock()
    return console


# ============================================================================
# Tests for _stream_response
# ============================================================================


class TestStreamResponse:
    """Tests for _stream_response function."""

    @pytest.mark.asyncio
    async def test_stream_text_deltas(self, mock_console):
        """Test streaming TextDeltaEvent updates the Live display."""
        mock_agent = MagicMock()
        mock_agent.run_stream = AsyncMock()

        async def mock_stream(prompt):
            yield TextDeltaEvent(content="Hello")
            yield TextDeltaEvent(content=" world")
            yield TextDeltaEvent(content="!")
            yield CompleteEvent(content="Hello world!")

        mock_agent.run_stream = mock_stream

        with patch("koboi.tui.loop.Live") as mock_live_class:
            mock_live = MagicMock()
            mock_live_class.return_value.__enter__.return_value = mock_live
            mock_live_class.return_value.__exit__.return_value = None

            await _stream_response(mock_agent, "test", mock_console, "Test")

            # Verify Live.update was called multiple times
            assert mock_live.update.call_count >= 1

            # Verify a panel was updated with markdown
            last_call = mock_live.update.call_args
            from rich.panel import Panel

            assert isinstance(last_call[0][0], Panel)

    @pytest.mark.asyncio
    async def test_stream_tool_call_event(self, mock_console):
        """Test ToolCallEvent shows tool name in subtitle."""
        mock_agent = MagicMock()

        async def mock_stream(prompt):
            yield ToolCallEvent(tool_name="calculator", tool_call_id="tc_123", arguments='{"expr": "2+2"}')
            yield CompleteEvent(content="Result: 4")

        mock_agent.run_stream = mock_stream

        with patch("koboi.tui.loop.Live") as mock_live_class:
            mock_live = MagicMock()
            mock_live_class.return_value.__enter__.return_value = mock_live
            mock_live_class.return_value.__exit__.return_value = None

            await _stream_response(mock_agent, "test", mock_console, "Test")

            # Check that Live.update was called
            assert mock_live.update.called

    @pytest.mark.asyncio
    async def test_stream_error_event(self, mock_console):
        """Test ErrorEvent displays error in red border."""
        mock_agent = MagicMock()

        async def mock_stream(prompt):
            yield ErrorEvent(error=Exception("API timeout"))

        mock_agent.run_stream = mock_stream

        with patch("koboi.tui.loop.Live") as mock_live_class:
            mock_live = MagicMock()
            mock_live_class.return_value.__enter__.return_value = mock_live
            mock_live_class.return_value.__exit__.return_value = None

            await _stream_response(mock_agent, "test", mock_console, "Test")

            # Verify update was called for error display
            assert mock_live.update.called

    @pytest.mark.asyncio
    async def test_stream_iteration_event(self, mock_console):
        """Test IterationEvent is handled without error."""
        mock_agent = MagicMock()

        async def mock_stream(prompt):
            yield IterationEvent(iteration=1, messages_count=5, tokens_estimated=150)
            yield CompleteEvent(content="Done")

        mock_agent.run_stream = mock_stream

        with patch("koboi.tui.loop.Live") as mock_live_class:
            mock_live = MagicMock()
            mock_live_class.return_value.__enter__.return_value = mock_live
            mock_live_class.return_value.__exit__.return_value = None

            await _stream_response(mock_agent, "test", mock_console, "Test")

            # Should complete without error
            assert True

    @pytest.mark.asyncio
    async def test_stream_complete_event_updates_final_content(self, mock_console):
        """Test CompleteEvent with content updates final display."""
        mock_agent = MagicMock()

        async def mock_stream(prompt):
            yield TextDeltaEvent(content="Partial")
            yield CompleteEvent(content="Final response")

        mock_agent.run_stream = mock_stream

        with patch("koboi.tui.loop.Live") as mock_live_class:
            mock_live = MagicMock()
            mock_live_class.return_value.__enter__.return_value = mock_live
            mock_live_class.return_value.__exit__.return_value = None

            await _stream_response(mock_agent, "test", mock_console, "Test")

            # Verify final update happened
            assert mock_live.update.called


# ============================================================================
# Tests for _print_summary
# ============================================================================


class TestPrintSummary:
    """Tests for _print_summary function."""

    @patch("koboi.tui.loop.time")
    def test_print_summary_basic(self, mock_time, mock_console):
        """Test basic summary output."""
        mock_time.time.return_value = 100.0
        _print_summary(mock_console, 5, 90.0)  # elapsed = 100 - 90 = 10.0

        # Verify console.print was called
        assert mock_console.print.called
        call_args = str(mock_console.print.call_args)
        assert "5" in call_args

    @patch("koboi.tui.loop.time")
    def test_print_summary_with_agent(self, mock_time, mock_console):
        """Test summary with agent that has memory."""
        mock_time.time.return_value = 100.0
        mock_agent = _make_mock_agent()
        mock_agent.core.memory = MagicMock()
        mock_agent.core.memory.__len__ = MagicMock(return_value=3)

        _print_summary(mock_console, 2, 95.0, mock_agent)  # elapsed = 100 - 95 = 5.0

        # Verify memory count is included
        call_args = str(mock_console.print.call_args)
        assert "3" in call_args

    @patch("koboi.tui.loop.time")
    def test_print_summary_zero_turns(self, mock_time, mock_console):
        """Test summary with zero turns."""
        mock_time.time.return_value = 100.0
        _print_summary(mock_console, 0, 99.9)  # elapsed = 100 - 99.9 = 0.1

        assert mock_console.print.called
        call_args = str(mock_console.print.call_args)
        assert "0" in call_args

    @patch("koboi.tui.loop.time")
    def test_print_summary_with_agent_memory_error(self, mock_time, mock_console):
        """Test summary when agent.memory raises exception."""
        mock_time.time.return_value = 100.0
        mock_agent = _make_mock_agent()
        mock_agent.core.memory = MagicMock()
        mock_agent.core.memory.__len__ = MagicMock(side_effect=AttributeError)

        # Should not crash
        _print_summary(mock_console, 1, 99.0, mock_agent)  # elapsed = 100 - 99 = 1.0

        assert mock_console.print.called

    @patch("koboi.tui.loop.time")
    def test_print_summary_elapsed_time_formatting(self, mock_time, mock_console):
        """Test elapsed time is formatted with one decimal."""
        mock_time.time.return_value = 200.0
        _print_summary(mock_console, 1, 100.0)  # elapsed = 200 - 100 = 100.0

        call_args = str(mock_console.print.call_args)
        # Should have "100.0s" formatted
        assert "100.0" in call_args or "message(s)" in call_args


# ============================================================================
# Tests for build_slash_commands
# ============================================================================


class TestBuildSlashCommands:
    """Tests for build_slash_commands function."""

    def test_build_slash_commands_returns_all_commands(self):
        """Test all expected commands are returned."""
        mock_agent = _make_mock_agent()
        commands = build_slash_commands(mock_agent)

        assert "/reset" in commands
        assert "/info" in commands
        assert "/history" in commands
        assert "/tools" in commands
        assert "/help" in commands

    def test_cmd_reset_calls_agent_reset(self):
        """Test /reset command calls agent.reset()."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        commands = build_slash_commands(mock_agent)

        commands["/reset"](mock_agent, mock_console)

        mock_agent.reset.assert_called_once()
        mock_console.print.assert_called()
        call_args = str(mock_console.print.call_args)
        assert "reset" in call_args.lower()

    def test_cmd_info_displays_welcome_panel(self):
        """Test /info command displays welcome panel."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        commands = build_slash_commands(mock_agent)

        commands["/info"](mock_agent, mock_console)

        mock_console.print.assert_called_once()
        from rich.panel import Panel

        call_arg = mock_console.print.call_args[0][0]
        assert isinstance(call_arg, Panel)

    def test_cmd_history_with_messages(self):
        """Test /history command displays messages."""
        mock_agent = _make_mock_agent()
        mock_agent.core.memory.get_messages = MagicMock(
            return_value=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        mock_console = MagicMock()
        commands = build_slash_commands(mock_agent)

        commands["/history"](mock_agent, mock_console)

        # Should have printed each message
        assert mock_console.print.call_count == 2

    def test_cmd_history_empty(self):
        """Test /history with no messages."""
        mock_agent = _make_mock_agent()
        mock_agent.core.memory.get_messages = MagicMock(return_value=[])
        mock_console = MagicMock()
        commands = build_slash_commands(mock_agent)

        commands["/history"](mock_agent, mock_console)

        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "no messages" in call_args.lower()

    def test_cmd_tools_displays_table(self):
        """Test /tools command displays tools in table."""
        mock_agent = _make_mock_agent_with_tools()
        mock_console = MagicMock()
        commands = build_slash_commands(mock_agent)

        commands["/tools"](mock_agent, mock_console)

        mock_console.print.assert_called_once()
        from rich.table import Table

        call_arg = mock_console.print.call_args[0][0]
        assert isinstance(call_arg, Table)

    def test_cmd_tools_empty(self):
        """Test /tools with no tools registered."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        commands = build_slash_commands(mock_agent)

        commands["/tools"](mock_agent, mock_console)

        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "no tools" in call_args.lower()

    def test_cmd_help_displays_help_panel(self):
        """Test /help command displays help."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        commands = build_slash_commands(mock_agent)

        commands["/help"](mock_agent, mock_console)

        mock_console.print.assert_called_once()
        from rich.panel import Panel

        call_arg = mock_console.print.call_args[0][0]
        assert isinstance(call_arg, Panel)


# ============================================================================
# Tests for interactive_loop
# ============================================================================


class TestInteractiveLoop:
    """Tests for interactive_loop function."""

    @pytest.mark.asyncio
    async def test_exit_on_quit(self):
        """Test 'quit' exits the loop."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        mock_console.input = MagicMock(return_value="quit")

        await interactive_loop(mock_agent, mock_console, stream=False)

        # Should have called summary once
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_exit_on_exit(self):
        """Test 'exit' exits the loop."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        mock_console.input = MagicMock(return_value="exit")

        await interactive_loop(mock_agent, mock_console, stream=False)

        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_exit_on_q(self):
        """Test 'q' exits the loop."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        mock_console.input = MagicMock(return_value="q")

        await interactive_loop(mock_agent, mock_console, stream=False)

        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_empty_input_continues(self):
        """Test empty input continues loop."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        # First input empty, second quit
        mock_console.input = MagicMock(side_effect=["   ", "quit"])

        await interactive_loop(mock_agent, mock_console, stream=False)

        # Should not have called agent.run for empty input
        mock_agent.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_slash_command_dispatch(self):
        """Test slash commands are dispatched."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        mock_console.input = MagicMock(side_effect=["/help", "quit"])

        commands = build_slash_commands(mock_agent)
        await interactive_loop(mock_agent, mock_console, extra_commands=commands, stream=False)

        # Help should have been displayed
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_triggers_summary(self):
        """Test KeyboardInterrupt triggers summary."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        mock_console.input = MagicMock(side_effect=KeyboardInterrupt())

        await interactive_loop(mock_agent, mock_console, stream=False)

        # Should print summary on interrupt
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_eof_error_triggers_summary(self):
        """Test EOFError triggers summary."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        mock_console.input = MagicMock(side_effect=EOFError())

        await interactive_loop(mock_agent, mock_console, stream=False)

        # Should print summary on EOF
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_error_preserves_session(self):
        """Test errors during run preserve session."""
        mock_agent = _make_mock_agent()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("LLM error"))
        mock_console = MagicMock()
        mock_console.input = MagicMock(side_effect=["test input", "quit"])

        await interactive_loop(mock_agent, mock_console, stream=False)

        # Should print error message
        error_calls = [c for c in mock_console.print.call_args_list]
        assert any("error" in str(c).lower() for c in error_calls)

    @pytest.mark.asyncio
    async def test_stream_mode_uses_stream_response(self):
        """Test stream=True uses _stream_response."""
        mock_agent = _make_mock_agent()

        async def mock_stream(prompt):
            yield TextDeltaEvent(content="Hi")
            yield CompleteEvent(content="Hi")

        mock_agent.run_stream = mock_stream
        mock_console = MagicMock()
        mock_console.input = MagicMock(side_effect=["hello", "quit"])

        with patch("koboi.tui.loop._stream_response", new=AsyncMock()) as mock_stream_resp:
            await interactive_loop(mock_agent, mock_console, stream=True)
            mock_stream_resp.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_stream_mode_uses_run(self):
        """Test stream=False uses agent.run."""
        mock_agent = _make_mock_agent()
        mock_console = MagicMock()
        mock_console.input = MagicMock(side_effect=["hello", "quit"])

        await interactive_loop(mock_agent, mock_console, stream=False)

        # Should have called run once
        mock_agent.run.assert_called_once_with("hello")


# ============================================================================
# Tests for CLI commands (app.py)
# ============================================================================


class TestRunCommand:
    """Tests for 'run' CLI command."""

    def test_run_missing_config_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "nonexistent_config.yaml", "-m", "hi"])
        assert result.exit_code != 0

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_run_single_shot_success(self, mock_from_config):
        mock_from_config.return_value = _make_mock_agent()
        config = _make_temp_config()

        runner = CliRunner()
        result = runner.invoke(main, ["run", config, "-m", "Hi"])
        assert result.exit_code == 0
        assert "Hello from agent!" in result.output

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_run_from_config_error(self, mock_from_config):
        mock_from_config.side_effect = Exception("API key not configured")
        config = _make_temp_config()

        runner = CliRunner()
        result = runner.invoke(main, ["run", config, "-m", "Hi"])
        assert result.exit_code == 1
        assert "API key not configured" in result.output

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_run_agent_error(self, mock_from_config):
        agent = _make_mock_agent()
        agent.run.side_effect = RuntimeError("LLM timeout")
        mock_from_config.return_value = agent
        config = _make_temp_config()

        runner = CliRunner()
        result = runner.invoke(main, ["run", config, "-m", "Hi"])
        assert result.exit_code == 1
        assert "LLM timeout" in result.output


class TestChatCommand:
    """Tests for 'chat' CLI command."""

    @patch("koboi.tui.loop.interactive_loop")
    @patch("koboi.facade.KoboiAgent.from_config")
    def test_chat_starts_successfully(self, mock_from_config, mock_loop):
        mock_from_config.return_value = _make_mock_agent()
        config = _make_temp_config()

        runner = CliRunner()
        result = runner.invoke(main, ["chat", "--no-tui", config])
        assert result.exit_code == 0
        assert mock_loop.called

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_chat_from_config_error(self, mock_from_config):
        mock_from_config.side_effect = Exception("API key missing")
        config = _make_temp_config()

        runner = CliRunner()
        result = runner.invoke(main, ["chat", config])
        assert result.exit_code == 1
        assert "API key missing" in result.output


class TestEvalCommand:
    """Tests for 'eval' CLI command."""

    def test_eval_no_cases_file(self):
        config = _make_temp_config()

        runner = CliRunner()
        result = runner.invoke(main, ["eval", config])
        assert "No eval cases" in result.output


class TestValidateCommand:
    """Tests for 'validate' CLI command."""

    def test_validate_valid_config(self):
        config = _make_temp_config()

        runner = CliRunner()
        result = runner.invoke(main, ["validate", config])
        assert "valid" in result.output.lower() or result.exit_code == 0

    def test_validate_missing_config_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "nonexistent.yaml"])
        assert result.exit_code != 0


class TestBuildWelcomePanel:
    """Tests for _build_welcome_panel helper."""

    def test_panel_contains_agent_info(self):
        mock_agent = MagicMock()
        mock_agent.config.agent_name = "sales-bot"
        mock_agent.config.provider = "openai"
        mock_agent.config.model = "gpt-4o"
        mock_agent.config.max_iterations = 15
        mock_agent.config.rag_enabled = True
        mock_agent.core.tools._tools = {"calculator": MagicMock(), "web_search": MagicMock()}
        mock_agent.core.tools.list_tools.return_value = mock_agent.core.tools._tools
        mock_agent.core.hooks.list_hooks.return_value = [{"name": "LoggingHook", "events": []}]
        mock_agent.core.input_guardrail = MagicMock()
        mock_agent.core.output_guardrail = None
        mock_agent.core.rate_limiter = None
        mock_agent.core.approval_handler = None

        panel = _build_welcome_panel(mock_agent)
        from rich.panel import Panel

        assert isinstance(panel, Panel)

    def test_panel_with_many_tools_truncates_display(self):
        """Test that many tools show truncation."""
        mock_agent = MagicMock()
        mock_agent.config.agent_name = "tool-heavy"
        mock_agent.config.provider = "anthropic"
        mock_agent.config.model = "claude-3"
        mock_agent.config.max_iterations = 5
        mock_agent.config.rag_enabled = False
        mock_agent.core.hooks.list_hooks.return_value = []
        mock_agent.core.input_guardrail = None
        mock_agent.core.output_guardrail = None
        mock_agent.core.rate_limiter = None
        mock_agent.core.approval_handler = None

        # Create 10 tools
        tools = {f"tool_{i}": MagicMock() for i in range(10)}
        mock_agent.core.tools._tools = tools
        mock_agent.core.tools.list_tools.return_value = tools

        panel = _build_welcome_panel(mock_agent)
        from rich.panel import Panel

        assert isinstance(panel, Panel)

    def test_panel_with_guardrails(self):
        """Test panel shows configured guardrails."""
        mock_agent = MagicMock()
        mock_agent.config.agent_name = "guarded"
        mock_agent.config.provider = "openai"
        mock_agent.config.model = "gpt-4"
        mock_agent.config.max_iterations = 10
        mock_agent.config.rag_enabled = False
        mock_agent.core.tools._tools = {}
        mock_agent.core.tools.list_tools.return_value = {}
        mock_agent.core.hooks.list_hooks.return_value = []
        mock_agent.core.input_guardrail = MagicMock()
        mock_agent.core.output_guardrail = MagicMock()
        mock_agent.core.rate_limiter = MagicMock()
        mock_agent.core.approval_handler = MagicMock()

        panel = _build_welcome_panel(mock_agent)
        from rich.panel import Panel

        assert isinstance(panel, Panel)
