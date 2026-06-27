"""textual_app.py -- Full-screen Textual application for koboi-agent."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer
from textual.worker import Worker

from koboi.modes import ModeManager
from koboi.tui.commands import CommandContext, build_registry
from koboi.tui.bridge import (
    StreamAgentDispatch,
    StreamAgentResult,
    StreamBridge,
    StreamComplete,
    StreamDelta,
    StreamError,
    StreamIteration,
    StreamOrchestrationComplete,
    StreamRoutingDecision,
    StreamToolCall,
    StreamToolResult,
)
from koboi.tui.widgets.chat_log import ChatLog
from koboi.tui.widgets.file_suggester import CompositeSuggester, FileSuggester
from koboi.tui.widgets.header_bar import HeaderBar
from koboi.tui.widgets.input_box import ChatSubmit, InputBox, VimModeChanged
from koboi.tui.widgets.slash_suggester import SlashSuggester
from koboi.tui.widgets.status_bar import StatusBar
from koboi.tui.widgets.tool_call import ToolCallWidget

if TYPE_CHECKING:
    from koboi.facade import KoboiAgent


class KoboiApp(App):
    """Full-screen Textual TUI for koboi-agent."""

    CSS_PATH = "app.tcss"

    # BINDINGS are loaded dynamically from config in __init__.
    # The class-level BINDINGS is a fallback; instance BINDINGS override it.
    BINDINGS = []

    def __init__(self, agent: KoboiAgent, **kwargs):
        super().__init__(**kwargs)
        self._agent = agent
        self._bridge = StreamBridge(self)
        self._turn_count = 0
        self._streaming = False
        self._current_task: Worker | None = None
        self._start_time: float = 0.0
        self._history: list[str] = []
        self._history_max = 500
        self._app_focused = True
        self._agent_states: dict[str, dict] = {}

        # Notification settings from config
        notif_conf = agent.config.get("harness", "notifications", default={})
        self._notify_enabled = notif_conf.get("enabled", True) if isinstance(notif_conf, dict) else True
        self._notify_sound = notif_conf.get("sound", False) if isinstance(notif_conf, dict) else False
        self._notify_sound_name = notif_conf.get("sound_name", "Ping") if isinstance(notif_conf, dict) else "Ping"

        # Load keybindings from config (overrides class-level BINDINGS)
        from koboi.tui.keybindings import load_keybindings
        from textual.binding import BindingsMap

        self._bindings_list = load_keybindings(agent.config)
        self._bindings = BindingsMap(self._bindings_list)

        # Phase 4: Mode manager (from agent or create default)
        self._mode_manager: ModeManager = agent.mode_manager or ModeManager()

        # Phase 4: TUI approval handler (wired to agent's approval_handler)
        self._setup_tui_approval()

        # Shared slash command registry
        self._commands = build_registry()

    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield HeaderBar(
            agent_name=self._agent.config.agent_name,
            model=f"{self._agent.config.provider}/{self._agent.config.model}",
            id="header-bar",
        )
        with VerticalScroll(id="chat-container"):
            yield ChatLog(id="chat-area")
        yield StatusBar(id="status-bar")
        with Horizontal(id="input-container"):
            yield InputBox(
                placeholder="Type a message, /command, or @file...",
                suggester=CompositeSuggester(
                    slash_suggester=SlashSuggester(self._get_all_commands()),
                    file_suggester=FileSuggester(base_dir="."),
                ),
                id="input-box",
            )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input box on startup and wire history."""
        self._start_time = time.time()
        input_box = self.query_one("#input-box", InputBox)
        input_box.set_history(self._history)
        input_box.focus()

        # Wire subagent UI hook for TUI progress display
        self._setup_subagent_hook()

        # Phase 4: Sync initial mode to header/status bars
        mode_str = self._mode_manager.current_mode.value
        self.query_one("#header-bar", HeaderBar).mode = mode_str.upper()
        self.query_one("#status-bar", StatusBar).mode = mode_str

        # Set max_tokens from agent config
        try:
            max_ctx = self._agent.core.max_context_tokens if self._agent.core else 8000
            if isinstance(max_ctx, int):
                self.query_one("#status-bar", StatusBar).max_tokens = max_ctx
        except (AttributeError, TypeError):
            pass

        # Phase 5: Register themes
        from koboi.tui.themes import register_themes

        saved_theme = self._agent.config.get("agent", "theme", default="koboi-dark")
        register_themes(self, default=saved_theme)

        # Phase 5: Skill count in status bar
        skills_reg = getattr(self._agent.core, "skills", None)
        if skills_reg:
            try:
                count = len(skills_reg.list_skills())
                self.query_one("#status-bar", StatusBar).skill_count = count
            except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
                pass

        # Phase 5: First-run welcome screen
        if self._is_first_run():
            from koboi.tui.screens.welcome_screen import WelcomeScreen

            self.push_screen(
                WelcomeScreen(
                    agent_name=self._agent.config.agent_name,
                    model=f"{self._agent.config.provider}/{self._agent.config.model}",
                )
            )

        # Phase 5: Ensure session record for SQLite backend
        if self._agent.core is None:
            return
        mem = self._agent.core.memory
        if hasattr(mem, "ensure_session_record"):
            try:
                mem.ensure_session_record(
                    agent_name=self._agent.config.agent_name,
                    model=f"{self._agent.config.provider}/{self._agent.config.model}",
                )
            except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
                pass

    def _setup_tui_approval(self) -> None:
        """Replace the agent's approval handler with a TUI-aware one.

        Only installs if the config explicitly sets guardrails.approval.handler
        to "cli" or "callback". When handler is "auto" (default) or missing,
        no approval handler is installed — all tools auto-approve.
        """
        if self._agent.core is None:
            return
        handler_conf = self._agent.config.get("guardrails", "approval", default={})
        handler_type = handler_conf.get("handler", "auto") if isinstance(handler_conf, dict) else "auto"
        if handler_type not in ("cli", "callback"):
            self._tui_approval = None
            return
        from koboi.tui.approval import TUIApprovalHandler

        tui_handler = TUIApprovalHandler(
            app=self,
            trust_db=self._agent.trust_db,
            audit_trail=self._agent.core.audit_trail,
        )
        self._agent.core.approval_handler = tui_handler
        self._tui_approval = tui_handler

    def _setup_subagent_hook(self) -> None:
        """Register SubagentUIHook to show subagent progress in TUI."""
        if self._agent.core is None:
            return
        from koboi.hooks.subagent_hook import SubagentUIHook

        hook = SubagentUIHook(app=self)
        self._agent.core.hooks.add(hook)

    def on_exit(self) -> None:
        """Print session summary to stderr on exit."""
        elapsed = time.time() - self._start_time
        parts = [f"{self._turn_count} message(s)", f"{elapsed:.1f}s"]
        try:
            mem_len = len(self._agent.core.memory)
            parts.append(f"{mem_len} in memory")
        except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
            pass
        print(f"\nGoodbye! {' | '.join(parts)}", file=sys.stderr)

    def on_focus(self) -> None:
        self._app_focused = True
        # Re-focus input box when terminal regains focus (unless a modal is open)
        if len(self.screen_stack) <= 1:
            self.query_one("#input-box").focus()

    def on_blur(self) -> None:
        self._app_focused = False

    def on_click(self) -> None:
        """Keep input box focused when clicking anywhere on the main screen."""
        if len(self.screen_stack) <= 1:
            self.query_one("#input-box").focus()

    # -- Input handling --------------------------------------------------------

    async def on_chat_submit(self, event: ChatSubmit) -> None:
        """Handle user submitting a message."""
        message = event.value
        if not message:
            return

        # Add to history
        self._add_to_history(message)

        # Check for slash commands before adding user message
        if await self._execute_slash_command(message):
            return

        chat = self.query_one("#chat-area", ChatLog)
        chat.add_message("user", message)

        # Build multimodal content if images are present
        content_for_agent: str | list = message
        if event.images:
            content_for_agent = [{"type": "text", "text": message}]
            for img in event.images:
                content_for_agent.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img['media_type']};base64,{img['data']}"},
                    }
                )
            img_names = [img.get("path", "image").rsplit("/", 1)[-1] for img in event.images]
            chat.add_system_message(f"[dim]Attached: {', '.join(img_names)}[/dim]")

        self._streaming = True
        self.query_one("#input-box", InputBox).disabled = True

        status = self.query_one("#status-bar", StatusBar)
        status.state = "streaming"
        self._turn_count += 1
        status.turn_count = self._turn_count

        chat.begin_stream()

        self._current_task = self.run_worker(self._process_stream(content_for_agent), exit_on_error=False)

    async def _process_stream(self, message: str | list) -> None:
        """Background worker: consume agent stream and post bridge events."""
        try:
            stream = self._agent.run_stream(message)
            await self._bridge.process_stream(stream)
        except asyncio.CancelledError:
            chat = self.query_one("#chat-area", ChatLog)
            chat.finalize_stream("")
            chat.add_system_message("[dim]Response cancelled.[/dim]")
        except Exception as e:
            chat = self.query_one("#chat-area", ChatLog)
            chat.finalize_stream("")
            chat.add_error(e)
        finally:
            self._streaming = False
            self._current_task = None
            input_box = self.query_one("#input-box", InputBox)
            input_box.disabled = False
            input_box.focus()
            self.query_one("#status-bar", StatusBar).state = "idle"
            chat = self.query_one("#chat-area", ChatLog)
            chat.scroll_end(animate=False)

    # -- Slash commands --------------------------------------------------------

    async def _execute_slash_command(self, message: str) -> bool:
        """Execute a slash command if the message is one. Returns True if handled."""
        stripped = message.strip()
        if not stripped.startswith("/"):
            return False

        parts = stripped.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Buffer output so clear_chat can happen before messages appear
        output_buf: list[str] = []
        ctx = CommandContext(
            agent=self._agent,
            output=output_buf.append,
            args=args,
            app=self,
        )
        result = await self._commands.dispatch(command, ctx)
        if result is None:
            chat = self.query_one("#chat-area", ChatLog)
            chat.add_system_message(f"Unknown command: {command}. Type /help for available commands.")
            return True

        chat = self.query_one("#chat-area", ChatLog)

        # TUI-specific post-command actions
        if result.clear_chat:
            chat.clear_messages()
            if result.repopulate_messages and self._agent.core is not None:
                for msg in self._agent.core.memory.get_messages():
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ("user", "assistant") and content:
                        chat.add_message(role, content)

        # Flush buffered output
        for line in output_buf:
            chat.add_system_message(line)

        # Handle commands that return a message to process (editor, run)
        if result.message:
            chat.add_message("user", result.message)
            self._streaming = True
            self.query_one("#input-box", InputBox).disabled = True
            status = self.query_one("#status-bar", StatusBar)
            status.state = "streaming"
            self._turn_count += 1
            status.turn_count = self._turn_count
            chat.begin_stream()
            self._current_task = self.run_worker(self._process_stream(result.message), exit_on_error=False)
        return True

    # (Slash command logic moved to koboi/tui/commands.py)

    # -- Command history -------------------------------------------------------

    def _add_to_history(self, entry: str) -> None:
        """Add entry to command history, capping at max size."""
        if entry and (not self._history or self._history[-1] != entry):
            self._history.append(entry)
            if len(self._history) > self._history_max:
                self._history = self._history[-self._history_max :]

    def get_history(self) -> list[str]:
        """Return the command history list."""
        return self._history

    # -- Keyboard actions ------------------------------------------------------

    def action_command_palette(self) -> None:
        """Open the command palette."""
        from koboi.tui.screens.command_palette import CommandPaletteScreen

        def on_select(command: str | None) -> None:
            if command:
                input_box = self.query_one("#input-box", InputBox)
                input_box.value = command
                input_box.focus()

        self.push_screen(CommandPaletteScreen(self._get_all_commands()), on_select)

    def action_history_search(self) -> None:
        """Open history search."""
        from koboi.tui.screens.history_search import HistorySearchScreen

        def on_select(entry: str | None) -> None:
            if entry:
                input_box = self.query_one("#input-box", InputBox)
                input_box.value = entry
                input_box.focus()

        self.push_screen(HistorySearchScreen(self._history), on_select)

    def action_clear_chat(self) -> None:
        """Clear the chat area."""
        chat = self.query_one("#chat-area", ChatLog)
        chat.clear_messages()

    def action_cancel_or_quit(self) -> None:
        """Cancel current streaming or quit."""
        if self._streaming and self._current_task:
            self._current_task.cancel()
        else:
            self.exit()

    def action_kill_subagents(self) -> None:
        """Cancel all running subagents (ctrl+k)."""
        from koboi.tools.builtin.subagent import get_manager

        manager = get_manager()
        chat = self.query_one("#chat-area", ChatLog)
        if manager:
            running = manager.list_running()
            if running:
                self.run_worker(self._cancel_all_subagents(manager))
                return
        chat.add_system_message("No running subagents to cancel.")

    async def _cancel_all_subagents(self, manager) -> None:
        """Worker: cancel all running subagents."""
        count = await manager.cancel_all()
        chat = self.query_one("#chat-area", ChatLog)
        chat.add_system_message(f"Cancelled {count} subagent(s).")

    def action_subagent_monitor(self) -> None:
        """Open the sub-agent monitoring panel."""
        from koboi.tui.screens.subagent_monitor import SubagentMonitorScreen

        self.push_screen(SubagentMonitorScreen(self._agent_states))

    def action_focus_input(self) -> None:
        """Focus the input box."""
        self.query_one("#input-box").focus()

    def action_cycle_mode(self) -> None:
        """Cycle to the next interaction mode."""
        new_mode = self._mode_manager.cycle_mode()
        mode_str = new_mode.value
        self.query_one("#header-bar", HeaderBar).mode = mode_str.upper()
        self.query_one("#status-bar", StatusBar).mode = mode_str
        chat = self.query_one("#chat-area", ChatLog)
        chat.add_system_message(f"Switched to {mode_str.upper()} mode.")

    def action_toggle_all_tools(self) -> None:
        """Toggle collapse/expand all tool call widgets."""
        # Don't intercept Tab when input box has focus (for autocomplete)
        input_box = self.query_one("#input-box", InputBox)
        if input_box.has_focus:
            return
        chat = self.query_one("#chat-area", ChatLog)
        any_expanded = any(not w.collapsed for w in chat.query(ToolCallWidget))
        if any_expanded:
            chat.collapse_all_tools()
        else:
            chat.expand_all_tools()

    # -- Phase 5 actions --------------------------------------------------------

    def action_cycle_theme(self) -> None:
        """Cycle to the next color theme."""
        from koboi.tui.themes import THEMES

        themes = list(THEMES.keys())
        idx = themes.index(self.theme) if self.theme in themes else 0
        self.theme = themes[(idx + 1) % len(themes)]

    def action_session_manager(self) -> None:
        """Open the session manager."""
        if self._agent.core is None:
            chat = self.query_one("#chat-area", ChatLog)
            chat.add_system_message("Session manager not available in orchestrated mode.")
            return
        from koboi.tui.screens.session_manager import SessionManagerScreen

        mem = self._agent.core.memory
        if not hasattr(mem, "db_path"):
            chat = self.query_one("#chat-area", ChatLog)
            chat.add_system_message("Session manager requires SQLite memory backend.")
            return

        def on_result(session_id: str | None) -> None:
            if session_id:
                self._resume_session(session_id)

        self.push_screen(SessionManagerScreen(mem.db_path), on_result)

    def _resume_session(self, session_id: str) -> None:
        """Load a previous session into the current chat."""
        if self._agent.core is None:
            return
        from koboi.memory_sqlite import SQLiteMemory

        mem = self._agent.core.memory
        if not isinstance(mem, SQLiteMemory):
            return
        messages = SQLiteMemory.get_session_messages(mem.db_path, session_id)
        chat = self.query_one("#chat-area", ChatLog)
        chat.clear_messages()
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                chat.add_message(role, content)
        mem.switch_session(session_id)
        chat.add_system_message(f"Resumed session {session_id[:8]}...")

    def action_transcript_viewer(self) -> None:
        """Open the transcript viewer."""
        if self._agent.core is None:
            chat = self.query_one("#chat-area", ChatLog)
            chat.add_system_message("Transcript viewer not available in orchestrated mode.")
            return
        from koboi.tui.screens.transcript_viewer import TranscriptViewerScreen

        messages = self._agent.core.memory.get_messages()
        self.push_screen(TranscriptViewerScreen(messages))

    def action_help_overlay(self) -> None:
        """Open the help overlay."""
        from koboi.tui.screens.help_overlay import HelpOverlayScreen
        from koboi.tui.keybindings import get_keybinding_display

        self.push_screen(
            HelpOverlayScreen(
                commands=self._get_all_commands(),
                bindings=self._bindings_list,
                current_mode=self._mode_manager.current_mode.value,
                keybinding_display=get_keybinding_display(self._agent.config),
            )
        )

    def _is_first_run(self) -> bool:
        """Check if this is the first run (no sessions in DB)."""
        if self._agent.core is None:
            return False
        try:
            mem = self._agent.core.memory
            if hasattr(mem, "db_path"):
                from koboi.memory_sqlite import SQLiteMemory

                sessions = SQLiteMemory.list_sessions(mem.db_path, limit=1)
                return len(sessions) == 0
        except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
            pass
        return False

    def _get_all_commands(self) -> list[str]:
        """Return all available slash commands for the palette."""
        return self._commands.get_all_names()

    # -- Permission dialog (Phase 4) -------------------------------------------

    def on_permission_request(self, event) -> None:
        """Handle permission request from TUIApprovalHandler."""
        from koboi.tui.screens.permission_dialog import PermissionDialog

        status = self.query_one("#status-bar", StatusBar)
        status.state = "waiting_approval"

        def on_result(result):
            from koboi.tui.approval import PermissionResponse

            if result is None:
                # User dismissed dialog (Esc)
                result_obj = PermissionResponse(approved=False, always_allow=False)
            else:
                result_obj = PermissionResponse(
                    approved=result.approved,
                    always_allow=result.always_allow,
                )
            self._tui_approval.resolve_pending(result_obj)
            status.state = "streaming" if self._streaming else "idle"

        self.push_screen(
            PermissionDialog(
                tool_name=event.tool_name,
                arguments=event.arguments,
                risk_level=event.risk_level,
            ),
            on_result,
        )

    # -- Bridge message handlers ----------------------------------------------

    def on_stream_delta(self, event: StreamDelta) -> None:
        chat = self.query_one("#chat-area", ChatLog)
        chat.append_delta(event.content)

    def on_stream_tool_call(self, event: StreamToolCall) -> None:
        chat = self.query_one("#chat-area", ChatLog)
        chat.add_tool_call(event.tool_name, event.tool_call_id, event.arguments)
        status = self.query_one("#status-bar", StatusBar)
        status.state = "running_tool"
        status.current_tool = event.tool_name

    def on_stream_tool_result(self, event: StreamToolResult) -> None:
        chat = self.query_one("#chat-area", ChatLog)
        chat.update_tool_result(event.tool_call_id, event.result)
        status = self.query_one("#status-bar", StatusBar)
        status.state = "streaming"
        status.current_tool = ""
        if not self._app_focused and self._notify_enabled:
            from koboi.notifications import notify

            notify("Koboi Agent", "Tool completed", sound=self._notify_sound)

    def on_stream_iteration(self, event: StreamIteration) -> None:
        status = self.query_one("#status-bar", StatusBar)
        status.tokens_used = event.tokens_estimated
        status.iteration = event.iteration
        try:
            if isinstance(status.max_tokens, int) and status.max_tokens > 0:
                status.context_pct = min(100.0, (event.tokens_estimated / status.max_tokens) * 100)
        except (TypeError, AttributeError):
            pass
        chat = self.query_one("#chat-area", ChatLog)
        chat.add_iteration_marker(event.iteration, event.messages_count)

        try:
            from koboi.tools.builtin.task import get_manager

            mgr = get_manager()
            status.task_summary = mgr.summary_short()
        except (RuntimeError, ImportError):
            pass

    def on_stream_complete(self, event: StreamComplete) -> None:
        chat = self.query_one("#chat-area", ChatLog)
        chat.finalize_stream(event.content)
        status = self.query_one("#status-bar", StatusBar)
        status.state = "idle"
        try:
            from koboi.tools.builtin.task import get_manager

            mgr = get_manager()
            status.task_summary = mgr.summary_short()
        except (RuntimeError, ImportError):
            pass
        if not self._app_focused and self._notify_enabled:
            from koboi.notifications import notify

            notify("Koboi Agent", "Response complete", sound=self._notify_sound)

    def on_stream_error(self, event: StreamError) -> None:
        chat = self.query_one("#chat-area", ChatLog)
        chat.add_error(event.error)
        status = self.query_one("#status-bar", StatusBar)
        status.state = "idle"

    def on_stream_routing_decision(self, event: StreamRoutingDecision) -> None:
        chat = self.query_one("#chat-area", ChatLog)
        chat.add_routing_decision(event.agents, event.method, event.confidence)
        status = self.query_one("#status-bar", StatusBar)
        status.state = "orchestrating"
        status.orchestration_agents = len(event.agents)
        # Reset agent monitor state for new orchestration
        self._agent_states.clear()
        for i, agent_name in enumerate(event.agents):
            self._agent_states[agent_name] = {
                "name": agent_name,
                "status": "pending",
                "index": i,
                "elapsed": 0.0,
                "answer_preview": "",
                "is_dynamic": False,
                "domain_label": None,
            }

    def on_stream_agent_dispatch(self, event: StreamAgentDispatch) -> None:
        chat = self.query_one("#chat-area", ChatLog)
        chat.add_agent_status(event.agent_name, "running")
        status = self.query_one("#status-bar", StatusBar)
        status.orchestration_current = event.agent_name
        # Update agent monitor
        if event.agent_name in self._agent_states:
            self._agent_states[event.agent_name]["status"] = "running"

    def on_stream_agent_result(self, event: StreamAgentResult) -> None:
        chat = self.query_one("#chat-area", ChatLog)
        status_text = "done" if not event.failed else "failed"
        chat.add_agent_status(event.agent_name, status_text, event.elapsed_seconds)
        status = self.query_one("#status-bar", StatusBar)
        status.orchestration_completed += 1
        status.orchestration_current = ""
        # Update agent monitor
        if event.agent_name in self._agent_states:
            self._agent_states[event.agent_name]["status"] = status_text
            self._agent_states[event.agent_name]["elapsed"] = event.elapsed_seconds
            self._agent_states[event.agent_name]["answer_preview"] = event.answer or ""

    def on_stream_orchestration_complete(self, event: StreamOrchestrationComplete) -> None:
        chat = self.query_one("#chat-area", ChatLog)
        chat.move_streaming_bubble_to_end()
        chat.finalize_stream(event.final_answer)
        chat.scroll_end(animate=False)
        status = self.query_one("#status-bar", StatusBar)
        status.state = "idle"
        status.orchestration_agents = 0
        status.orchestration_completed = 0
        status.orchestration_current = ""
        # Agent states kept for review; cleared on next routing decision

    # -- Subagent message handlers (from SubagentUIHook) ----------------------

    def on_subagent_dispatch(self, event) -> None:
        """Handle subagent dispatch from SubagentUIHook."""
        from koboi.hooks.subagent_hook import _SubagentDispatch

        if not isinstance(event, _SubagentDispatch):
            return
        chat = self.query_one("#chat-area", ChatLog)
        chat.add_agent_status(event.label, "running")
        status = self.query_one("#status-bar", StatusBar)
        status.state = "orchestrating"
        status.orchestration_current = event.label
        status.orchestration_agents = event.total
        status.orchestration_completed = event.index
        # Update agent monitor
        self._agent_states[event.label] = {
            "name": event.label,
            "status": "running",
            "index": event.index,
            "elapsed": 0.0,
            "answer_preview": "",
            "is_dynamic": False,
            "domain_label": None,
        }

    def on_subagent_result(self, event) -> None:
        """Handle subagent result from SubagentUIHook."""
        from koboi.hooks.subagent_hook import _SubagentResult

        if not isinstance(event, _SubagentResult):
            return
        chat = self.query_one("#chat-area", ChatLog)
        status_text = "done" if event.success else "failed"
        chat.add_agent_status(event.label, status_text, event.elapsed)
        status = self.query_one("#status-bar", StatusBar)
        status.orchestration_completed += 1
        status.orchestration_current = ""
        # Update agent monitor
        if event.label in self._agent_states:
            self._agent_states[event.label]["status"] = status_text
            self._agent_states[event.label]["elapsed"] = event.elapsed
            if event.error:
                self._agent_states[event.label]["answer_preview"] = event.error

    # -- Vim mode handler ----------------------------------------------------

    def on_vim_mode_changed(self, event: VimModeChanged) -> None:
        """Sync vim mode to the status bar."""
        status = self.query_one("#status-bar", StatusBar)
        status.vim_mode = event.mode
