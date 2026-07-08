"""koboi/tui/commands.py -- Shared slash command registry for TUI and console."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from koboi.facade import KoboiAgent


@dataclass
class CommandContext:
    """Context passed to every slash command handler."""

    agent: KoboiAgent
    output: Callable[[str], None]
    args: str = ""
    app: Any = None  # KoboiApp instance (None in console mode)


@dataclass
class CommandResult:
    """Return value from a command handler."""

    message: str | None = None  # user message to send (triggers agent response)
    clear_chat: bool = False  # clear the chat area (TUI-specific)
    repopulate_messages: bool = False  # reload messages from memory after clear
    handled: bool = True


@dataclass
class SlashCommand:
    """Metadata for a registered command."""

    name: str
    description: str
    aliases: list[str] = field(default_factory=list)


class SlashCommandRegistry:
    """Registry of slash commands with dispatch."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, command: SlashCommand, handler: Callable) -> None:
        self._commands[command.name] = command
        self._handlers[command.name] = handler
        for alias in command.aliases:
            self._commands[alias] = command

    async def dispatch(self, name: str, ctx: CommandContext) -> CommandResult | None:
        cmd = self._commands.get(name)
        if cmd is None:
            return None
        handler = self._handlers.get(cmd.name)
        if handler is None:
            return None
        return await handler(ctx)

    def get_all_names(self) -> list[str]:
        return sorted(self._commands.keys())

    def get_help_text(self) -> str:
        seen: set[str] = set()
        lines: list[str] = []
        for cmd in self._commands.values():
            if cmd.name in seen:
                continue
            seen.add(cmd.name)
            alias_str = f" (alias: {', '.join(cmd.aliases)})" if cmd.aliases else ""
            lines.append(f"  {cmd.name:<16} {cmd.description}{alias_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def _cmd_reset(ctx: CommandContext) -> CommandResult:
    ctx.agent.reset()
    ctx.output("Conversation reset.")
    return CommandResult(clear_chat=True)


async def _cmd_info(ctx: CommandContext) -> CommandResult:
    config = ctx.agent.config
    core = ctx.agent.core
    lines = [f"Agent: {config.agent_name}"]
    lines.append(f"Model: {config.provider}/{config.model}")
    lines.append(f"Max iterations: {config.max_iterations}")
    if core:
        tool_names = list(core.tools.list_tools().keys())
        if tool_names:
            display = ", ".join(sorted(tool_names)[:8])
            if len(tool_names) > 8:
                display += f" ... (+{len(tool_names) - 8})"
            lines.append(f"Tools ({len(tool_names)}): {display}")
        guards = []
        if core.input_guardrail:
            guards.append("input")
        if core.output_guardrail:
            guards.append("output")
        if core.rate_limiter:
            guards.append("rate_limit")
        if core.approval_handler:
            guards.append("approval")
        if guards:
            lines.append(f"Guardrails: {', '.join(guards)}")
    if config.rag_enabled:
        lines.append("RAG: enabled")
    hooks_cfg = config.get("hooks", default={})
    if hooks_cfg:
        lines.append(f"Hooks: {', '.join(hooks_cfg.keys())}")
    ctx.output("\n".join(lines))
    return CommandResult()


async def _cmd_history(ctx: CommandContext) -> CommandResult:
    messages = ctx.agent.core.memory.get_messages()
    if not messages:
        ctx.output("No messages in history.")
        return CommandResult()
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "") or ""
        preview = content[:120] + "..." if len(content) > 120 else content
        lines.append(f"  [{i}] {role}: {preview}")
    ctx.output("\n".join(lines))
    return CommandResult()


async def _cmd_tools(ctx: CommandContext) -> CommandResult:
    tools_dict = ctx.agent.core.tools.list_tools()
    if not tools_dict:
        ctx.output("No tools registered.")
        return CommandResult()
    lines = ["Registered Tools:"]
    for name in sorted(tools_dict.keys()):
        td = tools_dict[name]
        risk = str(td.risk_level.value) if hasattr(td, "risk_level") else "safe"
        lines.append(f"  {name} ({risk})")
    ctx.output("\n".join(lines))
    return CommandResult()


async def _cmd_help(ctx: CommandContext) -> CommandResult:
    if ctx.app is not None:
        ctx.app.action_help_overlay()
    else:
        ctx.output(
            "Commands: /reset, /info, /history, /tools, /run, /compact,\n"
            "  /model, /editor, /undo, /copy, /diagnostics, /help"
        )
    return CommandResult()


async def _cmd_theme(ctx: CommandContext) -> CommandResult:
    if ctx.app is None:
        ctx.output("Theme switching requires the TUI.")
        return CommandResult()
    from koboi.tui.themes import THEMES

    themes = list(THEMES.keys())
    idx = themes.index(ctx.app.theme) if ctx.app.theme in themes else 0
    ctx.app.theme = themes[(idx + 1) % len(themes)]
    ctx.output(f"Theme: {ctx.app.theme}")
    return CommandResult()


async def _cmd_sessions(ctx: CommandContext) -> CommandResult:
    if ctx.app is None:
        ctx.output("Session manager requires the TUI.")
        return CommandResult()
    ctx.app.action_session_manager()
    return CommandResult()


async def _cmd_fork(ctx: CommandContext) -> CommandResult:
    mem = ctx.agent.core.memory
    if not hasattr(mem, "fork_session") or not hasattr(mem, "db_path"):
        ctx.output("Fork requires SQLite memory backend.")
        return CommandResult()
    new_id = mem.fork_and_switch()  # type: ignore[attr-defined]  # guarded by the hasattr checks above (SQLiteMemory-only method)
    ctx.output(f"Conversation forked. New session: {new_id[:8]}...")
    return CommandResult(clear_chat=True, repopulate_messages=True)


async def _cmd_export(ctx: CommandContext) -> CommandResult:
    fmt = ctx.args.strip().lower() or "md"
    if fmt not in ("md", "markdown", "json", "html"):
        ctx.output("Usage: /export [md|json|html]")
        return CommandResult()
    from datetime import datetime
    from pathlib import Path
    from koboi.tui.export import export_markdown, export_json, export_html

    messages = ctx.agent.core.memory.get_messages()
    metadata = {
        "agent_name": ctx.agent.config.agent_name,
        "model": f"{ctx.agent.config.provider}/{ctx.agent.config.model}",
    }
    exporters = {"md": export_markdown, "markdown": export_markdown, "json": export_json, "html": export_html}
    content = exporters[fmt](messages, metadata)
    ext = {"md": "md", "markdown": "md", "json": "json", "html": "html"}[fmt]
    filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
    Path(filename).write_text(content)
    ctx.output(f"Exported to {filename}")
    return CommandResult()


async def _cmd_skills(ctx: CommandContext) -> CommandResult:
    core = ctx.agent.core
    skills_reg = getattr(core, "skills", None) if core else None
    if not skills_reg:
        ctx.output("No skills configured.")
        return CommandResult()
    skill_list = skills_reg.list_skills()
    if not skill_list:
        ctx.output("No skills discovered.")
        return CommandResult()
    lines = [f"Discovered Skills ({len(skill_list)}):"]
    for s in skill_list:
        desc = s.description[:80] + "..." if len(s.description) > 80 else s.description
        lines.append(f"  {s.name}: {desc}")
    ctx.output("\n".join(lines))
    return CommandResult()


async def _cmd_mode(ctx: CommandContext) -> CommandResult:
    from koboi.modes import ModeManager, AgentMode

    if not ctx.args:
        if ctx.app is not None:
            current = ctx.app._mode_manager.current_mode.value
        else:
            current = ctx.agent.mode_manager.current_mode.value if ctx.agent.mode_manager else "unknown"
        ctx.output(f"Current mode: {current.upper()}")
        return CommandResult()
    try:
        new_mode = ModeManager.from_string(ctx.args)
    except ValueError as e:
        ctx.output(str(e))
        return CommandResult()

    if new_mode == AgentMode.YOLO:
        confirmed = False
        if ctx.app is not None:
            from koboi.tui.screens.yolo_confirm import YoloConfirmDialog

            confirmed = await ctx.app.push_screen_wait(YoloConfirmDialog())
        else:
            ctx.output(
                "WARNING: YOLO mode bypasses rate limiting, approval prompts, "
                "and mode restrictions. Only hardcoded safety checks "
                "(sensitive paths, dangerous commands) remain active."
            )
            try:
                answer = input("Activate YOLO mode? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                pass
            else:
                confirmed = answer in ("y", "yes")
        if not confirmed:
            ctx.output("YOLO mode activation cancelled.")
            return CommandResult()

    if ctx.app is not None:
        ctx.app._mode_manager.switch_mode(new_mode)
        mode_str = new_mode.value
        ctx.app.query_one("#header-bar").mode = mode_str.upper()
        ctx.app.query_one("#status-bar").mode = mode_str
    elif ctx.agent.mode_manager:
        ctx.agent.mode_manager.switch_mode(new_mode)
    ctx.output(f"Switched to {new_mode.value.upper()} mode.")
    return CommandResult()


async def _cmd_tasks(ctx: CommandContext) -> CommandResult:
    core = ctx.agent.core
    mgr = core.tools.get_dep("task_manager") if core else None
    if not mgr:
        ctx.output("Task management not initialized. Add task tools to your config.")
        return CommandResult()
    status_filter = ctx.args.strip() if ctx.args.strip() in ("pending", "in_progress", "completed", "blocked") else None
    tasks = mgr.list_tasks(status_filter=status_filter)
    if not tasks:
        ctx.output("No tasks.")
    else:
        lines = [f"Tasks ({len(tasks)}):"]
        for t in tasks:
            desc = f" - {t.description[:60]}" if t.description else ""
            dep_info = f" [blocked by: {', '.join(t.blocked_by)}]" if t.blocked_by else ""
            lines.append(f"  [{t.status}] {t.id}: {t.subject}{desc}{dep_info}")
        ctx.output("\n".join(lines))
    return CommandResult()


async def _cmd_compact(ctx: CommandContext) -> CommandResult:
    core = ctx.agent.core
    if not core or not core.context_manager:
        ctx.output("No context strategy configured. Set context.strategy in YAML.")
        return CommandResult()
    from koboi.tokens import estimate_tokens

    messages = core.memory.get_messages()
    tokens_before = estimate_tokens(messages)
    compacted = await core.context_manager.manage(messages, max_tokens=0)
    tokens_after = estimate_tokens(compacted)
    removed = len(messages) - len(compacted)
    core.memory.replace_messages(compacted)
    ctx.output(
        f"Compacted: {len(messages)} -> {len(compacted)} messages "
        f"({tokens_before} -> {tokens_after} tokens, {removed} removed)"
    )
    return CommandResult()


async def _cmd_model(ctx: CommandContext) -> CommandResult:
    # Resolve client from core or orchestrator
    if ctx.agent.core is not None:
        client = ctx.agent.core.client
        client_attr = "core"
    elif ctx.agent._orchestrator is not None:
        client = ctx.agent._orchestrator.client
        client_attr = "orchestrator"
    else:
        ctx.output("No agent core or orchestrator available.")
        return CommandResult()
    if not ctx.args:
        ctx.output(f"Current model: {client.provider}/{client.model}")
        return CommandResult()
    raw = ctx.args.strip()
    try:
        from koboi.client import Client

        old_client = client
        # Parse "provider/model" format
        if "/" in raw:
            new_provider, new_model = raw.split("/", 1)
        else:
            new_provider = old_client.provider
            new_model = raw
        if new_provider != old_client.provider:
            # Provider changed — let Client re-resolve api_key/base_url for new provider
            new_client = Client(
                model=new_model,
                logger=old_client.logger,
                provider=new_provider,
                temperature=old_client.temperature,
            )
        else:
            new_client = Client(
                api_key=old_client.api_key,
                base_url=old_client.base_url,
                model=new_model,
                logger=old_client.logger,
                provider=old_client.provider,
                temperature=old_client.temperature,
            )
        if client_attr == "core":
            ctx.agent.core.client = new_client
        else:
            ctx.agent._orchestrator.client = new_client
            # Also update all specialist agents' clients
            for agent in ctx.agent._orchestrator._agents_map.values():
                agent.client = new_client
        ctx.agent.config.raw.setdefault("llm", {})["model"] = new_model
        ctx.agent.config.raw.setdefault("llm", {})["provider"] = new_provider
        if ctx.app is not None:
            ctx.app.query_one("#header-bar").model = f"{new_provider}/{new_model}"
        ctx.output(f"Switched model to: {new_provider}/{new_model}")
    except Exception as e:
        ctx.output(f"Error switching model: {e}")
    return CommandResult()


async def _cmd_editor(ctx: CommandContext) -> CommandResult:
    import os
    import subprocess
    import tempfile

    editor = os.environ.get("EDITOR", "vim")
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("")
        tmp_path = f.name
    try:
        result = subprocess.run([editor, tmp_path])
        if result.returncode == 0:
            content = open(tmp_path).read().strip()
            if content:
                return CommandResult(message=content)
            ctx.output("Editor closed with empty content.")
        else:
            ctx.output("Editor exited with error.")
    finally:
        os.unlink(tmp_path)
    return CommandResult()


async def _cmd_undo(ctx: CommandContext) -> CommandResult:
    import subprocess as _subprocess

    n = 1
    if ctx.args:
        try:
            n = int(ctx.args.strip())
        except ValueError:
            ctx.output("Usage: /undo [n] (n = number of commits to revert)")
            return CommandResult()
    if n < 1 or n > 10:
        ctx.output("Can only revert 1-10 commits at a time.")
        return CommandResult()
    try:
        log_result = _subprocess.run(["git", "log", f"-{n}", "--oneline"], capture_output=True, text=True, timeout=10)  # nosec B607 - intentional PATH-based launch of a user tool/editor
        if log_result.returncode != 0:
            ctx.output(f"Git error: {log_result.stderr.strip()}")
            return CommandResult()
        ctx.output(f"Reverting last {n} commit(s):\n{log_result.stdout.strip()}")
    except Exception as e:
        ctx.output(f"Git error: {e}")
        return CommandResult()
    reverted = 0
    for i in range(n):
        try:
            result = _subprocess.run(["git", "revert", "HEAD", "--no-edit"], capture_output=True, text=True, timeout=30)  # nosec B607 - intentional PATH-based launch of a user tool/editor
            if result.returncode == 0:
                reverted += 1
            else:
                ctx.output(f"Revert failed at commit {i + 1}: {result.stderr.strip()}")
                break
        except Exception as e:
            ctx.output(f"Revert error: {e}")
            break
    ctx.output(f"Reverted {reverted}/{n} commit(s).")
    return CommandResult()


async def _cmd_vim(ctx: CommandContext) -> CommandResult:
    if ctx.app is None:
        ctx.output("Vim mode requires the TUI.")
        return CommandResult()
    input_box = ctx.app.query_one("#input-box")
    input_box.vim_enabled = not input_box.vim_enabled
    if input_box.vim_enabled:
        input_box.vim_mode = "normal"
    else:
        input_box.vim_mode = "insert"
    status = ctx.app.query_one("#status-bar")
    status.vim_enabled = input_box.vim_enabled
    status.vim_mode = input_box.vim_mode
    state = "ON (normal mode)" if input_box.vim_enabled else "OFF"
    ctx.output(f"Vim mode: {state}")
    return CommandResult()


async def _cmd_copy(ctx: CommandContext) -> CommandResult:
    messages = ctx.agent.core.memory.get_messages()
    last_assistant = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            last_assistant = msg["content"]
            break
    if not last_assistant:
        ctx.output("No assistant message to copy.")
        return CommandResult()
    copied = False
    try:
        import pyperclip

        pyperclip.copy(last_assistant)
        copied = True
    except ImportError:
        pass
    if not copied:
        import shutil
        import subprocess as _sp

        if shutil.which("pbcopy"):
            _sp.run(["pbcopy"], input=last_assistant.encode(), check=True)  # nosec B607 - intentional PATH-based launch of a user tool/editor
            copied = True
        elif shutil.which("xclip"):
            _sp.run(["xclip", "-selection", "clipboard"], input=last_assistant.encode(), check=True)  # nosec B607 - intentional PATH-based launch of a user tool/editor
            copied = True
    if copied:
        preview = last_assistant[:80] + "..." if len(last_assistant) > 80 else last_assistant
        ctx.output(f"Copied to clipboard: {preview}")
    else:
        ctx.output("No clipboard backend found. Install pyperclip or xclip.")
    return CommandResult()


async def _cmd_run(ctx: CommandContext) -> CommandResult:
    if not ctx.args:
        ctx.output("Usage: /run <config_path> [message]")
        return CommandResult()
    run_parts = ctx.args.split(maxsplit=1)
    config_path = run_parts[0]
    run_message = run_parts[1] if len(run_parts) > 1 else ""
    from pathlib import Path

    if not Path(config_path).exists():
        ctx.output(f"Config not found: {config_path}")
        return CommandResult()
    try:
        from koboi.facade import KoboiAgent

        new_agent = KoboiAgent.from_config(config_path)
        if ctx.app is not None:
            # TUI mode: swap agent reference, close old agent
            old_agent = ctx.app._agent
            ctx.app._agent = new_agent
            ctx.app._mode_manager = new_agent.mode_manager or ctx.app._mode_manager
            ctx.app._setup_tui_approval()
            ctx.app.query_one("#header-bar").agent_name = new_agent.config.agent_name
            ctx.app.query_one("#header-bar").model = f"{new_agent.config.provider}/{new_agent.config.model}"
            import asyncio

            asyncio.get_event_loop().create_task(old_agent.close())
        else:
            # Console mode: update original agent in-place
            ctx.agent.replace_from(new_agent)
        ctx.output(f"Loaded config: {config_path}")
        if run_message:
            return CommandResult(message=run_message)
    except Exception as e:
        ctx.output(f"Error loading config: {e}")
    return CommandResult()


async def _cmd_kill(ctx: CommandContext) -> CommandResult:
    core = ctx.agent.core
    manager = core.tools.get_dep("subagent_manager") if core else None
    if not manager:
        ctx.output("Subagent system not initialized.")
        return CommandResult()
    if ctx.args:
        label = ctx.args.strip()
        if manager.cancel_task(label):
            ctx.output(f"Cancelled subagent: {label}")
        else:
            running = manager.list_running()
            if running:
                ctx.output(f"No running subagent '{label}'. Running: {', '.join(running)}")
            else:
                ctx.output(f"No running subagent '{label}'. No subagents active.")
    else:
        running = manager.list_running()
        if running:
            for label in running:
                manager.cancel_task(label)
            ctx.output(f"Cancelled {len(running)} subagent(s).")
        else:
            ctx.output("No running subagents to cancel.")
    return CommandResult()


async def _cmd_subagents(ctx: CommandContext) -> CommandResult:
    if ctx.app is not None and ctx.app._agent_states:
        ctx.app.action_subagent_monitor()
        return CommandResult()
    core = ctx.agent.core
    manager = core.tools.get_dep("subagent_manager") if core else None
    if not manager:
        ctx.output("Subagent system not initialized.")
        return CommandResult()
    running = manager.list_running()
    if running:
        lines = [f"Running subagents ({len(running)}):"]
        for label in running:
            lines.append(f"  - {label}")
        lines.append("\nUse /kill <label> or /kill to cancel.")
        ctx.output("\n".join(lines))
    else:
        ctx.output("No subagents currently running.")
    return CommandResult()


async def _cmd_diagnostics(ctx: CommandContext) -> CommandResult:
    from datetime import datetime
    from pathlib import Path
    from koboi.diagnostics import collect_diagnostics

    try:
        data = collect_diagnostics(ctx.agent)
        filename = f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        Path(filename).write_bytes(data)
        size_kb = len(data) / 1024
        ctx.output(f"Diagnostics exported to {filename} ({size_kb:.1f} KB)")
    except Exception as e:
        ctx.output(f"Error generating diagnostics: {e}")
    return CommandResult()


async def _cmd_quit(ctx: CommandContext) -> CommandResult:
    """Exit the TUI."""
    if ctx.app is not None:
        ctx.app.exit()
    else:
        raise SystemExit(0)
    return CommandResult()


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def build_registry() -> SlashCommandRegistry:
    """Build the default slash command registry with all built-in commands."""
    reg = SlashCommandRegistry()

    commands = [
        (SlashCommand("/reset", "Clear conversation memory"), _cmd_reset),
        (SlashCommand("/info", "Show agent configuration"), _cmd_info),
        (SlashCommand("/history", "Show conversation history"), _cmd_history),
        (SlashCommand("/tools", "List registered tools"), _cmd_tools),
        (SlashCommand("/help", "Show available commands"), _cmd_help),
        (SlashCommand("/theme", "Cycle color theme"), _cmd_theme),
        (SlashCommand("/sessions", "Open session manager"), _cmd_sessions),
        (SlashCommand("/fork", "Fork current session"), _cmd_fork),
        (SlashCommand("/export", "Export conversation [md|json|html]"), _cmd_export),
        (SlashCommand("/skills", "List discovered skills"), _cmd_skills),
        (SlashCommand("/mode", "Show/switch interaction mode"), _cmd_mode),
        (SlashCommand("/tasks", "List tasks"), _cmd_tasks),
        (SlashCommand("/compact", "Compact context window"), _cmd_compact),
        (SlashCommand("/model", "Show/switch LLM model"), _cmd_model),
        (SlashCommand("/editor", "Open $EDITOR for long messages", aliases=["/edit"]), _cmd_editor),
        (SlashCommand("/undo", "Revert last N git commits"), _cmd_undo),
        (SlashCommand("/vim", "Toggle vim mode"), _cmd_vim),
        (SlashCommand("/copy", "Copy last response to clipboard"), _cmd_copy),
        (SlashCommand("/run", "Hot-load a YAML config"), _cmd_run),
        (SlashCommand("/kill", "Cancel running subagents"), _cmd_kill),
        (SlashCommand("/subagents", "List/monitor subagents"), _cmd_subagents),
        (SlashCommand("/diagnostics", "Export session diagnostics"), _cmd_diagnostics),
        (SlashCommand("/quit", "Exit the TUI", aliases=["/exit"]), _cmd_quit),
    ]

    for cmd, handler in commands:
        reg.register(cmd, handler)

    return reg
