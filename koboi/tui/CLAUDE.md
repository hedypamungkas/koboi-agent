# koboi/tui/ -- Terminal UI (Textual)

## What this is
Textual-based terminal UI for interactive agent chat. The console-script
dispatch lives in `koboi/cli.py`; this package owns only the *interactive*
surface, exposed via `app.py:run_chat_interactive` (lazy-imported by
`cli._run_chat` for `koboi chat` without `--print`). `koboi chat --print`
(JSON-line output) is handled core-only in `koboi/cli_commands.py` and does
NOT touch this package. Requires the `[tui]` extra (`rich` for the legacy
`--no-tui` loop, `textual` for the default app).

## Key files
```
app.py              Interactive chat surface: run_chat_interactive() + _build_welcome_panel() + legacy _run_interactive() Rich loop (no click; dispatch is in cli.py)
commands.py          Slash-command registry (/help, /reset, /mode, /skills, /capture, etc.)
approval.py          TUIApprovalHandler -- Textual message-based handler that replaces CLIApprovalHandler's stdin prompt; swapped in by textual_app when guardrails.approval.handler is cli or callback
textual_app.py      Textual App subclass, screen management
keybindings.py       Configurable keybindings loaded from the YAML `keybindings:` section (14 defaults); load_keybindings()/get_keybinding_display()
bridge.py           Async bridge between Textual event loop and AgentCore
loop.py             TUI-specific agent loop wrapper
export.py           Conversation export (markdown/JSON/HTML)
notifications.py    Backward-compat re-export shim (real impl moved to koboi/notifications.py)
themes.py           Theme definitions (koboi-dark, koboi-light)
app.tcss            Textual CSS stylesheet
```

## Screens (`screens/`)
```
command_palette.py    Slash-command picker (/help, /reset, /mode, etc.)
help_overlay.py       Keyboard shortcut reference
history_search.py     Search through conversation history
permission_dialog.py  Tool approval dialog (modal launched by the TUIApprovalHandler flow; active when guardrails.approval.handler is cli/callback)
session_manager.py    Multi-session management
mcp_status.py         MCP server connection status (f2)
subagent_monitor.py   Sub-agent activity display
transcript_viewer.py  Full transcript view
welcome_screen.py     First-run welcome
yolo_confirm.py       YOLO-mode confirmation dialog (/mode yolo)
media_gallery.py      Generated media artifacts gallery (W5c; opened via the F3 keybinding)
```

## Widgets (`widgets/`)
```
chat_log.py           Main chat message list
message_bubble.py     Individual message rendering
input_box.py          User input with slash-command completion (exports ChatSubmit)
tool_call.py          Tool execution display
thinking_block.py     Collapsible thinking/reasoning display
diff_view.py          Side-by-side diff for file changes
file_suggester.py     File path autocomplete
plan_view.py          Plan mode display
risk_bar.py           Risk level indicator
slash_suggester.py    Slash command autocomplete
header_bar.py         Top bar with agent name/model
status_bar.py         Bottom status bar
```

## Conventions
- Widgets inherit from Textual `Widget` subclasses: `Widget`, `Static`, `VerticalScroll`, `Vertical`, `Input`, `Suggester`
- Screens are pushed on demand via `push_screen` from `textual_app.py` (most) and `commands.py` (yolo_confirm)
- Bridge pattern: `bridge.py` wraps async `AgentCore` calls for Textual's sync event loop
- Themes defined as Textual `Theme` objects (hex colors) in `themes.py`, registered via `register_themes(app)`; CSS rules live in `app.tcss`
- Widget exports in `widgets/__init__.py`: `ChatLog`, `InputBox`, `MessageBubble`, `StatusBar`, etc.
