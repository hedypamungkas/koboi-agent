# koboi/tui/ -- Terminal UI (Textual)

## What this is
Textual-based terminal UI for interactive agent chat. Entry point: `app.py:main`.

## Key files
```
app.py              CLI entry point (click + Textual setup), main() function
commands.py          Slash-command registry (/help, /reset, /mode, /skills, etc.)
approval.py          CLI approval handler bridge (guardrails.approval=cli)
textual_app.py      Textual App subclass, screen management
bridge.py           Async bridge between Textual event loop and AgentCore
loop.py             TUI-specific agent loop wrapper
export.py           Conversation export (markdown/JSON)
notifications.py    Toast notification system
themes.py           Theme definitions (koboi-dark, koboi-light)
app.tcss            Textual CSS stylesheet
```

## Screens (`screens/`)
```
command_palette.py    Slash-command picker (/help, /reset, /mode, etc.)
help_overlay.py       Keyboard shortcut reference
history_search.py     Search through conversation history
permission_dialog.py  Tool approval dialog (for guardrails.approval=cli)
session_manager.py    Multi-session management
subagent_monitor.py   Sub-agent activity display
transcript_viewer.py  Full transcript view
welcome_screen.py     First-run welcome
yolo_confirm.py       YOLO-mode confirmation dialog (/mode yolo)
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
- All widgets inherit from Textual `Widget` or `Static`
- Screens are registered in `textual_app.py`
- Bridge pattern: `bridge.py` wraps async `AgentCore` calls for Textual's sync event loop
- Themes defined as Textual CSS variables in `themes.py`
- Widget exports in `widgets/__init__.py`: `ChatLog`, `InputBox`, `MessageBubble`, `StatusBar`, etc.
