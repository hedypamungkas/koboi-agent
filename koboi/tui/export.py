"""export.py -- Conversation export formatters (markdown, JSON, HTML)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def export_markdown(messages: list[dict[str, Any]], metadata: dict[str, str] | None = None) -> str:
    """Export conversation as Markdown."""
    meta = metadata or {}
    lines = [
        "# Conversation Export",
        "",
        f"- **Agent:** {meta.get('agent_name', 'unknown')}",
        f"- **Model:** {meta.get('model', 'unknown')}",
        f"- **Exported:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "---",
        "",
    ]
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or ""
        if role == "system":
            continue
        if role == "user":
            lines.append(f"## User\n\n{content}\n")
        elif role == "assistant":
            lines.append(f"## Assistant\n\n{content}\n")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    lines.append(f"<details><summary>Tool Call: {fn.get('name', '?')}</summary>\n")
                    lines.append(f"```json\n{fn.get('arguments', '{}')}\n```\n</details>\n")
        elif role == "tool":
            lines.append(f"<details><summary>Tool Result</summary>\n\n```\n{content}\n```\n</details>\n")
    return "\n".join(lines)


def export_json(messages: list[dict[str, Any]], metadata: dict[str, str] | None = None) -> str:
    """Export conversation as JSON."""
    return json.dumps(
        {
            "metadata": metadata or {},
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "messages": messages,
        },
        indent=2,
        ensure_ascii=False,
    )


def export_html(messages: list[dict[str, Any]], metadata: dict[str, str] | None = None) -> str:
    """Export conversation as standalone HTML."""
    meta = metadata or {}
    agent = meta.get("agent_name", "unknown")
    model = meta.get("model", "unknown")
    ts = datetime.now(timezone.utc).isoformat()

    msg_html_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or ""
        if role == "system":
            continue
        css_class = "user" if role == "user" else "assistant" if role == "assistant" else "tool"
        escaped = _escape_html(content)
        label = role.capitalize()
        msg_html_parts.append(
            f'<div class="message {css_class}">'
            f'<div class="label">{label}</div>'
            f'<div class="content"><pre>{escaped}</pre></div>'
            f"</div>"
        )

    messages_html = "\n".join(msg_html_parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Koboi Agent Conversation Export</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; background: #0f0d1a; color: #e2e0f0; }}
h1 {{ color: #7c3aed; }}
.meta {{ color: #888; margin-bottom: 2rem; }}
.message {{ margin: 1rem 0; padding: 1rem; border-radius: 8px; }}
.message.user {{ background: #1e1b2e; border-left: 3px solid #7c3aed; }}
.message.assistant {{ background: #1a1830; border-left: 3px solid #10b981; }}
.message.tool {{ background: #1a1820; border-left: 3px solid #f59e0b; }}
.label {{ font-weight: bold; margin-bottom: 0.5rem; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; }}
.user .label {{ color: #7c3aed; }}
.assistant .label {{ color: #10b981; }}
.tool .label {{ color: #f59e0b; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; margin: 0; }}
</style>
</head>
<body>
<h1>Koboi Agent Conversation</h1>
<div class="meta">
<p>Agent: {_escape_html(agent)} | Model: {_escape_html(model)}</p>
<p>Exported: {_escape_html(ts)}</p>
</div>
{messages_html}
</body>
</html>"""


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
