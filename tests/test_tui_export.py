"""Tests for koboi/tui/export.py -- Conversation export formatters."""

from __future__ import annotations

import json

from koboi.tui.export import export_html, export_json, export_markdown, _escape_html


SAMPLE_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {
        "role": "assistant",
        "content": "Using tool",
        "tool_calls": [{"function": {"name": "read", "arguments": '{"path":"f.py"}'}}],
    },
    {"role": "tool", "content": "file contents"},
    {"role": "system", "content": "system prompt"},
]

METADATA = {"agent_name": "test-agent", "model": "gpt-4o"}


class TestExportMarkdown:
    def test_basic_structure(self):
        md = export_markdown(SAMPLE_MESSAGES, METADATA)
        assert "# Conversation Export" in md
        assert "test-agent" in md
        assert "gpt-4o" in md

    def test_user_message(self):
        md = export_markdown(SAMPLE_MESSAGES)
        assert "## User" in md
        assert "Hello" in md

    def test_assistant_message(self):
        md = export_markdown(SAMPLE_MESSAGES)
        assert "## Assistant" in md

    def test_tool_call_included(self):
        md = export_markdown(SAMPLE_MESSAGES)
        assert "Tool Call: read" in md

    def test_tool_result_included(self):
        md = export_markdown(SAMPLE_MESSAGES)
        assert "Tool Result" in md
        assert "file contents" in md

    def test_system_skipped(self):
        md = export_markdown(SAMPLE_MESSAGES)
        assert "system prompt" not in md

    def test_no_metadata(self):
        md = export_markdown(SAMPLE_MESSAGES)
        assert "unknown" in md


class TestExportJson:
    def test_valid_json(self):
        raw = export_json(SAMPLE_MESSAGES, METADATA)
        data = json.loads(raw)
        assert "metadata" in data
        assert "messages" in data
        assert "exported_at" in data

    def test_message_count(self):
        raw = export_json(SAMPLE_MESSAGES)
        data = json.loads(raw)
        assert len(data["messages"]) == 5

    def test_no_metadata(self):
        raw = export_json(SAMPLE_MESSAGES)
        data = json.loads(raw)
        assert data["metadata"] == {}


class TestExportHtml:
    def test_html_structure(self):
        html = export_html(SAMPLE_MESSAGES, METADATA)
        assert "<!DOCTYPE html>" in html
        assert "test-agent" in html
        assert "gpt-4o" in html

    def test_user_class(self):
        html = export_html(SAMPLE_MESSAGES)
        assert 'class="message user"' in html

    def test_assistant_class(self):
        html = export_html(SAMPLE_MESSAGES)
        assert 'class="message assistant"' in html

    def test_tool_class(self):
        html = export_html(SAMPLE_MESSAGES)
        assert 'class="message tool"' in html

    def test_system_skipped(self):
        html = export_html(SAMPLE_MESSAGES)
        assert "system prompt" not in html


class TestEscapeHtml:
    def test_ampersand(self):
        assert _escape_html("a&b") == "a&amp;b"

    def test_lt_gt(self):
        assert _escape_html("<tag>") == "&lt;tag&gt;"

    def test_plain(self):
        assert _escape_html("hello") == "hello"
