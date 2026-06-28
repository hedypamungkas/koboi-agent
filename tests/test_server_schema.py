"""Unit tests for koboi/server/schema.py (pure Pydantic, no FastAPI)."""

from __future__ import annotations

import pytest

from koboi.server.schema import ChatStreamRequest


class TestChatStreamRequest:
    def test_message_string(self):
        assert ChatStreamRequest(message="hello").user_message() == "hello"

    def test_messages_array_picks_user(self):
        req = ChatStreamRequest(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
        assert req.user_message() == "hi"

    def test_messages_picks_last_user_when_multiple(self):
        req = ChatStreamRequest(
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ]
        )
        assert req.user_message() == "second"

    def test_empty_message_raises(self):
        with pytest.raises(ValueError):
            ChatStreamRequest(message="").user_message()

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError):
            ChatStreamRequest(messages=[]).user_message()

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            ChatStreamRequest().user_message()

    def test_messages_no_user_role_raises(self):
        with pytest.raises(ValueError):
            ChatStreamRequest(messages=[{"role": "system", "content": "s"}]).user_message()

    def test_whitespace_message_raises(self):
        with pytest.raises(ValueError):
            ChatStreamRequest(message="   ").user_message()
