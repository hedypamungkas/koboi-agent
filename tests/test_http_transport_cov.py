"""koboi/llm/http_transport.py -- branch coverage via httpx.MockTransport."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from koboi.llm.auth import BearerAuth
from koboi.llm.base import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
)
from koboi.llm.http_transport import HttpTransport


def _make_transport(handler, max_retries=2) -> HttpTransport:
    t = HttpTransport("https://api.example.com/v1", BearerAuth("k"), max_retries=max_retries)
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return t


def _resp(status, payload=None, text=None, headers=None):
    if text is not None:
        return httpx.Response(status, text=text, headers=headers or {})
    return httpx.Response(status, json=payload if payload is not None else {}, headers=headers or {})


class TestPostSuccess:
    async def test_ok_returns_json(self):
        t = _make_transport(lambda req: _resp(200, {"ok": True}))
        assert await t.post("/chat", {"x": 1}) == {"ok": True}

    async def test_ok_non_json_raises_parse_error(self):
        t = _make_transport(lambda req: _resp(200, text="<<not json>>"))
        with pytest.raises(LLMResponseParseError):
            await t.post("/chat", {})


class TestPostErrors:
    async def test_auth_error(self):
        t = _make_transport(lambda req: _resp(401, {"error": {"message": "bad key"}}))
        with pytest.raises(LLMAuthenticationError):
            await t.post("/chat", {})

    async def test_bad_request(self):
        t = _make_transport(lambda req: _resp(400, {"error": {"message": "nope"}}))
        with pytest.raises(LLMInvalidRequestError):
            await t.post("/chat", {})

    async def test_rate_limit_no_retry(self):
        t = _make_transport(lambda req: _resp(429, {"error": "slow"}, headers={"retry-after": "2"}), max_retries=0)
        with pytest.raises(LLMRateLimitError):
            await t.post("/chat", {})

    async def test_server_error_5xx(self):
        t = _make_transport(lambda req: _resp(503, {"error": "down"}), max_retries=0)
        with pytest.raises(LLMServerError):
            await t.post("/chat", {})

    async def test_other_status_falls_through(self):
        # 404 -> not 401/403/429/400/>=500 -> final LLMInvalidRequestError "HTTP 404"
        t = _make_transport(lambda req: _resp(404, {"error": "missing"}), max_retries=0)
        with pytest.raises(LLMInvalidRequestError, match="HTTP 404"):
            await t.post("/chat", {})


class TestPostRetries:
    async def test_retry_then_success(self, monkeypatch):
        async def _noop(*_a):
            return None

        monkeypatch.setattr(asyncio, "sleep", _noop)
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            return _resp(429) if calls["n"] == 1 else _resp(200, {"ok": True})

        t = _make_transport(handler, max_retries=2)
        assert await t.post("/chat", {}) == {"ok": True}
        assert calls["n"] == 2

    async def test_retry_exhausted_raises_server(self, monkeypatch):
        async def _noop(*_a):
            return None

        monkeypatch.setattr(asyncio, "sleep", _noop)
        t = _make_transport(lambda req: _resp(500, {"error": "boom"}), max_retries=1)
        with pytest.raises(LLMServerError):
            await t.post("/chat", {})


class TestPostConnectionErrors:
    async def test_connect_error(self):
        def handler(req):
            raise httpx.ConnectError("nope")

        t = _make_transport(handler)
        with pytest.raises(LLMConnectionError):
            await t.post("/chat", {})

    async def test_timeout_error(self):
        def handler(req):
            raise httpx.TimeoutException("slow")

        t = _make_transport(handler)
        with pytest.raises(LLMConnectionError):
            await t.post("/chat", {})


class TestExtractErrorDetail:
    def test_dict_error_with_message(self):
        r = _resp(400, {"error": {"message": "bad"}})
        assert HttpTransport._extract_error_detail(r) == "bad"

    def test_non_dict_error(self):
        r = _resp(400, {"error": "flattened"})
        assert HttpTransport._extract_error_detail(r) == "flattened"

    def test_non_json_response(self):
        r = _resp(500, text="<<html>>")
        assert HttpTransport._extract_error_detail(r) == "<<html>>"


class TestPostStream:
    async def test_stream_yields_lines(self):
        body = "data: a\n\ndata: b\n\n"
        t = _make_transport(lambda req: httpx.Response(200, text=body))
        lines = [b async for b in t.post_stream("/chat", {"x": 1})]
        # post_stream sets body["stream"]=True before requesting
        assert any(b"a" in ln for ln in lines)

    async def test_stream_error_status(self):
        t = _make_transport(lambda req: httpx.Response(400, text="bad"))
        with pytest.raises(LLMInvalidRequestError):
            async for _ in t.post_stream("/chat", {}):
                pass

    async def test_stream_connect_error(self):
        def handler(req):
            raise httpx.ConnectError("dead")

        t = _make_transport(handler)
        with pytest.raises(LLMConnectionError):
            async for _ in t.post_stream("/chat", {}):
                pass

    async def test_stream_timeout(self):
        def handler(req):
            raise httpx.TimeoutException("slow")

        t = _make_transport(handler)
        with pytest.raises(LLMConnectionError):
            async for _ in t.post_stream("/chat", {}):
                pass


class TestContextManager:
    async def test_aenter_aexit(self):
        t = _make_transport(lambda req: _resp(200, {}))
        async with t as ctx:
            assert ctx is t
