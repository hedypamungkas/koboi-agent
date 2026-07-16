"""Tests for koboi.media.providers.mock + surplus (no network; transport mocked)."""

from __future__ import annotations

import base64
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from koboi.media.providers.mock import MockImageProvider
from koboi.media.providers.surplus import SurplusImageProvider
from koboi.media.types import MediaRequest, MediaUnit

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


class TestMockImageProvider:
    async def test_returns_valid_png_placeholder(self):
        provider = MockImageProvider()
        result = await provider.generate_image(MediaRequest(prompt="anything"))
        assert result.status == "ok"
        assert result.data and result.data.startswith(_PNG_SIG)
        assert result.content_type == "image/png"
        assert result.cost_usd == Decimal("0")
        assert result.billing_unit == MediaUnit.IMAGE
        assert result.billing_quantity == 1.0


class TestSurplusImageProvider:
    async def test_generates_from_b64(self):
        raw = _PNG_SIG + b"fakepixeldata"
        provider = SurplusImageProvider(api_key="inf_test", model="venice-z-image-turbo")
        provider._transport.post = AsyncMock(
            return_value={
                "data": [{"b64_json": base64.b64encode(raw).decode()}],
                "usage": {"cost_usd": 0.02, "images": 1},
            }
        )
        result = await provider.generate_image(MediaRequest(prompt="cat", n=1))
        assert result.status == "ok"
        assert result.data == raw
        assert result.cost_usd == Decimal("0.02")
        assert result.billing_unit == MediaUnit.IMAGE
        assert result.billing_quantity == 1.0
        await provider.close()

    async def test_token_metered_model(self):
        provider = SurplusImageProvider(api_key="k", model="gpt-5-image-1")
        provider._transport.post = AsyncMock(
            return_value={
                "data": [{"b64_json": base64.b64encode(b"x").decode()}],
                "usage": {"output_tokens": 1234, "buyer_cost_usd": 5000000},
            }
        )
        result = await provider.generate_image(MediaRequest(prompt="x"))
        assert result.billing_unit == MediaUnit.TOKEN
        assert result.billing_quantity == 1234.0
        assert result.cost_usd == Decimal("5")  # 5,000,000 microdollars / 1e6

    async def test_url_only_response(self):
        provider = SurplusImageProvider(api_key="k")
        provider._transport.post = AsyncMock(
            return_value={"data": [{"url": "https://example.com/img.png"}], "usage": {}}
        )
        result = await provider.generate_image(MediaRequest(prompt="x"))
        assert result.status == "ok"
        assert result.url == "https://example.com/img.png"
        assert result.data is None

    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("SURPLUS_API_KEY", raising=False)
        provider = SurplusImageProvider(api_key="")
        with pytest.raises(ValueError):
            await provider.generate_image(MediaRequest(prompt="x"))

    def test_unsupported_auth_mode(self):
        with pytest.raises(NotImplementedError):
            SurplusImageProvider(api_key="k", auth_mode="x402")

    async def test_empty_data_failed(self):
        provider = SurplusImageProvider(api_key="k")
        provider._transport.post = AsyncMock(return_value={"data": []})
        result = await provider.generate_image(MediaRequest(prompt="x"))
        assert result.status == "failed"

    async def test_close_calls_transport_close(self):
        provider = SurplusImageProvider(api_key="k")
        provider._transport.close = AsyncMock()
        await provider.close()
        provider._transport.close.assert_awaited()
