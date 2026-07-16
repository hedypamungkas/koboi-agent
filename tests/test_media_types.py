"""Tests for koboi.media.types (MediaBudget) + koboi.media.budget (CountingImageProvider)."""

from __future__ import annotations

from decimal import Decimal

from koboi.media.budget import CountingImageProvider
from koboi.media.providers.mock import MockImageProvider
from koboi.media.types import MediaBudget, MediaRequest, MediaResult, MediaUnit


class TestMediaBudget:
    def test_remaining_within_caps(self):
        budget = MediaBudget(max_cost_usd=5.0, max_images=10)
        assert budget.remaining("image") is True

    def test_remaining_usd_ceiling_exceeded(self):
        budget = MediaBudget(max_cost_usd=5.0, max_images=10, used_cost_usd=Decimal("5.0"))
        assert budget.remaining("image") is False

    def test_remaining_image_cap_exceeded(self):
        budget = MediaBudget(max_cost_usd=5.0, max_images=2, used_images=2)
        assert budget.remaining("image") is False

    def test_record_accrues_cost_and_images(self):
        budget = MediaBudget(max_cost_usd=5.0, max_images=10)
        result = MediaResult(
            request_id="x",
            modality="image",
            status="ok",
            cost_usd=Decimal("0.02"),
            billing_unit=MediaUnit.IMAGE,
            billing_quantity=2,
        )
        budget.record(result)
        assert budget.used_cost_usd == Decimal("0.02")
        assert budget.used_images == 2

    def test_record_ignores_non_ok(self):
        budget = MediaBudget()
        budget.record(
            MediaResult(
                request_id="x",
                modality="image",
                status="rejected",
                cost_usd=Decimal("1.0"),
                billing_quantity=1,
            )
        )
        assert budget.used_cost_usd == Decimal("0")
        assert budget.used_images == 0


class _SpyImageProvider(MockImageProvider):
    """MockImageProvider that counts generate_image calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def generate_image(self, req: MediaRequest) -> MediaResult:  # type: ignore[override]
        self.calls += 1
        return await super().generate_image(req)


class TestCountingImageProvider:
    async def test_delegates_and_records(self):
        inner = _SpyImageProvider()
        budget = MediaBudget(max_cost_usd=5.0, max_images=10)
        provider = CountingImageProvider(inner, budget)
        result = await provider.generate_image(MediaRequest(prompt="cat", n=1))
        assert result.status == "ok"
        assert inner.calls == 1
        assert budget.used_images == 1

    async def test_rejects_when_exhausted_without_calling_inner(self):
        inner = _SpyImageProvider()
        budget = MediaBudget(max_cost_usd=5.0, max_images=1, used_images=1)
        provider = CountingImageProvider(inner, budget)
        result = await provider.generate_image(MediaRequest(prompt="cat"))
        assert result.status == "rejected"
        assert "budget" in (result.rejection_reason or "")
        assert inner.calls == 0  # never billed the inner provider

    async def test_close_propagates(self):
        provider = CountingImageProvider(MockImageProvider(), MediaBudget())
        await provider.close()  # no exception
