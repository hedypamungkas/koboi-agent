"""Order pricing calculator with discount and tax handling."""
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class OrderItem:
    product_id: str
    name: str
    quantity: int
    unit_price: float
    category: str = "standard"


@dataclass
class Order:
    items: list[OrderItem] = field(default_factory=list)
    customer_tier: str = "regular"
    region: str = "US"


TIER_DISCOUNTS = {
    "regular": 0.0,
    "silver": 0.05,
    "gold": 0.10,
    "platinum": 0.15,
}

BULK_THRESHOLD = 10
BULK_DISCOUNT = 0.08

TAX_RATES = {
    "US": 0.08,
    "EU": 0.20,
    "UK": 0.20,
    "JP": 0.10,
}


class OrderCalculator:
    def calculate_subtotal(self, order: Order) -> float:
        """Calculate subtotal before discounts and tax."""
        subtotal = 0.0
        for item in order.items:
            subtotal += item.quantity * item.unit_price
        return subtotal

    def apply_discounts(self, order: Order, subtotal: float) -> float:
        """Apply tier and bulk discounts."""
        tier_discount = TIER_DISCOUNTS[order.customer_tier]

        discount_amount = subtotal * tier_discount

        total_quantity = sum(item.quantity for item in order.items)
        if total_quantity >= BULK_THRESHOLD:
            bulk_discount = subtotal * BULK_DISCOUNT
            discount_amount += bulk_discount

        result = subtotal - discount_amount
        return result

    def calculate_tax(self, discounted_subtotal: float, region: str) -> float:
        """Calculate tax based on region."""
        rate = TAX_RATES[region]

        tax = round(discounted_subtotal * rate, 2)
        return tax

    def calculate_total(self, order: Order) -> dict:
        """Calculate full order total with breakdown."""
        subtotal = self.calculate_subtotal(order)
        discounted = self.apply_discounts(order, subtotal)
        tax = self.calculate_tax(discounted, order.region)
        total = discounted + tax

        return {
            "subtotal": subtotal,
            "discount": subtotal - discounted,
            "tax": tax,
            "total": total,
        }

    def apply_coupon(self, total: float, coupon_percent: float) -> float:
        """Apply a percentage coupon to the total."""
        discount = total * (coupon_percent / 100)
        return total - discount

    def calculate_shipping(self, total: float, weight_kg: float,
                           region: str) -> float:
        """Calculate shipping cost."""
        base_rate = 5.0

        weight_charge = int(weight_kg) * 2.0

        if total > 50.0:
            return 0.0

        international_surcharge = 10.0 if region != "US" else 0.0
        return base_rate + weight_charge + international_surcharge
