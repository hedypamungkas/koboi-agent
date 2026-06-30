"""Skill-activation scenarios (category: skills).

Skills (hotel_receptionist, customer_service, code_review) are discovered and
model-invocable. From outside the box we cannot observe a "skill invoked" event,
so these scenarios assert domain *competence*: the response correctly handles a
domain task. Hotel/CS answers are also grounded by RAG docs, giving strong
keywords. Code-review scenarios reason over a pasted snippet (no tools needed).
"""

from __future__ import annotations

from tests.e2e.framework.scenario import Scenario, Turn

SCENARIOS: list[Scenario] = [
    # --- Hotel receptionist (5) ---
    Scenario("skill_hotel_inquiry", "skills", [
        Turn("Act as the Grand Plaza Hotel receptionist. A guest asks: what room types do you offer and the rates?", expect_keywords=["Standard", "Deluxe"]),
    ]),
    Scenario("skill_hotel_amenity_detail", "skills", [
        Turn("As the hotel receptionist, a guest wants to know what the Executive Suite includes.", expect_keywords=["Executive Lounge", "breakfast"]),
    ]),
    Scenario("skill_hotel_cancellation_guidance", "skills", [
        Turn("As the receptionist, a guest wants to cancel within 24 hours of check-in. Explain the policy.", expect_keywords=["night"]),
    ]),
    Scenario("skill_hotel_group_booking", "skills", [
        Turn("As the receptionist, a guest needs a room for 4 adults. Which room type fits, and what's the rate?", expect_keywords=["Presidential", "850"]),
    ]),
    Scenario("skill_hotel_pet_request", "skills", [
        Turn("As the receptionist, a guest asks if they can bring a 10kg dog. What do you tell them?", expect_keywords=["pet", "25"]),
    ]),
    # --- Customer service (5) ---
    Scenario("skill_cs_return_help", "skills", [
        Turn("Act as a ShopWave customer service agent. A customer wants to return an electronics item after 10 days. Can they?", expect_keywords=["14"]),
    ]),
    Scenario("skill_cs_refund_status", "skills", [
        Turn("As a ShopWave agent, a customer asks how long their refund will take.", expect_keywords=["business day"]),
    ]),
    Scenario("skill_cs_missing_package", "skills", [
        Turn("As a ShopWave agent, a customer says their tracking shows no update for 48 hours. What's the policy?", expect_keywords=["48"]),
    ]),
    Scenario("skill_cs_payment_methods", "skills", [
        Turn("As a ShopWave agent, list the payment methods you accept.", expect_keywords=["PayPal"]),
    ]),
    Scenario("skill_cs_return_shipping_fee", "skills", [
        Turn("As a ShopWave agent, a customer's order was $30 and they want to return it. Is return shipping free?", expect_keywords=["5.99"]),
    ]),
    # --- Code review (5) — reason over pasted snippet ---
    Scenario("skill_codereview_sql_injection", "skills", [
        Turn("Review this Python for security issues: `cursor.execute(\"SELECT * FROM users WHERE name='\" + name + \"'\")`. What's wrong?", expect_keywords=["injection", "parameterized"]),
    ]),
    Scenario("skill_codereview_mutable_default", "skills", [
        Turn("Review this Python: `def add(x, items=[]): items.append(x); return items`. What's the bug?", expect_keywords=["mutable", "default"]),
    ]),
    Scenario("skill_codereview_resource_leak", "skills", [
        Turn("Review this Python: opens a file with open() but never closes it. What should be improved?", expect_keywords=["close", "with", "context"]),
    ]),
    Scenario("skill_codereview_off_by_one", "skills", [
        Turn("Review this loop: `for i in range(1, len(items))` meant to process every element. Any issue?", expect_any_of=["off", "missing", "skips", "omits", "first element"]),
    ]),
    Scenario("skill_codereview_bare_except", "skills", [
        Turn("Review this Python error handling: `except: pass`. What's the concern?", expect_keywords=["broad", "silent", "swallow"]),
    ]),
]
