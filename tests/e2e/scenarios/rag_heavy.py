"""RAG-heavy scenarios (category: rag).

Single-turn factual questions grounded in the indexed documents. Distinctive
queries force retrieval (keyword retriever, top_k=5) rather than parametric
recall. Assertions require ONE doc-specific number/term. Doc facts:

  product_catalog.md : AcmeERP $15,000/yr · AcmePOS $500/mo · AcmeCRM $25/user/mo
                       · SaaS Starter $99/mo · Professional $299/mo
  company_policy.md  : remote 2 days/wk · annual leave 12 days · meal $150/mo
                       · transport $100/mo · internet $50/mo remote
  hotel_operations   : Standard $120 · Deluxe $180 · Executive $320 · Pres $850
                       · pet fee $25 · pet <15kg · valet $25 · checkin 3pm
  ecommerce_kb       : returns 30 days (14 electronics) · refund 5-7 biz days
                       · Express $9.99 · Overnight $19.99 · Affirm/Klarna
"""

from __future__ import annotations

from tests.e2e.framework.scenario import Scenario, Turn

SCENARIOS: list[Scenario] = [
    # --- Product catalog (5) ---
    Scenario("rag_acme_erp_price", "rag", [
        Turn("What is the annual price of the AcmeERP Enterprise product?", expect_any_of=["15,000", "15000"]),
    ]),
    Scenario("rag_acme_pos_price", "rag", [
        Turn("How much does AcmePOS Professional cost per month?", expect_keywords=["500"]),
    ]),
    Scenario("rag_acme_crm_users", "rag", [
        Turn("What is the minimum number of users for AcmeCRM Business, and its per-user price?", expect_keywords=["25"]),
    ]),
    Scenario("rag_saas_starter", "rag", [
        Turn("What does the Starter SaaS package cost per month and how many users?", expect_keywords=["99"]),
    ]),
    Scenario("rag_saas_professional", "rag", [
        Turn("How many users does the Professional SaaS package allow, and the price?", expect_keywords=["299"]),
    ]),
    # --- Company policy (5) ---
    Scenario("rag_remote_work_days", "rag", [
        Turn("How many days per week can employees work remotely under company policy?", expect_keywords=["2"]),
    ]),
    Scenario("rag_annual_leave", "rag", [
        Turn("How many annual leave days do permanent employees get?", expect_keywords=["12"]),
    ]),
    Scenario("rag_meal_allowance", "rag", [
        Turn("What is the monthly meal allowance amount?", expect_keywords=["150"]),
    ]),
    Scenario("rag_internet_allowance", "rag", [
        Turn("How much is the internet allowance for remote work days?", expect_keywords=["50"]),
    ]),
    Scenario("rag_transport_allowance", "rag", [
        Turn("What is the monthly transportation allowance?", expect_keywords=["100"]),
    ]),
    # --- Hotel operations (5) ---
    Scenario("rag_hotel_deluxe_rate", "rag", [
        Turn("What is the nightly rate for the Deluxe Room at Grand Plaza Hotel?", expect_keywords=["180"]),
    ]),
    Scenario("rag_hotel_presidential_amenities", "rag", [
        Turn("What special amenities does the Presidential Suite include at Grand Plaza Hotel?", expect_keywords=["butler"]),
    ]),
    Scenario("rag_hotel_pet_fee", "rag", [
        Turn("What is the nightly pet fee at Grand Plaza Hotel and what's the weight limit?", expect_keywords=["25"]),
    ]),
    Scenario("rag_hotel_cancellation", "rag", [
        Turn("How many hours before check-in is cancellation free at Grand Plaza Hotel?", expect_keywords=["48"]),
    ]),
    Scenario("rag_hotel_checkout", "rag", [
        Turn("What time is check-out at Grand Plaza Hotel?", expect_keywords=["11"]),
    ]),
    # --- E-commerce (5) ---
    Scenario("rag_ecom_return_window", "rag", [
        Turn("What is the standard return window for ShopWave items?", expect_keywords=["30"]),
    ]),
    Scenario("rag_ecom_refund_time", "rag", [
        Turn("How many business days does a ShopWave refund take to process?", expect_keywords=["business day"]),
    ]),
    Scenario("rag_ecom_express_shipping", "rag", [
        Turn("How much does Express shipping cost at ShopWave?", expect_keywords=["9.99"]),
    ]),
    Scenario("rag_ecom_bnop", "rag", [
        Turn("What Buy Now Pay Later options does ShopWave support?", expect_keywords=["Affirm", "Klarna"]),
    ]),
    Scenario("rag_ecom_overnight", "rag", [
        Turn("How much is overnight shipping and what's the order cutoff time?", expect_keywords=["19.99"]),
    ]),
]
