"""Multi-turn conversation scenarios (category: multi_turn).

Each scenario is a multi-LLM-call conversation in a SINGLE session. The primary
signal under test is *conversational memory*: facts stated in earlier turns must
be available later in the same session (ConversationMemory is SQLite-backed and
persists automatically — no tool needed). Assertions check that a distinctive
token from an earlier turn is echoed back.

GPT-4o-mini is variable, so keywords are chosen to be distinctive (low false-match
risk) and only ONE keyword is required per recall turn.
"""

from __future__ import annotations

from tests.e2e.framework.scenario import Scenario, Turn

# A library of distinctive facts to inject across memory chains. Each is unusual
# enough that the model echoing it back is strong evidence it retained context.
_FACTS = [
    ("My favorite color is turquoise.", "turquoise"),
    ("I work as a marine biologist in Lisbon.", "Lisbon"),
    ("My dog's name is Mochi and she is a shiba inu.", "Mochi"),
    ("I'm allergic to pineapples.", "pineapple"),
    ("My birthday is March 14th.", "March 14"),
    ("I prefer trains over planes when I travel.", "train"),
    ("My favorite cuisine is Ethiopian.", "Ethiopian"),
    ("I collect vintage typewriters.", "typewriter"),
    ("My middle name is Genevieve.", "Genevieve"),
    ("I speak conversational Finnish.", "Finnish"),
]


def _recall_chain(fact: str, keyword: str, idx: int) -> Scenario:
    """Store a fact, chat about something else, then recall it (3 turns)."""
    return Scenario(
        name=f"multiturn_memory_{idx:02d}",
        category="multi_turn",
        turns=[
            Turn(fact),
            Turn("That's interesting! Tell me a fun fact about the ocean."),
            Turn("Going back to what I told you earlier — what was it?", expect_keywords=[keyword]),
        ],
    )


SCENARIOS: list[Scenario] = [
    # --- Persona-driven multi-turn (hotel) ---
    Scenario(
        name="hotel_booking_5turn",
        category="multi_turn",
        turns=[
            Turn("Hi, I'd like to book a room at the Grand Plaza Hotel for 2 adults next weekend."),
            Turn("What room types do you have available?", expect_keywords=["Standard"]),
            Turn("What's the rate for the Executive Suite?", expect_keywords=["320"]),
            Turn("What amenities come with the Executive Suite?", expect_keywords=["lounge"]),
            Turn("Great. For now, please remember my booking is for 2 adults — confirm the party size.", expect_keywords=["2"]),
        ],
    ),
    Scenario(
        name="hotel_cancellation_3turn",
        category="multi_turn",
        turns=[
            Turn("I need to understand the Grand Plaza Hotel cancellation policy."),
            Turn("What happens if I cancel within 48 hours of check-in?", expect_keywords=["night"]),
            Turn("And what about a no-show?", expect_keywords=["charged", "full"]),
        ],
    ),
    Scenario(
        name="hotel_upgrade_4turn",
        category="multi_turn",
        turns=[
            Turn("I have a Standard Room booked at Grand Plaza. Rate is how much?", expect_keywords=["120"]),
            Turn("Can I upgrade to the Deluxe Room? What's the difference?", expect_keywords=["180"]),
            Turn("What about the Executive Suite rate?", expect_keywords=["320"]),
            Turn("Compare the Deluxe and Executive for me.", expect_keywords=["Executive"]),
        ],
    ),
    # --- Customer service multi-turn (e-commerce) ---
    Scenario(
        name="cs_refund_5turn",
        category="multi_turn",
        turns=[
            Turn("I want to return an item I bought from ShopWave."),
            Turn("What's your return window for a standard item?", expect_keywords=["30"]),
            Turn("It's electronics actually — does that change things?", expect_keywords=["14"]),
            Turn("How is the refund issued?", expect_keywords=["payment"]),
            Turn("And how long does the refund take?", expect_keywords=["business day"]),
        ],
    ),
    Scenario(
        name="cs_shipping_4turn",
        category="multi_turn",
        turns=[
            Turn("I have a question about ShopWave shipping."),
            Turn("What are my delivery options?", expect_keywords=["Standard", "Express"]),
            Turn("How much is Express shipping?", expect_keywords=["9.99"]),
            Turn("When will I get a tracking number?", expect_keywords=["24"]),
        ],
    ),
    Scenario(
        name="cs_payment_3turn",
        category="multi_turn",
        turns=[
            Turn("What payment methods does ShopWave accept?"),
            Turn("Do you accept PayPal?", expect_keywords=["PayPal"]),
            Turn("What about Buy Now Pay Later options?", expect_keywords=["Affirm", "Klarna"]),
        ],
    ),
    # --- Knowledge / tech multi-turn ---
    Scenario(
        name="coding_assistant_5turn",
        category="multi_turn",
        turns=[
            Turn("Explain what a Python decorator is, briefly."),
            Turn("Show me a tiny example.", expect_keywords=["def"]),
            Turn("How do I pass arguments to a decorator?"),
            Turn("What's a common use case for decorators?"),
            Turn("Summarize what I've learned about decorators so far.", expect_keywords=["decorator"]),
        ],
    ),
    Scenario(
        name="tech_support_5turn",
        category="multi_turn",
        turns=[
            Turn("My Python script keeps failing with a KeyError. What does that mean?"),
            Turn("I'm accessing a dictionary key that might not exist. Best fix?", expect_keywords=["get", "default"]),
            Turn("Can you show the .get() usage?", expect_keywords=["get"]),
            Turn("What if I want to add the key only if missing?"),
            Turn("Give me a one-line recap of the fix.", expect_keywords=["get"]),
        ],
    ),
    # --- Context switch (topic A → B → back to A) ---
    Scenario(
        name="context_switch_5turn",
        category="multi_turn",
        turns=[
            Turn("Remember: my project deadline is October 15th."),
            Turn("Now, unrelated — explain what an API gateway is."),
            Turn("How does an API gateway handle rate limiting?"),
            Turn("Switching topics again — what is serverless computing?"),
            Turn("Back to what matters: when is my project deadline?", expect_keywords=["October 15"]),
        ],
    ),
    # --- RAG follow-up chains (use hotel context, recall within session) ---
    Scenario(
        name="rag_followup_hotel_pets",
        category="multi_turn",
        turns=[
            Turn("What's the Grand Plaza Hotel pet policy?"),
            Turn("What's the pet fee per night?", expect_keywords=["25"]),
            Turn("And the weight limit for pets?", expect_keywords=["15"]),
        ],
    ),
    Scenario(
        name="rag_followup_hotel_parking",
        category="multi_turn",
        turns=[
            Turn("Tell me about parking at Grand Plaza Hotel."),
            Turn("How much is valet parking?", expect_keywords=["25"]),
            Turn("Is EV charging free for guests?", expect_keywords=["free"]),
        ],
    ),
    Scenario(
        name="rag_followup_acme_crm",
        category="multi_turn",
        turns=[
            Turn("Tell me about the AcmeCRM Business product."),
            Turn("What's its price per user?", expect_keywords=["25"]),
            Turn("What's the minimum number of users?", expect_keywords=["5"]),
        ],
    ),
    # --- Distinctive-fact memory chains (10 scenarios) ---
    *[_recall_chain(f, k, i) for i, (f, k) in enumerate(_FACTS, start=1)],
    # --- Multi-fact accumulation (recall 2 facts at end) ---
    Scenario(
        name="multiturn_accumulate_two_facts",
        category="multi_turn",
        turns=[
            Turn("Note: my flight number is AZ612."),
            Turn("Also, my hotel confirmation code is PLAZA-99."),
            Turn("Remind me of both my flight number and hotel code.", expect_keywords=["AZ612"]),
        ],
    ),
    Scenario(
        name="multiturn_preference_override",
        category="multi_turn",
        turns=[
            Turn("I usually prefer window seats on flights."),
            Turn("Actually, change that — I now prefer aisle seats."),
            Turn("Which seat do I prefer now?", expect_keywords=["aisle"]),
        ],
    ),
    Scenario(
        name="multiturn_numeric_detail",
        category="multi_turn",
        turns=[
            Turn("My customer ID is 8842 and my zip code is 90210."),
            Turn("What's the weather typically like in that zip area?"),
            Turn("Repeat my customer ID exactly.", expect_keywords=["8842"]),
        ],
    ),
    Scenario(
        name="multiturn_correction",
        category="multi_turn",
        turns=[
            Turn("I ordered the blue shirt."),
            Turn("Sorry, I meant the navy shirt, not blue."),
            Turn("Which shirt did I actually order?", expect_keywords=["navy"]),
        ],
    ),
    Scenario(
        name="multiturn_long_context_8turn",
        category="multi_turn",
        turns=[
            Turn("Let's plan a 3-day trip to Tokyo."),
            Turn("Day 1 should focus on Shibuya and Harajuku."),
            Turn("Day 2 — Asakusa and the Senso-ji temple."),
            Turn("Day 3 — teamLab and Odaiba."),
            Turn("What's the budget concern for a mid-range trip?"),
            Turn("Any transit tips?"),
            Turn("Recap Day 2 for me.", expect_keywords=["Asakusa"]),
            Turn("Which day did I assign teamLab?", expect_keywords=["Day 3"]),
        ],
    ),
    Scenario(
        name="multiturn_two_stakeholders",
        category="multi_turn",
        turns=[
            Turn("I'm planning for two guests: Alice (vegan) and Bob (gluten-free)."),
            Turn("Suggest a dinner that works for both."),
            Turn("Remind me of Alice's dietary restriction.", expect_keywords=["vegan"]),
        ],
    ),
    Scenario(
        name="multiturn_incremental_spec",
        category="multi_turn",
        turns=[
            Turn("I'm writing a function to validate an email. Start with the basic shape."),
            Turn("Now add a length check: max 254 characters."),
            Turn("What's the max length I asked for?", expect_keywords=["254"]),
        ],
    ),
    Scenario(
        name="multiturn_negotiation_state",
        category="multi_turn",
        turns=[
            Turn("I'm negotiating a salary. My floor is 95k."),
            Turn("They offered 88k. Is that above my floor?"),
            Turn("What number did I say is my absolute floor?", expect_keywords=["95"]),
        ],
    ),
]
