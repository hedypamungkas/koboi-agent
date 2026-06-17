"""Custom translate tool example -- mock translation for demo purposes."""

from __future__ import annotations

from koboi.tools.registry import tool

MOCK_TRANSLATIONS = {
    ("good morning", "id"): "Selamat pagi",
    ("good afternoon", "id"): "Selamat siang",
    ("good night", "id"): "Selamat malam",
    ("thank you", "id"): "Terima kasih",
    ("how are you", "id"): "Apa kabar",
    ("selamat pagi", "en"): "Good morning",
    ("terima kasih", "en"): "Thank you",
    ("apa kabar", "en"): "How are you",
}


@tool(
    name="translate_text",
    description="Translate text between English and Indonesian (mock/demo)",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to translate",
            },
            "target_lang": {
                "type": "string",
                "description": "Target language: 'en' for English, 'id' for Indonesian",
            },
        },
        "required": ["text", "target_lang"],
    },
)
def translate_text(text: str, target_lang: str = "en") -> str:
    key = (text.lower().strip(), target_lang.lower())
    if key in MOCK_TRANSLATIONS:
        return f"Translation: {MOCK_TRANSLATIONS[key]}"
    return f"[Demo] Translation of '{text}' to {target_lang} -- use a real translation API for accurate results."
