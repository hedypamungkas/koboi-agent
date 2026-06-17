"""Custom weather tool example -- mock data for demo purposes."""
from __future__ import annotations

from koboi.tools.registry import tool

MOCK_WEATHER = {
    "jakarta": {"temp": "32C", "condition": "Partly cloudy", "humidity": "75%"},
    "bandung": {"temp": "24C", "condition": "Light rain", "humidity": "85%"},
    "surabaya": {"temp": "34C", "condition": "Sunny", "humidity": "70%"},
    "yogyakarta": {"temp": "28C", "condition": "Cloudy", "humidity": "78%"},
    "bali": {"temp": "30C", "condition": "Sunny", "humidity": "72%"},
}


@tool(
    name="get_weather",
    description="Get weather information for Indonesian cities",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name (e.g. jakarta, bandung, surabaya)",
            },
        },
        "required": ["city"],
    },
)
def get_weather(city: str) -> str:
    key = city.lower().strip()
    if key in MOCK_WEATHER:
        data = MOCK_WEATHER[key]
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature: {data['temp']}\n"
            f"  Condition: {data['condition']}\n"
            f"  Humidity: {data['humidity']}"
        )
    available = ", ".join(sorted(MOCK_WEATHER.keys()))
    return f"City '{city}' not found. Available: {available}"
