"""koboi/rag/sample_documents.py — Sample document data for built-in agents.

Reads from data/sample/ files or provides inline fallbacks.
Used by the agent factory to populate knowledge for built-in HR, sales,
finance, and general agents.
"""

from __future__ import annotations

from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"


def _read_or_empty(filename: str) -> str:
    path = _DATA_DIR / filename
    if path.exists():
        return path.read_text()
    return ""


def get_company_policy() -> str:
    return _read_or_empty("company_policy.md")


def get_employee_handbook() -> str:
    return _read_or_empty("employee_handbook.md")


def get_product_catalog() -> str:
    return _read_or_empty("product_catalog.md")


def get_facilities_guide() -> str:
    return "Office Facilities Guide - Acme Corp"


def get_all_documents() -> list[tuple[str, str]]:
    return [
        ("Company Policy", get_company_policy()),
        ("Employee Handbook", get_employee_handbook()),
        ("Product Catalog", get_product_catalog()),
        ("Office Facilities Guide", get_facilities_guide()),
    ]
