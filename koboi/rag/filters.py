"""koboi/rag/filters.py -- metadata filtering for retrieval (relevance scoping).

**NOT a security/ACL boundary** -- a relevance filter that constrains which chunks a
retriever considers (e.g. ``year >= 2024``, ``source in [policy, handbook]``). For
access-control enforcement use a dedicated hardened layer; a metadata typo here must
never be the only thing keeping tenants apart.

Operators use Mongo-style ``$``-prefixed keys (won't collide with field names):

- scalar value          -> equality (``field == value``)
- ``{"$gte": x}`` / ``{"$gt": x}``   -> ``>=`` / ``>``
- ``{"$lte": y}`` / ``{"$lt": y}``   -> ``<=`` / ``<``
- ``{"$in": [...]}``                 -> membership

A chunk missing the field (or an incomparable type) does NOT match -- filters are
intentionally strict (exclude-on-doubt).
"""

from __future__ import annotations


def _cmp(actual: object, op: str, val: object) -> bool:
    if actual is None:
        return False
    try:
        if op == "$gte":
            return actual >= val  # type: ignore[operator]
        if op == "$gt":
            return actual > val  # type: ignore[operator]
        if op == "$lte":
            return actual <= val  # type: ignore[operator]
        if op == "$lt":
            return actual < val  # type: ignore[operator]
    except TypeError:
        return False  # incomparable types (e.g. str vs int) -> no match
    return False


def matches_filter(metadata: dict, filt: dict | None) -> bool:
    """True if ``metadata`` satisfies every clause in ``filt``. ``None``/empty -> match all."""
    if not filt:
        return True
    for field, clause in filt.items():
        actual = metadata.get(field)
        if isinstance(clause, dict):
            for op, val in clause.items():
                if op == "$in":
                    if actual not in val:
                        return False
                elif op in ("$gte", "$gt", "$lte", "$lt"):
                    if not _cmp(actual, op, val):
                        return False
                else:
                    return False  # unknown operator -> strict no-match
        elif actual != clause:
            return False  # equality clause
    return True
