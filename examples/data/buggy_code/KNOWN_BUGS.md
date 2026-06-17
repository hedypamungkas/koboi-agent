# KNOWN_BUGS.md -- Ground Truth Answer Key

This document is the authoritative reference for all known bugs in the SWE Bug
Hunter evaluation set. Every bug listed below has been verified against the
current file contents (after `# BUG:` comment removal). Line numbers refer to
the line where the buggy code **begins** in each file.

Severity levels:
- **P0** -- Critical (security vulnerability, data breach risk)
- **P1** -- High (incorrect results, crashes, race conditions)
- **P2** -- Medium (subtle logic errors, precision loss, weak defaults)

---

## auth_service.py (78 lines)

- Line 7: Hardcoded secret key `sk-abc123-default-key-do-not-use-in-production` | P0 | Security
- Line 21: SQL injection via f-string in query `f"SELECT id, password_hash FROM users WHERE username = '{username}'"` | P0 | Security
- Line 37-40: Race condition -- shared `_sessions` dict modified without lock during authenticate | P1 | Concurrency
- Line 47: Missing null check on `token` parameter -- `None` causes `sessions.get(None)` to silently return None instead of raising | P1 | Logic
- Line 37-40: Session never expires -- no TTL check on created_at during validate_session | P2 | Logic
- Line 61-77: No password strength validation in change_password | P2 | Security

## data_processor.py (75 lines)

- Line 16: Off-by-one error -- `range(0, len(records) - 1, batch_size)` should be `range(0, len(records), batch_size)` causing last record to be skipped | P1 | Logic
- Line 31-33: Exception swallowed without logging -- bare `except Exception: pass` hides all errors | P2 | Logic
- Line 42: Division by zero when `count=0` in `total / count` | P1 | Logic
- Line 44: Float precision in tax calculation -- `total * 0.11` uses binary float for financial data | P2 | Data
- Line 52-54: Resource leak -- file opened with `open()` instead of `with`, not closed on exception from `json.dump` | P1 | Resource
- Line 65: Returns reference to internal mutable state (`filtered` list could be mutated externally) | P2 | Logic
- Line 73: Division by zero in error_rate when `_processed=0` (guard `if self._processed` evaluates to falsy when _processed is 0, which is correct, but the condition `self._errors / self._processed` is guarded -- however if both are 0, `total` would be 0 and `error_rate` returns 0 correctly; the actual bug is the division on line 73 when `_processed` is 0 AND `_errors > 0`) | P1 | Logic

## cache_manager.py (83 lines)

- Line 18: No lock on read in `get()` -- `_cache.get(key)` reads without acquiring `_lock`, causing race with concurrent writes | P1 | Concurrency
- Line 24: TTL mismatch -- timestamp stored as `time.time() * 1000` (milliseconds) but `default_ttl` is in seconds (default 300), so TTL comparison is off by 1000x | P1 | Logic
- Line 34: Eviction only at exact `max_size` -- uses `==` instead of `>=`, so cache can exceed max_size if multiple threads pass the check simultaneously | P2 | Logic
- Line 58-60: Race condition between check and delete in `invalidate()` -- `if key in self._cache` check is not atomic with `del self._cache[key]`, and there is a `time.sleep(0.001)` between them widening the race window | P1 | Concurrency

## order_calculator.py (100 lines)

- Line 45: Floating point accumulation in subtotal -- iteratively adding floats `subtotal += item.quantity * item.unit_price` accumulates rounding error | P2 | Data
- Line 50: No validation of `customer_tier` -- `TIER_DISCOUNTS[order.customer_tier]` raises KeyError for unknown tiers | P1 | Logic
- Line 59: Discount can exceed subtotal producing negative result -- tier discount plus bulk discount can sum to more than 100% | P1 | Logic
- Line 64: Missing region causes KeyError -- `TAX_RATES[region]` fails for unsupported region codes | P1 | Logic
- Line 66: Rounding applied to individual tax calculation, not final amount -- `round(discounted_subtotal * rate, 2)` loses precision for downstream totals | P2 | Data
- Line 87: No validation of `coupon_percent` range -- allows values outside 0-100, producing negative totals or nonsensical discounts | P1 | Logic
- Line 95: Integer division loses precision on weight_charge -- `int(weight_kg) * 2.0` truncates fractional kg | P2 | Data
- Line 98: Wrong comparison operator -- `total > 50.0` should be `total >= 50.0` to give free shipping at exactly $50 | P2 | Logic

## payment_gateway.py

(Will be filled in after file is created)

## inventory_manager.py

(Will be filled in after file is created)

---

## Summary Statistics

| File | Lines | Bugs | P0 | P1 | P2 |
|---|---|---|---|---|---|
| auth_service.py | 78 | 6 | 2 | 2 | 2 |
| data_processor.py | 75 | 7 | 0 | 3 | 4 |
| cache_manager.py | 83 | 4 | 0 | 3 | 1 |
| order_calculator.py | 100 | 8 | 0 | 4 | 4 |
| payment_gateway.py | ~280 | TBD | - | - | - |
| inventory_manager.py | ~230 | TBD | - | - | - |

## Bug Distribution by Category

| Category | Count |
|---|---|
| Security | 4 |
| Logic | 12 |
| Concurrency | 4 |
| Data | 3 |
| Resource | 1 |
| **Total (documented)** | **24** |
