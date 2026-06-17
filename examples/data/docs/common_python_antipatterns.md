# Common Python Antipatterns and How to Avoid Them

> A reference guide for identifying and fixing frequent Python mistakes.
> Use this document to spot recurring bug categories in code reviews.

---

## 1. Resource Management

### 1.1: Not using context managers for files and connections

```python
# ANTIPATTERN - Resource leak on exception
f = open("output.json", "w")
json.dump(data, f, indent=2)
f.close()  # Never reached if json.dump raises

# CORRECT - Guaranteed cleanup via context manager
with open("output.json", "w") as f:
    json.dump(data, f, indent=2)
```

### 1.2: Returning references to internal mutable state

```python
# ANTIPATTERN - Caller can mutate internal data
class DataProcessor:
    def __init__(self):
        self._results = []

    def get_results(self):
        return self._results  # Returns direct reference

# CORRECT - Return a copy to prevent external mutation
class DataProcessor:
    def __init__(self):
        self._results = []

    def get_results(self):
        return list(self._results)  # Defensive copy
```

### 1.3: Mixing lock usage with unlocked access to shared state

```python
# ANTIPATTERN - Lock used for writes but not reads (race condition)
class Cache:
    def __init__(self):
        self._data = {}
        self._lock = Lock()

    def get(self, key):
        return self._data.get(key)  # No lock!

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

# CORRECT - Lock for both reads and writes
class Cache:
    def __init__(self):
        self._data = {}
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
```

### 1.4: Non-atomic check-then-act sequences

```python
# ANTIPATTERN - Race condition between check and delete
if key in cache:
    time.sleep(0.001)  # Another thread may delete key here
    del cache[key]

# CORRECT - Atomic operation
with lock:
    cache.pop(key, None)
```

---

## 2. Concurrency

### 2.1: Modifying shared dicts without synchronization

```python
# ANTIPATTERN - Concurrent dict mutation causes RuntimeError
sessions = {}

def logout(token):
    if token in sessions:
        del sessions[token]  # Raises if another thread modifies dict

# CORRECT - Protect all mutations with a lock
import threading

class SessionStore:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def logout(self, token):
        with self._lock:
            self._sessions.pop(token, None)
```

### 2.2: Unprotected counter increments

```python
# ANTIPATTERN - Non-atomic increment (lost updates)
self._hits += 1

# CORRECT - Use threading.Lock or atomic operations
with self._lock:
    self._hits += 1
```

### 2.3: Sleeping between check and mutation

```python
# ANTIPATTERN - Sleep creates a race window
if key in self._cache:
    time.sleep(0.001)  # Window for concurrent modification
    del self._cache[key]

# CORRECT - Perform atomically under lock
with self._lock:
    self._cache.pop(key, None)
```

---

## 3. Data Handling

### 3.1: Floating-point precision errors in financial calculations

```python
# ANTIPATTERN - Float accumulation loses precision
subtotal = 0.0
for item in items:
    subtotal += item.quantity * item.unit_price  # Accumulates error

# CORRECT - Use Decimal for financial math
from decimal import Decimal, ROUND_HALF_UP

subtotal = Decimal("0")
for item in items:
    subtotal += Decimal(str(item.quantity)) * Decimal(str(item.unit_price))
subtotal = subtotal.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

### 3.2: Premature rounding in multi-step calculations

```python
# ANTIPATTERN - Rounding at each step compounds error
tax = round(subtotal * rate, 2)       # Rounded too early
fee = round(tax * 0.1, 2)             # Error compounds

# CORRECT - Round only the final result
tax = subtotal * rate
fee = tax * 0.1
total = round(subtotal + tax + fee, 2)
```

### 3.3: Integer division losing precision

```python
# ANTIPATTERN - Floor division drops decimal part
weight_charge = int(weight_kg) * 2.0   # int() truncates
shipping = total_weight // 10           # // discards remainder

# CORRECT - Use float division or Decimal
weight_charge = float(weight_kg) * 2.0
shipping = total_weight / 10.0
```

### 3.4: Off-by-one errors in range and slicing

```python
# ANTIPATTERN - range stops one short
for i in range(0, len(records) - 1, batch_size):
    batch = records[i:i + batch_size]

# CORRECT - range should cover full length
for i in range(0, len(records), batch_size):
    batch = records[i:i + batch_size]
```

---

## 4. Error Handling

### 4.1: Swallowing exceptions silently

```python
# ANTIPATTERN - Errors hidden, impossible to debug
try:
    result = transform(record)
except Exception:
    pass  # Silently swallowed

# CORRECT - Log and either re-raise or handle explicitly
import logging
logger = logging.getLogger(__name__)

try:
    result = transform(record)
except ValueError as e:
    logger.warning("Invalid record %s: %s", record.get("id"), e)
    result = None
except Exception as e:
    logger.error("Transform failed: %s", e)
    raise
```

### 4.2: Missing null checks on optional values

```python
# ANTIPATTERN - None causes TypeError or AttributeError
def validate(token):
    session = sessions.get(token)
    return session["user_id"]  # Crashes if session is None

# CORRECT - Check for None explicitly
def validate(token):
    if token is None:
        return None
    session = sessions.get(token)
    if session is None:
        return None
    return session["user_id"]
```

### 4.3: Catching overly broad exceptions

```python
# ANTIPATTERN - Catches KeyboardInterrupt, SystemExit, etc.
except:
    pass

# CORRECT - Catch specific exceptions
except (ValueError, KeyError) as e:
    logger.warning("Data error: %s", e)
```

---

## 5. Performance

### 5.1: Unnecessary list creation for membership tests

```python
# ANTIPATTERN - O(n) lookup in a list
if tier in ["regular", "silver", "gold"]:
    ...

# CORRECT - O(1) lookup in a set
VALID_TIERS = {"regular", "silver", "gold"}
if tier in VALID_TIERS:
    ...
```

### 5.2: String concatenation in loops

```python
# ANTIPATTERN - O(n^2) due to immutable strings
result = ""
for item in items:
    result += str(item)

# CORRECT - Use join for efficient concatenation
result = "".join(str(item) for item in items)
```

### 5.3: Repeated len() calls on unchanging collections

```python
# ANTIPATTERN - Called every iteration
for i in range(len(records) - 1):
    process(records[i])

# CORRECT - Cache when the length is invariant
n = len(records)
for i in range(n):
    process(records[i])
```

---

## 6. Type Safety

### 6.1: Unvalidated dictionary key access

```python
# ANTIPATTERN - KeyError for unknown keys
discount = TIER_DISCOUNTS[customer_tier]
rate = TAX_RATES[region]

# CORRECT - Validate or use .get() with defaults
discount = TIER_DISCOUNTS.get(customer_tier, 0.0)
if customer_tier not in TIER_DISCOUNTS:
    raise ValueError(f"Unknown tier: {customer_tier}")
```

### 6.2: Missing range validation on numeric parameters

```python
# ANTIPATTERN - No bounds checking, allows negative or excessive values
def apply_coupon(total, coupon_percent):
    return total * (coupon_percent / 100)

# CORRECT - Validate ranges
def apply_coupon(total, coupon_percent):
    if not (0 <= coupon_percent <= 100):
        raise ValueError("coupon_percent must be between 0 and 100")
    return total * (coupon_percent / 100)
```

### 6.3: Assuming type without checking

```python
# ANTIPATTERN - Assumes integer input
weight_charge = int(weight_kg) * 2

# CORRECT - Validate and convert safely
if not isinstance(weight_kg, (int, float)):
    raise TypeError("weight_kg must be numeric")
if weight_kg < 0:
    raise ValueError("weight_kg must be non-negative")
weight_charge = float(weight_kg) * 2.0
```

### 6.4: Comparison operator mistakes in boundary conditions

```python
# ANTIPATTERN - Off-by-one from wrong operator
if total > FREE_SHIPPING_THRESHOLD:
    return 0.0  # Misses exact threshold amount

# CORRECT - Inclusive comparison
if total >= FREE_SHIPPING_THRESHOLD:
    return 0.0  # Covers exact threshold
```

---

## Bug Category Quick Reference

When reviewing code, watch for these categories:

| Category | Key Indicators | Common Locations |
|---|---|---|
| Resource Leak | `open()` without `with`, missing `finally` | File I/O, network connections |
| Race Condition | Shared state without locks, check-then-act | Caches, session stores, counters |
| Logic Error | Off-by-one, wrong operator, missing validation | Loops, conditionals, dict lookups |
| Data Precision | Float math, premature rounding, int division | Financial calculations, statistics |
| Security | Hardcoded secrets, f-string SQL, no input validation | Auth, database queries, config |
| Silent Failure | Bare `except`, `pass` in exception handler | Transform pipelines, batch processing |
