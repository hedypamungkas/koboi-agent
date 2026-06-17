# Security Coding Standards for Python Applications

> This document defines the security coding standards for Python applications.
> All developers must follow these guidelines to ensure code integrity and
> protect against common vulnerabilities as identified by OWASP.

---

## 1. Input Validation

All external input must be validated and sanitized before use. Never trust data from users, APIs, files, or environment variables.

### Rule 1.1: Always validate and bound-check input parameters

```python
# INCORRECT - No validation, arbitrary values accepted
def process_order(quantity, price):
    total = quantity * price
    return total

# CORRECT - Validate types and ranges
def process_order(quantity, price):
    if not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("quantity must be a positive integer")
    if not isinstance(price, (int, float)) or price <= 0:
        raise ValueError("price must be a positive number")
    if quantity > MAX_ORDER_QUANTITY:
        raise ValueError(f"quantity exceeds maximum of {MAX_ORDER_QUANTITY}")
    return quantity * price
```

### Rule 1.2: Use parameterized queries for all database operations (OWASP A03:2021)

```python
# INCORRECT - SQL injection via string formatting
query = f"SELECT * FROM users WHERE username = '{username}'"
cursor.execute(query)

# CORRECT - Parameterized query with placeholders
query = "SELECT * FROM users WHERE username = ?"
cursor.execute(query, (username,))
```

### Rule 1.3: Sanitize data used in file paths

```python
# INCORRECT - Path traversal vulnerability
filepath = os.path.join(base_dir, user_input)

# CORRECT - Resolve and verify the path stays within base_dir
filepath = os.path.realpath(os.path.join(base_dir, user_input))
if not filepath.startswith(os.path.realpath(base_dir)):
    raise ValueError("Invalid file path")
```

### Rule 1.4: Validate enum-like inputs against allowed values

```python
# INCORRECT - Unvalidated string used as dict key (KeyError risk)
discount = TIER_DISCOUNTS[customer_tier]

# CORRECT - Explicit validation with safe fallback
VALID_TIERS = {"regular", "silver", "gold", "platinum"}
if customer_tier not in VALID_TIERS:
    raise ValueError(f"Invalid customer_tier: {customer_tier}")
discount = TIER_DISCOUNTS[customer_tier]
```

### Rule 1.5: Always null-check optional or externally-provided values

```python
# INCORRECT - No null check, will crash on None
def validate_session(token):
    session = sessions.get(token)
    return session["user_id"]

# CORRECT - Explicit None handling
def validate_session(token):
    if token is None:
        raise ValueError("Token cannot be None")
    session = sessions.get(token)
    if session is None:
        return None
    return session["user_id"]
```

---

## 2. SQL Security

SQL injection remains one of the most critical web application vulnerabilities (OWASP A03:2021 - Injection).

### Rule 2.1: Never use f-strings or string concatenation for SQL queries

```python
# INCORRECT - SQL injection vulnerable
query = f"SELECT id, name FROM products WHERE category = '{category}'"
cursor.execute(query)

# CORRECT - Use parameterized queries
query = "SELECT id, name FROM products WHERE category = ?"
cursor.execute(query, (category,))
```

### Rule 2.2: Use ORM or query builders when available

```python
# INCORRECT - Raw SQL with manual interpolation
cursor.execute(f"UPDATE users SET email = '{email}' WHERE id = {user_id}")

# CORRECT - Use SQLAlchemy or similar ORM
session.query(User).filter(User.id == user_id).update({"email": email})
session.commit()
```

### Rule 2.3: Apply the principle of least privilege to database connections

```python
# INCORRECT - Using root/admin credentials for application queries
conn = sqlite3.connect("app.db")  # Often runs with excessive privileges

# CORRECT - Use a read-only connection for queries, restricted user for writes
def get_read_connection():
    conn = sqlite3.connect("app.db")
    conn.execute("PRAGMA query_only = ON")
    return conn
```

---

## 3. Authentication and Session Management

Authentication failures are ranked #7 in OWASP Top 10 (A07:2021).

### Rule 3.1: Never hardcode secrets, keys, or credentials

```python
# INCORRECT - Hardcoded secret key
SECRET_KEY = "sk-abc123-default-key-do-not-use-in-production"

# CORRECT - Load from environment or secret manager
import os
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is required")
```

### Rule 3.2: Enforce session expiration (TTL)

```python
# INCORRECT - Session never expires
sessions[token] = {"user_id": user_id}

# CORRECT - Store creation time and check TTL on validation
SESSION_TIMEOUT = 3600  # 1 hour

sessions[token] = {"user_id": user_id, "created_at": time.time()}

def validate_session(token):
    session = sessions.get(token)
    if not session:
        return None
    if time.time() - session["created_at"] > SESSION_TIMEOUT:
        del sessions[token]
        return None
    return session
```

### Rule 3.3: Enforce password strength requirements

```python
# INCORRECT - No password validation
def change_password(user, new_password):
    update_password(user, new_password)

# CORRECT - Validate password meets strength requirements
import re

def validate_password_strength(password: str) -> bool:
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain a lowercase letter")
    if not re.search(r"[0-9]", password):
        raise ValueError("Password must contain a digit")
    if not re.search(r"[^A-Za-z0-9]", password):
        raise ValueError("Password must contain a special character")
    return True
```

### Rule 3.4: Use thread-safe data structures for shared session stores

```python
# INCORRECT - Race condition: dict modified without lock
self._sessions[token] = {"user_id": user_id}

# CORRECT - Use locking for all reads and writes
import threading

class SessionStore:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def get(self, token):
        with self._lock:
            return self._sessions.get(token)

    def set(self, token, data):
        with self._lock:
            self._sessions[token] = data

    def delete(self, token):
        with self._lock:
            self._sessions.pop(token, None)
```

---

## 4. Cryptography

### Rule 4.1: Use modern hashing algorithms

```python
# INCORRECT - SHA-256 is fast and unsuitable for passwords
password_hash = hashlib.sha256(password.encode()).hexdigest()

# CORRECT - Use bcrypt or argon2 with salt
import bcrypt
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))
```

### Rule 4.2: Use secrets module for token generation

```python
# INCORRECT - Predictable token from hash
token = hashlib.sha256(f"{user_id}{time.time()}".encode()).hexdigest()

# CORRECT - Cryptographically random token
import secrets
token = secrets.token_urlsafe(32)
```

### Rule 4.3: Never roll your own crypto

```python
# INCORRECT - Custom encryption scheme
def encrypt(data, key):
    return bytes(b ^ key for b in data.encode())

# CORRECT - Use established library
from cryptography.fernet import Fernet
fernet = Fernet(key)
encrypted = fernet.encrypt(data.encode())
```

---

## 5. Error Handling and Logging

### Rule 5.1: Never swallow exceptions silently

```python
# INCORRECT - Exception caught and ignored
try:
    result = transform_record(record)
except Exception:
    pass

# CORRECT - Log the exception and handle appropriately
import logging
logger = logging.getLogger(__name__)

try:
    result = transform_record(record)
except Exception as e:
    logger.error("Failed to transform record %s: %s", record.get("id"), e)
    raise
```

### Rule 5.2: Do not expose sensitive data in error messages

```python
# INCORRECT - Exposes database details to caller
except sqlite3.Error as e:
    return {"error": f"Database error: {e}"}

# CORRECT - Log internally, return generic message
except sqlite3.Error as e:
    logger.error("Database error during auth: %s", e)
    return {"error": "Authentication failed"}
```

### Rule 5.3: Always clean up resources in finally blocks

```python
# INCORRECT - File not closed on exception
def export_data(data, filepath):
    f = open(filepath, "w")
    json.dump(data, f, indent=2)
    f.close()

# CORRECT - Use context manager for guaranteed cleanup
def export_data(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
```

---

## 6. File Operations

### Rule 6.1: Always use context managers for file I/O

```python
# INCORRECT - Resource leak if exception occurs before close
f = open(filepath, "w")
json.dump(data, f)
f.close()

# CORRECT - Context manager ensures cleanup
with open(filepath, "w") as f:
    json.dump(data, f, indent=2)
```

### Rule 6.2: Validate file paths against directory traversal

```python
# INCORRECT - User-supplied filename allows traversal
def read_report(filename):
    return open(f"/reports/{filename}").read()

# CORRECT - Sanitize and validate path
def read_report(filename):
    base = os.path.realpath("/reports")
    filepath = os.path.realpath(os.path.join(base, filename))
    if not filepath.startswith(base + os.sep):
        raise ValueError("Invalid filename")
    with open(filepath) as f:
        return f.read()
```

### Rule 6.3: Handle encoding explicitly

```python
# INCORRECT - Platform-dependent encoding
with open(filepath, "r") as f:
    data = f.read()

# CORRECT - Explicit UTF-8 encoding
with open(filepath, "r", encoding="utf-8") as f:
    data = f.read()
```

---

## 7. HTTP Security

### Rule 7.1: Always use HTTPS for external requests

```python
# INCORRECT - Unencrypted HTTP
response = requests.get(f"http://api.example.com/users/{user_id}")

# CORRECT - HTTPS with certificate verification
response = requests.get(
    f"https://api.example.com/users/{user_id}",
    verify=True,
    timeout=30,
)
```

### Rule 7.2: Set security headers on responses

```python
# INCORRECT - No security headers
return Response(content)

# CORRECT - Include standard security headers
headers = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
}
return Response(content, headers=headers)
```

### Rule 7.3: Validate and limit redirect chains

```python
# INCORRECT - Unlimited redirects
response = requests.get(url, allow_redirects=True)

# CORRECT - Limit redirects
response = requests.get(url, allow_redirects=True, max_redirects=3)
```

---

## 8. Dependency Management

### Rule 8.1: Pin dependency versions

```python
# INCORRECT - Unpinned dependency in requirements.txt
flask
requests

# CORRECT - Pinned with hash verification
flask==3.0.0
requests==2.31.0
# Use pip-compile with --generate-hashes for hash pinning
```

### Rule 8.2: Regularly audit dependencies for vulnerabilities

```bash
# Run security audits in CI/CD pipeline
pip audit
safety check --full-report
```

### Rule 8.3: Use virtual environments to isolate dependencies

```bash
# INCORRECT - Installing globally
pip install package

# CORRECT - Use isolated environment
python -m venv .venv
source .venv/bin/activate
pip install package
```

---

## Quick Reference: OWASP Top 10 (2021) Mapping

| OWASP Category | Relevant Sections |
|---|---|
| A01 - Broken Access Control | Sections 3, 6 |
| A02 - Cryptographic Failures | Section 4 |
| A03 - Injection | Sections 1, 2 |
| A04 - Insecure Design | Sections 1, 3, 7 |
| A05 - Security Misconfiguration | Sections 3, 8 |
| A06 - Vulnerable Components | Section 8 |
| A07 - Auth Failures | Section 3 |
| A08 - Data Integrity Failures | Sections 1, 8 |
| A09 - Logging Failures | Section 5 |
| A10 - SSRF | Section 7 |
