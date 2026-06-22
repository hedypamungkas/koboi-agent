---
name: bug-hunter
description: >
  Systematic bug hunting and vulnerability detection workflow.
  Use when the user asks to find bugs, analyze code for issues,
  debug problems, or perform security audits.
allowed-tools: search filesystem
---

# Bug Hunter Skill

Specialized bug-hunting patterns and investigation strategies for Python code.

## Bug Signature Patterns (Quick Reference)

| Pattern | grep regex | Typical Severity |
|---------|-----------|-----------------|
| SQL injection | `f".*SELECT.*\{` or `f".*INSERT.*\{` | P0 |
| Hardcoded secrets | `(password|secret|key|token)\s*=\s*["\'][^"\']` | P0 |
| Unvalidated webhooks | `webhook.*signature\|verify.*webhook` | P0 |
| Race conditions | `self\.\w+\[.*\]\s*=` without lock context | P1 |
| Division by zero | `/\s*\w+\s*(?!\s*if\s*\w)` | P1 |
| Missing cleanup | `open\(.*\)\s*$` without `with` | P1 |
| Unclosed connections | `\.(connect|post|get)\(` without `finally` | P1 |
| Float currency | `\*\s*0\.\d+` in money context | P2 |
| TTL unit mismatch | `(time|timestamp)\s*[\*\+].*1000` | P2 |
| Off-by-one | `range\(.*len\(.+\)\s*-\s*1` | P1 |
| Swallowed exceptions | `except.*:\s*\n\s*pass` | P2 |
| Missing validation | `(tier|region|category)\[.*\]` without `.get(` | P1 |

## Investigation Strategy

1. **SCOPE** -- Use `glob_find` to discover all relevant files in the directory
2. **SCAN** -- Use `grep_search` with the signature patterns above
3. **READ** -- Use `read_file` on flagged files for full context
4. **TRACE** -- Follow data flow from inputs to outputs
5. **REPORT** -- Use the structured output format from system prompt

## Common False Positives to Avoid

- String formatting with parameterized queries (`?` placeholders) is NOT SQL injection
- `time.sleep()` in a lock context is valid; outside a lock suggests race conditions
- Integer division (`//`) is intentional when floor behavior is needed
- `round()` on intermediate values is fine; on final monetary amounts it loses precision
- `try/except Exception` is valid when followed by logging; only flag empty `pass`
- Constants like `MERCHANT_API_KEY` in module scope may be intentional config; flag only if clearly hardcoded secrets
