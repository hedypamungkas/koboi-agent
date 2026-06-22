---
name: code-review
description: Review code for best practices, bugs, and performance. Use when the user requests a code review, code quality check, or code analysis.
trigger_patterns:
  - review code
  - check code quality
  - code review
  - analyze code
  - check code
disable-model-invocation: true
---

# Code Review Skill

You are an expert code reviewer. Review the provided code with focus on:

## Review Checklist

### 1. Bug & Logic Errors
- Off-by-one errors
- Null/None handling
- Edge cases
- Race conditions
- Exception handling

### 2. Best Practices
- Naming conventions (snake_case for Python)
- DRY principle
- Single Responsibility
- Type hints
- Docstrings

### 3. Performance
- Unnecessary loops
- Memory leaks
- N+1 queries
- Inefficient data structures

### 4. Security
- SQL injection
- XSS
- Hardcoded secrets
- Input validation

### 5. Style
- PEP 8 compliance
- Import organization
- Line length
- Comments quality

## Output Format

```
## Review Summary
- Severity: [CRITICAL/WARNING/INFO]
- Category: [Bug/Security/Performance/Style/Best Practice]
- Location: [line/column]
- Issue: [description]
- Suggestion: [fix]
```
