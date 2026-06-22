---
name: code-review
description: Systematic code review focusing on security, quality, and performance
license: MIT
disable-model-invocation: true
disallowed-tools: shell
---

# Code Review Skill

## Instructions

When this skill is activated, review code systematically:

1. **Read**: Use `read_file` to examine the code
2. **Security**: Check for injection, XSS, auth issues, sensitive data exposure
3. **Quality**: Check naming, structure, error handling, test coverage
4. **Performance**: Check for N+1 queries, memory leaks, unnecessary allocations
5. **Report**: Provide structured feedback

## Output Format

- **Overall Assessment**: PASS / NEEDS CHANGES / REJECT
- **Security Issues**: Critical/High/Medium/Low
- **Quality Issues**: With specific line references
- **Performance Issues**: With suggested fixes
- **Recommendations**: Ordered by priority
