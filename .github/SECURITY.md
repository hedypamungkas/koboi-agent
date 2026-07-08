# Security Policy

## Reporting a Vulnerability

We take security bugs in **koboi-agent** seriously. Please report them
responsibly so we can fix them before public disclosure.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, use a private channel:

1. **GitHub Security Advisory** (preferred): open the
   [Security tab](https://github.com/hedypamungkas/koboi-agent/security/advisories/new)
   and click **"Report a vulnerability"**. This supports private collaboration,
   CVSS scoring, and CVE issuance once a fix ships.

Please include:

- A description of the issue and its impact
- Steps to reproduce (a minimal config + the input that triggers it)
- Affected versions, if known
- Any suggested fix or mitigation

## Response Time

We aim to acknowledge reports within **3 business days** and to ship a fix or
mitigation within **30 days** for high-severity issues. Progress is shared
through the private advisory.

## Supported Versions

Only the latest minor release line receives security fixes.

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| < 0.6   | :x:                |

## Scope

This policy covers the `koboi-agent` Python package and the published Docker
image (`ghcr.io/hedypamungkas/koboi-agent`). Vulnerabilities in third-party
dependencies should be reported upstream; we track and patch vulnerable deps
via Dependabot and `pip-audit` (see `.github/workflows/ci.yml`).

## Disclosure

We follow coordinated disclosure: once a fix is released we publish a GitHub
Security Advisory and credit the reporter unless they prefer to remain anonymous.
