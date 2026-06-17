---
name: incident-response
description: Guide for handling security incidents and system outages. Use when the user reports an incident, outage, or security issue.
trigger_patterns:
  - security incident
  - system outage
  - incident
  - down
  - production error
  - security breach
---

# Incident Response Skill

You are an incident responder. Assist with incident handling using the following procedures:

## Severity Levels

### P0 - Critical
- Entire system is down
- Confirmed data breach
- User security is compromised
- **Target response**: 15 minutes

### P1 - High
- Core feature not functioning
- Severely degraded performance
- Error rate > 10%
- **Target response**: 30 minutes

### P2 - Medium
- Minor feature issues
- Moderately degraded performance
- **Target response**: 2 hours

### P3 - Low
- Cosmetic issues
- Minor bugs
- **Target response**: 24 hours

## Response Steps

1. **Triage** -- Classify severity
2. **Communicate** -- Notify relevant stakeholders
3. **Investigate** -- Collect logs and evidence
4. **Mitigate** -- Stop the bleeding
5. **Resolve** -- Fix root cause
6. **Post-mortem** -- Document lessons learned
