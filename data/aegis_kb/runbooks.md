# Northwind Cloud -- Infra Runbooks

## Checkout service 500s

If `checkout-service` returns HTTP 500 on `/api/checkout`, the most common cause
is a stale database connection pool after a deploy. First check
`git_status`/`git_log` on the deploy workdir to confirm the last release, then
inspect recent error logs. Restarting the `checkout-service` pod clears a stale
pool in >90% of cases. Do not run destructive commands (`rm -rf`, force-push)
against production directly from this agent -- file a ticket instead if a
restart does not resolve it.

## Database connection pool exhaustion

Symptoms: 500s clustered right after a deploy, `ECONNREFUSED` in logs. Fix:
restart the affected service; if it recurs within an hour, escalate to a human
operator (this is the standard trigger for handover).

## Auth service outage

`auth-service` down means every downstream service (checkout, billing,
api-gateway) will report cascading errors. Check `auth-service` health first
before diagnosing anything downstream.
