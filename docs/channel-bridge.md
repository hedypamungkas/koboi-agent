# Channel-Bridge — Omnichannel Handover Architecture (Wave 3 B5)

koboi is the **bot-leg** inside a host CS platform's switchboard (Zendesk/Flex/Genesys/Twilio).
It speaks HTTP/SSE; it does **not** own channel joining (WhatsApp/SMS/email/voice) — that is the
host platform's job. This doc describes how the pieces connect for an omnichannel CS deployment
with confidence-based handover.

## The two webhook surfaces

| Path | Config | Event | Fires when |
|---|---|---|---|
| **Chat handover** (`/chat/stream`) | `handover.webhooks` | `handover.requested` | A live chat yields to a human (B1 tool / B1.5 structural). Mid-conversation. |
| **Job handover** (`/v1/jobs`) | `jobs.webhooks` | `job.awaiting_human` | An autonomous job yields (B1 in a job). Terminal status. |

Both are HMAC-SHA256-signed (`X-Koboi-Signature: sha256=...`), fire-and-forget, retried on 5xx/network.
Config shape (mirror):

```yaml
handover:
  webhooks:
    - url: "https://cs-platform.example.com/koboi-handover"
      secret: "${HANDOVER_WEBHOOK_SECRET}"
      timeout: 15
```

## The end-to-end handover flow

```
Customer (any channel) → [host adapter] → POST /v1/chat/stream (session_id=S, X-Session-Id: S)
   bot answers simple turns (RAG + A3 grounding)
   complex/low-confidence → B1/B1.5 yields → AgentHandoverError
      → HandoverEvent{handover_id, reason, summary(redacted; B4 digest if enabled)} on the SSE stream
      → session_events buffer (B2 replay)
      → handover.webhooks fires "handover.requested" → host CS platform router
          → router picks an operator (skill/presence/SLA)
   operator console: GET /v1/sessions/S/stream (B2 replay: history + digest)
                    POST /v1/sessions/S/transfer (ownership → operator)
                    POST /v1/chat/stream (S) → drives the conversation
```

## Inbound channel adapters (the host's job — NOT built into koboi)

koboi has **no inbound receiver** (no WhatsApp/SMS/email route). An adapter is an external bridge:

1. **Map channel identity → stable `session_id`.** e.g. WhatsApp `from_number` → a deterministic
   `session_id` (so the same customer's messages land in the same koboi session across turns).
2. **Inbound message → koboi.** For **synchronous** channels (web chat, live): `POST /v1/chat/stream`
   with `X-Session-Id: <session_id>` (SSE — the adapter streams the reply back to the channel). For
   **asynchronous** channels (WhatsApp/SMS/email — no live SSE): `POST /v1/jobs` with
   `session_id: <session_id>` (202; the adapter polls `GET /v1/jobs/{id}` or consumes the terminal
   `jobs.webhooks` POST, then forwards the reply to the channel).
3. **Outbound reply → channel.** The adapter forwards koboi's reply (from the SSE stream or the job
   result) to the channel's messaging API. The existing **outbound** pattern is
   `examples/33_command_hook_messaging.py` (a command hook forwarding each LLM response to a
   WhatsApp/Telegram/Slack stand-in).

`session_id` is the **omnichannel anchor** — identity lives above the channel, on `session_id`
(memory, journal, ownership, B2 replay buffer all key on it). `/transfer` reassigns ownership of a
`session_id`, so the operator's turns continue the same conversation regardless of channel.

## What koboi ships vs the host platform

| koboi (bot-leg) | host CS platform |
|---|---|
| confidence-aware abstain (A3) + handover (B1/B1.5) | channel joining (WhatsApp/SMS/email/voice) |
| `HandoverEvent` + B2 replay + B4 digest | operator routing (skill/presence/SLA), inbox |
| `handover.webhooks` / `jobs.webhooks` (notify) | operator UI, case-card rendering |
| `/transfer` (ownership) + `/chat/stream` (take over) | conversation continuity across channels |
| calibration harness (A5, offline) | labeled-data ownership (PPI) |

koboi **emits** the handover signal + provides the replay/take-over API; the host platform **routes**
+ **renders** + **joins channels**. This is the "bot-leg only" positioning — koboi does not compete
with Zendesk/Twilio on channel infrastructure.
