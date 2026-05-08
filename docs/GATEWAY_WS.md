# OpenClaw gateway WebSocket protocol

Notes from probing the gateway on 2026-05-06 (gateway v2026.5.5, source at
`/usr/lib/node_modules/openclaw/dist/`). Captured here so we don't have
to reverse-engineer it again if we ever decide to move the bridge off
SSE.

**The bridge currently uses SSE** (`gateway_chat_stream` →
`/v1/chat/completions` with `stream: true`). This document describes the
*alternative* WS path. See "Why we didn't switch" at the bottom.

## Endpoint

The gateway upgrades any HTTP path on its listening port to a WebSocket
connection — `/`, `/ws`, `/v1/realtime`, `/v1/chat/completions` all work
identically. There is one handler that does protocol negotiation
post-upgrade; the URL is not used for routing.

```
ws://127.0.0.1:18789/
```

The same listener serves the OpenAI-compatible HTTP routes and the WS
protocol. Auth lives in the WS frames, not in headers, so this isn't
the same as "websocketifying" `/v1/chat/completions`.

## Frame envelope

Every frame is a JSON object with a `type` discriminator (defined in
`server-ws-runtime-B7UE28UM.js` / `protocol-ByTcB0og.js`):

| `type`  | Direction | Shape |
|---------|-----------|-------|
| `req`   | client → server | `{type:"req", id, method, params?}` — RPC request, the server replies with a `res` carrying the same `id` |
| `res`   | server → client | `{type:"res", id, ok, payload?, error?}` |
| `event` | server → client | `{type:"event", event, payload?, seq?, stateVersion?}` |

`error` shape: `{code, message, details?, retryable?, retryAfterMs?}`.
`code` is one of `ErrorCodes` (e.g. `INVALID_REQUEST`, `UNAVAILABLE`,
`NOT_PAIRED`).

## Handshake

1. **Open the WS.** Server immediately pushes:
   ```json
   {"type":"event","event":"connect.challenge",
    "payload":{"nonce":"<uuid>","ts":<epoch_ms>}}
   ```
   The `nonce` is **only** consumed by the device-pair signature flow
   (`device.nonce` in the connect params, with a server-side check that
   `providedNonce === connectNonce`). For bearer-token auth on loopback
   the nonce can be ignored.

2. **Client sends `connect` request** (must be the very first frame
   from client to server — anything else closes with code 1008 / "invalid
   handshake"):

   ```json
   {
     "type": "req",
     "id": "<uuid>",
     "method": "connect",
     "params": {
       "minProtocol": 3,
       "maxProtocol": 3,
       "client": {
         "id": "gateway-client",
         "mode": "backend",
         "version": "0.0.1",
         "platform": "linux"
       },
       "role": "operator",
       "scopes": ["operator.read", "operator.write"],
       "auth": {"token": "<gateway bearer token>"}
     }
   }
   ```

   `ConnectParamsSchema` is in `protocol-ByTcB0og.js`. Notable fields
   beyond the above: optional `caps`, `commands`, `permissions`,
   `device` (publicKey/signature/signedAt/nonce — paired-device flow
   only), `auth.bootstrapToken` / `auth.deviceToken` / `auth.password`
   alternatives, `locale`, `userAgent`.

   **Protocol version is `3`** as of v2026.5.5 — a v1 connect gets
   rejected with `INVALID_REQUEST: "protocol mismatch"` and
   `details.expectedProtocol: 3`.

3. **Server replies with `hello-ok`**:
   ```json
   {
     "type":"res","id":"<same as connect>","ok":true,
     "payload":{
       "type":"hello-ok",
       "protocol":3,
       "server":{"version":"2026.5.5","connId":"<uuid>"},
       "features":{"methods":[...],"events":[...]},
       "snapshot":{...},
       "auth":{"role":"operator","scopes":["operator.read","operator.write"], ...},
       "policy":{"maxPayload":..., "maxBufferedBytes":..., "tickIntervalMs":...}
     }
   }
   ```

   Verify `payload.auth.scopes` actually contains what you asked for.
   The gateway silently strips scopes the auth method isn't allowed to
   bind (see next section).

After `hello-ok` you can call any method listed in
`payload.features.methods`, subject to scopes. Server starts emitting
periodic `event:"health"` and `event:"tick"` frames; ignore them.

## Why `client.id="gateway-client"` and `mode="backend"`

This is the key to making the bridge work without the device-pair
flow. The connect handler clears any requested scopes when there's no
device identity — except when `shouldSkipLocalBackendSelfPairing`
(in `message-handler-DZdD0nqB.js`) returns true, which requires:

1. `client.id === "gateway-client"` AND `client.mode === "backend"`
2. Connection locality is `direct_local` or `shared_secret_loopback_local`
   (loopback peer + no proxy/origin headers)
3. Token (or password / device-token) auth succeeded

When all three hold, the requested `scopes` are preserved — so a
local backend on 127.0.0.1 with a valid bearer can self-grant
`operator.read` + `operator.write`.

Other client IDs we tried (`cli` / `cli` mode) connect successfully but
end up with `scopes: []`, so every subsequent method call returns
`INVALID_REQUEST: "missing scope: operator.read"`.

## chat.send and the chat event stream

Send the user message:

```json
{
  "type":"req","id":"<uuid>","method":"chat.send",
  "params":{
    "sessionKey":"agent:main:voice-bridge",
    "message":"<user text>",
    "idempotencyKey":"<uuid>"
  }
}
```

Schema `ChatSendParamsSchema` (also accepts: `sessionId`, `thinking`,
`deliver`, `originatingChannel/To/AccountId/ThreadId`, `attachments`,
`timeoutMs`, `systemInputProvenance`, `systemProvenanceReceipt`).
`idempotencyKey` is **required** and is also used as the run ID — the
server uses it to dedupe re-sends.

Server replies `res ok=true` almost immediately (the run is queued/
started — the actual model output comes via events).

Then the server emits `event:"chat"` frames whose `payload` matches
`ChatEventSchema`:

```json
{
  "type":"event","event":"chat",
  "payload":{
    "runId":"<uuid>",
    "sessionKey":"agent:main:voice-bridge",
    "seq": <int>,
    "state":"delta" | "final" | "aborted" | "error",
    "message": {"role":"assistant","content":[{"type":"text","text":"..."}],
                "timestamp":<epoch_ms>},
    "errorMessage"?: "...",
    "errorKind"?: "refusal"|"timeout"|"rate_limit"|"context_length"|"unknown",
    "usage"?: {...},
    "stopReason"?: "..."
  }
}
```

`final` (or `aborted` / `error`) closes the run.

### CRITICAL: deltas are cumulative, not incremental

Unlike OpenAI SSE (where each delta carries the next few tokens),
each WS `state:"delta"` event carries the **entire assistant message
accumulated so far**. Sample run for "Spiega in tre frasi cosa è il
sole.":

| seq | text length | text snippet |
|-----|-------------|--------------|
| 2   | 9           | `Il Sole è` |
| 6   | 63          | `Il Sole è una stella gialla al centro del nostro sistema solare` |
| 9   | 109         | `… composta principalmente …` |
| 14  | 169         | `… continued …` |
| 20  | 238         | `…` |
| 26  | 296         | `…` |
| 28  | 298         | (final, `state:"final"`) |

To feed an incremental TTS pipeline, the consumer must keep the
last-seen length and slice each new delta:

```python
last = ""
def incremental(payload):
    global last
    full = "".join(p["text"] for p in payload["message"]["content"]
                   if p.get("type") == "text")
    delta, last = full[len(last):], full
    return delta
```

This is the main reason the WS path is more code than the SSE path —
SSE deltas are already incremental.

## Other useful methods

From `server-methods-list-DT1gCczU.js` — relevant ones for a future
voice bridge:

- `chat.history` — paginated history fetch (READ scope)
- `chat.abort` — cancel an in-flight run by `runId` or `sessionKey`
- `sessions.subscribe` / `sessions.messages.subscribe` — push session
  events onto this connection
- `sessions.list` / `sessions.describe` / `sessions.preview`
- `health` (no scope required)

## Why we didn't switch

Measured against the same gateway, same prompt
(`"Rispondi con una sola parola: ok"`):

| Path | TTFT | full latency |
|------|-----:|-------------:|
| SSE (`gateway_chat_stream`)             |  8.5 s |  8.7 s |
| WS  (`chat.send` + `chat` event stream) | 13.5 s | 13.7 s |

SSE actually beats WS on time-to-first-byte for a single turn, because
`chat.send` queues the run and yields control back before the model
starts generating. The WS path's theoretical advantages are:

- One persistent connection across turns (skip per-turn TLS + auth
  handshake) — only matters if we keep state across turns, which adds
  reconnect / ping handling.
- Access to other gateway methods over the same socket (history,
  abort, session subscriptions). Not currently needed by the bridge.

For the current single-turn-per-button-press model, SSE wins on
simplicity (no handshake, no protocol version pinning, no
cumulative→incremental diffing, no reconnect logic) at no latency cost.
Revisit if any of these shifts:

- We start chaining multiple turns inside one wake event.
- We want barge-in (user-press-mid-playback → `chat.abort` over WS).
- The gateway's SSE path regresses or is removed.
