"""Gateway legs and turn-text helpers.

Three wire protocols, one iterator contract (yield reply text deltas):
`gateway_chat_stream` (OpenClaw SSE), `gateway_chat_stream_zeroclaw`
(`/webhook`, one JSON reply) and `gateway_chat_stream_zeroclaw_ws`
(`/ws/chat`, answer chunks only). `gateway_chat` is the non-streaming
OpenClaw call kept for tests and ad-hoc use.

The text helpers classify a turn: `_filter_no_reply` / `_is_no_reply`
(the agent chose silence), `_is_non_speech` (STT heard only noise tags)
and `_expects_answer` (the reply leaves the user something to answer).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Iterable, Iterator

log = logging.getLogger("voice-bridge")

GATEWAY_FALLBACK_REPLY = "Mi dispiace, ho avuto un problema di connessione."


def _openclaw_request(base_url: str, token: str, text: str, voice_model: str,
                      session_key: str, stream: bool) -> "urllib.request.Request":
    """`POST /v1/chat/completions` for one user turn (OpenClaw leg)."""
    payload = json.dumps({
        "model": voice_model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 500,
        "stream": stream,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if stream:
        headers["Accept"] = "text/event-stream"
    if session_key:
        headers["X-OpenClaw-Session-Key"] = session_key
    return urllib.request.Request(f"{base_url}/v1/chat/completions",
                                  data=payload, headers=headers)


def gateway_chat(base_url: str, token: str, text: str, voice_model: str, session_key: str = "voice-bridge") -> str:
    """Send user transcript to OpenClaw gateway and get response text."""
    req = _openclaw_request(base_url, token, text, voice_model, session_key, stream=False)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            log.info('HTTP Request: POST %s "HTTP/1.1 %d %s"', req.full_url, resp.status, resp.reason)
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as exc:
        log.error("Gateway error: %s", exc)
        return GATEWAY_FALLBACK_REPLY

# Tokens the agent emits to signal "stay silent on this turn" (e.g. when the
# user utterance was just background noise). The bridge intercepts these
# before they hit TTS and plays a short low beep instead.
NO_REPLY_SENTINELS: frozenset[str] = frozenset({"NO_REPLY", "NOREPLY", "NO-REPLY"})
_NO_REPLY_MAX_LEN = max(len(s) for s in NO_REPLY_SENTINELS)


# The voice agent opens every spoken reply with "... " (a silent breath for
# eleven_v3, see the voice-reply-style skill), so "... NO_REPLY" must count.
_NO_REPLY_LEAD = " \t\n.…"


def _is_no_reply(text: str) -> bool:
    return (text or "").strip().lstrip(_NO_REPLY_LEAD).strip() in NO_REPLY_SENTINELS


def _filter_no_reply(stream: Iterable[str]) -> Iterator[str]:
    """Wrap a delta stream and swallow it entirely if it strips to a
    NO_REPLY sentinel. Otherwise yield deltas unchanged.

    Buffers up to a few characters (enough to distinguish a sentinel from
    a real reply) before it commits to a passthrough — this only delays
    first-audio by one or two SSE deltas in the normal case, and avoids a
    wasted TTS HTTP call when the agent decided to stay silent.
    """
    buf = ""
    holding = True
    for delta in stream:
        if not holding:
            yield delta
            continue
        buf += delta
        if len(buf.lstrip(_NO_REPLY_LEAD)) > _NO_REPLY_MAX_LEN + 2:
            yield buf
            buf = ""
            holding = False
    if holding and buf:
        if _is_no_reply(buf):
            return
        yield buf


# Non-verbal events an STT provider transcribes instead of returning an empty
# string: ElevenLabs emits bracketed audio-event tags ("[click]", "[rumore di
# fogli]", "[rumore di sottofondo]"), other engines use parentheses. They are
# non-empty strings, so without this guard they reach the gateway as a real
# user turn and the agent answers a noise — which the speaker replays into the
# still-open mic (`play_pcm` unmutes for external speech), transcribes as more
# noise, and the loop feeds itself. Same "nothing to say" decision as a
# NO_REPLY reply, taken one stage earlier and without the round-trip.
_NON_SPEECH_TAG_RE = re.compile(r"[\[(][^\])]*[\])]")


def _is_non_speech(text: str) -> bool:
    """True when `text` is only audio-event tags and punctuation, i.e. the
    utterance carried no words at all. A tag mixed with real speech
    ("[click] accendi la luce") is speech and passes through unchanged."""
    return not any(ch.isalnum() for ch in _NON_SPEECH_TAG_RE.sub(" ", text))


# Closing invitations that expect an answer without a "?": "Dimmi tu il
# titolo e la metto!", "Rifammi il pensiero, che ci sono."
_INVITE_RE = re.compile(
    r"\b(dimmi|ditemi|dimmelo|ditemelo|raccontami|fammi sapere|fatemi sapere|"
    r"rifammi|ripetimi|ripeti|riprova\w*|prova a)\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def _expects_answer(text: str) -> bool:
    """True when the spoken reply leaves the user something to answer: a "?"
    in its last two sentences ("Vuoi la remix? Te la metto se dici sì.") or
    an invitation in the last one ("Dimmi tu il titolo!"). eleven_v3 tone
    tags ("[warm]") are ignored."""
    stripped = _NON_SPEECH_TAG_RE.sub(" ", text or "")
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(stripped.strip())
                 if any(ch.isalnum() for ch in s)]
    if not sentences:
        return False
    return any("?" in s for s in sentences[-2:]) or bool(_INVITE_RE.search(sentences[-1]))


def gateway_chat_stream(
    base_url: str,
    token: str,
    text: str,
    voice_model: str,
    session_key: str = "voice-bridge",
) -> Iterator[str]:
    """Same shape as `gateway_chat`, but yields content deltas as they arrive.

    Posts with `stream: true` and parses the OpenAI-style SSE response
    (`data: {...}\\n\\n`, terminated by `data: [DONE]`). Yields each
    `choices[0].delta.content` string. Malformed or non-data lines are
    skipped silently — the OpenAI spec allows comments and keep-alives.

    On transport error this yields the same fallback string `gateway_chat`
    returns, so the downstream TTS still has *something* to speak. This
    means an empty stream really does mean "no content" (the model said
    nothing), distinct from "the request blew up."
    """
    req = _openclaw_request(base_url, token, text, voice_model, session_key, stream=True)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            log.info('HTTP Request: POST %s "HTTP/1.1 %d %s"', req.full_url, resp.status, resp.reason)
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                if data == "[DONE]":
                    return
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                try:
                    delta = ev["choices"][0].get("delta", {}).get("content")
                except (KeyError, IndexError, TypeError):
                    delta = None
                if delta:
                    yield delta
    except Exception as exc:
        log.error("Gateway streaming error: %s", exc)
        yield GATEWAY_FALLBACK_REPLY


def gateway_chat_stream_zeroclaw(
    base_url: str,
    token: str,
    text: str,
) -> Iterator[str]:
    """zeroclaw gateway leg — same iterator contract as `gateway_chat_stream`.

    zeroclaw's gateway is NOT OpenAI-compatible: it exposes `POST /webhook`
    expecting ``{"message": "..."}`` with ``Authorization: Bearer <token>``
    and replies with a single, non-streaming JSON ``{"model": ..., "response":
    "..."}``. There is no SSE and no session header — conversational context
    is keyed by the bearer token itself (the paired token *is* the session),
    so `voice_model` and `session_key` have no place here.

    The whole reply text is yielded as one delta; the downstream sentence
    buffer (`http_sentence` TTS mode) splits it for synthesis. On transport
    error this yields the same fallback string the OpenClaw leg uses, so the
    TTS stage always has something to speak.
    """
    url = f"{base_url}/webhook"
    payload = json.dumps({"message": text}).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            log.info('HTTP Request: POST %s "HTTP/1.1 %d %s"', url, resp.status, resp.reason)
            body = resp.read().decode("utf-8", errors="replace")
        try:
            reply = (json.loads(body).get("response") or "").strip()
        except json.JSONDecodeError:
            log.error("zeroclaw gateway: non-JSON body: %.200s", body)
            reply = ""
        if reply:
            yield reply
    except Exception as exc:
        log.error("Gateway streaming error: %s", exc)
        yield GATEWAY_FALLBACK_REPLY


# Spoken when the WS dropped AFTER the message reached the agent: it may
# already have acted (started a song, sent a mail), so don't pretend
# nothing happened and don't silently retry (that would act twice).
GATEWAY_LOST_REPLY = ("Ho perso la risposta per strada. Se mi avevi chiesto di fare "
                      "qualcosa, controlla se è partita.")
# Spoken when the agent could not be reached at all, even after a retry.
GATEWAY_UNREACHABLE_REPLY = "Non riesco a raggiungere l'assistente in questo momento. Riprova tra poco."


def gateway_chat_stream_zeroclaw_ws(
    base_url: str,
    token: str,
    text: str,
    agent: str = "default",
    session_id: str = "voice-bridge",
    on_event=None,
    connect_fn=None,
) -> Iterator[str]:
    """zeroclaw `/ws/chat` WebSocket leg — same iterator contract as the others.

    Unlike `/webhook` (non-streaming, which returns the tool loop's whole
    `accumulated_display_text` — for a reasoning model that string is the
    chain-of-thought narration *concatenated* with the answer), the WS leg
    streams typed frames and keeps reasoning separate:
      - ``{"type":"chunk","content":...}``    → the actual answer (yielded)
      - ``{"type":"thinking","content":...}`` → reasoning (dropped — never TTS'd)
      - ``tool_call`` / ``tool_result``       → tool activity (dropped)
      - ``approval_request``                   → auto-denied (voice can't approve)
      - ``done``                               → end of turn (its `full_response`
                                                 is the same lossy string; ignored)

    Every non-chunk frame is also reported to `on_event(type, frame)` if
    given (the bridge plays a "un attimo" cue on the first `tool_call`).

    Failure handling depends on how far the turn got:
      - before the message was sent (connect/upgrade failed): retried once,
        then `GATEWAY_UNREACHABLE_REPLY` — nothing reached the agent, so a
        retry can't make it act twice;
      - after the message was sent but before any answer text:
        `GATEWAY_LOST_REPLY`, no retry (the agent may already have acted);
      - after some answer text was yielded: stop quietly, the user already
        heard part of the answer and an apology tacked on would be noise.

    Connects to ``ws(s)://<host>/ws/chat?agent=<agent>&session_id=<id>`` with
    the paired token as the ``?token=`` query param. Conversational context is
    keyed by ``session_id`` (the gateway namespaces it as ``gw_<id>``), so a
    stable id keeps every turn in the same session. `connect_fn` is injectable
    for tests (defaults to ``websockets.sync.client.connect``).
    """
    from urllib.parse import urlsplit, urlunsplit, urlencode
    if connect_fn is None:
        from websockets.sync.client import connect as connect_fn

    parts = urlsplit(base_url)
    ws_scheme = "wss" if parts.scheme == "https" else "ws"
    query = urlencode({"agent": agent, "session_id": session_id, "token": token})
    url = urlunsplit((ws_scheme, parts.netloc, "/ws/chat", query, ""))

    for attempt in (1, 2):
        sent = yielded = False
        try:
            # open_timeout caps the upgrade; the per-recv timeout below caps
            # each frame wait so a stalled turn can't hang the worker forever.
            with connect_fn(url, max_size=None, open_timeout=30) as ws:
                log.info("WS connect: %s/ws/chat (agent=%s session=%s)",
                         base_url, agent, session_id)
                ws.send(json.dumps({"type": "message", "content": text}))
                sent = True
                while True:
                    raw = ws.recv(timeout=120)
                    try:
                        frame = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    ftype = frame.get("type")
                    if ftype == "chunk":
                        delta = frame.get("content")
                        if delta:
                            yielded = True
                            yield delta
                        continue
                    if on_event is not None:
                        try:
                            on_event(ftype, frame)
                        except Exception:
                            log.exception("WS on_event callback failed")
                    if ftype == "done":
                        return
                    elif ftype == "error":
                        log.error("WS gateway error frame: %s",
                                  frame.get("message", str(raw)[:200]))
                        if not yielded:
                            yield GATEWAY_LOST_REPLY
                        return
                    elif ftype == "approval_request":
                        # Voice has no interactive approval path; deny so the
                        # turn finishes instead of blocking on a prompt.
                        log.info("WS approval_request for tool %r → auto-deny",
                                 frame.get("tool"))
                        ws.send(json.dumps({
                            "type": "approval_response",
                            "request_id": frame.get("request_id"),
                            "decision": "deny",
                        }))
                    # thinking / tool_call / tool_result / session_start /
                    # chunk_reset / agent_end etc. are intentionally dropped.
        except Exception as exc:
            log.error("Gateway WS streaming error (attempt %d, sent=%s, answered=%s): %s",
                      attempt, sent, yielded, exc)
            if yielded:
                return
            if sent:
                yield GATEWAY_LOST_REPLY
                return
            if attempt == 1:
                log.info("WS: message never reached the agent — retrying once")
                continue
            yield GATEWAY_UNREACHABLE_REPLY
            return
        return
