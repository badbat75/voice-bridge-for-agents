#!/usr/bin/env python3
"""
Binary Voice Bridge v3 — always-on mic, async pipeline.

Trigger: voice activity (no wake word, no PTT).
         Jabra HID button = hard-cancel toggle (stops/resumes everything).
STT:     Deepgram or ElevenLabs Scribe (configurable).
TTS:     Deepgram Aura or ElevenLabs (configurable, streaming).
Output:  ALSA aplay.

Four worker threads connected by queues:

    Recorder ──audio_q──▶ Endpointer ──utt_q──▶ Worker ──playback_q──▶ Player

Recorder keeps PyAudio open while `recording` is set; endpointer runs
RMS VAD per chunk and commits utterances on a configurable pause; worker
drives STT → gateway SSE → TTS streaming; player drives one aplay
subprocess per utterance. The bridge auto-idles (closes the mic stream)
after `idle_timeout_ms` of pure silence; an HID press resumes it. CPU is
low by design — recorder blocks in `stream.read`, endpointer does a
single RMS per ~64 ms chunk, worker/player are idle off-turn.
"""

from __future__ import annotations

import array
import contextlib
import json
import logging
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Iterable, Iterator

import pyaudio

from deepgram_voice import DeepgramVoice
from deezer_connect_plugin import DeezerConnectPlugin
from elevenlabs_voice import ElevenLabsVoice
from jabra_hid import HidMuteMonitor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("voice-bridge")
logging.basicConfig(
    level=logging.INFO,
    format="[voice-bridge] %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "voice-bridge.json")
# Local, gitignored secrets file (API keys + gateway token). The bridge
# is self-contained: everything it needs lives in this folder. See
# voice-bridge.secrets.example.json for the expected shape.
SECRETS_PATH = os.path.join(_HERE, "voice-bridge.secrets.json")

VALID_PROVIDERS = ("elevenlabs", "deepgram")
VALID_TTS_STREAM_MODES = ("http_sentence", "websocket")
VALID_GATEWAY_BACKENDS = ("openclaw", "zeroclaw")


def _camel_to_snake_keys(d: dict | None) -> dict | None:
    """Convert camelCase dict keys to snake_case (one level deep).

    The openclaw gateway config uses camelCase (`similarityBoost`,
    `useSpeakerBoost`) but the ElevenLabs Python SDK expects snake_case
    (`similarity_boost`, `use_speaker_boost`). We translate at the
    config-loading boundary so the rest of the code never has to think
    about it.
    """
    import re
    if not d:
        return d
    out = {}
    for k, v in d.items():
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", k).lower()
        out[snake] = v
    return out


def _read_json(path: str) -> dict:
    """Read a JSON object from `path`, or {} if it's missing/unreadable.

    Used for the optional local secrets file, which is allowed to be
    absent (e.g. a fresh checkout before keys are filled in).
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    # Self-contained config: secrets live in a local, gitignored file in
    # this folder. Everything else is in voice-bridge.json above.
    secrets = _read_json(SECRETS_PATH)

    # --- Secrets: local file first, env override for Deepgram ---
    cfg["gateway_token"] = secrets.get("gateway_token", "")
    cfg["deepgram_key"] = (
        os.environ.get("DEEPGRAM_API_KEY", "")
        or secrets.get("deepgram_api_key", "")
    )

    # --- Deepgram STT settings ---
    # The `deepgram` block in voice-bridge.json: `sttOptions` are extra
    # kwargs (smart_format, punctuate, ...) passed verbatim, `sttModel`
    # is the model name. Only used if a Deepgram provider is selected.
    dg_local = cfg.get("deepgram") or {}
    cfg["deepgram_stt_options"] = dg_local.get("sttOptions") or {}
    cfg["deepgram_stt_model"] = dg_local.get("sttModel") or ""

    # --- ElevenLabs TTS settings ---
    # The `elevenlabs` block in voice-bridge.json holds the non-secret
    # bits (voice id, model, language, voice_settings, text
    # normalization); the API key comes from the secrets file. Keys stay
    # camelCase here (`modelId`, not `model`); voiceSettings are
    # translated to the SDK's snake_case at this boundary.
    el = cfg.get("elevenlabs") or {}
    cfg["elevenlabs_key"] = secrets.get("elevenlabs_api_key", "")
    cfg["elevenlabs_voice"] = el.get("voiceId", "") or ""
    cfg["elevenlabs_model"] = el.get("modelId", "") or ""
    cfg["elevenlabs_language"] = el.get("languageCode")
    cfg["elevenlabs_voice_settings"] = _camel_to_snake_keys(el.get("voiceSettings")) or None
    cfg["elevenlabs_text_normalization"] = el.get("applyTextNormalization")

    cfg["voice_model"] = cfg.get("voice_model") or "openclaw"

    # Which gateway protocol the worker speaks. `openclaw` (default) posts
    # to `/v1/chat/completions` with OpenAI-style SSE; `zeroclaw` posts to
    # `/webhook` and parses a single non-streaming JSON reply. Selecting a
    # backend also implies which gateway `gateway_base_url`/`gateway_token`
    # point at — they are not interchangeable.
    cfg["gateway_backend"] = str(cfg.get("gateway_backend", "openclaw")).strip().lower()
    if cfg["gateway_backend"] not in VALID_GATEWAY_BACKENDS:
        raise ValueError(
            f"voice-bridge.json: unknown gateway_backend "
            f"{cfg['gateway_backend']!r}; valid: {VALID_GATEWAY_BACKENDS}"
        )

    # Session key for the voice bridge: from voice-bridge.json, with a
    # literal fallback.
    cfg["session_key"] = cfg.get("session_key") or "voice-bridge"

    # Provider selection lives in voice-bridge.json itself
    # (`stt_provider`, `tts_provider`). Missing keys → default to
    # ElevenLabs for both roles.
    cfg["stt_provider"] = str(cfg.get("stt_provider", "elevenlabs")).strip().lower()
    cfg["tts_provider"] = str(cfg.get("tts_provider", "elevenlabs")).strip().lower()
    for role in ("stt", "tts"):
        if cfg[f"{role}_provider"] not in VALID_PROVIDERS:
            raise ValueError(
                f"voice-bridge.json: unknown {role}_provider "
                f"{cfg[f'{role}_provider']!r}; valid: {VALID_PROVIDERS}"
            )

    # Output sample rate must be agreed upon by TTS request, the
    # synth library, and the aplay invocation. One number, one place.
    cfg["tts_sample_rate"] = int(cfg.get("tts_sample_rate", 22050))

    # Endpointer / VAD knobs. All configurable so the bridge can be
    # retuned per environment without touching code.
    #
    # - `vad_rms_threshold`: per-chunk energy above which a chunk is
    #   counted as speech. Same dimensionless metric the legacy
    #   `record_until_silence` used (sum(s²)/sqrt(N), not true RMS) so
    #   prior calibrations carry over. Quiet rooms typically need ~300;
    #   noisy ones higher.
    # - `silence_timeout_ms`: pause after speech that ends an utterance
    #   and pushes it down the pipeline. Don't push this below ~600 ms
    #   or natural between-word pauses get split into separate turns.
    # - `idle_timeout_ms`: total silence (no speech) after which the
    #   recording stream is closed entirely. Set to 0 to disable
    #   auto-idle (mic stays open until SIGTERM or HID press).
    cfg["vad_rms_threshold"] = float(cfg.get("vad_rms_threshold", 300))
    cfg["silence_timeout_ms"] = int(cfg.get("silence_timeout_ms", 800))
    cfg["idle_timeout_ms"] = int(cfg.get("idle_timeout_ms", 10000))
    # Trailing silence preserved in the committed PCM. The full
    # silence_timeout_ms window is captured to *detect* end-of-speech,
    # but only `silence_keep_ms` of it is included in the audio handed
    # to STT — the rest is trimmed. Keeping a small tail (default
    # 500 ms) helps STT models that use trailing silence as a
    # word-boundary cue without bloating each utterance with the full
    # detection window.
    cfg["silence_keep_ms"] = int(cfg.get("silence_keep_ms", 500))
    # Pre-roll: how much audio captured *before* the threshold-crossing
    # to prepend to the committed PCM. Helps STT catch the very first
    # phoneme, which often dips below the VAD threshold (the leading
    # consonant of a word can be quieter than its vowel). The bridge
    # keeps a rolling window of the last `pre_speech_keep_ms` of
    # below-threshold audio and pastes it in at speech onset.
    cfg["pre_speech_keep_ms"] = int(cfg.get("pre_speech_keep_ms", 100))

    # TTS streaming strategy. `http_sentence` (default) buffers gateway
    # deltas to sentence boundaries and calls the HTTP streaming endpoint
    # per sentence — works on every account tier. `websocket` feeds
    # deltas straight into ElevenLabs' realtime websocket for token-level
    # latency, but requires a paid tier (free accounts get HTTP 403 on
    # the upgrade). Only consulted when tts_provider == "elevenlabs".
    cfg["tts_streaming_mode"] = str(
        cfg.get("tts_streaming_mode", "http_sentence")
    ).strip().lower()
    if cfg["tts_streaming_mode"] not in VALID_TTS_STREAM_MODES:
        raise ValueError(
            f"voice-bridge.json: unknown tts_streaming_mode "
            f"{cfg['tts_streaming_mode']!r}; valid: {VALID_TTS_STREAM_MODES}"
        )

    return cfg


def _build_voice_provider(role: str, cfg: dict):
    """Construct the provider class chosen for `role` ('stt' or 'tts').

    Both classes expose the same transcribe/synthesize surface, so the
    caller doesn't have to care which one came back. The full TTS-side
    settings (voice id, model, voice_settings, language, text
    normalization) come straight from the openclaw config — we don't
    invent defaults beyond what the SDK itself uses.
    """
    name = cfg[f"{role}_provider"]
    if name == "elevenlabs":
        return ElevenLabsVoice(
            api_key=cfg["elevenlabs_key"],
            voice_id=cfg["elevenlabs_voice"],
            tts_model=cfg["elevenlabs_model"],
            tts_sample_rate=cfg["tts_sample_rate"],
            tts_language=cfg.get("elevenlabs_language"),
            tts_voice_settings=cfg.get("elevenlabs_voice_settings"),
            tts_text_normalization=cfg.get("elevenlabs_text_normalization"),
            tts_stream_mode=cfg.get("tts_streaming_mode", "http_sentence"),
        )
    if name == "deepgram":
        kwargs = {
            "api_key": cfg["deepgram_key"],
            "stt_options": cfg.get("deepgram_stt_options") or {},
            "tts_sample_rate": cfg["tts_sample_rate"],
        }
        if cfg.get("deepgram_stt_model"):
            kwargs["stt_model"] = cfg["deepgram_stt_model"]
        return DeepgramVoice(**kwargs)
    raise ValueError(f"unknown provider: {name}")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def find_input_device(pa: pyaudio.PyAudio) -> int | None:
    """Find Jabra SPEAK 510 input device index."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "jabra" in info["name"].lower() and info["maxInputChannels"] > 0:
            return i
    return None


_AUDIO_DEBUG = os.environ.get("VOICE_BRIDGE_DEBUG_AUDIO") == "1"


def _drain_aplay_stderr(stream) -> None:
    """Forward aplay's stderr line-by-line to the Python logger.

    Runs as a daemon thread; exits when aplay closes stderr.
    """
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                log.warning("aplay: %s", line)
    finally:
        with contextlib.suppress(Exception):
            stream.close()


def _aplay_popen(device: str, sample_rate: int, *, bufsize: int = -1) -> subprocess.Popen:
    """Spawn aplay for raw S16LE mono PCM at `sample_rate`.

    With `VOICE_BRIDGE_DEBUG_AUDIO=1`, drops `-q` and forwards aplay's
    stderr to the logger — that's where ALSA prints `underrun!!!` lines.
    """
    cmd = ["aplay", "-D", device, "-f", "S16_LE", "-r", str(sample_rate), "-c", "1"]
    if not _AUDIO_DEBUG:
        cmd.insert(1, "-q")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE if _AUDIO_DEBUG else subprocess.DEVNULL,
        bufsize=bufsize,
    )
    if _AUDIO_DEBUG and proc.stderr is not None:
        threading.Thread(
            target=_drain_aplay_stderr,
            args=(proc.stderr,),
            daemon=True,
        ).start()
    return proc


def play_audio(device: str, audio_data: bytes, sample_rate: int) -> None:
    """Play raw S16LE mono PCM bytes through aplay at `sample_rate`.

    `sample_rate` MUST match what the TTS provider produced (we ask it
    for `pcm_<rate>` / `linear16` at the same number) — otherwise aplay
    plays back at the wrong speed/pitch."""
    proc = _aplay_popen(device, sample_rate)
    proc.communicate(input=audio_data)


def play_audio_stream(device: str, audio_iter: Iterable[bytes], sample_rate: int) -> bool:
    """Pipe audio chunks through aplay as they arrive.

    aplay is started immediately (so ALSA acquires the device early) and
    each chunk is written to its stdin the moment it's pulled from
    `audio_iter`. ALSA's own period buffer absorbs short pauses in the
    producer (e.g. waiting on the next sentence from TTS). With
    `bufsize=0`, every write goes straight to the pipe — no Python-level
    buffering between TTS chunks and ALSA.

    Returns True if at least one chunk was written (i.e. something was
    actually played), so callers can distinguish "speech happened" from
    "stream produced nothing".
    """
    proc = _aplay_popen(device, sample_rate, bufsize=0)
    wrote_any = False
    try:
        for chunk in audio_iter:
            if not chunk:
                continue
            try:
                proc.stdin.write(chunk)
            except BrokenPipeError:
                # aplay died (device gone, ALSA error). Stop pulling
                # from the upstream iterators — still need to drain the
                # process below.
                break
            wrote_any = True
    finally:
        with contextlib.suppress(Exception):
            proc.stdin.close()
        proc.wait()
    return wrote_any


def _make_beep_pcm(sample_rate: int, freq: float, duration: float, amplitude: int = 16000) -> bytes:
    """Generate a fade-out sine beep as S16LE mono PCM bytes."""
    n_samples = int(sample_rate * duration)
    buf = array.array("h")
    for i in range(n_samples):
        env = 1.0 - (i / n_samples)
        val = int(amplitude * env * (0.5 + 0.5 * math.sin(2 * math.pi * freq * i / sample_rate)))
        buf.append(max(-32768, min(32767, val)))
    return buf.tobytes()


def play_beep(device: str, sample_rate: int) -> None:
    """Play a short ~80 ms 880 Hz beep at the given output sample rate."""
    play_audio(device, _make_beep_pcm(sample_rate, freq=880, duration=0.08), sample_rate)


# ---------------------------------------------------------------------------
# Gateway — send transcript, get response
# ---------------------------------------------------------------------------
def gateway_chat(base_url: str, token: str, text: str, voice_model: str, session_key: str = "voice-bridge") -> str:
    """Send user transcript to OpenClaw gateway and get response text."""
    import urllib.request

    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": voice_model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 500,
        "stream": False,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if session_key:
        headers["X-OpenClaw-Session-Key"] = session_key

    req = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            log.info('HTTP Request: POST %s "HTTP/1.1 %d %s"', url, resp.status, resp.reason)
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as exc:
        log.error("Gateway error: %s", exc)
        return "Mi dispiace, ho avuto un problema di connessione."


GATEWAY_FALLBACK_REPLY = "Mi dispiace, ho avuto un problema di connessione."

# Tokens the agent emits to signal "stay silent on this turn" (e.g. when the
# user utterance was just background noise). The bridge intercepts these
# before they hit TTS and plays a short low beep instead.
NO_REPLY_SENTINELS: frozenset[str] = frozenset({"NO_REPLY", "NOREPLY", "NO-REPLY"})
_NO_REPLY_MAX_LEN = max(len(s) for s in NO_REPLY_SENTINELS)


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
        if len(buf) > _NO_REPLY_MAX_LEN + 2:
            yield buf
            buf = ""
            holding = False
    if holding and buf:
        if buf.strip() in NO_REPLY_SENTINELS:
            return
        yield buf


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
    import urllib.request

    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": voice_model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 500,
        "stream": True,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream",
    }
    if session_key:
        headers["X-OpenClaw-Session-Key"] = session_key

    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            log.info('HTTP Request: POST %s "HTTP/1.1 %d %s"', url, resp.status, resp.reason)
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
    import urllib.request

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


# ---------------------------------------------------------------------------
# Async pipeline orchestrator
# ---------------------------------------------------------------------------
# Sentinel pushed into `playback_q` after each utterance's audio chunks
# so the player thread knows to close the current aplay process and
# wait for the next utterance. Plain None would conflict with empty-
# chunk filtering elsewhere; an explicit object is unambiguous.
class _EndOfUtterance:
    pass


_END_OF_UTTERANCE = _EndOfUtterance()


class VoiceBridge:
    """Always-on mic + 4-stage async pipeline + HID hard-cancel toggle.

    Threads (all daemons):

      - `_hid_loop`        polls HidMuteMonitor; triggers state toggles
      - `_recorder_loop`   PyAudio open/read while `recording` is set
      - `_endpointer_loop` RMS VAD; emits utterances on `silence_timeout_ms`
                           pauses, triggers auto-idle on `idle_timeout_ms`
      - `_worker_loop`     STT → gateway SSE → TTS streaming
      - `_player_loop`     one aplay subprocess per utterance

    Cancellation has two grades:

      - **Hard cancel** (HID press while recording): bumps `_gen`,
        drains every queue, kills the active aplay. Pipeline items
        carry the generation they were produced under; downstream
        stages drop anything whose generation has been superseded.
      - **Soft idle** (silence > `idle_timeout_ms`): clears `recording`
        (closes the mic stream) but does NOT bump `_gen`. Anything
        already queued continues to flow — the user gets the reply they
        were waiting for even though the mic is now idle.

    Resuming from idle/cancel is always an HID press: it sets
    `recording` and bumps `_gen` so any leftover stale chunks are shed.
    """

    def __init__(self, cfg: dict, stt, tts, hid: HidMuteMonitor,
                 deezer: DeezerConnectPlugin | None = None) -> None:
        self.cfg = cfg
        self.stt = stt
        self.tts = tts
        self.hid = hid
        # Optional deezer-connect ducking plugin. No-op unless enabled
        # in voice-bridge.json under `deezer_connect`. Constructed by
        # main() so tests can inject a fake.
        self.deezer = deezer or DeezerConnectPlugin(cfg.get("deezer_connect"))

        # `audio_q` is unbounded: the endpointer is O(N) over a 1024-
        # sample chunk per ~64 ms — easily faster than the recorder, so
        # the queue should stay near-empty in practice. The other queues
        # are also unbounded; backpressure is naturally bounded by an
        # utterance's duration (~10s of audio = ~250KB at 24kHz).
        self.audio_q: "queue.Queue[tuple[int, bytes]]" = queue.Queue()
        self.utterance_q: "queue.Queue[tuple[int, bytes, int]]" = queue.Queue()
        self.playback_q: "queue.Queue[tuple[int, bytes | _EndOfUtterance]]" = queue.Queue()

        self.shutdown_event = threading.Event()
        # `recording` gates the recorder thread. With HID enabled (the
        # canonical deployment), the bridge boots muted — the
        # HidMuteMonitor's engage write puts the device into firmware-
        # mute (LED red, USB capture silenced) and `recording` stays
        # clear until the user presses the button. Cleared/set in pairs
        # with `hid.set_led()` so device state always tracks `recording`.
        # If HID is disabled, fall back to "always-on" boot so there's
        # still a way to use the bridge — without HID there's nothing to
        # un-mute it from a muted boot.
        self.recording = threading.Event()
        if not cfg.get("hid_mute_enabled"):
            self.recording.set()

        self._gen = 0
        self._gen_lock = threading.Lock()

        # Set by the player after a playback completes so the endpointer
        # resets its silence counter — otherwise the 10s playback eats
        # into the idle window and the bridge auto-idles right after the
        # reply finishes. Effect: idle timer measures silence *after* the
        # last interaction (user speech OR our reply), not just user speech.
        self._idle_reset_pending = threading.Event()

        # Set by an HID-press while recording to tell the endpointer to
        # commit any in-progress speech buffer right now, instead of
        # waiting for `silence_timeout_ms`. The companion to "soft mute":
        # the user pressed mute, so we still send what they were saying
        # but stop listening for new input.
        self._force_commit = threading.Event()

        # Set by `_enter_idle` (auto-idle on silence), cleared on the
        # next user transition. The player checks it when the first PCM
        # chunk of a reply arrives: if the bridge auto-idled mid-turn
        # (silence timeout while the worker was still processing), the
        # player un-idles itself so the user can talk back the moment
        # the reply ends. If the user explicitly muted via HID, this
        # flag stays clear and the player respects the press — playback
        # happens but the mic stays muted afterwards.
        self._auto_idled = threading.Event()

        # The active aplay process, if any. Held under `_player_lock`
        # so a hard-cancel from the HID thread can `kill()` it without
        # racing the player thread's setup/teardown.
        self._player_proc: subprocess.Popen | None = None
        self._player_lock = threading.Lock()

        self._threads: list[threading.Thread] = []

    # -- generation helpers --------------------------------------------
    def _current_gen(self) -> int:
        with self._gen_lock:
            return self._gen

    def _bump_gen(self) -> int:
        with self._gen_lock:
            self._gen += 1
            return self._gen

    @staticmethod
    def _drain_queue(q: "queue.Queue") -> int:
        n = 0
        try:
            while True:
                q.get_nowait()
                n += 1
        except queue.Empty:
            return n

    def _kill_player(self) -> None:
        with self._player_lock:
            proc = self._player_proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception as exc:
            log.warning("kill aplay failed: %s", exc)

    def _is_playing(self) -> bool:
        with self._player_lock:
            return self._player_proc is not None

    # -- state transitions ---------------------------------------------
    def _enter_idle(self, source: str) -> None:
        """Hard idle: close mic, firmware-mute, LED red.

        Fires after `idle_timeout_ms` of silence following the last
        "transaction" — either a user utterance commit or the end of
        a TTS playback. The 10 s window is owned by the endpointer's
        `silence_count`, which is reset on both commit and playback
        end so the timer always measures silence *after* the last
        interaction, not just after the last user speech.

        We deliberately don't bump `_gen` here, so any utterance the
        worker is processing (and any audio the player is still
        flushing) finishes naturally. The audio_q is drained because
        anything captured after the silence threshold won't change
        the outcome; sparing the endpointer the work of filtering it
        out chunk-by-chunk on resume.
        """
        if not self.recording.is_set():
            return
        log.info("Idle (%s): closing mic, in-flight pipeline continues", source)
        self.recording.clear()
        self._drain_queue(self.audio_q)
        # Mark this idle as auto so the player un-idles itself when the
        # in-flight reply starts playing — see `_player_loop`.
        self._auto_idled.set()
        self.hid.set_led(muted=True)

    def _on_hid_press(self) -> None:
        if self.recording.is_set():
            log.info("HID press: commit-and-mute (in-flight pipeline continues)")
            # Soft mute: tell the endpointer to commit any in-progress
            # speech buffer right now (don't wait for silence_timeout_ms),
            # stop the recorder, write LED on. The worker still picks the
            # committed utterance off `utterance_q` and runs STT → gateway
            # → TTS as usual; the player still plays the reply. We just
            # stop listening for new input until the next press resumes.
            # No queue drain, no gen bump, no aplay kill — those would
            # discard the very thing the user pressed mute to send.
            self._force_commit.set()
            self.recording.clear()
            self.hid.set_led(muted=True)
        else:
            log.info("HID press: resume recording")
            # Bump gen so any stragglers from before (e.g. an old
            # in-progress speech buffer the endpointer might have under
            # the previous gen) are shed by downstream stages.
            self._auto_idled.clear()
            self._bump_gen()
            self.recording.set()
            self.hid.set_led(muted=False)

    # -- thread loops --------------------------------------------------
    def _hid_loop(self) -> None:
        while not self.shutdown_event.wait(0.05):
            if self.hid.consume_unmute_event():
                self._on_hid_press()

    def _recorder_loop(self) -> None:
        pa = pyaudio.PyAudio()
        stream: pyaudio.Stream | None = None
        sr = self.cfg["sample_rate"]
        chunk = self.cfg["chunk_size"]
        try:
            while not self.shutdown_event.is_set():
                if not self.recording.is_set():
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
                        stream = None
                        log.info("Recorder: stream closed")
                    # Park on `recording`. A timeout lets us notice
                    # shutdown even if no toggle ever arrives.
                    self.recording.wait(0.2)
                    continue

                if stream is None:
                    idx = find_input_device(pa)
                    try:
                        stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=sr,
                            input=True,
                            input_device_index=idx,
                            frames_per_buffer=chunk,
                        )
                        log.info("Recorder: opened (idx=%s rate=%dHz chunk=%d)",
                                 idx, sr, chunk)
                    except Exception as exc:
                        log.warning("Recorder: cannot open: %s — retry in 1s", exc)
                        if self.shutdown_event.wait(1.0):
                            return
                        continue

                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                except Exception as exc:
                    log.warning("Recorder: read failed: %s — reopening", exc)
                    try:
                        stream.close()
                    except Exception:
                        pass
                    stream = None
                    continue
                self.audio_q.put((self._current_gen(), data))
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            try:
                pa.terminate()
            except Exception:
                pass

    def _endpointer_loop(self) -> None:
        """RMS VAD. Two timers run off the same per-chunk silence count:

          - speech-then-silence ≥ `silence_timeout_ms` → commit utterance
          - cumulative silence ≥ `idle_timeout_ms` (no speech) → enter idle

        The silence count is NOT reset by a commit, so the idle timer
        accounts for the trailing silence of the last utterance too.
        """
        sr = self.cfg["sample_rate"]
        chunk = self.cfg["chunk_size"]
        chunk_ms = (chunk / sr) * 1000.0
        rms_threshold = float(self.cfg["vad_rms_threshold"])
        commit_chunks = max(1, int(self.cfg["silence_timeout_ms"] / chunk_ms))
        keep_chunks = max(0, int(self.cfg.get("silence_keep_ms", 500) / chunk_ms))
        pre_chunks = max(0, int(self.cfg.get("pre_speech_keep_ms", 100) / chunk_ms))
        prebuf: "deque[bytes] | None" = (
            deque(maxlen=pre_chunks) if pre_chunks > 0 else None
        )
        idle_ms = int(self.cfg.get("idle_timeout_ms", 0))
        idle_chunks = int(idle_ms / chunk_ms) if idle_ms > 0 else 0

        seen_gen = self._current_gen()
        in_speech = False
        silence_count = 0
        buf: list[bytes] = []
        gen_at_start = seen_gen
        speech_tick = 0

        while not self.shutdown_event.is_set():
            # Reset on resume: a gen bump means the user pressed HID
            # to resume from idle, so any half-built speech buffer
            # captured under the old gen must be discarded.
            cur_gen = self._current_gen()
            if cur_gen != seen_gen:
                in_speech = False
                silence_count = 0
                buf.clear()
                if prebuf is not None:
                    prebuf.clear()
                seen_gen = cur_gen

            # Reset silence_count when the player finishes a reply, so
            # the time spent listening to TTS doesn't count toward
            # idle_timeout_ms. Also drain audio_q: the recorder was
            # filling it while aplay was running, and those chunks
            # (silence — the user was listening to the reply) would
            # otherwise be processed back-to-back right after the reset
            # and burn the idle counter down to nearly the threshold
            # before the first wall-clock-fresh chunk even arrives. The
            # whole point of this reset is "start counting from end-of-
            # aplay", which means dropping the queued past too.
            if self._idle_reset_pending.is_set():
                self._idle_reset_pending.clear()
                silence_count = 0
                self._drain_queue(self.audio_q)

            # HID press while recording = "send what I said and stop
            # listening". If we're in the middle of a speech buffer,
            # commit it now (don't wait for silence_timeout_ms); the
            # worker will pick it up off utterance_q normally. If
            # there's nothing in flight, the press is just a soft mute.
            if self._force_commit.is_set():
                self._force_commit.clear()
                if in_speech and buf:
                    trim = max(0, min(silence_count - keep_chunks, len(buf)))
                    speech_buf = buf[:-trim] if trim > 0 else buf
                    pcm = b"".join(speech_buf)
                    duration_s = len(speech_buf) * chunk_ms / 1000.0
                    log.info("Endpointer: force-commit on HID press "
                             "(%d chunks ≈ %.2fs)",
                             len(speech_buf), duration_s)
                    self.utterance_q.put((gen_at_start, pcm, sr))
                    buf = []
                    in_speech = False
                    silence_count = 0

            try:
                gen, data = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != cur_gen:
                continue

            samples = array.array("h", data)
            if not samples:
                continue
            # Same not-quite-RMS metric the legacy `record_until_silence`
            # used: `sum(s²) / sqrt(N)`. Operator precedence makes this
            # `sum(s²) / len(samples)**0.5`. Keeping the formula as-is
            # so the threshold default (`vad_rms_threshold`) carries
            # over from prior calibrations.
            rms = sum(s * s for s in samples) / len(samples) ** 0.5

            if rms >= rms_threshold:
                if not in_speech:
                    in_speech = True
                    gen_at_start = gen
                    speech_tick = 0
                    if prebuf:
                        buf.extend(prebuf)
                        prebuf.clear()
                    log.info("Endpointer: sound detected (rms=%.0f ≥ %g)",
                             rms, rms_threshold)
                buf.append(data)
                silence_count = 0
                speech_tick += 1
                if speech_tick % 32 == 0:
                    log.info("Endpointer: still in_speech tick=%d rms=%.0f",
                             speech_tick, rms)
            else:
                if in_speech:
                    buf.append(data)
                    if silence_count == 0:
                        log.info("Endpointer: silence onset (rms=%.0f < %g, "
                                 "need %d chunks ≈ %dms to commit)",
                                 rms, rms_threshold, commit_chunks,
                                 self.cfg["silence_timeout_ms"])
                elif prebuf is not None:
                    # Rolling pre-roll window for the next utterance.
                    prebuf.append(data)
                silence_count += 1

                if in_speech and silence_count >= commit_chunks:
                    # Keep only `keep_chunks` of the trailing silence
                    # in the committed audio: the rest of the detection
                    # window is trimmed so STT doesn't see a full
                    # silence_timeout_ms tail. With keep_chunks=0 the
                    # cut is exactly at the last above-threshold chunk.
                    trim = max(0, min(silence_count - keep_chunks, len(buf)))
                    speech_buf = buf[:-trim] if trim > 0 else buf
                    pcm = b"".join(speech_buf)
                    duration_s = len(speech_buf) * chunk_ms / 1000.0
                    log.info("Endpointer: commit (%d chunks ≈ %.2fs, "
                             "kept %d trailing silence, trimmed %d)",
                             len(speech_buf), duration_s,
                             min(keep_chunks, silence_count), trim)
                    self.utterance_q.put((gen_at_start, pcm, sr))
                    buf = []
                    in_speech = False
                    # Treat commit as a "transaction" boundary: the
                    # idle 10 s window starts counting from here, not
                    # from the trailing-silence chunks already absorbed
                    # to detect end-of-speech. (Playback end resets it
                    # again via _idle_reset_pending — whichever lands
                    # later wins.)
                    silence_count = 0

                if (not in_speech
                        and idle_chunks > 0
                        and silence_count >= idle_chunks
                        and not self._is_playing()):
                    self._enter_idle(source=f"silence>{idle_ms}ms")
                    silence_count = 0
                    buf = []

    def _worker_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                gen, pcm, sr = self.utterance_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._current_gen():
                continue

            log.info("Worker: STT (%d bytes ≈ %.2fs)", len(pcm), len(pcm) / (sr * 2))
            text = self.stt.transcribe(pcm, sr)
            if not text:
                log.info("Worker: empty transcription, skipping")
                continue
            if gen != self._current_gen():
                continue
            log.info("User: %s", text)

            backend = self.cfg.get("gateway_backend", "openclaw")
            if backend == "zeroclaw":
                log.info("Worker: → gateway %s (backend=zeroclaw)",
                         self.cfg["gateway_base_url"])
                text_stream = gateway_chat_stream_zeroclaw(
                    self.cfg["gateway_base_url"],
                    self.cfg["gateway_token"],
                    text,
                )
            else:
                log.info("Worker: → gateway %s (backend=openclaw model=%s session=%s)",
                         self.cfg["gateway_base_url"],
                         self.cfg["voice_model"],
                         self.cfg.get("session_key", "voice-bridge"))
                text_stream = gateway_chat_stream(
                    self.cfg["gateway_base_url"],
                    self.cfg["gateway_token"],
                    text,
                    self.cfg["voice_model"],
                    self.cfg.get("session_key", "voice-bridge"),
                )
            collected: list[str] = []

            def _tee(s: Iterable[str]) -> Iterator[str]:
                for delta in s:
                    if gen != self._current_gen():
                        return
                    collected.append(delta)
                    yield delta

            try:
                for chunk in self.tts.synthesize_stream(_filter_no_reply(_tee(text_stream))):
                    if gen != self._current_gen():
                        break
                    if not chunk:
                        continue
                    self.playback_q.put((gen, chunk))
            except Exception as exc:
                log.error("Worker: TTS pipeline error: %s", exc)
            finally:
                full_reply = "".join(collected).strip()
                if full_reply in NO_REPLY_SENTINELS and gen == self._current_gen():
                    # Agent said "stay silent" — play a short low beep so
                    # the user gets feedback that the turn was processed
                    # but nothing needed saying.
                    log.info("Binary: %s (sentinel — low beep)", full_reply)
                    beep = _make_beep_pcm(
                        self.cfg["tts_sample_rate"],
                        freq=220,
                        duration=0.18,
                    )
                    self.playback_q.put((gen, beep))
                # Always emit the end-of-utterance marker (even after
                # cancel) so the player can release the current aplay
                # cleanly. Stale gen → player drops it harmlessly.
                self.playback_q.put((gen, _END_OF_UTTERANCE))

            if collected and full_reply not in NO_REPLY_SENTINELS:
                log.info("Binary: %s", "".join(collected)[:200])

    def _player_loop(self) -> None:
        device = self.cfg["output_device"]
        sample_rate = self.cfg["tts_sample_rate"]
        while not self.shutdown_event.is_set():
            try:
                gen, item = self.playback_q.get(timeout=0.2)
            except queue.Empty:
                continue
            # Skip end-of-utterance markers that arrive with no
            # preceding audio (e.g. worker bailed before producing any
            # PCM), and stale-gen anything.
            if isinstance(item, _EndOfUtterance) or gen != self._current_gen():
                continue

            # If the bridge auto-idled while this turn was still being
            # processed (worker → TTS), the device is currently muted
            # and the mic is closed. Resume recording before playing the
            # reply so the user can talk back the moment it ends; reset
            # the idle silence counter so the next idle window measures
            # from end-of-playback. Only fires for *auto*-idle — if the
            # user explicitly pressed HID to mute, `_auto_idled` is
            # clear and we leave the mic muted (the press wins).
            if self._auto_idled.is_set():
                log.info("Player: un-idling for playback (auto-idle, "
                         "resume mic + LED off)")
                self._auto_idled.clear()
                self.recording.set()
                self.hid.set_led(muted=False)
                self._idle_reset_pending.set()

            # Duck deezer-connect (no-op if disabled or BFF unreachable).
            # Paired with unduck() in the finally below; the lock inside
            # the plugin keeps the pair safe under back-to-back replies.
            self.deezer.duck()
            proc = _aplay_popen(device, sample_rate, bufsize=0)
            with self._player_lock:
                self._player_proc = proc
            try:
                if not self._write_chunk(proc, item):
                    continue
                while not self.shutdown_event.is_set():
                    try:
                        gen2, item2 = self.playback_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if gen2 != self._current_gen():
                        # Hard-cancel happened mid-utterance; the proc
                        # has likely already been killed, but break
                        # explicitly so we close it cleanly.
                        break
                    if isinstance(item2, _EndOfUtterance):
                        break
                    if not self._write_chunk(proc, item2):
                        break
            finally:
                # Close stdin first so aplay knows there's no more PCM
                # coming, then wait for it to drain its ALSA buffer.
                # Crucially, keep `_player_proc` set throughout the wait:
                # `_is_playing()` is the endpointer's guard against firing
                # auto-idle prematurely, and clearing the handle before
                # `proc.wait()` returns would let the idle timer fire
                # during the audible tail of the reply (the mic would
                # mute the instant the user hears the end of the message).
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
                with self._player_lock:
                    self._player_proc = None
                self.deezer.unduck()
                self._idle_reset_pending.set()

    @staticmethod
    def _write_chunk(proc: subprocess.Popen, chunk: bytes) -> bool:
        if not chunk:
            return True
        try:
            proc.stdin.write(chunk)
            return True
        except BrokenPipeError:
            return False

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        for name, fn in (
            ("hid", self._hid_loop),
            ("recorder", self._recorder_loop),
            ("endpointer", self._endpointer_loop),
            ("worker", self._worker_loop),
            ("player", self._player_loop),
        ):
            t = threading.Thread(target=fn, name=f"vb-{name}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self.shutdown_event.set()
        # Wake any thread parked on `recording.wait()` so it notices
        # shutdown immediately instead of waiting out its timeout.
        self.recording.set()
        # Kill the active aplay so the player loop's wait returns
        # without hitting the 2s timeout.
        self._kill_player()
        for t in self._threads:
            t.join(timeout=3.0)
        # Restore deezer-connect's volume if we were ducked when stop
        # arrived (SIGTERM mid-playback). No-op when the plugin is
        # disabled or wasn't currently ducking.
        self.deezer.unduck()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()

    # Validate only the API keys actually needed for the chosen
    # providers (voice-bridge.json decides which).
    needed_providers = {cfg["stt_provider"], cfg["tts_provider"]}
    if "elevenlabs" in needed_providers and not cfg.get("elevenlabs_key"):
        log.error("ElevenLabs selected but no ElevenLabs API key in gateway config")
        sys.exit(1)
    if cfg["tts_provider"] == "elevenlabs":
        if not cfg.get("elevenlabs_voice"):
            log.error("messages.tts.providers.elevenlabs.voiceId missing in openclaw config")
            sys.exit(1)
        if not cfg.get("elevenlabs_model"):
            log.error("messages.tts.providers.elevenlabs.modelId missing in openclaw config")
            sys.exit(1)
    if "deepgram" in needed_providers and not cfg.get("deepgram_key"):
        log.error("Deepgram selected but no Deepgram API key (env DEEPGRAM_API_KEY or gateway config)")
        sys.exit(1)
    if not cfg.get("gateway_token"):
        log.error("No gateway token in config")
        sys.exit(1)

    log.info("Config loaded")
    log.info("STT provider: %s", cfg["stt_provider"])
    log.info("TTS provider: %s (voice=%s model=%s rate=%dHz stream=%s)",
             cfg["tts_provider"], cfg["elevenlabs_voice"], cfg["elevenlabs_model"],
             cfg["tts_sample_rate"], cfg["tts_streaming_mode"])
    log.info("Output: %s @ %d Hz", cfg["output_device"], cfg["tts_sample_rate"])
    log.info("VAD: rms_threshold=%g pause_commit=%dms idle=%dms",
             cfg["vad_rms_threshold"], cfg["silence_timeout_ms"], cfg["idle_timeout_ms"])
    if not cfg.get("hid_mute_enabled") and cfg["idle_timeout_ms"] > 0:
        log.warning("HID disabled but idle_timeout_ms>0 — auto-idle will be unrecoverable; "
                    "set idle_timeout_ms=0 or hid_mute_enabled=true")

    stt = _build_voice_provider("stt", cfg)
    tts = _build_voice_provider("tts", cfg)

    hid = HidMuteMonitor()
    if cfg.get("hid_mute_enabled"):
        hid.start()

    bridge = VoiceBridge(cfg, stt, tts, hid)

    def _sigterm(_signum, _frame):
        log.info("Shutdown requested")
        bridge.shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    bridge.start()
    if cfg.get("hid_mute_enabled"):
        log.info("Ready — device starts muted, press the Jabra button to begin")
    else:
        log.info("Ready — listening (always-on mic, HID button disabled)")

    # Block here until SIGTERM/SIGINT. Worker threads do all the work;
    # main is only around to own the signal handlers and the cleanup.
    try:
        while not bridge.shutdown_event.wait(1.0):
            pass
    finally:
        bridge.stop()
        hid.stop()
        log.info("Stopped")


if __name__ == "__main__":
    main()
