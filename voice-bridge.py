#!/usr/bin/env python3
"""
Binary Voice Bridge v2 — minimal, ARM-friendly.

Trigger: Jabra SPEAK 510 HID unmute (no wake word).
STT:     Deepgram REST API (streaming).
TTS:     ElevenLabs REST API.
Output:  ALSA aplay.

CPU idle: ~0% (sleep loop, no audio stream open).
"""

from __future__ import annotations

import array
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Iterable, Iterator

import pyaudio

from deepgram_voice import DeepgramVoice
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

VALID_PROVIDERS = ("elevenlabs", "deepgram")
VALID_TTS_STREAM_MODES = ("http_sentence", "websocket")


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


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    # Secrets from gateway config
    gw_path = os.path.expanduser("~/.openclaw/openclaw.json")
    with open(gw_path) as f:
        gw = json.load(f)

    cfg["gateway_token"] = gw.get("gateway", {}).get("auth", {}).get("token", "") or gw.get("token", "")
    cfg["deepgram_key"] = os.environ.get("DEEPGRAM_API_KEY", "") or gw.get("stt", {}).get("deepgram", {}).get("apiKey", "")

    # Pull Deepgram STT settings from the openclaw `tools.media.audio`
    # section: model name from `models[where provider==deepgram]`, and
    # all extra kwargs (smart_format, punctuate, detect_language, ...)
    # straight from `providerOptions.deepgram`. Already snake_case in
    # openclaw — no key translation needed (unlike ElevenLabs).
    audio = gw.get("tools", {}).get("media", {}).get("audio", {}) or {}
    cfg["deepgram_stt_options"] = audio.get("providerOptions", {}).get("deepgram") or {}
    cfg["deepgram_stt_model"] = next(
        (m.get("model") for m in audio.get("models") or [] if m.get("provider") == "deepgram"),
        "",
    )

    # Pull the entire ElevenLabs TTS block from the openclaw gateway
    # config — voice id, model, language, voice_settings, text
    # normalization. The bridge does not invent its own values: whatever
    # openclaw says about TTS is what we use. (Note the key is `modelId`
    # in openclaw's schema, not `model`.)
    el = gw.get("messages", {}).get("tts", {}).get("providers", {}).get("elevenlabs", {})
    cfg["elevenlabs_key"] = el.get("apiKey", "")
    cfg["elevenlabs_voice"] = el.get("voiceId", "")
    cfg["elevenlabs_model"] = el.get("modelId", "")
    cfg["elevenlabs_language"] = el.get("languageCode")
    cfg["elevenlabs_voice_settings"] = _camel_to_snake_keys(el.get("voiceSettings")) or None
    cfg["elevenlabs_text_normalization"] = el.get("applyTextNormalization")

    cfg["voice_model"] = gw.get("agents", {}).get("defaults", {}).get("voiceBridgeModel", "openclaw")

    # Session key for the voice bridge: local voice-bridge.json wins,
    # then gateway config, then a literal fallback.
    cfg["session_key"] = cfg.get("session_key") or gw.get("voice", {}).get("sessionKey") or "voice-bridge"

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


def play_beep(device: str, sample_rate: int) -> None:
    """Play a short ~80 ms 880 Hz beep at the given output sample rate."""
    duration = 0.08
    freq = 880
    n_samples = int(sample_rate * duration)
    buf = array.array("h")
    for i in range(n_samples):
        # Fade out
        env = 1.0 - (i / n_samples)
        val = int(16000 * env * (0.5 + 0.5 * __import__("math").sin(2 * __import__("math").pi * freq * i / sample_rate)))
        buf.append(max(-32768, min(32767, val)))
    play_audio(device, buf.tobytes(), sample_rate)


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
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as exc:
        log.error("Gateway error: %s", exc)
        return "Mi dispiace, ho avuto un problema di connessione."


GATEWAY_FALLBACK_REPLY = "Mi dispiace, ho avuto un problema di connessione."


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


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def record_until_silence(
    pa: pyaudio.PyAudio,
    stream: pyaudio.Stream,
    chunk_size: int,
    silence_ms: int,
    max_s: int,
    sample_rate: int,
) -> list[bytes]:
    """Record audio chunks until silence or max duration."""
    frames: list[bytes] = []
    silence_threshold = 300  # RMS energy threshold
    silence_count = 0
    silence_limit = int(silence_ms / (chunk_size / sample_rate * 1000))
    max_frames = int(max_s * sample_rate / chunk_size)

    log.info("Recording...")
    while len(frames) < max_frames:
        try:
            data = stream.read(chunk_size, exception_on_overflow=False)
        except Exception:
            break

        frames.append(data)

        # Check silence (RMS energy)
        samples = array.array("h", data)
        if len(samples) == 0:
            continue
        rms = sum(s * s for s in samples) / len(samples) ** 0.5

        if rms < silence_threshold:
            silence_count += 1
            if silence_count >= silence_limit:
                log.info("Silence detected (%d frames)", silence_count)
                break
        else:
            silence_count = 0

    log.info("Recorded %d frames", len(frames))
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()

    # Validate only the API keys actually needed for the chosen
    # providers (voice-bridge.conf decides which).
    needed_providers = {cfg["stt_provider"], cfg["tts_provider"]}
    if "elevenlabs" in needed_providers and not cfg.get("elevenlabs_key"):
        log.error("ElevenLabs selected but no ElevenLabs API key in gateway config")
        sys.exit(1)
    if cfg["tts_provider"] == "elevenlabs":
        # We use whatever voice/model the openclaw config specifies.
        # If those are missing we'd silently fall through to SDK defaults
        # (random English voice etc.) — fail loudly instead.
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

    stt = _build_voice_provider("stt", cfg)
    tts = _build_voice_provider("tts", cfg)

    # Start HID mute monitor. The monitor's poll thread owns the device
    # lifecycle: it parks (zero CPU) until the Jabra is plugged in, and
    # transparently reconnects across unplug/replug. The bridge stays
    # up indefinitely; only SIGTERM/SIGINT exits.
    hid = HidMuteMonitor()
    if cfg.get("hid_mute_enabled"):
        hid.start()

    # Graceful shutdown
    shutdown_event = threading.Event()

    def _sigterm(signum, frame):
        log.info("Shutdown requested")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    log.info("Ready — waiting for unmute...")

    # Main loop — edge-triggered on mute→unmute transitions. Exits only
    # on SIGTERM/SIGINT; device unplug is handled inside the HID monitor.
    while not shutdown_event.is_set():
        if not hid.consume_unmute_event():
            time.sleep(0.2)
            continue

        log.info("Wake: unmute detected")
        play_beep(cfg["output_device"], cfg["tts_sample_rate"])

        # Open audio stream
        pa = pyaudio.PyAudio()
        input_dev = find_input_device(pa)
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=cfg["sample_rate"],
                input=True,
                input_device_index=input_dev,
                frames_per_buffer=cfg["chunk_size"],
            )
        except Exception as exc:
            log.error("Cannot open audio: %s", exc)
            pa.terminate()
            time.sleep(1)
            continue

        log.info("Input device: %s", input_dev)

        # Record
        frames = record_until_silence(
            pa, stream, cfg["chunk_size"],
            cfg["silence_timeout_ms"], cfg["max_recording_s"],
            cfg["sample_rate"],
        )

        # Close audio stream immediately
        stream.close()
        stream = None
        pa.terminate()
        pa = None

        if not frames:
            log.info("No audio captured")
            continue

        # Transcribe
        log.info("Transcribing...")
        text = stt.transcribe(b"".join(frames), cfg["sample_rate"])

        if not text:
            log.info("No speech detected")
            continue

        log.info("User: %s", text)

        # Streaming pipeline: gateway SSE → TTS (mode-dependent) → aplay
        # stdin. Each stage is a lazy iterator pulled by the next; aplay
        # writes the first PCM bytes as soon as the TTS produces them.
        # First-audio latency is one sentence under `http_sentence` mode
        # and one token under `websocket` mode (paid tier only).
        log.info("Streaming response...")
        text_stream = gateway_chat_stream(
            cfg["gateway_base_url"],
            cfg["gateway_token"],
            text,
            cfg["voice_model"],
            cfg.get("session_key", "voice-bridge"),
        )
        # Tee deltas through a list so we can still log the full assembled
        # reply once playback ends. The list lives on the stack of this
        # turn — no shared state with future turns.
        collected: list[str] = []

        def _tee(stream: Iterable[str]) -> Iterator[str]:
            for delta in stream:
                collected.append(delta)
                yield delta

        audio_stream = tts.synthesize_stream(_tee(text_stream))
        played = play_audio_stream(cfg["output_device"], audio_stream, cfg["tts_sample_rate"])
        log.info("Binary: %s", "".join(collected)[:200])
        if not played:
            log.info("No audio produced")

    hid.stop()
    log.info("Stopped")


if __name__ == "__main__":
    main()
