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


def play_audio(device: str, audio_data: bytes, sample_rate: int) -> None:
    """Play raw S16LE mono PCM bytes through aplay at `sample_rate`.

    `sample_rate` MUST match what the TTS provider produced (we ask it
    for `pcm_<rate>` / `linear16` at the same number) — otherwise aplay
    plays back at the wrong speed/pitch."""
    proc = subprocess.Popen(
        ["aplay", "-q", "-D", device, "-f", "S16_LE", "-r", str(sample_rate), "-c", "1"],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    proc.communicate(input=audio_data)


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
    log.info("TTS provider: %s (voice=%s model=%s rate=%dHz)",
             cfg["tts_provider"], cfg["elevenlabs_voice"], cfg["elevenlabs_model"],
             cfg["tts_sample_rate"])
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

        # Gateway chat
        log.info("Getting response...")
        reply = gateway_chat(cfg["gateway_base_url"], cfg["gateway_token"], text, cfg["voice_model"], cfg.get("session_key", "voice-bridge"))
        log.info("Binary: %s", reply[:100])

        # TTS
        log.info("Synthesizing...")
        audio = tts.synthesize(reply)

        if audio:
            play_audio(cfg["output_device"], audio, cfg["tts_sample_rate"])
            log.info("Playback done")

    hid.stop()
    log.info("Stopped")


if __name__ == "__main__":
    main()
