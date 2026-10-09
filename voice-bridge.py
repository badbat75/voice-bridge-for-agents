#!/usr/bin/env python3
"""
Binary Voice Bridge v3 — always-on mic, async pipeline.

Trigger: `activation` = "button" (Jabra HID press unmutes) or "wake_word"
         (RMS-gated Whistle keyword spotting while idle; see wake_word.py).
         Either way the HID button mutes, and voice activity drives turns.
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
import fcntl
import json
import logging
import math
import os
import queue
import re
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
import wake_word
from stt_compare import SttComparer

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
# resources/voice-bridge.secrets.example.json for the expected shape.
SECRETS_PATH = os.path.join(_HERE, "voice-bridge.secrets.json")

VALID_PROVIDERS = ("elevenlabs", "deepgram")
VALID_TTS_STREAM_MODES = ("http_sentence", "websocket")
VALID_GATEWAY_BACKENDS = ("openclaw", "zeroclaw", "zeroclaw_ws")
VALID_ACTIVATIONS = ("button", "wake_word")


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
    # Telegram bot token, used only by the MCP sidecar's `send_voice_telegram`
    # / `say_to_telegram` tools. The bridge itself ignores it; loaded here so
    # the same shared `load_config()` keeps every secret in one place. The
    # default destination chat_id is non-secret config (`telegram.chat_id` in
    # voice-bridge.json) and exposed as `telegram_chat_id` — callers may still
    # override per-tool-call.
    cfg["telegram_bot_token"] = secrets.get("telegram_bot_token", "")
    tg_local = cfg.get("telegram") or {}
    cfg["telegram_chat_id"] = str(tg_local.get("chat_id") or "").strip()

    # --- Deepgram STT/TTS settings ---
    # The `deepgram` block in voice-bridge.json: `sttOptions` are extra
    # kwargs (smart_format, punctuate, ...) passed verbatim, `sttModel`
    # is the STT model name, `ttsModel` is the Aura TTS voice model.
    # Deepgram TTS has no separate `language` param — the language is
    # baked into the voice model name (e.g. `aura-2-thalia-en` is English,
    # `aura-2-livia-it` is Italian), so picking an Italian voice IS how you
    # get Italian speech. Only used if a Deepgram provider is selected.
    dg_local = cfg.get("deepgram") or {}
    cfg["deepgram_stt_options"] = dg_local.get("sttOptions") or {}
    cfg["deepgram_stt_model"] = dg_local.get("sttModel") or ""
    cfg["deepgram_tts_model"] = dg_local.get("ttsModel") or ""

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

    # Output softvol level re-asserted at startup (see `_apply_output_volume`).
    # The `output_volume` block targets the `voice_out` softvol control —
    # independent from deezer-connect's player gain. ALSA resets a softvol
    # control to 100% when it first re-creates it after a reboot, so the
    # bridge pins the configured level on every start. Missing block / keys
    # default to 100% on the conventional VoiceBridge control / card "USB".
    # `card` is passed to `amixer -c` verbatim: use the Jabra's card *name*
    # ("USB"), not an index — adding an HDMI display renumbers ALSA cards.
    ov = cfg.get("output_volume") or {}
    cfg["output_volume_control"] = ov.get("control", "VoiceBridge")
    cfg["output_volume_card"] = ov.get("card", "USB")
    cfg["output_volume_percent"] = int(ov.get("percent", 100))

    cfg["voice_model"] = cfg.get("voice_model") or "openclaw"

    # Which gateway protocol the worker speaks. `openclaw` (default) posts
    # to `/v1/chat/completions` with OpenAI-style SSE; `zeroclaw` posts to
    # `/webhook` and parses a single non-streaming JSON reply; `zeroclaw_ws`
    # streams over the `/ws/chat` WebSocket and consumes only `chunk` frames
    # (the reasoning arrives in separate `thinking` frames and is dropped —
    # this is why `zeroclaw_ws` keeps deepseek-reasoner's chain-of-thought
    # out of TTS, where the lossy `/webhook` leg cannot). Selecting a backend
    # also implies which gateway `gateway_base_url`/`gateway_token` point at —
    # they are not interchangeable.
    cfg["gateway_backend"] = str(cfg.get("gateway_backend", "openclaw")).strip().lower()
    if cfg["gateway_backend"] not in VALID_GATEWAY_BACKENDS:
        raise ValueError(
            f"voice-bridge.json: unknown gateway_backend "
            f"{cfg['gateway_backend']!r}; valid: {VALID_GATEWAY_BACKENDS}"
        )

    # Agent alias for the `/ws/chat` WebSocket leg (`zeroclaw_ws` backend
    # only). zeroclaw requires an explicit `?agent=<alias>`; the runtime
    # synthesizes a `default` agent even when `[agents]` is empty in its
    # config, so `default` is the safe fallback. Ignored by other backends.
    cfg["gateway_agent"] = (cfg.get("gateway_agent") or "default").strip()

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

    # How an idle bridge comes back: "button" (HID press) or "wake_word"
    # (mic stays open while idle; the `wake_word` block configures it).
    cfg["activation"] = str(cfg.get("activation", "button")).strip().lower()
    if cfg["activation"] not in VALID_ACTIVATIONS:
        raise ValueError(
            f"voice-bridge.json: unknown activation {cfg['activation']!r}; "
            f"valid: {VALID_ACTIVATIONS}"
        )

    # Whistle settings fail at load, not on the first wake/utterance.
    if cfg["activation"] == "wake_word":
        wcfg = wake_word.wake_config(cfg)
        if not wcfg["phrases"]:
            raise ValueError("voice-bridge.json: wake_word.phrases is empty")
        wake_word.validate_language("wake_word.language", wcfg["language"])
    if (cfg.get("stt_compare") or {}).get("enabled"):
        wake_word.validate_language("stt_compare.language", cfg["stt_compare"].get("language", "it"))

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
    # Minimum above-threshold audio for an utterance to reach STT. A key
    # click, a bump or a speaker echo crosses the threshold for one or two
    # chunks; real speech stays above it far longer. Shorter bursts are
    # dropped at commit time (no STT call, no idle-timer reset). 0 = off.
    cfg["min_speech_ms"] = int(cfg.get("min_speech_ms", 0))
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
    # `tts_whole_reply: true` disables per-sentence splitting on the HTTP
    # path: the whole gateway reply is buffered and sent to TTS in ONE
    # call, so tone/prosody stay consistent (eleven_v3 rejects request
    # stitching, so this is the only consistency lever there). Trades
    # first-audio latency for it. Default false = split per sentence.
    cfg["tts_whole_reply"] = bool(cfg.get("tts_whole_reply", False))
    # With `tts_whole_reply`, speak the first sentence as soon as it is
    # complete and the rest in one call (ElevenLabs only).
    cfg["tts_first_sentence_early"] = bool(cfg.get("tts_first_sentence_early", False))

    # Idle window after a reply finished playing. Defaults to the plain
    # idle window; a reply ending in "?" gets the (usually longer)
    # question window, since the user is expected to answer.
    cfg["idle_after_reply_ms"] = int(cfg.get("idle_after_reply_ms", cfg["idle_timeout_ms"]))
    cfg["idle_after_question_ms"] = int(
        cfg.get("idle_after_question_ms", cfg["idle_after_reply_ms"]))
    # Idle window right after a wake word / button resume: time to start
    # talking after "Dimmi". Defaults to the plain idle window.
    cfg["idle_after_resume_ms"] = int(cfg.get("idle_after_resume_ms", cfg["idle_timeout_ms"]))
    # An utterance still going after this long is people talking among
    # themselves, not a request: it is dropped and the bridge goes idle.
    # 0 = no limit.
    cfg["max_utterance_ms"] = int(cfg.get("max_utterance_ms", 0))

    # PCM buffered before the first write to aplay, so a TTS stream that
    # stalls right after its first chunk doesn't underrun (audible click).
    cfg["playback_prebuffer_ms"] = int(cfg.get("playback_prebuffer_ms", 0))

    # Frame-level voice check (webrtcvad) at commit: utterances with less
    # than `min_voiced_ms` of voiced frames are dropped before STT.
    sf = cfg.get("speech_filter") or {}
    cfg["speech_filter"] = {
        "enabled": bool(sf.get("enabled", False)),
        "aggressiveness": int(sf.get("aggressiveness", 2)),
        "min_voiced_ms": int(sf.get("min_voiced_ms", 200)),
    }
    if not 0 <= cfg["speech_filter"]["aggressiveness"] <= 3:
        raise ValueError("voice-bridge.json: speech_filter.aggressiveness must be 0..3")

    # "Still working" feedback between commit and the first reply audio.
    tc = cfg.get("thinking_cue") or {}
    cfg["thinking_cue"] = {
        "enabled": bool(tc.get("enabled", False)),
        "delay_ms": int(tc.get("delay_ms", 3000)),
        "repeat_ms": int(tc.get("repeat_ms", 5000)),
        "phrases": list(tc.get("phrases") or []),
    }

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
            tts_whole_reply=cfg.get("tts_whole_reply", False),
            tts_first_sentence_early=cfg.get("tts_first_sentence_early", False),
        )
    if name == "deepgram":
        kwargs = {
            "api_key": cfg["deepgram_key"],
            "stt_options": cfg.get("deepgram_stt_options") or {},
            "tts_sample_rate": cfg["tts_sample_rate"],
        }
        if cfg.get("deepgram_stt_model"):
            kwargs["stt_model"] = cfg["deepgram_stt_model"]
        if cfg.get("deepgram_tts_model"):
            kwargs["tts_model"] = cfg["deepgram_tts_model"]
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
    except ValueError:
        pass  # the stream was closed under us when aplay was reaped
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


def _drain_aplay(
    proc: subprocess.Popen,
    sample_rate: int,
    abort: "threading.Event | None" = None,
) -> None:
    """Close aplay's stdin and wait for it to play out its buffered PCM,
    then make sure it has exited.

    Network TTS feeds PCM faster than realtime, so at stdin-close the
    unplayed tail is the kernel pipe buffer (queried exactly via
    `F_GETPIPE_SZ`) plus aplay's ALSA ring buffer. We derive a
    generous-but-bounded drain budget from that buffer size and the stream
    rate, then poll until aplay exits on its own — we never kill a
    still-draining process, which would chop the reply's tail and, because
    `voice_out` runs through `sw_dmix` (which *mixes* streams), briefly
    overlap the next utterance's aplay. The budget scales with
    `sample_rate`, so it stays correct if `tts_sample_rate` changes.

    `abort`, if given, cuts the wait short (bridge shutdown). A genuinely
    hung aplay is killed once the budget is spent, so the caller can never
    block forever.
    """
    bytes_per_sec = sample_rate * 2  # S16LE mono
    try:
        pipe_bytes = fcntl.fcntl(proc.stdin.fileno(), fcntl.F_GETPIPE_SZ)
    except Exception:
        pipe_bytes = 65536  # Linux default pipe capacity
    # pipe drain time + ALSA buffer headroom & safety margin.
    drain_budget = pipe_bytes / bytes_per_sec + 2.0
    with contextlib.suppress(Exception):
        proc.stdin.close()
    deadline = time.monotonic() + drain_budget
    while abort is None or not abort.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            proc.wait(timeout=min(0.2, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    if proc.poll() is None:
        # Budget exhausted (hung aplay) or aborted — drop it so the
        # caller can move on.
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=1.0)


def _apply_output_volume(cfg: dict) -> None:
    """Re-assert the configured softvol level on the output control.

    The `voice_out` softvol control gives the bridge a volume independent
    from deezer-connect's player gain. ALSA creates that control lazily on
    first PCM open and resets it to 100% after a reboot, so we (1) open the
    device with a brief silent buffer to instantiate the control, then
    (2) set it with `amixer`. Best-effort: any failure is logged and
    swallowed — a missing mixer must never take the bridge down (it just
    means the level stays at ALSA's 100% default)."""
    control = cfg.get("output_volume_control")
    card = cfg.get("output_volume_card")
    percent = int(cfg.get("output_volume_percent", 100))
    if not control or card is None:
        return
    device = cfg["output_device"]
    rate = int(cfg["tts_sample_rate"])
    # ~50 ms of silence: enough to make ALSA open the PCM and create the
    # softvol control element, inaudible since every sample is zero.
    silence = b"\x00\x00" * (rate // 20)
    try:
        proc = _aplay_popen(device, rate)
        proc.communicate(input=silence, timeout=5)
    except Exception as exc:
        log.warning("output volume: could not prime %s (%s); "
                    "amixer set may fail", device, exc)
    try:
        subprocess.run(
            ["amixer", "-c", str(card), "sset", control, f"{percent}%"],
            check=True, capture_output=True, timeout=5,
        )
        log.info("Output volume: %s on card %s → %d%%", control, card, percent)
    except Exception as exc:
        log.warning("output volume: amixer set %s on card %s failed: %s",
                    control, card, exc)


def play_audio(device: str, audio_data: bytes, sample_rate: int) -> None:
    """Play raw S16LE mono PCM bytes through aplay at `sample_rate`.

    `sample_rate` MUST match what the TTS provider produced (we ask it
    for `pcm_<rate>` / `linear16` at the same number) — otherwise aplay
    plays back at the wrong speed/pitch."""
    proc = _aplay_popen(device, sample_rate)
    # Not `communicate()`: with VOICE_BRIDGE_DEBUG_AUDIO=1 a drain thread
    # already owns aplay's stderr, and two readers race to EBADF.
    with contextlib.suppress(BrokenPipeError):
        proc.stdin.write(audio_data)
    _drain_aplay(proc, sample_rate)


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
        # Bounded drain (see `_drain_aplay`): play out the buffered tail
        # without chopping it, but don't hang on a dead aplay.
        _drain_aplay(proc, sample_rate)
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


def _make_tick_pcm(sample_rate: int) -> bytes:
    """Soft, short "still working" tick — quiet enough to sit under music."""
    return _make_beep_pcm(sample_rate, freq=520, duration=0.06, amplitude=4000)


def _make_sleep_tone_pcm(sample_rate: int) -> bytes:
    """Two soft falling notes: the bridge dozed off without a conversation."""
    return (_make_beep_pcm(sample_rate, freq=660, duration=0.09, amplitude=6000)
            + _make_beep_pcm(sample_rate, freq=440, duration=0.12, amplitude=6000))


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


_VAD_RATES = (8000, 16000, 32000, 48000)
_VAD_FRAME_MS = 30


def _voiced_ms(pcm: bytes, sample_rate: int, vad) -> int | None:
    """Milliseconds of voiced 30 ms frames in S16LE mono `pcm`, per a
    `webrtcvad.Vad`. None when the rate is one webrtcvad can't take, so the
    caller can skip the check instead of dropping speech."""
    if sample_rate not in _VAD_RATES:
        return None
    frame_bytes = sample_rate * _VAD_FRAME_MS // 1000 * 2
    voiced = 0
    for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        if vad.is_speech(pcm[i:i + frame_bytes], sample_rate):
            voiced += _VAD_FRAME_MS
    return voiced


def _make_speech_vad(cfg: dict):
    """A `webrtcvad.Vad` when `speech_filter.enabled`, else None. A missing
    module only logs: the filter is an optimization, never a hard dep."""
    sf = cfg.get("speech_filter") or {}
    if not sf.get("enabled"):
        return None
    try:
        import webrtcvad
    except ImportError:
        log.warning("speech_filter enabled but webrtcvad is not installed — filter off")
        return None
    return webrtcvad.Vad(int(sf.get("aggressiveness", 2)))


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


# A whole externally-supplied utterance (the MCP `say_to_speaker` tool),
# enqueued as ONE atomic item rather than chunk+marker so it can never
# interleave with the worker's streamed reply chunks on `playback_q`. The
# player plays `pcm` end-to-end as its own utterance and fires `done` when
# the playback (and no-clip drain) finishes, releasing a blocked caller.
class _ExternalUtterance:
    """`duck`: lower the music while it plays. `cue`: a thinking cue (tick
    or "un attimo") — it doesn't count as a reply for the idle window.
    `question`: the speech expects an answer (see `_after_external`)."""

    def __init__(self, pcm: bytes, done: "threading.Event", *,
                 label: str = "say_to_speaker", duck: bool = True,
                 cue: bool = False, question: bool = False) -> None:
        self.pcm = pcm
        self.done = done
        self.label = label
        self.duck = duck
        self.cue = cue
        self.question = question


class _ThinkingCue:
    """Per-turn "still working" feedback between pick-up and the first
    reply audio, so the user never sits through 10–25 s of dead air.

    - the first `tool_call` frame plays a spoken cue ("Un attimo.") once;
    - after `delay_ms` with no reply audio, a soft tick, repeated every
      `repeat_ms`.

    Cues go through the player queue (serialized, never mixed with the
    reply) and don't duck the music. `stop()` is called before the first
    reply chunk is queued; the lock makes "check stopped + enqueue"
    atomic, so no cue can land behind the reply."""

    def __init__(self, bridge: "VoiceBridge", gen: int) -> None:
        self._bridge = bridge
        self._gen = gen
        tc = bridge.cfg.get("thinking_cue") or {}
        self._enabled = bool(tc.get("enabled"))
        self._delay = max(0, int(tc.get("delay_ms", 3000))) / 1000.0
        self._repeat = max(100, int(tc.get("repeat_ms", 5000))) / 1000.0
        self._stop = threading.Event()
        self._tool = threading.Event()
        self._lock = threading.Lock()
        self.cues_played = 0

    def start(self) -> "_ThinkingCue":
        if self._enabled:
            threading.Thread(target=self._run, name="vb-thinking", daemon=True).start()
        return self

    def on_event(self, ftype: str, _frame: dict) -> None:
        if ftype == "tool_call":
            self._tool.set()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()

    def _enqueue(self, pcm: bytes | None, label: str) -> bool:
        if not pcm:
            return False
        with self._lock:
            if self._stop.is_set():
                return False
            self._bridge.playback_q.put((self._gen, _ExternalUtterance(
                pcm, threading.Event(), label=label, duck=False, cue=True)))
            self.cues_played += 1
            return True

    def _run(self) -> None:
        rate = int(self._bridge.cfg["tts_sample_rate"])
        next_tick = time.monotonic() + self._delay
        spoke = False
        while not self._stop.wait(0.05):
            now = time.monotonic()
            if self._tool.is_set() and not spoke:
                spoke = True
                bank = self._bridge._thinking_acks
                clip = bank.pick() if bank else None
                if self._enqueue(clip[2] if clip else None, "thinking-ack"):
                    log.info("Thinking cue: %r", clip[1])
                    next_tick = now + self._repeat
                    continue
            if now >= next_tick:
                self._enqueue(_make_tick_pcm(rate), "thinking-tick")
                next_tick = now + self._repeat


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
        # Ducking holds — see `_duck_acquire`.
        self._duck_lock = threading.Lock()
        self._duck_holds = 0

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
        # Set from an utterance commit until the reply starts playing (or the
        # turn ends without one): the mic is not recorded while the agent
        # is processing, so nothing piles up behind the turn.
        self._processing = threading.Event()

        # The active aplay process, if any. Held under `_player_lock`
        # so a hard-cancel from the HID thread can `kill()` it without
        # racing the player thread's setup/teardown.
        self._player_proc: subprocess.Popen | None = None
        self._player_lock = threading.Lock()

        # Wake-word activation. `_wake_armed` set = idle but listening for
        # the wake phrase: the firmware mic stays unmuted and the recorder
        # routes chunks to `wake_q` instead of `audio_q`. Cleared by an HID
        # mute (privacy: firmware-muted, only the button resumes) and while
        # recording. Always clear in "button" mode.
        self.wake_mode = cfg.get("activation") == "wake_word"
        self.wake_q: "queue.Queue[bytes]" = queue.Queue()
        self._wake_armed = threading.Event()
        # Set while a wake/sleep clip plays so the wake loop ignores our own
        # voice coming back through the mic.
        self._speaking_ack = threading.Event()
        # Set by the worker from picking an utterance up until it loops back
        # for the next one — i.e. a turn is in flight. A goodbye on auto-idle
        # is skipped while a reply is still on its way.
        self._worker_busy = threading.Event()
        self._wake_cfg = self._wake_detector = self._wake_acks = self._sleep_acks = None
        if self.wake_mode:
            self._wake_cfg = wake_word.wake_config(cfg)
            self._wake_detector = wake_word.WakeDetector(self._wake_cfg)

            def build_tts(name):
                return _build_voice_provider("tts", {**cfg, "tts_provider": name})

            # Acks speak in the reply voice: the configured tts_provider only.
            w = self._wake_cfg
            voice = [cfg.get("tts_provider", "elevenlabs")]
            self._wake_acks = wake_word.AckBank(
                cfg, w["acks"], voice, w["ack_cache_dir"], build_tts, kind="wake")
            self._sleep_acks = wake_word.AckBank(
                cfg, w["sleep_acks"], voice, w["ack_cache_dir"], build_tts, kind="sleep")
            if not self.recording.is_set():
                self._wake_armed.set()

        # Spoken "un attimo" clips for the thinking cue (any activation mode).
        self._thinking_acks = None
        tphrases = (cfg.get("thinking_cue") or {}).get("phrases") or []
        if (cfg.get("thinking_cue") or {}).get("enabled") and tphrases:
            self._thinking_acks = wake_word.AckBank(
                cfg, tphrases, [cfg.get("tts_provider", "elevenlabs")],
                wake_word.wake_config(cfg)["ack_cache_dir"],
                lambda name: _build_voice_provider("tts", {**cfg, "tts_provider": name}),
                kind="thinking")

        # Frame-level voice check at commit (None = off).
        self._speech_vad = _make_speech_vad(cfg)

        # Idle window currently in force (ms). Reset to `idle_timeout_ms` on
        # resume/commit; widened to `idle_after_reply_ms` /
        # `idle_after_question_ms` when a reply finishes playing.
        self._idle_window_ms = int(cfg.get("idle_timeout_ms", 0))
        # Set by the worker when the reply text is complete: does it end
        # with a question? Read by the player when the reply finishes.
        self._reply_is_question = False
        # Replies played since the last wake/resume — a goodbye is only
        # spoken after an actual conversation (wake_word mode).
        self._replies_since_resume = 0
        # Set after a say_to_speaker announcement that asked nothing: the
        # next auto-idle goes back to sleep without a goodbye or tone.
        self._quiet_idle = False
        # Pick-up time of the turn in flight, for the timing log.
        self._turn_t0: float | None = None
        # External utterances that arrived while a streamed reply was
        # playing; the player plays them right after it.
        self._player_backlog: "deque[tuple[int, object]]" = deque()

        # Shadow STT comparison (`stt_compare.enabled`): the remote STT stays
        # authoritative; Whistle's take on the same PCM is only logged.
        ccfg = cfg.get("stt_compare") or {}
        self._stt_compare = (
            SttComparer(ccfg, _HERE, cfg.get("stt_provider", "?")) if ccfg.get("enabled") else None
        )

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
    # -- deezer ducking --------------------------------------------------
    # Music is ducked while someone is talking: the user (the endpointer
    # holds from the first above-threshold chunk until the utterance is
    # committed to STT) or the bridge (one hold per aplay — reply, ack,
    # goodbye, beep, say_to_speaker). Holds overlap (talking over a reply,
    # an ack racing the player), so they're counted: duck on 0→1, unduck
    # on 1→0. Mute/idle state no longer affects ducking.
    def _duck_acquire(self, reason: str) -> None:
        with self._duck_lock:
            self._duck_holds += 1
            if self._duck_holds == 1:
                log.info("Ducking on (%s)", reason)
                self.deezer.duck()

    def _duck_release(self, reason: str) -> None:
        with self._duck_lock:
            self._duck_holds = max(0, self._duck_holds - 1)
            if self._duck_holds == 0:
                log.info("Ducking off (%s)", reason)
                self.deezer.unduck()

    @contextlib.contextmanager
    def _ducked(self, reason: str):
        self._duck_acquire(reason)
        try:
            yield
        finally:
            self._duck_release(reason)

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
        if self.wake_mode:
            # Idle-but-listening: the firmware mic must stay open or the
            # wake loop would hear only zeros, so the LED stays off too.
            self._wake_armed.set()
            # Say goodbye — unless a reply is still on its way, in which
            # case the player un-idles for it and "a dopo" would be a lie.
            # Only after a conversation, though: a wake that got no reply
            # (nobody spoke, or only noise) dozes off with a soft tone.
            if self._quiet_idle:
                # Only an announcement was spoken: no conversation to close.
                self._quiet_idle = False
            elif not self._turn_in_flight():
                clip = None
                if (self._replies_since_resume > 0
                        or not self._wake_cfg.get("goodbye_after_turn_only", True)):
                    clip = self._sleep_acks.pick() if self._sleep_acks else None
                if clip is None:
                    rate = int(self.cfg["tts_sample_rate"])
                    clip = ("tone", "(sleep tone)", _make_sleep_tone_pcm(rate))
                threading.Thread(target=self._say_goodbye, args=(clip,),
                                 name="vb-goodbye", daemon=True).start()
        else:
            self.hid.set_led(muted=True)

    def _turn_in_flight(self) -> bool:
        return (self._worker_busy.is_set() or not self.utterance_q.empty()
                or not self.playback_q.empty() or self._is_playing())

    def _play_clip(self, clip) -> None:
        provider, text, pcm = clip
        log.info("Ack (%s): %r", provider, text)
        self._speaking_ack.set()
        try:
            with self._ducked("ack"):
                play_audio(self.cfg["output_device"], pcm, int(self.cfg["tts_sample_rate"]))
        finally:
            self._speaking_ack.clear()

    def _say_goodbye(self, clip) -> None:
        try:
            self._play_clip(clip)
        except Exception:
            log.exception("Goodbye playback failed")

    def _reply_playing(self) -> bool:
        """A reply (or other player output) is audible or queued to be."""
        return self._is_playing() or not self.playback_q.empty() or bool(self._player_backlog)

    def _drain_playback(self) -> None:
        """Empty the player queues, releasing any blocked `play_pcm` caller."""
        items = list(self._player_backlog)
        self._player_backlog.clear()
        while True:
            try:
                items.append(self.playback_q.get_nowait())
            except queue.Empty:
                break
        for _gen, item in items:
            if isinstance(item, _ExternalUtterance):
                item.done.set()

    def _stop_reply(self) -> None:
        """HID press during playback: cut the reply and listen right away.

        Bumping the gen first stops the worker from queueing more of the
        reply (its tee checks the gen) and makes the player drop whatever
        is in flight; then the queues are drained and aplay is killed."""
        log.info("HID press during playback: stop reply, listening")
        self._resume()
        self._drain_playback()
        self._drain_queue(self.utterance_q)
        self._kill_player()

    def _on_hid_press(self) -> None:
        if self._processing.is_set():
            # Mic is only paused for the agent; a press here means "mute".
            # No gen bump, so the reply still plays — the mic stays muted.
            log.info("HID press while processing: mute (reply still plays)")
            self._processing.clear()
            self._auto_idled.clear()
            self._wake_armed.clear()
            self.hid.set_led(muted=True)
            return
        if self.recording.is_set():
            if self._reply_playing():
                self._stop_reply()
                return
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
            # An explicit mute is a privacy mute in wake mode too: firmware
            # silenced, wake word off, only the button resumes.
            self._wake_armed.clear()
            self.hid.set_led(muted=True)
        elif self._reply_playing():
            # Muted earlier (press while processing) and the reply is now
            # playing: a second press means "stop talking, I'm listening".
            self._stop_reply()
        else:
            log.info("HID press: resume recording")
            self._resume()

    def _resume(self, prefill: "Iterable[bytes]" = ()) -> None:
        """Idle/muted → recording. Shared by HID press and wake word.

        `prefill`: mic chunks captured before the resume (the wake word's
        trailing command) that the endpointer must see first."""
        # Bump gen so any stragglers from before (e.g. an old
        # in-progress speech buffer the endpointer might have under
        # the previous gen) are shed by downstream stages.
        self._auto_idled.clear()
        self._processing.clear()
        self._replies_since_resume = 0
        self._quiet_idle = False
        self._idle_window_ms = int(self.cfg.get(
            "idle_after_resume_ms", self.cfg.get("idle_timeout_ms", 0)))
        gen = self._bump_gen()
        # Queued before `recording` is set, so they land ahead of the
        # recorder's first live chunk.
        for chunk in prefill:
            self.audio_q.put((gen, chunk))
        # `recording` before clearing `_wake_armed`: the recorder checks
        # `recording` first, so no chunk falls between the two routes.
        self.recording.set()
        self._wake_armed.clear()
        self.hid.set_led(muted=False)

    def _on_wake(self, text: str, segment: bytes = b"", backlog: "Iterable[bytes]" = (),
                 command: str = "") -> None:
        """Wake phrase heard while idle: listen at once, ack in parallel.

        The mic opens before the ack plays, so "Hey Binary… metti la
        musica" said in one breath isn't cut. `backlog` (audio queued while
        Whistle transcribed) always goes to the endpointer; `segment` (the
        wake segment itself) too when Whistle heard words after the phrase
        (`command`) — STT then gets the whole sentence. With a command under
        way a soft tick replaces the spoken ack, so it doesn't talk over it."""
        if not self._wake_armed.is_set() or self.recording.is_set():
            return
        log.info("Wake word: %r%s", text, f" + command {command!r}" if command else "")
        prefill = list(self._split_chunks(segment)) if command else []
        prefill += list(backlog)
        speaking = self._is_playing()
        self._resume(prefill=prefill)
        if speaking:
            return  # a reply is already speaking; the resume is the ack
        rate = int(self.cfg["tts_sample_rate"])
        if command:
            clip = ("tone", "(wake tick)", _make_tick_pcm(rate))
        else:
            clip = self._wake_acks.pick() if self._wake_acks else None
            if clip is None:
                clip = ("tone", "(beep)", _make_beep_pcm(rate, freq=880, duration=0.08))
        threading.Thread(target=self._play_wake_ack, args=(clip,),
                         name="vb-wake-ack", daemon=True).start()

    def _play_wake_ack(self, clip) -> None:
        try:
            self._play_clip(clip)
        except Exception:
            # The mic is already open: a failed ack costs nothing else.
            log.exception("Wake ack playback failed")

    def _split_chunks(self, pcm: bytes) -> "Iterator[bytes]":
        """`pcm` in recorder-sized chunks (the VAD metric depends on N)."""
        step = int(self.cfg["chunk_size"]) * 2
        for i in range(0, len(pcm), step):
            yield pcm[i:i + step]

    def play_pcm(self, pcm: bytes, *, block: bool = True, text: str | None = None) -> float:
        """Play externally-supplied PCM through the bridge's player so it
        behaves like a normal spoken reply.

        Used by the MCP `say_to_speaker` tool. The PCM (S16LE mono at
        `tts_sample_rate`) is enqueued as one utterance under the current
        generation, so it serializes behind any reply already playing and
        reuses the player's no-clip drain and the `_is_playing()` guard (the
        endpointer won't auto-idle mid-speech).

        Like a real reply the device unmutes (mic open, LED off) and the
        player ducks deezer-connect for exactly the playback window.
        Afterwards the player's end-of-playback reset starts the idle window,
        so the usual `idle_timeout_ms` silence re-mutes — exactly the tail of
        a normal speech. Ducking is a no-op unless `deezer_connect` is enabled.

        `text` (what the PCM says) decides the tail: a question gets the
        answer window, a plain announcement goes back to sleep without a
        goodbye — see `_after_external`.

        Returns the audio duration in seconds; when `block` (the default),
        waits until the player has finished this utterance (bounded so a
        stuck player can't pin the caller).
        """
        sample_rate = int(self.cfg["tts_sample_rate"])
        seconds = len(pcm) / (sample_rate * 2) if pcm else 0.0
        if not pcm:
            return 0.0

        # Unmute like a normal speech. Mirrors `_on_hid_press` resume and
        # the player's auto-resume; idempotent if already unmuted.
        if not self.recording.is_set():
            log.info("play_pcm: unmuting for external speech (mic open, LED off)")
            self._wake_armed.clear()
            self.recording.set()
            self.hid.set_led(muted=False)
        self._auto_idled.clear()
        self._processing.clear()

        # One atomic item under the current gen → serializes behind any reply
        # already playing, never interleaves with the worker's chunks. The
        # player fires `done` when playback finishes (or immediately if a HID
        # press bumps the gen and the item goes stale).
        done = threading.Event()
        self.playback_q.put((self._current_gen(), _ExternalUtterance(
            pcm, done, question=_expects_answer(text or ""))))

        if block:
            # Generous bound: playback is realtime, plus drain + margin.
            done.wait(timeout=seconds + 15.0)
        return seconds

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

        def _safe_terminate(inst) -> None:
            if inst is None:
                return
            try:
                inst.terminate()
            except Exception:
                pass

        def _rebuild_pa(old):
            # Tear the old PyAudio instance down and build a fresh one so a
            # hot-plugged Jabra becomes visible (PyAudio snapshots its device
            # list at construction). Both the teardown and the rebuild are
            # guarded and the rebuild retries: a transient ALSA error during an
            # unplug must not kill the recorder thread — that would leave the
            # mic dead until a service restart. Returns the new instance, or
            # None when shutdown is requested while retrying.
            _safe_terminate(old)
            while not self.shutdown_event.wait(1.0):
                try:
                    return pyaudio.PyAudio()
                except Exception as exc:
                    log.warning("Recorder: PyAudio rebuild failed: %s — retry in 1s", exc)
            return None

        try:
            while not self.shutdown_event.is_set():
                listening = self.recording.is_set() or self._wake_armed.is_set()
                if not listening:
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
                    if idx is None:
                        # A Jabra plugged in after we started is invisible until
                        # we rebuild the instance — same reconnect philosophy the
                        # HID monitor uses, so a hot-plug needs no restart.
                        log.warning("Recorder: Jabra input not found — retry in 1s")
                        pa = _rebuild_pa(pa)
                        if pa is None:
                            return
                        continue
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
                        pa = _rebuild_pa(pa)
                        if pa is None:
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
                if self.recording.is_set():
                    self.audio_q.put((self._current_gen(), data))
                elif self._wake_armed.is_set():
                    self.wake_q.put(data)
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            _safe_terminate(pa)

    def _wake_loop(self) -> None:
        """Idle wake-word spotting: gate chunks on energy, Whistle the rest."""
        try:
            self._wake_detector.load()
        except Exception:
            log.exception("Wake word: Whistle failed to load — only the HID button can resume")
            return
        log.info("Wake word: listening for %s (lang=%s, rms>%g)",
                 self._wake_cfg["phrases"], self._wake_cfg["language"],
                 self._wake_cfg["rms_threshold"])
        gate = wake_word.SpeechGate(self.cfg["sample_rate"], self.cfg["chunk_size"], self._wake_cfg)
        while not self.shutdown_event.is_set():
            try:
                chunk = self.wake_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self._wake_armed.is_set() or self._speaking_ack.is_set():
                gate.reset()
                self._drain_queue(self.wake_q)
                continue
            segment = gate.feed(chunk)
            if segment is None:
                continue
            t0 = time.monotonic()
            try:
                text = self._wake_detector.transcribe(segment)
            except Exception as exc:
                log.warning("Wake word: transcribe failed: %s", exc)
                continue
            hit = self._wake_detector.heard(text)
            log.info("Wake segment %.1fs → %r in %.2fs%s",
                     len(segment) / (2 * self.cfg["sample_rate"]), text,
                     time.monotonic() - t0, " (wake)" if hit else "")
            if hit:
                # Audio queued while Whistle transcribed is the start of
                # what the user says next: hand it to the endpointer.
                backlog = []
                while True:
                    try:
                        backlog.append(self.wake_q.get_nowait())
                    except queue.Empty:
                        break
                self._on_wake(text, segment, backlog, self._wake_detector.command(text))
                gate.reset()

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
        min_speech_chunks = round(self.cfg.get("min_speech_ms", 0) / chunk_ms)
        keep_chunks = max(0, int(self.cfg.get("silence_keep_ms", 500) / chunk_ms))
        pre_chunks = max(0, int(self.cfg.get("pre_speech_keep_ms", 100) / chunk_ms))
        prebuf: "deque[bytes] | None" = (
            deque(maxlen=pre_chunks) if pre_chunks > 0 else None
        )
        # idle_timeout_ms = 0 disables auto-idle entirely; otherwise the
        # window in force is `self._idle_window_ms` (widened after a reply).
        idle_enabled = int(self.cfg.get("idle_timeout_ms", 0)) > 0
        min_voiced_ms = int((self.cfg.get("speech_filter") or {}).get("min_voiced_ms", 0))
        max_utt_ms = int(self.cfg.get("max_utterance_ms", 0))
        max_utt_chunks = int(max_utt_ms / chunk_ms) if max_utt_ms > 0 else 0

        seen_gen = self._current_gen()
        in_speech = False
        silence_count = 0
        buf: list[bytes] = []
        gen_at_start = seen_gen
        speech_tick = 0
        levels: list[float] = []  # per-chunk energy of the current utterance
        # Music is ducked from the first above-threshold chunk until the
        # utterance goes to STT (commit), is dropped, or is discarded.
        ducked = False

        def unduck_speech(reason: str) -> None:
            nonlocal ducked
            if ducked:
                ducked = False
                self._duck_release(reason)

        def level_stats() -> str:
            if not levels:
                return "n/a"
            lv = sorted(levels)
            pick = lambda q: lv[min(len(lv) - 1, int(q * len(lv)))]
            return f"p50={pick(0.5):.3g} p90={pick(0.9):.3g} max={lv[-1]:.3g}"

        while not self.shutdown_event.is_set():
            # Reset on resume: a gen bump means the user pressed HID
            # to resume from idle, so any half-built speech buffer
            # captured under the old gen must be discarded.
            cur_gen = self._current_gen()
            if cur_gen != seen_gen:
                unduck_speech("speech discarded")
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
                    unduck_speech("speech committed")
                    self._enqueue_utterance(gen_at_start, pcm, sr)
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
                    levels = []
                    if prebuf:
                        buf.extend(prebuf)
                        prebuf.clear()
                    log.info("Endpointer: sound detected (rms=%.0f ≥ %g)",
                             rms, rms_threshold)
                    if not ducked:
                        ducked = True
                        self._duck_acquire("speech")
                buf.append(data)
                levels.append(rms)
                silence_count = 0
                speech_tick += 1
                if speech_tick % 32 == 0:
                    log.info("Endpointer: still in_speech tick=%d rms=%.0f",
                             speech_tick, rms)
                if max_utt_chunks and len(buf) >= max_utt_chunks:
                    # Nobody talks to an assistant this long in one go: it's
                    # people talking among themselves (seen: 19–34 s of
                    # family chat sent as one turn). Drop it and go idle —
                    # in wake mode only the wake word brings it back.
                    log.info("Endpointer: utterance over max_utterance_ms=%d — side "
                             "conversation, dropped; going idle", max_utt_ms)
                    unduck_speech("utterance too long")
                    buf = []
                    in_speech = False
                    silence_count = 0
                    self._enter_idle(source=f"utterance>{max_utt_ms}ms")
                    continue
            else:
                if in_speech:
                    buf.append(data)
                    if silence_count < commit_chunks:
                        levels.append(rms)
                    if silence_count == 0:
                        log.info("Endpointer: silence onset (rms=%.0f < %g, "
                                 "need %d chunks ≈ %dms to commit)",
                                 rms, rms_threshold, commit_chunks,
                                 self.cfg["silence_timeout_ms"])
                elif prebuf is not None:
                    # Rolling pre-roll window for the next utterance.
                    prebuf.append(data)
                silence_count += 1

                if in_speech and silence_count >= commit_chunks and speech_tick < min_speech_chunks:
                    # Too short to be speech (click, bump, echo blip):
                    # drop it before STT. silence_count keeps running, so
                    # noise never postpones auto-idle.
                    log.info("Endpointer: dropped %d-chunk burst (< min_speech_ms=%d) levels %s",
                             speech_tick, self.cfg["min_speech_ms"], level_stats())
                    unduck_speech("burst dropped")
                    buf = []
                    in_speech = False
                elif (in_speech and silence_count >= commit_chunks
                        and self._speech_vad is not None
                        and self._too_little_voice(buf, sr, min_voiced_ms, level_stats)):
                    # Loud, long enough, but not a voice (keyboard, TV hum,
                    # a door): dropped like a burst, no STT call, no pause.
                    unduck_speech("no voice")
                    buf = []
                    in_speech = False
                elif in_speech and silence_count >= commit_chunks:
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
                             "kept %d trailing silence, trimmed %d) levels %s",
                             len(speech_buf), duration_s,
                             min(keep_chunks, silence_count), trim, level_stats())
                    unduck_speech("speech committed")
                    self._idle_window_ms = int(self.cfg.get("idle_timeout_ms", 0))
                    if self._enqueue_utterance(gen_at_start, pcm, sr):
                        self._pause_for_processing()
                    buf = []
                    in_speech = False
                    # Treat commit as a "transaction" boundary: the
                    # idle 10 s window starts counting from here, not
                    # from the trailing-silence chunks already absorbed
                    # to detect end-of-speech. (Playback end resets it
                    # again via _idle_reset_pending — whichever lands
                    # later wins.)
                    silence_count = 0

                idle_ms = self._idle_window_ms
                idle_chunks = int(idle_ms / chunk_ms) if idle_enabled and idle_ms > 0 else 0
                if (not in_speech
                        and idle_chunks > 0
                        and silence_count >= idle_chunks
                        and not self._is_playing()
                        and not self._idle_reset_pending.is_set()):
                    # The `_idle_reset_pending` guard closes a race at the
                    # tail of a reply: the player clears `_player_proc`
                    # (so `_is_playing()` flips False) and sets the reset
                    # flag as a pair, but the endpointer could observe the
                    # cleared proc one tick before consuming the reset —
                    # with `silence_count` already past the threshold from
                    # the whole playback — and fire idle the instant the
                    # reply finished. Holding off while a reset is pending
                    # lets the next loop tick zero the counter first, so
                    # the idle window truly starts at end-of-playback.
                    self._enter_idle(source=f"silence>{idle_ms}ms")
                    silence_count = 0
                    buf = []

    def _too_little_voice(self, buf: list[bytes], sr: int, min_voiced_ms: int,
                          level_stats) -> bool:
        """webrtcvad check at commit. Logs the voiced ms of every commit so
        `speech_filter.min_voiced_ms` can be tuned from the journal."""
        voiced = _voiced_ms(b"".join(buf), sr, self._speech_vad)
        if voiced is None:
            return False
        if voiced < min_voiced_ms:
            log.info("Endpointer: dropped %.2fs utterance, only %d ms voiced "
                     "(< speech_filter.min_voiced_ms=%d) levels %s",
                     len(buf) * len(buf[0]) / (2 * sr) if buf else 0.0,
                     voiced, min_voiced_ms, level_stats())
            return True
        log.info("Endpointer: %d ms voiced", voiced)
        return False

    def _enqueue_utterance(self, gen: int, pcm: bytes, sr: int) -> bool:
        """Hand a committed utterance to the worker, holding at most ONE
        message in line behind the turn in flight. While the worker is busy
        (gateway or playback), the first utterance waits; anything said after
        it is dropped — otherwise interjections, side talk and speaker echo
        pile into a muddled turn."""
        if self._worker_busy.is_set() and not self.utterance_q.empty():
            log.info("Endpointer: one message already queued, dropping %.2fs utterance",
                     len(pcm) / (sr * 2))
            return False
        self.utterance_q.put((gen, pcm, sr))
        return True

    def _pause_for_processing(self) -> None:
        """Stop recording while the committed utterance is processed.

        The player resumes the mic when the reply starts (via `_auto_idled`,
        same as an auto-idle), and the worker resumes it if the turn ends
        without a reply (noise, empty STT, gateway error). The LED and
        firmware mute are left alone: this is not a mute, just not listening.
        """
        if not self.recording.is_set():
            return
        log.info("Processing: mic paused until the reply starts")
        self.recording.clear()
        self._drain_queue(self.audio_q)
        self._processing.set()
        self._auto_idled.set()

    def _resume_after_processing(self) -> None:
        """Turn ended with nothing to play: listen again."""
        if not self._processing.is_set():
            return
        self._processing.clear()
        self._auto_idled.clear()
        log.info("Processing: no reply, resuming mic")
        self._idle_reset_pending.set()
        self.recording.set()

    def _drain_utterances(self, gen: int, into: list[bytes]) -> int:
        """Pull every currently-queued same-gen utterance onto `into`.

        Non-blocking. Stale-gen items (a hard-cancel resume bumped `_gen`)
        are dropped. Returns how many segments were appended.
        """
        added = 0
        while True:
            try:
                g, pcm, _sr = self.utterance_q.get_nowait()
            except queue.Empty:
                break
            if g != gen:
                continue  # stale generation — drop
            into.append(pcm)
            added += 1
        return added

    def _worker_loop(self) -> None:
        while not self.shutdown_event.is_set():
            # Every `continue` below lands back here, so busy spans exactly
            # pick-up → reply handed to the player (or turn dropped).
            self._worker_busy.clear()
            if (self._processing.is_set() and self.utterance_q.empty()
                    and self.playback_q.empty() and not self._is_playing()):
                self._resume_after_processing()
            try:
                gen, pcm, sr = self.utterance_q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._worker_busy.set()
            if gen != self._current_gen():
                continue

            # Only ONE message waits behind the reply being delivered: hold
            # this utterance until the previous reply has finished playing,
            # and drop anything else said meanwhile (the endpointer already
            # refuses a second one while we're busy; this catches the rest).
            # Well-behaved turn-taking pays no latency: when no reply is in
            # flight the wait loop doesn't run and the utterance is sent
            # immediately.
            extra: list[bytes] = []
            self._drain_utterances(gen, extra)
            while not self.playback_q.empty() or self._is_playing():
                if self.shutdown_event.is_set() or gen != self._current_gen():
                    break
                self.shutdown_event.wait(0.1)
                self._drain_utterances(gen, extra)
            if gen != self._current_gen():
                continue
            if extra:
                log.info("Worker: dropped %d extra utterance(s) queued behind the reply",
                         len(extra))

            t0 = time.monotonic()
            self._turn_t0 = t0
            timing: dict[str, float] = {}
            # Dead-air feedback from pick-up until the first reply audio.
            cue = _ThinkingCue(self, gen).start()
            try:
                self._run_turn(gen, pcm, sr, t0, timing, cue)
            finally:
                cue.stop()

    def _run_turn(self, gen: int, pcm: bytes, sr: int, t0: float,
                  timing: dict, cue: "_ThinkingCue") -> None:
        """One turn: STT → gateway → TTS → playback_q. Returns early (turn
        dropped) on empty/non-speech STT or a stale generation."""
        log.info("Worker: STT (%d bytes ≈ %.2fs)", len(pcm), len(pcm) / (sr * 2))
        t_stt = time.monotonic()
        text = self.stt.transcribe(pcm, sr)
        timing["stt"] = time.monotonic() - t0
        if self._stt_compare:
            self._stt_compare.submit(pcm, sr, text, time.monotonic() - t_stt)
        if not text:
            log.info("Worker: empty transcription, skipping")
            return
        if _is_non_speech(text):
            log.info("Worker: non-speech transcription %s, skipping", text)
            return
        if gen != self._current_gen():
            return
        log.info("User: %s", text)

        backend = self.cfg.get("gateway_backend", "openclaw")
        if backend == "zeroclaw_ws":
            log.info("Worker: → gateway %s (backend=zeroclaw_ws agent=%s session=%s)",
                     self.cfg["gateway_base_url"],
                     self.cfg.get("gateway_agent", "default"),
                     self.cfg.get("session_key", "voice-bridge"))
            text_stream = gateway_chat_stream_zeroclaw_ws(
                self.cfg["gateway_base_url"],
                self.cfg["gateway_token"],
                text,
                self.cfg.get("gateway_agent", "default"),
                self.cfg.get("session_key", "voice-bridge"),
                on_event=cue.on_event,
            )
        elif backend == "zeroclaw":
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
                timing.setdefault("first_token", time.monotonic() - t0)
                collected.append(delta)
                yield delta
            timing["gateway_done"] = time.monotonic() - t0

        try:
            for chunk in self.tts.synthesize_stream(_filter_no_reply(_tee(text_stream))):
                if gen != self._current_gen():
                    break
                if not chunk:
                    continue
                if "first_pcm" not in timing:
                    timing["first_pcm"] = time.monotonic() - t0
                    cue.stop()  # before the first chunk: no cue lands behind it
                self.playback_q.put((gen, chunk))
        except Exception as exc:
            log.error("Worker: TTS pipeline error: %s", exc)
        finally:
            cue.stop()
            full_reply = "".join(collected).strip()
            self._reply_is_question = _expects_answer(full_reply)
            if _is_no_reply(full_reply) and gen == self._current_gen():
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

        if collected and not _is_no_reply(full_reply):
            log.info("Binary: %s", "".join(collected)[:200])
        log.info("Turn timing (s from pick-up): %s, cues=%d",
                 " ".join(f"{k}={v:.2f}" for k, v in timing.items()) or "n/a",
                 cue.cues_played)

    def _next_playback(self, timeout: float):
        """Next player item: deferred external utterances first, then the queue."""
        if self._player_backlog:
            return self._player_backlog.popleft()
        return self.playback_q.get(timeout=timeout)

    def _after_reply(self, *, cue: bool = False) -> None:
        """A reply (or say_to_speaker speech) finished playing: widen the
        idle window so the user has time to answer — more if it asked."""
        if cue:
            return
        self._replies_since_resume += 1
        self._quiet_idle = False
        self._idle_window_ms = int(
            self.cfg.get("idle_after_question_ms" if self._reply_is_question
                         else "idle_after_reply_ms",
                         self.cfg.get("idle_timeout_ms", 0)))
        self._reply_is_question = False

    def _after_external(self, question: bool) -> None:
        """say_to_speaker finished. A question opens a conversation: the
        answer window, and a goodbye later if nobody answers. A plain
        announcement keeps the normal window and, unless a conversation was
        already going, dozes off silently — no "Ciao ciao!" after "La lavatrice
        ha finito"."""
        if question:
            self._replies_since_resume += 1
            self._quiet_idle = False
            self._idle_window_ms = int(self.cfg.get(
                "idle_after_question_ms", self.cfg.get("idle_timeout_ms", 0)))
            return
        self._idle_window_ms = int(self.cfg.get("idle_timeout_ms", 0))
        if not self._replies_since_resume:
            self._quiet_idle = True

    def _player_loop(self) -> None:
        device = self.cfg["output_device"]
        sample_rate = self.cfg["tts_sample_rate"]
        while not self.shutdown_event.is_set():
            try:
                gen, item = self._next_playback(timeout=0.2)
            except queue.Empty:
                continue
            # Externally-supplied speech (MCP say_to_speaker) and thinking
            # cues arrive as one atomic item; play it as its own utterance
            # via the same path, then release the blocked caller.
            # play_pcm() already did the unmute transition before enqueuing.
            if isinstance(item, _ExternalUtterance):
                if gen == self._current_gen():
                    ctx = self._ducked(item.label) if item.duck else contextlib.nullcontext()
                    with ctx:
                        self._play_blob(item.pcm, device, sample_rate)
                    if gen == self._current_gen() and not item.cue:  # not cut by a barge-in
                        self._after_external(item.question)
                item.done.set()
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
                self._processing.clear()
                self.recording.set()
                self.hid.set_led(muted=False)
                self._idle_reset_pending.set()

            with self._ducked("reply"):
                if self._play_streamed(item, device, sample_rate, gen):
                    self._after_reply()

    def _prebuffer(self, first: bytes, gen: int, sample_rate: int) -> "tuple[list[bytes], bool, bool]":
        """Collect `playback_prebuffer_ms` of PCM before the first aplay
        write. ElevenLabs v3 often sends one chunk and then stalls ~0.5 s;
        writing that chunk alone makes aplay start and underrun (click).
        Waits at most prebuffer + 1 s, so a slow stream still starts.

        Returns (chunks, ended, stale): `ended` when the end-of-utterance
        marker was reached, `stale` when the gen changed meanwhile."""
        ms = int(self.cfg.get("playback_prebuffer_ms", 0))
        chunks = [first]
        if ms <= 0:
            return chunks, False, False
        want = sample_rate * 2 * ms // 1000
        have = len(first)
        deadline = time.monotonic() + ms / 1000.0 + 1.0
        while have < want and not self.shutdown_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                gen2, item2 = self.playback_q.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if gen2 != self._current_gen() or gen2 != gen:
                if isinstance(item2, _ExternalUtterance):
                    item2.done.set()
                return chunks, False, True
            if isinstance(item2, _EndOfUtterance):
                return chunks, True, False
            if isinstance(item2, _ExternalUtterance):
                self._player_backlog.append((gen2, item2))
                continue
            chunks.append(item2)
            have += len(item2)
        return chunks, False, False

    def _play_streamed(self, item: bytes, device: str, sample_rate: int,
                       gen: int | None = None) -> bool:
        """Play one reply: `item` is its first PCM chunk, the rest is pulled
        off `playback_q` until the end-of-utterance marker. Returns True when
        the reply played to its end (not cut by a stale gen / barge-in)."""
        if gen is None:
            gen = self._current_gen()
        chunks, ended, stale = self._prebuffer(item, gen, sample_rate)
        if stale:
            return False
        proc = _aplay_popen(device, sample_rate, bufsize=0)
        with self._player_lock:
            self._player_proc = proc
        if self._turn_t0 is not None:
            log.info("Turn: first audio %.2fs after pick-up", time.monotonic() - self._turn_t0)
            self._turn_t0 = None
        completed = False
        try:
            if not self._write_chunk(proc, b"".join(chunks)):
                return False
            if ended:
                completed = True
                return True
            while not self.shutdown_event.is_set():
                try:
                    gen2, item2 = self.playback_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if gen2 != self._current_gen():
                    # Hard-cancel happened mid-utterance; the proc
                    # has likely already been killed, but break
                    # explicitly so we close it cleanly.
                    if isinstance(item2, _ExternalUtterance):
                        item2.done.set()
                    break
                if isinstance(item2, _EndOfUtterance):
                    completed = True
                    break
                if isinstance(item2, _ExternalUtterance):
                    # say_to_speaker while a reply streams: play it next,
                    # never write the object into aplay.
                    self._player_backlog.append((gen2, item2))
                    continue
                if not self._write_chunk(proc, item2):
                    break
        finally:
            # Close stdin and play out aplay's buffered tail without
            # chopping it (bounded so a hung aplay can't pin the
            # thread). See `_drain_aplay`: a fixed timeout + kill would
            # clip the reply's tail and, via `sw_dmix`'s mixing, overlap
            # the next utterance. Crucially, `_player_proc` stays set
            # throughout the drain — it's only cleared below, after the
            # wait returns — so `_is_playing()` keeps the endpointer
            # from firing auto-idle during the audible tail of the reply.
            _drain_aplay(proc, sample_rate, abort=self.shutdown_event)
            # Order matters: signal the end-of-playback idle reset
            # BEFORE clearing the player handle. The endpointer's idle
            # check also gates on `_idle_reset_pending`, so as long as
            # the flag is already set whenever `_player_proc` becomes
            # None, the endpointer can never see "not playing + stale
            # silence_count" and auto-idle the instant the reply ends.
            # The flag is consumed on the endpointer's next tick, which
            # zeroes `silence_count` — restarting the idle window from
            # here (end of playback), as intended.
            self._idle_reset_pending.set()
            with self._player_lock:
                self._player_proc = None
        return completed and gen == self._current_gen()

    def _play_blob(self, pcm: bytes, device: str, sample_rate: int) -> None:
        """Play one complete PCM blob as a single utterance (external speech).

        Mirrors the per-utterance body of `_player_loop`: registers the aplay
        proc in `_player_proc` (so `_is_playing()` blocks auto-idle during the
        speech), writes the whole blob, then runs the bounded no-clip drain
        and starts the end-of-playback idle window via `_idle_reset_pending`.
        """
        proc = _aplay_popen(device, sample_rate, bufsize=0)
        with self._player_lock:
            self._player_proc = proc
        try:
            self._write_chunk(proc, pcm)
        finally:
            _drain_aplay(proc, sample_rate, abort=self.shutdown_event)
            self._idle_reset_pending.set()
            with self._player_lock:
                self._player_proc = None

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
        # Pin deezer-connect to its configured baseline volume on boot
        # (no-op unless the plugin is enabled).
        self.deezer.apply_default_volume()
        loops = [
            ("hid", self._hid_loop),
            ("recorder", self._recorder_loop),
            ("endpointer", self._endpointer_loop),
            ("worker", self._worker_loop),
            ("player", self._player_loop),
        ]
        if self.wake_mode:
            # Boot idle-but-listening: firmware mic open (the HID monitor
            # boots muted), acks synthesized/loaded off the hot path.
            if self._wake_armed.is_set():
                self.hid.set_led(muted=False)
            loops += [("wake", self._wake_loop),
                      ("wake-acks", lambda: self._wake_acks.prepare(self.shutdown_event)),
                      ("sleep-acks", lambda: self._sleep_acks.prepare(self.shutdown_event))]
        if self._thinking_acks:
            loops.append(("thinking-acks", lambda: self._thinking_acks.prepare(self.shutdown_event)))
        if self._stt_compare:
            self._stt_compare.start()
        for name, fn in loops:
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
        # arrived (SIGTERM mid-playback or mid-speech). No-op when the
        # plugin is disabled or wasn't currently ducking.
        with self._duck_lock:
            self._duck_holds = 0
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
    log.info("TTS provider: %s (voice=%s model=%s rate=%dHz stream=%s whole_reply=%s)",
             cfg["tts_provider"], cfg["elevenlabs_voice"], cfg["elevenlabs_model"],
             cfg["tts_sample_rate"], cfg["tts_streaming_mode"], cfg["tts_whole_reply"])
    log.info("Output: %s @ %d Hz", cfg["output_device"], cfg["tts_sample_rate"])
    _apply_output_volume(cfg)
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
    if cfg["activation"] == "wake_word":
        log.info("Ready — say the wake word to begin (Jabra button mutes/resumes)")
    elif cfg.get("hid_mute_enabled"):
        log.info("Ready — device starts muted, press the Jabra button to begin")
    else:
        log.info("Ready — listening (always-on mic, HID button disabled)")

    # Serve the MCP voice tools from inside this process when
    # `mcp_server.enabled` is true, sharing this process's config and the
    # same provider instances. Best-effort: a failure to start the MCP server
    # is caught and logged so the always-on voice client keeps running.
    if (cfg.get("mcp_server") or {}).get("enabled"):
        try:
            import mcp_voice_server
            mcp_voice_server.configure(cfg, stt=stt, tts=tts, bridge=bridge)
            mcp_voice_server.serve_background()
            log.info("MCP server: http://%s:%d",
                     mcp_voice_server._HOST, mcp_voice_server._PORT)
        except Exception:
            log.exception(
                "MCP server failed to start; voice client continues"
            )

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
