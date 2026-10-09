"""Config loading for the voice bridge.

`load_config()` reads `voice-bridge.json` (non-secret settings) and
`voice-bridge.secrets.json` (gitignored keys) from this folder, validates
them and returns one flat dict; `_build_voice_provider()` turns the
`stt_provider` / `tts_provider` choice into a provider instance. See
AGENTS.md → *Config* for every key.
"""

from __future__ import annotations

import json
import os
import re

import speaker_socket
import wake_word
from deepgram_voice import DeepgramVoice
from elevenlabs_voice import VALID_TTS_STREAM_MODES, ElevenLabsVoice

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "voice-bridge.json")
# Local, gitignored secrets file (API keys + gateway token). The bridge
# is self-contained: everything it needs lives in this folder. See
# resources/voice-bridge.secrets.example.json for the expected shape.
SECRETS_PATH = os.path.join(_HERE, "voice-bridge.secrets.json")

VALID_PROVIDERS = ("elevenlabs", "deepgram")
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

    # Unix socket the bridge serves `play_pcm` on for the MCP server's
    # `say_to_speaker` (see speaker_socket.py). Read by both processes.
    mcp_cfg = cfg.get("mcp_server") or {}
    cfg["speaker_socket"] = mcp_cfg.get("speaker_socket") or speaker_socket.default_path()

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
