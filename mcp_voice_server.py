"""MCP server exposing the voice-bridge STT/TTS providers as tools.

Lets an MCP client (e.g. zeroclaw) transcribe inbound voice messages and
synthesize spoken replies when relaying audio to/from a social platform.
The two providers (`ElevenLabsVoice`, `DeepgramVoice`) are reused verbatim
from the bridge; this module is a thin wrapper around them plus an `ffmpeg`
boundary that handles every container format (ogg/opus, mp3, wav) so the
providers can stay PCM-centric, exactly as the bridge uses them.

A third tool, `send_voice_telegram`, delivers a synthesized Opus/Ogg file
to a Telegram chat via the Bot API. It exists here (not in zeroclaw)
because the bridge's TTS already produces the OGG, and a one-shot
`sendVoice` POST does NOT conflict with zeroclaw's `getUpdates` poller —
the 409 only applies to long-polling. Requires `telegram_bot_token` in
voice-bridge.secrets.json.

A `say_to_speaker` tool is the speaker-side analogue of `say_to_telegram`:
it synthesizes text and plays it aloud on the bridge's ALSA output device
(`output_device`) via aplay, reusing the bridge's own playback helpers.
Output mixes through sw_dmix, so it coexists with whatever the bridge is
playing; it needs `aplay` on PATH and `audio`-group access.

Design notes:
- **All container <-> PCM conversion goes through ffmpeg, uniformly.** STT
  decodes whatever the client sent to S16LE mono PCM, then calls the
  provider's `transcribe(pcm, rate)`. TTS calls `synthesize(text)` (which
  returns S16LE PCM at `tts_sample_rate`) and pipes that into ffmpeg to
  produce the requested container. We do NOT rely on the provider/SDK to
  emit mp3/ogg directly — ffmpeg (with libmp3lame + libopus) is the single,
  tier-independent path.
- **Config is shared, not duplicated.** `voice-bridge.py` owns
  `load_config()` and `_build_voice_provider()`; we import that module by
  path (its hyphenated filename blocks a normal `import`) so the secrets,
  provider selection, and ElevenLabs/Deepgram settings come from the same
  `voice-bridge.json` / `voice-bridge.secrets.json` the bridge reads.
- **Transport is streamable-http** so zeroclaw connects by URL. Host/port
  (and the output directory for synthesized files) come from an optional
  `mcp_server` block in `voice-bridge.json`; sane localhost defaults
  otherwise.

Run:
    .venv/bin/python mcp_voice_server.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import uuid

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("mcp_voice_server")

HERE = os.path.dirname(os.path.abspath(__file__))

# STT decode target. 16 kHz mono is plenty for both Scribe and nova-3 and
# keeps the WAV the providers build internally small.
_STT_RATE = 16000

# Output containers we can produce. Each maps to the ffmpeg encoder args and
# the MIME type the client should label the file with. `ogg` is Opus-in-Ogg,
# the WhatsApp/Telegram voice-note format.
_OUTPUT_FORMATS = {
    "ogg": (["-c:a", "libopus", "-b:a", "32k"], "audio/ogg", ".ogg"),
    "mp3": (["-c:a", "libmp3lame", "-q:a", "4"], "audio/mpeg", ".mp3"),
    "wav": (["-c:a", "pcm_s16le"], "audio/wav", ".wav"),
}


def _load_bridge_module():
    """Import voice-bridge.py by path (hyphen blocks `import voice-bridge`)."""
    path = os.path.join(HERE, "voice-bridge.py")
    spec = importlib.util.spec_from_file_location("voice_bridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Runtime state, populated by `configure()`. Kept module-global because the
# `@mcp.tool()` functions read them at call time. The server can run two ways:
#   - standalone (`python mcp_voice_server.py`): `configure()` loads its own
#     config and builds its own providers.
#   - embedded in the bridge process: the bridge calls `configure(cfg,
#     stt=..., tts=..., bridge=...)` to inject the SAME config and provider
#     instances it already built, then `serve_background()`. One process,
#     one config load, shared providers — see the bridge's main().
_bridge = None
_cfg: dict = {}
_stt = None
_tts = None
_TTS_RATE = 0
# ALSA output PCM name for the `say_to_speaker` tool — the same
# `output_device` the bridge plays replies through (e.g. `voice_out`).
# Playback goes through sw_dmix, so it mixes with whatever the bridge is
# already playing; the device's mic-mute state does not gate output.
_OUT_DEVICE = ""
_HOST = "127.0.0.1"
_PORT = 9080
# Telegram bot token for the optional `send_voice_telegram` /
# `say_to_telegram` tools. Empty string disables them (they raise on
# call). Lives in the shared voice-bridge.secrets.json so all secrets
# stay in one place. The default `chat_id` is plain config (non-secret),
# in voice-bridge.json → `telegram.chat_id`; tool callers may override.
_TG_TOKEN = ""
_TG_CHAT_ID = ""
# Where synthesized files land. Default to a per-run temp dir; the client is
# expected to read the returned path (it shares the Pi's filesystem) and is
# responsible for cleanup once the file has been relayed.
_OUT_DIR = ""

# The FastMCP app is created at import time (the `@mcp.tool()` decorators
# below need it), but host/port are NOT bound here — `serve_*()` pass them
# to uvicorn from the resolved config, so a single instance works for both
# standalone and embedded modes.
mcp = FastMCP("voice-bridge-voice")


def configure(cfg: dict | None = None, *, stt=None, tts=None, bridge=None) -> "FastMCP":
    """Resolve config + providers into the module globals the tools read.

    Standalone: call with no args — loads `voice-bridge.json` via the bridge
    module and builds its own providers. Embedded: the bridge passes its
    already-loaded `cfg` and built `stt`/`tts` instances (and itself as
    `bridge`) so nothing is loaded or constructed twice. Idempotent enough
    to call once at startup. Returns the configured `mcp` app.
    """
    global _bridge, _cfg, _stt, _tts, _TTS_RATE, _OUT_DEVICE
    global _HOST, _PORT, _TG_TOKEN, _TG_CHAT_ID, _OUT_DIR

    _bridge = bridge or _load_bridge_module()
    _cfg = cfg if cfg is not None else _bridge.load_config()
    # Build any provider not injected. Each provider class constructs its own
    # SDK client per call, so a single shared instance is safe to use
    # concurrently across FastMCP's threadpool-dispatched tool calls.
    _stt = stt if stt is not None else _bridge._build_voice_provider("stt", _cfg)
    _tts = tts if tts is not None else _bridge._build_voice_provider("tts", _cfg)
    _TTS_RATE = int(_cfg["tts_sample_rate"])
    _OUT_DEVICE = str(_cfg["output_device"])

    mcp_cfg = _cfg.get("mcp_server") or {}
    _HOST = str(mcp_cfg.get("host", "127.0.0.1"))
    _PORT = int(mcp_cfg.get("port", 9080))
    _TG_TOKEN = str(_cfg.get("telegram_bot_token") or "")
    _TG_CHAT_ID = str(_cfg.get("telegram_chat_id") or "")
    _OUT_DIR = os.path.abspath(
        mcp_cfg.get("out_dir") or os.path.join(tempfile.gettempdir(), "voice-bridge-mcp")
    )

    log.info(
        "providers: stt=%s tts=%s | tts_rate=%d | out_dir=%s | http=%s:%d",
        _cfg["stt_provider"], _cfg["tts_provider"], _TTS_RATE, _OUT_DIR, _HOST, _PORT,
    )
    return mcp


def _make_uvicorn_server():
    """Build a uvicorn Server bound to the configured host/port for the
    streamable-http ASGI app. Shared by both serve paths."""
    import uvicorn  # lazy: only needed when actually serving

    app = mcp.streamable_http_app()
    config = uvicorn.Config(app, host=_HOST, port=_PORT, log_level="info")
    return uvicorn.Server(config)


def serve_background() -> threading.Thread:
    """Run the MCP HTTP server in a daemon thread (embedded-in-bridge mode).

    uvicorn installs SIGINT/SIGTERM handlers, which only work in the main
    thread — so we disable them here and let the host process (the bridge)
    own the signals. The server runs its own asyncio loop inside the thread.
    Returns the started thread.
    """
    server = _make_uvicorn_server()
    server.install_signal_handlers = lambda: None  # bridge owns the signals

    def _run() -> None:
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, name="mcp-http", daemon=True)
    t.start()
    return t


def serve_blocking() -> None:
    """Run the MCP HTTP server in the foreground (standalone mode).

    uvicorn owns SIGINT/SIGTERM and blocks until shutdown. This replaces the
    old `mcp.run(transport="streamable-http")` so both serve paths share the
    same host/port resolution.
    """
    _make_uvicorn_server().run()


def _decode_to_pcm(path: str, rate: int = _STT_RATE) -> bytes:
    """Decode any audio file (ogg/opus, mp3, wav, ...) to S16LE mono PCM.

    ffmpeg auto-detects the input container/codec, so the client's declared
    mime_type is advisory only. Raises CalledProcessError on a decode
    failure (corrupt/empty/unsupported input) — surfaced to the caller as a
    tool error rather than silently returning "".
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(rate),
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return proc.stdout


def _encode_from_pcm(pcm: bytes, src_rate: int, fmt: str, out_path: str) -> None:
    """Encode raw S16LE mono PCM into `fmt`'s container at `out_path`."""
    enc_args, _mime, _ext = _OUTPUT_FORMATS[fmt]
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(src_rate), "-ac", "1", "-i", "pipe:0",
            *enc_args,
            out_path,
        ],
        input=pcm,
        capture_output=True,
        check=True,
    )


@mcp.tool()
def speech_to_text(audio_path: str, mime_type: str | None = None) -> str:
    """Transcribe a voice-message audio file to text (Italian).

    Use this when relaying an inbound voice note from a social platform:
    pass the path to the downloaded audio file. Any common container is
    accepted (Opus/Ogg as used by WhatsApp & Telegram, MP3, WAV, ...) —
    the file is decoded with ffmpeg and run through the configured STT
    provider. `mime_type` is advisory (ffmpeg auto-detects). Returns the
    transcript, or "" if the audio was silent/unintelligible.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"audio_path not found: {audio_path}")
    try:
        pcm = _decode_to_pcm(audio_path, _STT_RATE)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg decode failed: {stderr}") from exc
    text = _stt.transcribe(pcm, _STT_RATE)
    log.info("speech_to_text: %s -> %d chars", audio_path, len(text))
    return text


# Emoji + pittogrammi (range Unicode comuni) da togliere prima del TTS:
# vengono pronunciati male o bloccano il flusso audio del provider.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # simboli & pittogrammi, emoticon, oggetti
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # bandiere
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U00002190-\U000021FF"  # frecce
    "\U00002B00-\U00002BFF"  # frecce/simboli misc
    "\U0000200D"             # zero-width joiner
    "]+",
    flags=re.UNICODE,
)


def _sanitize_for_tts(text: str) -> str:
    """Toglie markdown ed emoji dal testo prima della sintesi vocale.

    Il modello (deepseek) ogni tanto produce `**grassetto**`, `_corsivo_`,
    `# titoli`, backtick ed emoji nonostante le istruzioni di prompt. Gli
    asterischi in particolare mandano in errore / bloccano il flusso audio
    del TTS. Qui li rimuoviamo in modo deterministico: il client può fidarsi
    che qualunque testo passato a TTS venga ripulito.
    """
    if not text:
        return text
    t = text
    # Rimuovi recinti di codice ``` e backtick singoli (tieni il contenuto).
    t = t.replace("```", " ").replace("`", "")
    # Grassetto/corsivo markdown: **x**, *x*, __x__, _x_, ~~x~~ -> x
    t = re.sub(r"\*{1,3}([^*]+?)\*{1,3}", r"\1", t)
    t = re.sub(r"_{1,3}([^_]+?)_{1,3}", r"\1", t)
    t = re.sub(r"~~([^~]+?)~~", r"\1", t)
    # Asterischi/underscore/tilde/cancelletti rimasti spaiati -> via.
    t = re.sub(r"[*_~#`>]", "", t)
    # Link markdown [testo](url) -> testo
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    # Emoji e pittogrammi
    t = _EMOJI_RE.sub("", t)
    # Normalizza spazi multipli generati dalle sostituzioni
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


@mcp.tool()
def text_to_speech(
    text: str,
    format: str = "ogg",
    out_path: str | None = None,
) -> dict:
    """Synthesize spoken audio from text and write it to a file.

    Use this to produce a voice-message reply for a social platform.
    `format` is one of "ogg" (Opus/Ogg — WhatsApp/Telegram voice notes),
    "mp3", or "wav". If `out_path` is omitted a file is created under the
    server's output directory. Returns {"path", "mime_type", "format",
    "bytes"}; the client reads the file from that path and is responsible
    for deleting it after relaying.
    """
    fmt = (format or "ogg").strip().lower()
    if fmt not in _OUTPUT_FORMATS:
        raise ValueError(
            f"unknown format {format!r}; valid: {sorted(_OUTPUT_FORMATS)}"
        )
    if not text or not text.strip():
        raise ValueError("text is empty")

    text = _sanitize_for_tts(text)
    if not text:
        raise ValueError("text is empty after sanitization")

    pcm = _tts.synthesize(text)
    if not pcm:
        raise RuntimeError("TTS produced no audio (provider error — check logs)")

    _enc_args, mime, ext = _OUTPUT_FORMATS[fmt]
    if out_path is None:
        os.makedirs(_OUT_DIR, exist_ok=True)
        out_path = os.path.join(_OUT_DIR, f"tts-{uuid.uuid4().hex}{ext}")
    out_path = os.path.abspath(out_path)

    try:
        _encode_from_pcm(pcm, _TTS_RATE, fmt, out_path)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg encode failed: {stderr}") from exc

    size = os.path.getsize(out_path)
    log.info("text_to_speech: %d chars -> %s (%d bytes, %s)",
             len(text), out_path, size, fmt)
    return {"path": out_path, "mime_type": mime, "format": fmt, "bytes": size}


def _post_multipart(
    url: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    timeout: float = 30.0,
) -> dict:
    """POST a multipart/form-data request and return the parsed JSON body.

    Kept on `urllib.request` to match the rest of the codebase's "minimal
    deps on ARM" preference (the bridge uses urllib for gateway/TTS HTTP).
    `files` values are (filename, bytes, mime_type) tuples.
    """
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        parts.append(str(value).encode("utf-8"))
        parts.append(b"\r\n")
    for name, (filename, content, mime) in files.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_chat_id(chat_id: str | None) -> str:
    """Pick the per-call `chat_id` override, else fall back to config."""
    cid = (chat_id or "").strip() or _TG_CHAT_ID
    if not cid:
        raise ValueError(
            "chat_id not provided and telegram.chat_id is empty in "
            "voice-bridge.json — set one or pass chat_id explicitly"
        )
    return cid


def _upload_voice_to_telegram(
    audio_bytes: bytes,
    filename: str,
    chat_id: str,
    caption: str | None,
    *,
    log_tag: str,
) -> dict:
    """POST `sendVoice` to the Telegram Bot API and shape the response.

    Shared by `send_voice_telegram` (uploads an existing file) and
    `say_to_telegram` (uploads freshly-synthesized PCM-turned-OGG bytes),
    so the multipart + error handling + result shape stay in one place.
    """
    if not _TG_TOKEN:
        raise RuntimeError(
            "telegram_bot_token not set in voice-bridge.secrets.json"
        )

    fields = {"chat_id": str(chat_id)}
    if caption:
        fields["caption"] = caption
    files = {"voice": (filename, audio_bytes, "audio/ogg")}

    url = f"https://api.telegram.org/bot{_TG_TOKEN}/sendVoice"
    try:
        payload = _post_multipart(url, fields, files)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"Telegram API HTTP {exc.code}: {err_body}"
        ) from exc

    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API error: {payload}")
    result = payload.get("result") or {}
    log.info(
        "%s: chat=%s file=%s (%d bytes) -> message_id=%s",
        log_tag, chat_id, filename, len(audio_bytes), result.get("message_id"),
    )
    return {
        "ok": True,
        "message_id": result.get("message_id"),
        "chat_id": (result.get("chat") or {}).get("id"),
        "date": result.get("date"),
    }


@mcp.tool()
def send_voice_telegram(
    audio_path: str,
    chat_id: str | None = None,
    caption: str | None = None,
) -> dict:
    """Deliver an existing voice-note (.ogg Opus) to a Telegram chat.

    Use this when you already have an OGG file on disk (e.g. one returned
    by `text_to_speech`) and just want to send it. For the common case of
    "synthesize then send", call `say_to_telegram` instead — it does both
    in one MCP roundtrip and cleans up the temp file.

    `chat_id` is optional: if omitted, falls back to `telegram.chat_id`
    from voice-bridge.json (the bridge's bound recipient). Pass it
    explicitly only to override that default. `caption` is optional plain
    text. Returns `{ok, message_id, chat_id, date}` on success; raises
    with the Telegram error body on failure.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"audio_path not found: {audio_path}")
    cid = _resolve_chat_id(chat_id)
    with open(audio_path, "rb") as f:
        data = f.read()
    filename = os.path.basename(audio_path) or "voice.ogg"
    return _upload_voice_to_telegram(
        data, filename, cid, caption, log_tag="send_voice_telegram",
    )


@mcp.tool()
def say_to_telegram(
    text: str,
    chat_id: str | None = None,
    caption: str | None = None,
) -> dict:
    """Synthesize `text` and deliver it as a Telegram voice-note in one shot.

    Equivalent to `text_to_speech(text, format="ogg")` followed by
    `send_voice_telegram(path, chat_id, caption)`, but a single MCP call
    and the temp file is deleted after upload (success OR failure), so no
    path ever leaks back to the client. This is the tool to use when
    relaying a spoken reply on Telegram — the two-step variants exist for
    when you need to inspect or re-use the audio.

    `chat_id` is optional: defaults to `telegram.chat_id` from
    voice-bridge.json. Returns `{ok, message_id, chat_id, date}` on
    success.
    """
    if not _TG_TOKEN:
        raise RuntimeError(
            "telegram_bot_token not set in voice-bridge.secrets.json"
        )
    if not text or not text.strip():
        raise ValueError("text is empty")
    cid = _resolve_chat_id(chat_id)

    text = _sanitize_for_tts(text)
    if not text:
        raise ValueError("text is empty after sanitization")

    pcm = _tts.synthesize(text)
    if not pcm:
        raise RuntimeError("TTS produced no audio (provider error — check logs)")

    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, f"tts-{uuid.uuid4().hex}.ogg")
    try:
        _encode_from_pcm(pcm, _TTS_RATE, "ogg", out_path)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg encode failed: {stderr}") from exc

    try:
        with open(out_path, "rb") as f:
            data = f.read()
        return _upload_voice_to_telegram(
            data,
            os.path.basename(out_path),
            cid,
            caption,
            log_tag=f"say_to_telegram[text={len(text)}c]",
        )
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def _play_pcm_blocking(pcm: bytes) -> float:
    """Play raw S16LE mono PCM on the configured ALSA output device.

    Reuses the bridge's `_aplay_popen` + `_drain_aplay`, so the wire format
    (S16_LE mono @ `tts_sample_rate`) and the no-clip drain behaviour match
    the bridge exactly. Network TTS feeds the whole buffer faster than
    realtime; the `stdin.write` blocks until aplay has consumed it (i.e. for
    the duration of playback), then `_drain_aplay` plays out the buffered
    tail without chopping it. Returns the audio duration in seconds.
    """
    proc = _bridge._aplay_popen(_OUT_DEVICE, _TTS_RATE)
    try:
        proc.stdin.write(pcm)
        proc.stdin.flush()
    except BrokenPipeError:
        # aplay died early (e.g. ALSA device busy/unavailable). Fall through
        # to drain/reap so we can surface its exit status below.
        pass
    _bridge._drain_aplay(proc, _TTS_RATE)
    if proc.returncode not in (0, None):
        raise RuntimeError(
            f"aplay exited with status {proc.returncode} "
            f"(device={_OUT_DEVICE!r}) — is the ALSA output available?"
        )
    return len(pcm) / (_TTS_RATE * 2)


@mcp.tool()
def say_to_speaker(text: str) -> dict:
    """Synthesize `text` and play it aloud on the bridge's speaker in one shot.

    The speaker-side analogue of `say_to_telegram`: synthesize -> play via
    aplay on the configured ALSA output device (`output_device` in
    voice-bridge.json, e.g. `voice_out`). Nothing is written to disk.
    Playback goes through sw_dmix, so it mixes with any reply the bridge is
    already speaking, and the device's mic-mute state does NOT gate output
    (mute only silences capture). Blocks until playback finishes. Returns
    `{ok, chars, seconds}`.
    """
    if not text or not text.strip():
        raise ValueError("text is empty")

    text = _sanitize_for_tts(text)
    if not text:
        raise ValueError("text is empty after sanitization")

    pcm = _tts.synthesize(text)
    if not pcm:
        raise RuntimeError("TTS produced no audio (provider error — check logs)")

    seconds = _play_pcm_blocking(pcm)
    log.info("say_to_speaker: %d chars -> %.1fs on %s",
             len(text), seconds, _OUT_DEVICE)
    return {"ok": True, "chars": len(text), "seconds": round(seconds, 2)}


if __name__ == "__main__":
    # Standalone mode: load our own config + providers, then serve in the
    # foreground. When embedded in the bridge process, the bridge calls
    # configure(...) + serve_background() instead (see voice-bridge.py).
    configure()
    serve_blocking()
