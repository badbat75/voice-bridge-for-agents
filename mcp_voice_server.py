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
it synthesizes text here and hands the PCM to the running bridge over its
local speaker socket (`speaker_socket.py`), so it plays through the bridge's
player exactly like a reply.

Design notes:
- **Its own process, started on demand.** `voice-bridge-mcp.socket` makes
  systemd listen on the MCP port; the first connection starts
  `voice-bridge-mcp.service` (this module's `main()`), which loads the same
  `voice-bridge.json` and builds its own providers. After
  `mcp_server.idle_exit_s` (default 600) with no request in flight it exits,
  and systemd starts it again on the next call (~4–5 s cold start on the Pi
  3B+, well inside zeroclaw's 60 s tool timeout). The bridge no longer carries
  mcp/uvicorn/pydantic (~45 MB) all day. The server is `stateless_http`, so a
  client's session survives the restarts.
- **All container <-> PCM conversion goes through ffmpeg, uniformly.** STT
  decodes whatever the client sent to S16LE mono PCM, then calls the
  provider's `transcribe(pcm, rate)`. TTS calls `synthesize(text)` (which
  returns S16LE PCM at `tts_sample_rate`) and pipes that into ffmpeg to
  produce the requested container. ffmpeg (with libmp3lame + libopus) is the
  single, tier-independent path for emitting mp3/ogg/wav.
- **Transport is streamable-http** so an MCP client (e.g. zeroclaw) connects
  by URL. Host/port (and the output directory for synthesized files) come
  from the `mcp_server` block in `voice-bridge.json`; localhost defaults
  otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

from mcp.server.fastmcp import FastMCP

# Logging is configured by the host process (the bridge's main()).
log = logging.getLogger("mcp_voice_server")

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


# Runtime state, populated by `configure()`. Kept module-global because the
# `@mcp.tool()` functions read them at call time. `main()` calls
# `configure(cfg, stt=..., tts=..., bridge=...)` with this process's
# providers and a `SpeakerClient` (anything with the bridge's `play_pcm`).
_bridge = None
_cfg: dict = {}
_stt = None
_tts = None
_TTS_RATE = 0
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
# below need it). Stateless: every request stands alone, so the idle exit
# never strands a client's session id.
mcp = FastMCP("voice-bridge-voice", stateless_http=True)


def configure(cfg: dict, *, stt, tts, bridge) -> "FastMCP":
    """Resolve the config + providers into the module globals the tools
    read. `bridge` is anything with `play_pcm(pcm, text=...)` — in
    production a `speaker_socket.SpeakerClient`. Returns the `mcp` app."""
    global _bridge, _cfg, _stt, _tts, _TTS_RATE
    global _HOST, _PORT, _TG_TOKEN, _TG_CHAT_ID, _OUT_DIR

    # The providers are safe under FastMCP's threadpool-dispatched tool
    # calls: ElevenLabsVoice reuses one thread-safe httpx-backed SDK client,
    # DeepgramVoice builds a fresh async client per call.
    _bridge = bridge
    _cfg = cfg
    _stt = stt
    _tts = tts
    _TTS_RATE = int(_cfg["tts_sample_rate"])

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


class _IdleTracker:
    """ASGI wrapper counting HTTP requests in flight and the time the last
    one ended, for the idle exit."""

    def __init__(self, app) -> None:
        self.app = app
        self.active = 0
        self.last = time.monotonic()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        self.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.active -= 1
            self.last = time.monotonic()


def _exit_when_idle(server, tracker: "_IdleTracker", idle_s: float) -> None:
    while not server.should_exit:
        time.sleep(min(10.0, idle_s))
        if tracker.active == 0 and time.monotonic() - tracker.last >= idle_s:
            log.info("idle for %.0fs: exiting (systemd restarts on the next call)", idle_s)
            server.should_exit = True


def _systemd_socket() -> "socket.socket | None":
    """The listening socket systemd passed (socket activation), if any."""
    if os.environ.get("LISTEN_PID") != str(os.getpid()) or os.environ.get("LISTEN_FDS") != "1":
        return None
    return socket.socket(fileno=3)


def main() -> None:
    """Standalone entry point (`voice-bridge-mcp.service`)."""
    import uvicorn

    import speaker_socket
    from bridge_config import _build_voice_provider, load_config

    logging.basicConfig(level=logging.INFO,
                        format="[voice-bridge-mcp] %(levelname)s %(message)s")
    cfg = load_config()
    configure(cfg, stt=_build_voice_provider("stt", cfg), tts=_build_voice_provider("tts", cfg),
              bridge=speaker_socket.SpeakerClient(cfg["speaker_socket"]))
    tracker = _IdleTracker(mcp.streamable_http_app())
    server = uvicorn.Server(uvicorn.Config(tracker, host=_HOST, port=_PORT, log_level="warning"))
    idle_s = float((cfg.get("mcp_server") or {}).get("idle_exit_s", 600))
    if idle_s > 0:
        threading.Thread(target=_exit_when_idle, args=(server, tracker, idle_s),
                         name="mcp-idle", daemon=True).start()
    sock = _systemd_socket()
    log.info("serving %s (idle exit %s)",
             "on the systemd socket" if sock else f"http://{_HOST}:{_PORT}",
             f"{idle_s:.0f}s" if idle_s > 0 else "off")
    server.run(sockets=[sock] if sock else None)


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


def _encode_from_pcm(pcm: bytes, src_rate: int, fmt: str, out_path: str | None = None) -> bytes:
    """Encode raw S16LE mono PCM into `fmt`'s container.

    Written to `out_path` when given; otherwise ffmpeg writes to its stdout
    and the encoded bytes are returned (no temp file). Raises RuntimeError
    with ffmpeg's stderr on failure."""
    enc_args, _mime, _ext = _OUTPUT_FORMATS[fmt]
    out = ["-f", fmt, "pipe:1"] if out_path is None else [out_path]
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "s16le", "-ar", str(src_rate), "-ac", "1", "-i", "pipe:0",
                *enc_args,
                *out,
            ],
            input=pcm,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg encode failed: {stderr}") from exc
    return proc.stdout


def _synthesize(text: str) -> "tuple[str, bytes]":
    """Validate + sanitize `text` and synthesize it: (spoken text, PCM).
    Shared by every tool that speaks; raises on empty text or TTS failure."""
    if not text or not text.strip():
        raise ValueError("text is empty")
    text = _sanitize_for_tts(text)
    if not text:
        raise ValueError("text is empty after sanitization")
    pcm = _tts.synthesize(text)
    if not pcm:
        raise RuntimeError("TTS produced no audio (provider error — check logs)")
    return text, pcm


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
    text, pcm = _synthesize(text)

    _enc_args, mime, ext = _OUTPUT_FORMATS[fmt]
    if out_path is None:
        os.makedirs(_OUT_DIR, exist_ok=True)
        out_path = os.path.join(_OUT_DIR, f"tts-{uuid.uuid4().hex}{ext}")
    out_path = os.path.abspath(out_path)

    _encode_from_pcm(pcm, _TTS_RATE, fmt, out_path)

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
    and the OGG is encoded in memory, so no file is ever written. This is the tool to use when
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
    cid = _resolve_chat_id(chat_id)
    text, pcm = _synthesize(text)
    data = _encode_from_pcm(pcm, _TTS_RATE, "ogg")
    return _upload_voice_to_telegram(
        data,
        f"tts-{uuid.uuid4().hex}.ogg",
        cid,
        caption,
        log_tag=f"say_to_telegram[text={len(text)}c]",
    )


@mcp.tool()
def say_to_speaker(text: str) -> dict:
    """Synthesize `text` and play it aloud on the bridge's speaker in one shot.

    The speaker-side analogue of `say_to_telegram`: synthesize -> hand the
    PCM to the bridge's player via `play_pcm()`, so it behaves exactly like a
    normal spoken reply. The bridge unmutes (mic open, LED off) and ducks
    deezer-connect for exactly the playback window; after the speech the
    usual idle timer re-mutes. The audio serializes
    behind any reply already playing rather than overlapping it. Blocks until
    playback finishes. Returns `{ok, chars, seconds}`.
    """
    text, pcm = _synthesize(text)
    seconds = _bridge.play_pcm(pcm, text=text)
    log.info("say_to_speaker: %d chars -> %.1fs", len(text), seconds)
    return {"ok": True, "chars": len(text), "seconds": round(seconds, 2)}


if __name__ == "__main__":
    main()
