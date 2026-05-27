"""MCP server exposing the voice-bridge STT/TTS providers as tools.

Lets an MCP client (e.g. zeroclaw) transcribe inbound voice messages and
synthesize spoken replies when relaying audio to/from a social platform.
The two providers (`ElevenLabsVoice`, `DeepgramVoice`) are reused verbatim
from the bridge; this module is a thin wrapper around them plus an `ffmpeg`
boundary that handles every container format (ogg/opus, mp3, wav) so the
providers can stay PCM-centric, exactly as the bridge uses them.

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

import importlib.util
import logging
import os
import subprocess
import tempfile
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


_bridge = _load_bridge_module()
_cfg = _bridge.load_config()

# Build both providers up front. Each provider class constructs its own SDK
# client per call, so a single shared instance is safe to use concurrently
# across FastMCP's threadpool-dispatched tool calls.
_stt = _bridge._build_voice_provider("stt", _cfg)
_tts = _bridge._build_voice_provider("tts", _cfg)
_TTS_RATE = int(_cfg["tts_sample_rate"])

_mcp_cfg = _cfg.get("mcp_server") or {}
_HOST = str(_mcp_cfg.get("host", "127.0.0.1"))
_PORT = int(_mcp_cfg.get("port", 9080))
# Where synthesized files land. Default to a per-run temp dir; the client is
# expected to read the returned path (it shares the Pi's filesystem) and is
# responsible for cleanup once the file has been relayed.
_OUT_DIR = os.path.abspath(
    _mcp_cfg.get("out_dir") or os.path.join(tempfile.gettempdir(), "voice-bridge-mcp")
)

log.info(
    "providers: stt=%s tts=%s | tts_rate=%d | out_dir=%s | http=%s:%d",
    _cfg["stt_provider"], _cfg["tts_provider"], _TTS_RATE, _OUT_DIR, _HOST, _PORT,
)

mcp = FastMCP("voice-bridge-voice", host=_HOST, port=_PORT)


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


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
