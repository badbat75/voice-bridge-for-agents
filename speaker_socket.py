"""Local speaker endpoint: lets another process play PCM through the bridge.

The MCP voice server (`mcp_voice_server.py`) runs as its own socket-activated
unit, so its `say_to_speaker` tool can't call `VoiceBridge.play_pcm()`
directly. The bridge serves it on a Unix socket instead (stdlib only, a few
hundred KB, no threads while idle beyond the accept loop):

    request:  one JSON line {"text": str, "bytes": N}, then N bytes of
              S16LE mono PCM at `tts_sample_rate`
    response: one JSON line {"ok": true, "seconds": float}
              or {"ok": false, "error": str}

The response is sent when playback has finished (`play_pcm` blocks), so the
tool keeps its "returns when spoken" contract. The socket lives in
`$XDG_RUNTIME_DIR/voice-bridge/` (user-private, tmpfs), mode 0600.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import socketserver
import tempfile
import threading

log = logging.getLogger("voice-bridge")

# Bigger than any sane announcement (~5 min at 24 kHz); a guard against a
# garbage length field, not a product limit.
_MAX_BYTES = 16 * 1024 * 1024


def default_path() -> str:
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return os.path.join(base, "voice-bridge", "speaker.sock")


def _read_exact(f, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-request")
        buf += chunk
    return bytes(buf)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            head = json.loads(self.rfile.readline(4096) or b"{}")
            n = int(head.get("bytes", 0))
            if not 0 < n <= _MAX_BYTES:
                raise ValueError(f"bad PCM length {n}")
            pcm = _read_exact(self.rfile, n)
            seconds = self.server.play_pcm(pcm, text=head.get("text") or None)
            reply = {"ok": True, "seconds": seconds}
        except Exception as exc:
            log.warning("speaker socket: request failed: %s", exc)
            reply = {"ok": False, "error": str(exc)}
        try:
            self.wfile.write(json.dumps(reply).encode() + b"\n")
        except OSError:
            pass


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def serve(path: str, play_pcm, poll_interval: float = 5.0) -> "_Server":
    """Serve `play_pcm(pcm, text=...)` on `path` in a daemon thread.

    `poll_interval` is how often the accept loop wakes to check for
    `shutdown()` (connections wake it at once); the bridge never calls
    `shutdown()`, so it is long."""
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    try:
        os.unlink(path)  # stale socket from a previous run
    except FileNotFoundError:
        pass
    old = os.umask(0o177)
    try:
        server = _Server(path, _Handler)
    finally:
        os.umask(old)
    server.play_pcm = play_pcm
    threading.Thread(target=server.serve_forever, args=(poll_interval,),
                     name="vb-speaker-sock", daemon=True).start()
    log.info("Speaker socket: %s", path)
    return server


class SpeakerClient:
    """`play_pcm()` with the bridge's signature, over the socket."""

    def __init__(self, path: str, timeout: float = 600.0) -> None:
        self.path = path
        self.timeout = timeout

    def play_pcm(self, pcm: bytes, *, text: str | None = None) -> float:
        head = json.dumps({"text": text or "", "bytes": len(pcm)}).encode() + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect(self.path)
                s.sendall(head)
                s.sendall(pcm)
                reply = json.loads(s.makefile("rb").readline() or b"{}")
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise RuntimeError(f"voice-bridge is not running ({self.path}: {exc})") from exc
        if not reply.get("ok"):
            raise RuntimeError(f"voice-bridge speaker error: {reply.get('error', 'no reply')}")
        return float(reply.get("seconds", 0.0))
