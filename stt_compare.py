"""Shadow STT comparison: run every utterance through Whistle as well.

Enabled with `"stt_compare": {"enabled": true}` in voice-bridge.json. The
configured remote STT stays authoritative — its text is what reaches the
gateway. After it returns, the worker hands the same PCM and the remote result
to `SttComparer.submit()`, which queues it for a single background thread: that
thread runs Whistle (local, on-device) and appends one TSV row to `log_path`
(default `data/stt-compare.tsv`). The turn never waits on Whistle: `submit()`
never blocks (a full queue drops the comparison), and the thread runs at
nice +10 so the recorder/endpointer/player always win the CPU.

Columns: timestamp, audio seconds, remote provider, remote seconds, remote text,
Whistle seconds, Whistle text. Tabs/newlines inside texts are flattened so
every utterance is exactly one line.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from datetime import datetime

import wake_word

log = logging.getLogger("voice-bridge")

MAX_PENDING = 8
HEADER = "timestamp\taudio_s\tstt_provider\tstt_s\tstt_text\twhistle_s\twhistle_text\n"


def _flat(text: str | None) -> str:
    return " ".join((text or "").split())


class SttComparer:
    def __init__(self, ccfg: dict, base_dir: str, provider: str) -> None:
        self.language = ccfg.get("language", "it") or None
        path = ccfg.get("log_path") or "data/stt-compare.tsv"
        self.path = path if os.path.isabs(path) else os.path.join(base_dir, path)
        self.provider = provider
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=MAX_PENDING)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="vb-stt-compare", daemon=True)
        self._thread.start()
        log.info("STT compare: %s vs Whistle (lang=%s) → %s", self.provider, self.language, self.path)

    def submit(self, pcm: bytes, sample_rate: int, stt_text: str | None, stt_seconds: float) -> None:
        if sample_rate != wake_word.WHISTLE_RATE:
            log.warning("STT compare: Whistle needs %d Hz, mic is %d Hz — skipped",
                        wake_word.WHISTLE_RATE, sample_rate)
            return
        try:
            self._q.put_nowait((datetime.now(), pcm, sample_rate, stt_text, stt_seconds))
        except queue.Full:
            log.warning("STT compare: %d comparisons pending, dropping this one", MAX_PENDING)

    def _loop(self) -> None:
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except OSError as exc:
            log.warning("STT compare: cannot lower thread priority: %s", exc)
        while True:
            stamp, pcm, sr, stt_text, stt_s = self._q.get()
            t0 = time.monotonic()
            try:
                whistle_text = wake_word.whistle_transcribe(pcm, self.language)
            except Exception as exc:
                whistle_text = f"[error: {exc}]"
            whistle_s = time.monotonic() - t0
            row = "\t".join((
                stamp.strftime("%Y-%m-%d %H:%M:%S"), f"{len(pcm) / (2 * sr):.2f}",
                self.provider, f"{stt_s:.2f}", _flat(stt_text),
                f"{whistle_s:.2f}", _flat(whistle_text),
            )) + "\n"
            try:
                new = not os.path.exists(self.path)
                with open(self.path, "a", encoding="utf-8") as f:
                    if new:
                        f.write(HEADER)
                    f.write(row)
            except OSError as exc:
                log.warning("STT compare: cannot write %s: %s", self.path, exc)
            log.info("STT compare (%.1fs audio): %s=%r (%.2fs) | whistle=%r (%.2fs)",
                     len(pcm) / (2 * sr), self.provider, _flat(stt_text), stt_s,
                     _flat(whistle_text), whistle_s)
