"""Utterance endpointing, one mic chunk at a time.

`Endpointer` is the pure state machine behind `VoiceBridge._endpointer_loop`:
no threads, no queues, no bridge state. The loop feeds it chunks and acts on
what `feed()` returns; generation changes, ducking and auto-idle stay in the
loop, which reads `in_speech` and `silence` to drive them.

Per chunk the energy metric is `sum(s²) / sqrt(N)` — NOT true RMS. It is the
metric the legacy `record_until_silence` used, kept so `vad_rms_threshold`
calibrations carry over (see AGENTS.md → *Architecture notes*).
"""

from __future__ import annotations

import array
import logging
import math
from collections import deque

from audio_io import _voiced_ms

log = logging.getLogger("voice-bridge")

# `feed()` / `force_commit()` results: (kind, pcm). The kinds double as the
# reason logged when the utterance's ducking hold is released.
COMMITTED = "speech committed"
BURST = "burst dropped"
NO_VOICE = "no voice"
TOO_LONG = "utterance too long"
DISCARDED = "speech discarded"


class Endpointer:
    """Turns a stream of mic chunks into committed utterances.

    - speech-then-silence ≥ `silence_timeout_ms` → `(COMMITTED, pcm)`, unless
      it was a burst shorter than `min_speech_ms` (`BURST`) or webrtcvad found
      less than `speech_filter.min_voiced_ms` of voice in it (`NO_VOICE`);
    - an utterance reaching `max_utterance_ms` → `(TOO_LONG, None)`.

    `silence` counts consecutive below-threshold chunks. It is NOT reset by a
    dropped burst, so noise never postpones auto-idle; a commit resets it, so
    the idle window starts at the commit."""

    def __init__(self, cfg: dict, speech_vad=None) -> None:
        self.sample_rate = sr = int(cfg["sample_rate"])
        chunk = int(cfg["chunk_size"])
        self.chunk_ms = chunk_ms = chunk / sr * 1000.0
        self._threshold = float(cfg["vad_rms_threshold"])
        self._silence_timeout_ms = cfg["silence_timeout_ms"]
        self._commit_chunks = max(1, int(cfg["silence_timeout_ms"] / chunk_ms))
        self._min_speech_ms = cfg.get("min_speech_ms", 0)
        self._min_speech_chunks = round(self._min_speech_ms / chunk_ms)
        self._keep_chunks = max(0, int(cfg.get("silence_keep_ms", 500) / chunk_ms))
        pre_chunks = max(0, int(cfg.get("pre_speech_keep_ms", 100) / chunk_ms))
        self._prebuf: "deque[bytes] | None" = deque(maxlen=pre_chunks) if pre_chunks else None
        self._vad = speech_vad
        self._min_voiced_ms = int((cfg.get("speech_filter") or {}).get("min_voiced_ms", 0))
        self._max_utt_ms = int(cfg.get("max_utterance_ms", 0))
        self._max_utt_chunks = int(self._max_utt_ms / chunk_ms) if self._max_utt_ms > 0 else 0

        self.silence = 0
        self.in_speech = False
        self._buf: list[bytes] = []
        self._speech_chunks = 0
        self._levels: list[float] = []  # per-chunk energy of the current utterance

    # -- control ---------------------------------------------------------
    def reset(self) -> None:
        """Forget everything (a resume bumped the generation)."""
        self._end()
        self.silence = 0
        if self._prebuf is not None:
            self._prebuf.clear()

    def force_commit(self) -> "tuple[str, bytes] | None":
        """HID press: commit the utterance in progress now, without waiting
        for the pause and without the voice check (the user asked for it)."""
        if not (self.in_speech and self._buf):
            return None
        pcm, kept, _trim = self._speech_pcm()
        log.info("Endpointer: force-commit on HID press (%d chunks ≈ %.2fs)",
                 kept, kept * self.chunk_ms / 1000.0)
        self._end()
        self.silence = 0
        return COMMITTED, pcm

    # -- per chunk -------------------------------------------------------
    def feed(self, data: bytes) -> "tuple[str, bytes | None] | None":
        samples = array.array("h", data)
        if not samples:
            return None
        rms = math.sumprod(samples, samples) / len(samples) ** 0.5

        if rms >= self._threshold:
            return self._loud(data, rms)

        if self.in_speech:
            self._buf.append(data)
            if self.silence < self._commit_chunks:
                self._levels.append(rms)
            if self.silence == 0:
                log.info("Endpointer: silence onset (rms=%.0f < %g, need %d chunks ≈ %dms to commit)",
                         rms, self._threshold, self._commit_chunks, self._silence_timeout_ms)
        elif self._prebuf is not None:
            self._prebuf.append(data)  # rolling pre-roll for the next utterance
        self.silence += 1

        if self.in_speech and self.silence >= self._commit_chunks:
            return self._at_pause()
        return None

    def _loud(self, data: bytes, rms: float) -> "tuple[str, None] | None":
        if not self.in_speech:
            self.in_speech = True
            self._speech_chunks = 0
            self._levels = []
            if self._prebuf:
                self._buf.extend(self._prebuf)
                self._prebuf.clear()
            log.info("Endpointer: sound detected (rms=%.0f ≥ %g)", rms, self._threshold)
        self._buf.append(data)
        self._levels.append(rms)
        self.silence = 0
        self._speech_chunks += 1
        if self._speech_chunks % 32 == 0:
            log.info("Endpointer: still in_speech tick=%d rms=%.0f", self._speech_chunks, rms)
        if self._max_utt_chunks and len(self._buf) >= self._max_utt_chunks:
            # Nobody talks to an assistant this long in one go: it's people
            # talking among themselves (seen: 19–34 s of family chat sent as
            # one turn). Dropped; the caller goes idle.
            log.info("Endpointer: utterance over max_utterance_ms=%d — side "
                     "conversation, dropped; going idle", self._max_utt_ms)
            self._end()
            return TOO_LONG, None
        return None

    def _at_pause(self) -> "tuple[str, bytes | None]":
        if self._speech_chunks < self._min_speech_chunks:
            # Click, bump, echo blip: dropped before STT. `silence` keeps
            # running, so noise never postpones auto-idle.
            log.info("Endpointer: dropped %d-chunk burst (< min_speech_ms=%d) levels %s",
                     self._speech_chunks, self._min_speech_ms, self._level_stats())
            self._end()
            return BURST, None
        pcm, kept, trim = self._speech_pcm()
        if self._too_little_voice(pcm):
            # Loud, long enough, but not a voice (keyboard, TV hum, a door).
            self._end()
            return NO_VOICE, None
        log.info("Endpointer: commit (%d chunks ≈ %.2fs, kept %d trailing silence, "
                 "trimmed %d) levels %s", kept, kept * self.chunk_ms / 1000.0,
                 min(self._keep_chunks, self.silence), trim, self._level_stats())
        self._end()
        # A commit is a "transaction" boundary: the idle window counts from
        # here, not from the trailing silence that detected end-of-speech.
        self.silence = 0
        return COMMITTED, pcm

    # -- helpers ---------------------------------------------------------
    def _end(self) -> None:
        self._buf = []
        self.in_speech = False

    def _speech_pcm(self) -> "tuple[bytes, int, int]":
        """The utterance as PCM, keeping only `silence_keep_ms` of the
        trailing silence (the rest of the detection window is trimmed so STT
        doesn't get a full `silence_timeout_ms` tail). Returns (pcm, chunks
        kept, chunks trimmed)."""
        trim = max(0, min(self.silence - self._keep_chunks, len(self._buf)))
        kept = self._buf[:-trim] if trim > 0 else self._buf
        return b"".join(kept), len(kept), trim

    def _too_little_voice(self, pcm: bytes) -> bool:
        """webrtcvad check at commit. Logs the voiced ms of every commit so
        `speech_filter.min_voiced_ms` can be tuned from the journal."""
        if self._vad is None:
            return False
        voiced = _voiced_ms(pcm, self.sample_rate, self._vad)
        if voiced is None:
            return False
        if voiced < self._min_voiced_ms:
            log.info("Endpointer: dropped %.2fs utterance, only %d ms voiced "
                     "(< speech_filter.min_voiced_ms=%d) levels %s",
                     len(pcm) / (2 * self.sample_rate), voiced, self._min_voiced_ms,
                     self._level_stats())
            return True
        log.info("Endpointer: %d ms voiced", voiced)
        return False

    def _level_stats(self) -> str:
        if not self._levels:
            return "n/a"
        lv = sorted(self._levels)
        pick = lambda q: lv[min(len(lv) - 1, int(q * len(lv)))]  # noqa: E731
        return f"p50={pick(0.5):.3g} p90={pick(0.9):.3g} max={lv[-1]:.3g}"
