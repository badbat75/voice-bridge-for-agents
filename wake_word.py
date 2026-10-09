"""Wake-word activation: RMS-gated Whistle keyword spotting + cached spoken acks.

Selected with `"activation": "wake_word"` in voice-bridge.json (the default,
`"button"`, keeps the Jabra-HID-press-to-unmute flow). While the bridge is
idle the recorder keeps the mic open and hands chunks to `VoiceBridge._wake_loop`,
which runs them through:

  - `SpeechGate` — pure-Python energy gate. Whistle never sees idle audio: a
    segment opens when a chunk's true RMS crosses `rms_threshold`, closes after
    `hang_ms` under it (or at `max_segment_ms`), and is dropped unless at least
    `min_loud_ms` of it was above threshold. Continuous streaming through
    Whistle cost ~230% CPU on the Pi 3B+ and fell behind real time; gated, it's
    ~3–6%. It also keeps Whistle away from room noise, which it otherwise
    hallucinates into "Thank you." / "Vielen Dank.".
  - `WakeDetector` — one-shot Whistle transcription of the gated segment with
    `language` forced and the wake phrases passed as keyword bias (auto-detect
    once heard "Hey, Binary" as Spanish; forced language + bias caught every
    take in testing). A hit is a normalized substring match.

`AckBank` holds short spoken clips picked at random — `acks` played on a hit,
`sleep_acks` played when the bridge dozes off — each phrase synthesized with
every TTS provider in `ack_providers` and cached on disk as raw PCM so later
boots make no API calls. The cache is keyed to the voice: one folder per
`<kind>/<provider>-<voice>-<model>`, and a clip's file name hashes the text
with the rate, language and voice settings. Changing any of them re-downloads
the affected clips and prunes the ones no voice uses any more.

Unlike `vad_rms_threshold`, `rms_threshold` here is TRUE RMS of S16 samples
(speech at the Jabra measured ~14k–27k peak, room noise <3k).
"""

from __future__ import annotations

import array
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
from collections import deque

log = logging.getLogger("voice-bridge")

DEFAULTS = {
    "phrases": ["hey binary"],
    "language": "en",
    "rms_threshold": 5000,
    "min_loud_ms": 300,
    "hang_ms": 300,
    "pre_roll_ms": 300,
    "max_segment_ms": 3000,
    "acks": [],
    "sleep_acks": [],
    "ack_providers": ["elevenlabs", "deepgram"],
    "ack_cache_dir": "~/.cache/voice-bridge/acks",
}


def wake_config(cfg: dict) -> dict:
    """`cfg["wake_word"]` merged over DEFAULTS, phrases lower-cased."""
    out = {**DEFAULTS, **(cfg.get("wake_word") or {})}
    out["phrases"] = [normalize(p) for p in out["phrases"] if normalize(p)]
    return out


def normalize(text: str) -> str:
    """Lower-case, letters/digits/spaces only, single-spaced."""
    return " ".join(re.sub(r"[^\w\s]|_", " ", text.lower()).split())


def matches(text: str, phrases: list[str]) -> bool:
    heard = f" {normalize(text)} "
    return any(f" {p} " in heard for p in phrases)


def _rms(pcm: bytes) -> float:
    samples = array.array("h", pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class SpeechGate:
    """Cuts loud segments out of a stream of S16LE mono chunks."""

    def __init__(self, sample_rate: int, chunk_size: int, wcfg: dict) -> None:
        chunk_ms = chunk_size / sample_rate * 1000.0
        self.threshold = float(wcfg["rms_threshold"])
        self.hang = max(1, round(wcfg["hang_ms"] / chunk_ms))
        self.min_loud = max(1, round(wcfg["min_loud_ms"] / chunk_ms))
        self.max_chunks = max(1, round(wcfg["max_segment_ms"] / chunk_ms))
        self._pre: deque[bytes] = deque(maxlen=max(0, round(wcfg["pre_roll_ms"] / chunk_ms)))
        self.reset()

    def reset(self) -> None:
        self._seg: list[bytes] = []
        self._loud = 0
        self._quiet = 0
        self._pre.clear()

    def feed(self, chunk: bytes) -> bytes | None:
        """Returns a finished segment's PCM (pre-roll included), else None."""
        loud = _rms(chunk) > self.threshold
        if not self._seg:
            if loud:
                self._seg = [*self._pre, chunk]
                self._loud, self._quiet = 1, 0
            else:
                self._pre.append(chunk)
            return None
        self._seg.append(chunk)
        if loud:
            self._loud += 1
            self._quiet = 0
        else:
            self._quiet += 1
        if self._quiet < self.hang and len(self._seg) < self.max_chunks:
            return None
        seg, enough = b"".join(self._seg), self._loud >= self.min_loud
        self.reset()
        return seg if enough else None


# Whistle's languages (needle.agent.whistle.LANGUAGES), copied so config
# validation doesn't have to load the engine.
WHISTLE_LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")


def validate_language(where: str, language) -> None:
    """None/"" means auto-detect; anything else must be a Whistle language."""
    if language and language not in WHISTLE_LANGUAGES:
        raise ValueError(f"voice-bridge.json: {where} {language!r} is not a Whistle "
                         f"language; valid: {WHISTLE_LANGUAGES} (or null to auto-detect)")


# One Whistle per process (cactus-needle keeps a single loaded model and is
# not thread-safe), shared by the wake loop and the STT comparison.
_whistle = None
_whistle_lock = threading.Lock()
WHISTLE_RATE = 16000
_MAX_PASS = 30 * WHISTLE_RATE  # Whistle takes at most 30 s per pass


def load_whistle() -> None:
    global _whistle
    with _whistle_lock:
        if _whistle is None:
            # cactus-needle reports usage to its vendor unless told not to.
            os.environ.setdefault("NEEDLE_TELEMETRY", "0")
            from needle import Whistle
            _whistle = Whistle()


def whistle_transcribe(pcm: bytes, language: str | None = None, keywords=None) -> str:
    """Transcribe 16 kHz S16LE mono PCM; longer than 30 s is split into passes."""
    load_whistle()
    samples = array.array("f", (s / 32768.0 for s in array.array("h", pcm)))
    texts = []
    with _whistle_lock:
        for i in range(0, len(samples), _MAX_PASS):
            result = _whistle.transcribe(samples[i:i + _MAX_PASS], language=language, keywords=keywords)
            texts.append((result.get("text") or "").strip())
    return " ".join(t for t in texts if t)


class WakeDetector:
    """Whistle keyword spotting on gated segments. Loads lazily on first use."""

    def __init__(self, wcfg: dict) -> None:
        self.phrases = wcfg["phrases"]
        self.language = wcfg.get("language") or None

    def load(self) -> None:
        load_whistle()

    def transcribe(self, pcm: bytes) -> str:
        return whistle_transcribe(pcm, self.language, self.phrases)

    def heard(self, text: str) -> bool:
        return matches(text, self.phrases)


RETRY_MIN_S, RETRY_MAX_S = 30.0, 600.0


class AckBank:
    """Short spoken clips: every phrase × every ack provider, disk-cached per voice."""

    def __init__(self, cfg: dict, phrases, providers, cache_dir: str, build_tts, kind: str = "wake") -> None:
        self.cfg = cfg
        self.kind = kind
        self.phrases = list(phrases)
        self.providers = list(providers)
        self.cache_dir = os.path.expanduser(cache_dir)
        self._build_tts = build_tts
        self._clips: list[tuple[str, str, bytes]] = []
        self._lock = threading.Lock()

    def _voice_dir(self, provider: str) -> str:
        """Folder for one voice: changing voice/model starts an empty folder."""
        if provider == "elevenlabs":
            name = f"elevenlabs-{self.cfg.get('elevenlabs_voice')}-{self.cfg.get('elevenlabs_model')}"
        else:
            name = f"{provider}-{self.cfg.get('deepgram_tts_model') or 'default'}"
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    def _render_key(self, provider: str) -> str:
        """Everything besides the voice id that changes how a clip sounds."""
        extra = {"rate": int(self.cfg["tts_sample_rate"])}
        if provider == "elevenlabs":
            extra.update(language=self.cfg.get("elevenlabs_language"),
                         settings=self.cfg.get("elevenlabs_voice_settings"),
                         normalization=self.cfg.get("elevenlabs_text_normalization"))
        return json.dumps(extra, sort_keys=True, default=str)

    def _path(self, provider: str, text: str) -> str:
        digest = hashlib.sha1(f"{self._render_key(provider)}|{text}".encode()).hexdigest()
        return os.path.join(self.cache_dir, self.kind, self._voice_dir(provider), digest + ".pcm")

    def _available(self, provider: str) -> bool:
        if provider == "elevenlabs":
            return bool(self.cfg.get("elevenlabs_key") and self.cfg.get("elevenlabs_voice")
                        and self.cfg.get("elevenlabs_model"))
        if provider == "deepgram":
            return bool(self.cfg.get("deepgram_key"))
        return False

    def _prune(self, keep: set[str]) -> None:
        """Delete clips of voices/phrases no longer configured (ours only)."""
        root = os.path.join(self.cache_dir, self.kind)
        removed = 0
        for dirpath, _dirs, files in os.walk(root, topdown=False):
            for name in files:
                path = os.path.join(dirpath, name)
                if name.endswith((".pcm", ".tmp")) and path not in keep:
                    with contextlib.suppress(OSError):
                        os.remove(path)
                        removed += 1
            if dirpath != root:
                with contextlib.suppress(OSError):
                    os.rmdir(dirpath)  # only succeeds when empty
        # Clips from the pre-per-voice layout sat directly in cache_dir.
        with contextlib.suppress(OSError):
            for name in os.listdir(self.cache_dir):
                if name.endswith((".pcm", ".tmp")):
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(self.cache_dir, name))
                        removed += 1
        if removed:
            log.info("Wake acks (%s): pruned %d stale clips", self.kind, removed)

    def _load_or_synthesize(self, provider: str, text: str, path: str, tts_box: list) -> bytes | None:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                return f.read()
        try:
            if not tts_box:
                tts_box.append(self._build_tts(provider))
            pcm = tts_box[0].synthesize(text)
        except Exception as exc:
            log.warning("Wake acks (%s): %s failed on %r: %s", self.kind, provider, text, exc)
            return None
        if pcm:
            # Write-then-rename: a crash mid-write never leaves a truncated
            # clip that later boots would trust as cached.
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "wb") as f:
                f.write(pcm)
            os.replace(tmp, path)
        return pcm or None

    def prepare(self, stop: "threading.Event | None" = None) -> None:
        """Load cached clips; synthesize (and cache) the missing ones.

        Clips that fail (network down, API error) are retried in the
        background with backoff until the whole set is ready or `stop` is
        set; wakes meanwhile use whatever is ready, or a beep if nothing is.
        """
        stop = stop or threading.Event()
        wanted = []
        for provider in self.providers:
            if not self._available(provider):
                log.warning("Wake acks (%s): %s not configured, skipping", self.kind, provider)
                continue
            wanted += [(provider, text, self._path(provider, text)) for text in self.phrases]
        self._prune({path for _p, _t, path in wanted})
        delay = RETRY_MIN_S
        while True:
            tts_boxes: dict[str, list] = {}
            missing = []
            for provider, text, path in wanted:
                pcm = self._load_or_synthesize(provider, text, path, tts_boxes.setdefault(provider, []))
                if pcm:
                    with self._lock:
                        self._clips.append((provider, text, pcm))
                else:
                    missing.append((provider, text, path))
            wanted = missing
            log.info("Wake acks (%s): %d clips ready%s", self.kind, len(self._clips),
                     f", {len(wanted)} missing — retry in {delay:.0f}s" if wanted else "")
            if not wanted or stop.wait(delay):
                return
            delay = min(delay * 2, RETRY_MAX_S)

    def pick(self) -> tuple[str, str, bytes] | None:
        with self._lock:
            return random.choice(self._clips) if self._clips else None
