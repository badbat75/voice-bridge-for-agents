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
    take in testing). A hit is an exact word-boundary match on a phrase or
    alias, else a fuzzy match (`fuzzy_threshold`) on a phrase.

`AckBank` holds short spoken clips picked at random — `acks` played on a hit,
`sleep_acks` played when the bridge dozes off — each phrase synthesized in the
reply voice (the bridge's own TTS instance) and cached on disk as raw PCM so
later boots make no API calls. Phrases may carry eleven_v3 tone tags like
`[warm]`; they're stripped for any other voice, which would read them aloud. The cache is keyed to the voice: one folder per
`<kind>/<provider>-<voice>-<model>`, and a clip's file name hashes the text
with the rate, language and voice settings. Changing any of them re-downloads
the affected clips and prunes the ones no voice uses any more.

Unlike `vad_rms_threshold`, `rms_threshold` here is TRUE RMS of S16 samples
(speech at the Jabra measured ~14k–27k peak, room noise <3k).
"""

from __future__ import annotations

import array
import contextlib
import difflib
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
import wave
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
    # Whistle often hears a non-native "hey binary" as "hey binally" or
    # "okay binary". A heard window whose letters are at least this similar
    # (difflib ratio, spaces removed) to a phrase counts as a hit. 0 = exact only.
    "fuzzy_threshold": 0.8,
    # Extra phrases that count as a hit but are NOT passed to Whistle as
    # keyword bias (biasing towards a mishearing would make it more likely).
    "aliases": [],
    # Speak a `sleep_acks` goodbye only if a reply was played since the last
    # wake/resume; an unanswered wake dozes off with a soft tone instead.
    "goodbye_after_turn_only": True,
    "acks": [],
    "sleep_acks": [],
    # Keep the audio of every segment Whistle was asked about (hit or not),
    # to tune matching on real takes instead of guessing from transcripts.
    # Room audio on disk: off by default, bounded, under the gitignored data/.
    "capture": {"enabled": False, "dir": "data/wake-segments", "max_files": 300},
    "ack_cache_dir": "~/.cache/voice-bridge/acks",
}


def wake_config(cfg: dict) -> dict:
    """`cfg["wake_word"]` merged over DEFAULTS, phrases lower-cased."""
    out = {**DEFAULTS, **(cfg.get("wake_word") or {})}
    out["phrases"] = [normalize(p) for p in out["phrases"] if normalize(p)]
    out["aliases"] = [normalize(p) for p in out.get("aliases") or [] if normalize(p)]
    out["fuzzy_threshold"] = float(out.get("fuzzy_threshold") or 0.0)
    out["capture"] = {**DEFAULTS["capture"], **(out.get("capture") or {})}
    return out


def normalize(text: str) -> str:
    """Lower-case, letters/digits/spaces only, single-spaced."""
    return " ".join(re.sub(r"[^\w\s]|_", " ", text.lower()).split())


def _phrase_end(words: list[str], phrases, aliases=(), fuzzy: float = 0.0) -> int | None:
    return _phrase_span(words, phrases, aliases, fuzzy)[0]


def is_only_phrase(text: str, phrases, aliases=(), fuzzy: float = 0.0) -> bool:
    """`text` is the wake phrase and nothing else (one stray word allowed
    before it: "oh hey binary")."""
    words = normalize(text).split()
    end, _exact = _phrase_span(words, phrases, aliases, fuzzy)
    longest = max((len(normalize(p).split()) for p in (*phrases, *aliases)), default=0)
    return end is not None and end == len(words) and len(words) <= longest + 1


def _phrase_span(words: list[str], phrases, aliases=(), fuzzy: float = 0.0) -> tuple[int | None, bool]:
    """(index just past the wake phrase in `words` (normalized) or None,
    whether the match was exact).

    An exact phrase/alias on word boundaries wins; else, when `fuzzy` > 0,
    the window of 1..n+1 heard words whose letters (spaces dropped, so "hey
    bin ally" scores like "hey binally") are most similar to a phrase, if
    that difflib ratio reaches `fuzzy`. Aliases never match fuzzily."""
    for p in (*phrases, *aliases):
        pw = normalize(p).split()
        for i in range(len(words) - len(pw) + 1):
            if pw and words[i:i + len(pw)] == pw:
                return i + len(pw), True
    if fuzzy <= 0:
        return None, False
    best, end = 0.0, None
    for p in phrases:
        target = normalize(p).replace(" ", "")
        for n in range(1, len(normalize(p).split()) + 2):
            for i in range(len(words) - n + 1):
                r = difflib.SequenceMatcher(None, "".join(words[i:i + n]), target).ratio()
                if r >= fuzzy and r > best:
                    best, end = r, i + n
    return end, False


def matches(text: str, phrases: list[str], aliases=(), fuzzy: float = 0.0) -> bool:
    """Exact word-boundary match on phrases + aliases, else a fuzzy match
    on the phrases when `fuzzy` > 0."""
    return _phrase_end(normalize(text).split(), phrases, aliases, fuzzy) is not None


class SegmentCapture:
    """Saves wake segments as WAV + one `index.tsv` row each (time, file,
    seconds, hit, what Whistle heard); keeps the newest `max_files`.
    Failures are logged once and never reach the wake loop."""

    def __init__(self, ccfg: dict, sample_rate: int, base_dir: str = "") -> None:
        self.enabled = bool(ccfg.get("enabled"))
        self.dir = os.path.join(base_dir, ccfg.get("dir") or "data/wake-segments")
        self.max_files = max(1, int(ccfg.get("max_files") or 300))
        self.sample_rate = sample_rate
        self._warned = False

    def save(self, pcm: bytes, text: str, hit: bool) -> str | None:
        if not self.enabled:
            return None
        try:
            os.makedirs(self.dir, exist_ok=True)
            now = time.time()
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
            name = f"{stamp}-{int(now * 1000) % 1000:03d}-{'hit' if hit else 'miss'}.wav"
            with wave.open(os.path.join(self.dir, name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes(pcm)
            with open(os.path.join(self.dir, "index.tsv"), "a", encoding="utf-8") as f:
                f.write("\t".join((stamp, name, f"{len(pcm) / (2 * self.sample_rate):.2f}",
                                   "hit" if hit else "miss",
                                   " ".join(text.split()))) + "\n")
            wavs = sorted(n for n in os.listdir(self.dir) if n.endswith(".wav"))
            for old in wavs[:-self.max_files]:
                os.unlink(os.path.join(self.dir, old))
            return name
        except Exception as exc:
            if not self._warned:
                self._warned = True
                log.warning("Wake capture: could not save segment: %s", exc)
            return None


def _rms(pcm: bytes) -> float:
    samples = array.array("h", pcm)
    if not samples:
        return 0.0
    return math.sqrt(math.sumprod(samples, samples) / len(samples))


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
            # Offline first: with the weights cached, needle still made three
            # Hugging Face round-trips per boot (a download-counter ping and
            # a config check), and without network the wake word waited on
            # their timeouts. Only a cold cache needs the Hub: retry online
            # (huggingface_hub reads the flag per call, so flipping it works).
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from needle import Whistle
            try:
                _whistle = Whistle()
            except Exception as exc:
                if os.environ.get("HF_HUB_OFFLINE") != "1":
                    raise
                log.warning("Whistle: offline load failed (%s) — retrying online", exc)
                import huggingface_hub.constants as hf_constants
                hf_constants.HF_HUB_OFFLINE = False
                _whistle = Whistle()


_S16_SCALE = 1.0 / 32768.0


def _to_float(samples: "array.array") -> "array.array":
    """S16 samples → float32 in [-1, 1), the format Whistle takes as-is.

    A list comprehension is the fastest pure-Python form on the Pi (~20%
    faster than a generator; numpy is not in the venv)."""
    k = _S16_SCALE
    return array.array("f", [s * k for s in samples])


def whistle_transcribe(pcm: bytes, language: str | None = None, keywords=None) -> str:
    """Transcribe 16 kHz S16LE mono PCM; longer than 30 s is split into passes.

    Each pass is converted to float on its own, outside the lock, so the
    float copy never exceeds one 30 s pass and the wake loop is not held
    up by a long comparison's conversion."""
    load_whistle()
    shorts = array.array("h", pcm)
    texts = []
    for i in range(0, len(shorts), _MAX_PASS):
        samples = _to_float(shorts[i:i + _MAX_PASS])
        with _whistle_lock:
            result = _whistle.transcribe(samples, language=language, keywords=keywords)
        texts.append((result.get("text") or "").strip())
    return " ".join(t for t in texts if t)


class WakeDetector:
    """Whistle keyword spotting on gated segments. Loads lazily on first use."""

    def __init__(self, wcfg: dict) -> None:
        self.phrases = wcfg["phrases"]
        self.aliases = wcfg.get("aliases") or []
        self.fuzzy = float(wcfg.get("fuzzy_threshold") or 0.0)
        self.language = wcfg.get("language") or None

    def load(self) -> None:
        load_whistle()

    def transcribe(self, pcm: bytes) -> str:
        return whistle_transcribe(pcm, self.language, self.phrases)

    def heard(self, text: str) -> bool:
        return matches(text, self.phrases, self.aliases, self.fuzzy)

    def is_only_phrase(self, text: str) -> bool:
        return is_only_phrase(text, self.phrases, self.aliases, self.fuzzy)


RETRY_MIN_S, RETRY_MAX_S = 30.0, 600.0

# A clip ends at its first pause this long; that much is kept after the
# last loud window so the final syllable isn't clipped.
CLIP_PAUSE_MS, CLIP_TAIL_MS = 700, 150


def trim_clip(pcm: bytes, rate: int) -> bytes:
    """Cut a clip at its first long pause.

    eleven_v3 sometimes pads a one-second phrase with several seconds of
    near-silence, or says it a second time after the gap (seen 2026-10-10:
    "Un attimo." came back 7.1 s and 8.1 s long). The player holds the
    speaker, and the music down, for the whole clip, and the reply waits
    behind it. "Loud" is relative to the clip's own peak, so the generated
    noise floor doesn't count as speech."""
    samples = array.array("h")
    samples.frombytes(pcm[:len(pcm) & ~1])
    win = max(1, rate // 20)  # 50 ms
    levels = []
    for i in range(0, len(samples), win):
        seg = samples[i:i + win]
        levels.append(math.sqrt(sum(x * x for x in seg) / len(seg)))
    if not levels:
        return pcm
    quiet = max(levels) * 0.1
    pause = max(1, CLIP_PAUSE_MS // 50)
    last_loud = None
    for i, level in enumerate(levels):
        if level > quiet:
            last_loud = i
        elif last_loud is not None and i - last_loud >= pause:
            break
    if last_loud is None:
        return pcm
    end = (last_loud + 1) * win + rate * CLIP_TAIL_MS // 1000
    return pcm[:end * 2] if end * 2 < len(pcm) else pcm


class AckBank:
    """Short spoken clips in the reply voice, disk-cached per voice.

    `tts` is the bridge's own reply provider (named by `provider`), so acks
    never sound like a different speaker and no extra SDK client is built."""

    def __init__(self, cfg: dict, phrases, provider: str, cache_dir: str, tts, kind: str = "wake") -> None:
        self.cfg = cfg
        self.kind = kind
        self.phrases = list(phrases)
        self.provider = provider
        self.cache_dir = os.path.expanduser(cache_dir)
        self._tts = tts
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

    def _spoken(self, provider: str, text: str) -> str:
        """Drop `[tone]` tags unless the voice is eleven_v3, which acts them."""
        model = str(self.cfg.get("elevenlabs_model") or "")
        if provider == "elevenlabs" and "v3" in model:
            return text
        return " ".join(re.sub(r"\[[^\]]*\]", " ", text).split())

    def _load_or_synthesize(self, text: str, path: str) -> bytes | None:
        pcm = self._cached_or_synthesized(text, path)
        if not pcm:
            return None
        # Trimmed on the way out, so the cache keeps what the voice sent.
        trimmed = trim_clip(pcm, int(self.cfg["tts_sample_rate"]))
        if len(trimmed) < len(pcm):
            log.info("Wake acks (%s): %r trimmed %.1fs → %.1fs", self.kind, text,
                     len(pcm) / 2 / int(self.cfg["tts_sample_rate"]),
                     len(trimmed) / 2 / int(self.cfg["tts_sample_rate"]))
        return trimmed

    def _cached_or_synthesized(self, text: str, path: str) -> bytes | None:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                return f.read()
        try:
            pcm = self._tts.synthesize(self._spoken(self.provider, text))
        except Exception as exc:
            log.warning("Wake acks (%s): %s failed on %r: %s", self.kind, self.provider, text, exc)
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
        wanted = [(text, self._path(self.provider, text)) for text in self.phrases]
        self._prune({path for _t, path in wanted})
        delay = RETRY_MIN_S
        while True:
            missing = []
            for text, path in wanted:
                pcm = self._load_or_synthesize(text, path)
                if pcm:
                    with self._lock:
                        self._clips.append((self.provider, text, pcm))
                else:
                    missing.append((text, path))
            wanted = missing
            log.info("Wake acks (%s): %d clips ready%s", self.kind, len(self._clips),
                     f", {len(wanted)} missing — retry in {delay:.0f}s" if wanted else "")
            if not wanted or stop.wait(delay):
                return
            delay = min(delay * 2, RETRY_MAX_S)

    def pick(self) -> tuple[str, str, bytes] | None:
        with self._lock:
            return random.choice(self._clips) if self._clips else None
