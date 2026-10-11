"""Who is talking: local speaker identification + the spoken wizard that
learns a voice ("far conoscenza").

`SpeakerId` turns an utterance into a voice embedding (sherpa-onnx, CAM++
by default: 28 MB ONNX, 192 floats) and compares it, by cosine similarity,
with the profiles saved by the wizard. Measured on the Pi 3B+ (2 threads):
model load 4–6 s and ~90–100 MB RSS, 2 s of audio in 0.5 s, 4 s in ~1.0 s,
10 s in 2.3 s — hence `max_audio_s`: a turn is identified on its first
seconds, in parallel with the remote STT, so it adds next to nothing.
Nothing is loaded until a profile exists or the wizard starts; the model
file is downloaded once to `model_dir`.

`Enrollment` is the wizard's state machine, pure (no audio, no threads):
`start()` / `on_utterance()` / `cancel()` return a `Step` — what to say,
and whether to listen afterwards. `VoiceBridge` plays the steps and feeds
it the user's utterances instead of sending them to the agent.

Profiles are voiceprints (biometric data): they live under the gitignored
`data/`. This is recognition for convenience, not authentication — a
recording of the voice passes.
"""

from __future__ import annotations

import array
import datetime
import json
import logging
import math
import os
import random
import threading
import urllib.request
from typing import NamedTuple

import wake_word

log = logging.getLogger("voice-bridge")

DEFAULTS = {
    "enabled": False,
    "model_url": ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                  "speaker-recongition-models/"
                  "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"),
    "model_dir": "~/.cache/voice-bridge/speaker-id",
    "profiles_path": "data/speakers.json",
    # Cosine similarity at or above which a voice is a known speaker. Every
    # turn logs its score, so this can be tuned from the journal.
    "threshold": 0.5,
    "num_threads": 2,
    # A turn is identified on its first seconds only (see module docstring).
    "max_audio_s": 4.0,
    # How long a turn waits for the identification after STT has returned.
    "wait_ms": 2500,
    "unknown_name": "Sconosciuto",
    # Prefixed to every user message sent to the agent.
    "tag": "[Sta parlando: {name}] ",
    # Said after the wake word, these start the wizard instead of a turn.
    "enroll_phrases": ["facciamo conoscenza", "impara la mia voce",
                       "impara a riconoscermi", "registra la mia voce"],
    # One take per sentence.
    "sentences": [
        "Oggi è una bella giornata e ho voglia di ascoltare un po' di musica.",
        "Per favore, accendi la luce del salotto e abbassa il volume.",
        "Che tempo farà domani mattina, quando esco di casa?",
        "Ricordami di chiamare il dottore giovedì alle quattro e mezza.",
        "Mi piace cucinare la pasta con il pomodoro fresco e il basilico.",
    ],
    # A take shorter than this is asked again.
    "min_take_s": 2.0,
    # A take less similar than this to the ones before it is asked again
    # (somebody else talking, or mostly noise).
    "take_consistency": 0.35,
}

# The wizard's fixed lines. `[tone]` tags are acted by eleven_v3 and
# stripped for any other voice, like the wake acks.
INTRO = ("[warm] Facciamo conoscenza! Ti farò ripetere qualche frase, "
         "così imparo a riconoscere la tua voce.")
ASK_NAME = "[curious] Come ti chiami?"
NAME_AGAIN = "[warm] Non ho capito il nome. Dimmi solo come ti chiami."
GREET = "[cheerful] Piacere!"
GOOD = ["[cheerful] Perfetto.", "[warm] Benissimo.", "[cheerful] Ottimo."]
TOO_SHORT = "[warm] Era un po' corta. Riproviamo."
UNCLEAR = "[warm] Non ti ho sentito bene. Riproviamo."
DONE = "[cheerful] Fatto! D'ora in poi riconoscerò la tua voce."
CANCELLED = "[calm] Va bene, lasciamo stare."
TIMEOUT = "[calm] Non sento nessuno, ci fermiamo qui. Riproviamo quando vuoi."
GIVE_UP = "[calm] Non riesco a sentirti bene. Riproviamo più tardi."
FAILED = "[calm] Mi dispiace, adesso non riesco a imparare le voci. Riprova più tardi."

_CANCEL_WORDS = ("annulla", "basta", "stop", "lascia stare", "lasciamo stare", "smetti")
_NAME_LEADS = ("io mi chiamo", "mi chiamo", "il mio nome è", "io sono", "sono", "ciao")
_MAX_NAME_TRIES = 3
_MAX_BAD_TAKES = 4


def config(cfg: dict) -> dict:
    """`cfg["speaker_id"]` merged over DEFAULTS."""
    out = {**DEFAULTS, **(cfg.get("speaker_id") or {})}
    out["enroll_phrases"] = [wake_word.normalize(p) for p in out["enroll_phrases"]
                             if wake_word.normalize(p)]
    return out


def take_prompt(sentence: str) -> str:
    return f"[calm] Ripeti dopo di me: {sentence}"


def prompts(scfg: dict) -> list[str]:
    """Every fixed line the wizard can say — what the bridge caches."""
    return ([INTRO, ASK_NAME, NAME_AGAIN, GREET, *GOOD, TOO_SHORT, UNCLEAR, DONE,
             CANCELLED, TIMEOUT, GIVE_UP, FAILED]
            + [take_prompt(s) for s in scfg["sentences"]])


def _unit(vec) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a, b) -> float:
    """Dot product: both are unit vectors."""
    return sum(x * y for x, y in zip(a, b))


def _mean(vectors) -> list[float]:
    return _unit([sum(col) for col in zip(*vectors)])


class _Pending:
    """A result computed on a background thread."""

    def __init__(self, fn, default) -> None:
        self._value = default
        self._done = threading.Event()
        threading.Thread(target=self._run, args=(fn,), name="vb-speaker-id", daemon=True).start()

    def _run(self, fn) -> None:
        try:
            self._value = fn()
        except Exception:
            log.exception("Speaker id failed")
        finally:
            self._done.set()

    def wait(self, timeout: float):
        """The result, or the default if it failed or isn't ready in time."""
        return self._value if self._done.wait(timeout) else None


class SpeakerId:
    def __init__(self, scfg: dict, base_dir: str = "") -> None:
        self.cfg = scfg
        self.threshold = float(scfg["threshold"])
        self.unknown = str(scfg["unknown_name"])
        self.model_path = os.path.join(os.path.expanduser(scfg["model_dir"]),
                                       os.path.basename(scfg["model_url"]))
        self.profiles_path = os.path.join(base_dir, scfg["profiles_path"])
        self._extractor = None
        self._lock = threading.Lock()  # model load + inference, one at a time
        self.profiles: dict[str, list[float]] = {}
        self._read_profiles()

    # -- profiles --------------------------------------------------------
    def _read_profiles(self) -> None:
        try:
            with open(self.profiles_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("Speaker id: cannot read %s: %s", self.profiles_path, exc)
            return
        if data.get("model") != os.path.basename(self.model_path):
            # Embeddings of different models don't compare.
            log.warning("Speaker id: profiles were made with %s, ignoring them",
                        data.get("model"))
            return
        self.profiles = {name: _unit(p["embedding"])
                         for name, p in (data.get("speakers") or {}).items()}
        log.info("Speaker id: %d known voice(s): %s", len(self.profiles),
                 ", ".join(self.profiles) or "-")

    def save_profile(self, name: str, embeddings) -> None:
        """Store `name` as the mean of its takes (replaces an older profile)."""
        try:
            with open(self.profiles_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        model = os.path.basename(self.model_path)
        if data.get("model") != model:
            data = {"model": model, "speakers": {}}
        mean = _mean(embeddings)
        data["speakers"][name] = {
            "embedding": mean,
            "takes": len(embeddings),
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(os.path.dirname(self.profiles_path) or ".", exist_ok=True)
        tmp = f"{self.profiles_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.profiles_path)
        self.profiles[name] = mean
        log.info("Speaker id: saved the voice of %s (%d takes)", name, len(embeddings))

    # -- model -----------------------------------------------------------
    def download(self) -> bool:
        """Fetch the model file if it isn't cached yet."""
        if os.path.exists(self.model_path):
            return True
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        tmp = f"{self.model_path}.tmp"
        try:
            log.info("Speaker id: downloading %s", self.cfg["model_url"])
            with urllib.request.urlopen(self.cfg["model_url"], timeout=30) as r, \
                    open(tmp, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            os.replace(tmp, self.model_path)
            return True
        except Exception as exc:
            log.warning("Speaker id: model download failed: %s", exc)
            return False

    def load(self) -> bool:
        """Load the model (downloading it first if needed); False if it can't."""
        with self._lock:
            if self._extractor is not None:
                return True
            if not self.download():
                return False
            try:
                import sherpa_onnx
                self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                    sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=self.model_path, num_threads=int(self.cfg["num_threads"])))
            except Exception as exc:
                log.warning("Speaker id: model failed to load: %s", exc)
                return False
            log.info("Speaker id: model loaded")
            return True

    def prepare(self, stop: "threading.Event | None" = None) -> None:
        """Boot, off the hot path: have the model file on disk for the
        wizard, and in memory only if there is a voice to recognize."""
        stop = stop or threading.Event()
        delay = wake_word.RETRY_MIN_S
        while not self.download():
            if stop.wait(delay):
                return
            delay = min(delay * 2, wake_word.RETRY_MAX_S)
        if self.profiles:
            self.load()

    def warm(self) -> None:
        """Start loading the model without waiting for it."""
        threading.Thread(target=self.load, name="vb-speaker-id-load", daemon=True).start()

    # -- embeddings ------------------------------------------------------
    def embed(self, pcm: bytes, sample_rate: int, max_s: float = 0.0) -> list[float] | None:
        """Unit voice embedding of S16LE mono `pcm` (its first `max_s`
        seconds when given), or None if the model is unavailable."""
        if not self.load():
            return None
        if max_s > 0:
            pcm = pcm[:int(max_s * sample_rate) * 2]
        shorts = array.array("h")
        shorts.frombytes(pcm[:len(pcm) & ~1])
        samples = [s / 32768.0 for s in shorts]
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()
            if not self._extractor.is_ready(stream):
                return None  # too short for the model
            return _unit(self._extractor.compute(stream))

    def embed_async(self, pcm: bytes, sample_rate: int) -> _Pending:
        return _Pending(lambda: self.embed(pcm, sample_rate), None)

    def best(self, embedding) -> tuple[str | None, float]:
        """Closest profile and its similarity (any score, even a poor one)."""
        scored = [(_cosine(embedding, p), name) for name, p in self.profiles.items()]
        if not scored:
            return None, 0.0
        score, name = max(scored)
        return name, score

    def identify(self, pcm: bytes, sample_rate: int) -> str:
        """Name of the enrolled speaker of `pcm`, or `unknown_name`."""
        if not self.profiles:
            return self.unknown
        emb = self.embed(pcm, sample_rate, float(self.cfg["max_audio_s"]))
        if emb is None:
            return self.unknown
        name, score = self.best(emb)
        known = score >= self.threshold
        log.info("Speaker: %s (closest %s, %.2f, threshold %.2f)",
                 name if known else self.unknown, name, score, self.threshold)
        return name if known else self.unknown

    def identify_async(self, pcm: bytes, sample_rate: int) -> _Pending:
        return _Pending(lambda: self.identify(pcm, sample_rate), self.unknown)

    def tag(self, name: str | None) -> str:
        return str(self.cfg["tag"]).format(name=name or self.unknown)


class Step(NamedTuple):
    """What the wizard does next. `say` items are a fixed line from
    `prompts()`, or `("tts", text, fallback line)` for one made on the fly."""
    say: list
    listen: bool


class Enrollment:
    """The "far conoscenza" wizard: ask the name, take one recording per
    sentence, save the profile. One session at a time."""

    def __init__(self, sid: SpeakerId, scfg: dict) -> None:
        self.sid = sid
        self.cfg = scfg
        self._lock = threading.Lock()
        self.active = False
        self._stage = ""
        self._name = ""
        self._takes: list[list[float]] = []
        self._name_tries = self._bad = 0

    def is_request(self, text: str) -> bool:
        """`text` (a transcript) asks to start the wizard."""
        return (len(wake_word.normalize(text).split()) <= 8
                and wake_word.matches(text, self.cfg["enroll_phrases"]))

    @property
    def wants_voice(self) -> bool:
        """The next utterance is a take: it needs its embedding."""
        return self.active and self._stage == "takes"

    def start(self, name: str = "") -> Step:
        with self._lock:
            self.active = True
            self._name = _clean_name(name)
            self._takes = []
            self._name_tries = self._bad = 0
            self.sid.warm()  # loads while the intro plays
            if self._name:
                log.info("Enrollment: started for %s", self._name)
                return self._first_take([INTRO])
            log.info("Enrollment: started, asking the name")
            self._stage = "name"
            return Step([INTRO, ASK_NAME], True)

    def cancel(self, reason: str) -> Step:
        """Stop the session; `reason` "timeout" picks the line that says so."""
        with self._lock:
            if not self.active:
                return Step([], False)
            return self._end(TIMEOUT if reason == "timeout" else CANCELLED, reason)

    def on_utterance(self, text: str, embedding, seconds: float) -> Step:
        """The user said `text` (`embedding`: its voice, when `wants_voice`
        was set; `seconds`: its length)."""
        with self._lock:
            if not self.active:
                return Step([], False)
            words = wake_word.normalize(text)
            if len(words.split()) <= 3 and any(
                    wake_word.matches(words, [w]) for w in _CANCEL_WORDS):
                return self._end(CANCELLED, "cancelled by voice")
            if self._stage == "name":
                return self._on_name(text)
            return self._on_take(embedding, seconds)

    # -- stages (called with the lock held) --------------------------------
    def _end(self, line: str, why: str) -> Step:
        log.info("Enrollment: ended (%s)", why)
        self.active = False
        self._stage = ""
        return Step([line], False)

    def _first_take(self, before: list) -> Step:
        self._stage = "takes"
        return Step(before + [take_prompt(self.cfg["sentences"][0])], True)

    def _on_name(self, text: str) -> Step:
        name = _parse_name(text)
        if not name:
            self._name_tries += 1
            if self._name_tries >= _MAX_NAME_TRIES:
                return self._end(GIVE_UP, "no name understood")
            return Step([NAME_AGAIN], True)
        self._name = name
        log.info("Enrollment: name %r", name)
        return self._first_take([("tts", f"[cheerful] Piacere, {name}!", GREET)])

    def _on_take(self, embedding, seconds: float) -> Step:
        sentences = self.cfg["sentences"]
        again = take_prompt(sentences[len(self._takes)])
        if embedding is None and seconds >= float(self.cfg["min_take_s"]):
            return self._end(FAILED, "no voice model")
        problem = None
        if embedding is None or seconds < float(self.cfg["min_take_s"]):
            problem = TOO_SHORT
        elif self._takes:
            score = _cosine(embedding, _mean(self._takes))
            log.info("Enrollment: take %d, %.1fs, %.2f like the previous ones",
                     len(self._takes) + 1, seconds, score)
            if score < float(self.cfg["take_consistency"]):
                problem = UNCLEAR
        if problem:
            self._bad += 1
            if self._bad >= _MAX_BAD_TAKES:
                return self._end(GIVE_UP, "too many bad takes")
            return Step([problem, again], True)
        self._takes.append(embedding)
        if len(self._takes) < len(sentences):
            return Step([random.choice(GOOD), take_prompt(sentences[len(self._takes)])], True)
        other, score = self.sid.best(_mean(self._takes))
        if other and other != self._name and score >= self.sid.threshold:
            log.warning("Enrollment: %s sounds like %s (%.2f) — they may be confused",
                        self._name, other, score)
        try:
            self.sid.save_profile(self._name, self._takes)
        except OSError as exc:
            log.error("Enrollment: cannot save the profile: %s", exc)
            return self._end(FAILED, "profile not saved")
        return self._end(DONE, f"learned {self._name}")


def _clean_name(name: str) -> str:
    return " ".join(w.capitalize() for w in wake_word.normalize(name or "").split()[:3])


def _parse_name(text: str) -> str:
    """The name in an answer to "Come ti chiami?" ("Mi chiamo Anna." → Anna),
    or "" when the answer doesn't look like one."""
    words = wake_word.normalize(text)
    stripped = True
    while stripped:  # "ciao, io sono Anna"
        stripped = False
        for lead in _NAME_LEADS:
            if words.startswith(lead + " "):
                words, stripped = words[len(lead) + 1:], True
    return _clean_name(words) if 1 <= len(words.split()) <= 3 else ""
