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

Two things cover what one embedding cannot (2026-10-11: phrases of 5–9 s
scored 0.72–0.82, "ok" / "grazie" / "ciao" 0.19–0.49 against a 0.50
threshold, so every short follow-up came out unknown):

- **Conversation memory** (`carry_max_s`): once a voice is recognized, a
  later utterance too short to tell keeps that name until the bridge goes
  idle or is muted (`end_conversation()`). A long utterance that matches
  nobody is somebody else, and clears it.
- **Adapted voiceprint** (`adapt`, off by default): embeddings of real
  turns, labelled by the conversation rather than by their own score — a
  turn counts only in a conversation where a phrase matched the *enrolled*
  voiceprint with margin (`confident`) — are averaged into a second
  voiceprint per speaker, and a voice is scored against the better of the
  two. The enrolled one is the anchor: an adapted voiceprint that drifts
  away from it (`min_anchor_similarity`) is thrown away. Only vectors are
  kept, never audio.

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
    # An utterance no longer than this that matches nobody keeps the name
    # recognized earlier in the same conversation (0 = off).
    "carry_max_s": 2.5,
    # The adapted voiceprint (see module docstring).
    "adapt": {
        "enabled": False,
        # A conversation labels its turns once one of them scores this
        # against the enrolled voiceprint.
        "confident": 0.70,
        # A turn is kept as a sample if it is at least this long and this
        # close to the enrolled voiceprint (below: noise, or somebody else).
        "min_s": 1.0,
        "min_score": 0.35,
        # Used from `min_samples` on; the oldest go after `max_samples`.
        "min_samples": 5,
        "max_samples": 40,
        # Adapted vs enrolled voiceprint: below this the samples are dropped.
        "min_anchor_similarity": 0.60,
    },
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
    user = cfg.get("speaker_id") or {}
    out = {**DEFAULTS, **user}
    out["adapt"] = {**DEFAULTS["adapt"], **(user.get("adapt") or {})}
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


class Match(NamedTuple):
    """One utterance against the profiles: the closest `name`, its `score`
    (the better of `enrolled` and `adapted`, None while there is no adapted
    voiceprint), the embedding and the utterance's length."""
    name: str
    score: float
    enrolled: float
    adapted: float | None
    embedding: list
    seconds: float


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
        self.profiles: dict[str, list[float]] = {}  # enrolled voiceprints
        self._adapt = scfg["adapt"]
        self._samples: dict[str, list[list[float]]] = {}
        self._adapted: dict[str, list[float]] = {}
        # This conversation, worker thread only: who was recognized, who
        # with margin, and the samples waiting for that.
        self._conv_name: str | None = None
        self._conv_confident: str | None = None
        self._conv_pending: list[tuple[str, list[float]]] = []
        self._read_profiles()

    # -- profiles --------------------------------------------------------
    def _read_file(self) -> dict:
        """The profiles file, empty if missing or made with another model."""
        model = os.path.basename(self.model_path)
        try:
            with open(self.profiles_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        except (OSError, ValueError) as exc:
            log.warning("Speaker id: cannot read %s: %s", self.profiles_path, exc)
            data = {}
        if data and data.get("model") != model:
            # Embeddings of different models don't compare.
            log.warning("Speaker id: profiles were made with %s, ignoring them",
                        data.get("model"))
            data = {}
        return data or {"model": model, "speakers": {}}

    def _write_file(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.profiles_path) or ".", exist_ok=True)
        tmp = f"{self.profiles_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.profiles_path)

    def _read_profiles(self) -> None:
        speakers = self._read_file()["speakers"]
        self.profiles = {name: _unit(p["embedding"]) for name, p in speakers.items()}
        if not self.profiles:
            return
        log.info("Speaker id: %d known voice(s): %s", len(self.profiles),
                 ", ".join(self.profiles))
        if self._adapt["enabled"]:
            for name, p in speakers.items():
                self._samples[name] = (p.get("adapted") or {}).get("samples") or []
                self._refresh_adapted(name)

    def save_profile(self, name: str, embeddings) -> None:
        """Store `name` as the mean of its takes (replaces an older profile,
        adapted samples included: they belonged to the old voiceprint)."""
        data = self._read_file()
        mean = _mean(embeddings)
        data["speakers"][name] = {
            "embedding": mean,
            "takes": len(embeddings),
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        self._write_file(data)
        self.profiles[name] = mean
        self._samples.pop(name, None)
        self._adapted.pop(name, None)
        log.info("Speaker id: saved the voice of %s (%d takes)", name, len(embeddings))

    # -- adapted voiceprint ----------------------------------------------
    def _refresh_adapted(self, name: str) -> None:
        """Rebuild `name`'s adapted voiceprint from its samples; drop them
        if it has drifted from the enrolled one."""
        samples = self._samples.get(name) or []
        self._adapted.pop(name, None)
        if len(samples) < int(self._adapt["min_samples"]):
            if samples:
                log.info("Speaker id: adapted voice of %s: %d of %d samples, not used yet",
                         name, len(samples), int(self._adapt["min_samples"]))
            return
        adapted = _mean(samples)
        near = _cosine(adapted, self.profiles[name])
        if near < float(self._adapt["min_anchor_similarity"]):
            log.warning("Speaker id: adapted voice of %s is %.2f from the enrolled one "
                        "(min %.2f), dropping its %d samples", name, near,
                        float(self._adapt["min_anchor_similarity"]), len(samples))
            self._samples[name] = []
            return
        self._adapted[name] = adapted
        log.info("Speaker id: adapted voice of %s: %d samples, %.2f to the enrolled one",
                 name, len(samples), near)

    def _add_samples(self, name: str, embeddings) -> None:
        samples = self._samples.setdefault(name, [])
        samples += [[round(x, 5) for x in e] for e in embeddings]
        del samples[:-int(self._adapt["max_samples"])]
        self._refresh_adapted(name)
        data = self._read_file()
        if name in data["speakers"]:
            data["speakers"][name]["adapted"] = {
                "samples": self._samples[name],
                "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            self._write_file(data)

    def _collect(self, m: Match, named: bool) -> None:
        """Keep `m` as a sample of its speaker if this conversation vouches
        for it: a turn in it matched the enrolled voiceprint with margin.
        Turns heard before that one wait in `_conv_pending`."""
        a = self._adapt
        if not a["enabled"]:
            return
        if named and m.seconds >= float(a["min_s"]) and m.enrolled >= float(a["min_score"]):
            self._conv_pending.append((m.name, m.embedding))
        if m.enrolled >= float(a["confident"]):
            self._conv_confident = m.name
        who = self._conv_confident  # `end_conversation` runs on other threads
        if who:
            take = [e for name, e in self._conv_pending if name == who]
            self._conv_pending = []
            if take:
                self._add_samples(who, take)

    def end_conversation(self) -> None:
        """The bridge went idle or was muted: the next voice starts fresh."""
        self._conv_name = self._conv_confident = None
        self._conv_pending = []

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

    def _score(self, embedding, name: str) -> tuple[float, float, float | None]:
        """(`score`, enrolled, adapted) of `embedding` against `name`."""
        enrolled = _cosine(embedding, self.profiles[name])
        adapted = _cosine(embedding, self._adapted[name]) if name in self._adapted else None
        return max(enrolled, adapted if adapted is not None else enrolled), enrolled, adapted

    def best(self, embedding) -> tuple[str | None, float]:
        """Closest profile and its similarity (any score, even a poor one)."""
        scored = [(self._score(embedding, name)[0], name) for name in self.profiles]
        if not scored:
            return None, 0.0
        score, name = max(scored)
        return name, score

    def match(self, pcm: bytes, sample_rate: int) -> Match | None:
        """`pcm` against the profiles; None with nobody enrolled or no model."""
        if not self.profiles:
            return None
        emb = self.embed(pcm, sample_rate, float(self.cfg["max_audio_s"]))
        if emb is None:
            return None
        name, _ = self.best(emb)
        return Match(name, *self._score(emb, name), emb, len(pcm) / (sample_rate * 2))

    def match_async(self, pcm: bytes, sample_rate: int) -> _Pending:
        return _Pending(lambda: self.match(pcm, sample_rate), None)

    def resolve(self, m: Match | None) -> str:
        """Who said the turn `m` was computed on, with the conversation's
        memory applied. Call once per turn sent to the agent, in order."""
        if m is None:
            return self.unknown
        carry = float(self.cfg["carry_max_s"])
        note = ""
        if m.score >= self.threshold:
            if m.name != self._conv_name:
                self.end_conversation()  # somebody else took over
            name = self._conv_name = m.name
        elif self._conv_name and m.seconds <= carry:
            name = self._conv_name
            note = f"; {m.seconds:.1f}s, kept from this conversation"
        else:
            name = None
            self.end_conversation()  # long enough to tell, and it is nobody
        if m.adapted is not None:
            note = f"; enrolled {m.enrolled:.2f}, adapted {m.adapted:.2f}" + note
        log.info("Speaker: %s (closest %s, %.2f, threshold %.2f%s)",
                 name or self.unknown, m.name, m.score, self.threshold, note)
        self._collect(m, named=name == m.name)
        return name or self.unknown

    def identify(self, pcm: bytes, sample_rate: int) -> str:
        """Name of the enrolled speaker of `pcm`, or `unknown_name`."""
        return self.resolve(self.match(pcm, sample_rate))

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
