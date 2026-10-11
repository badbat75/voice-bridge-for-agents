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
  voiceprint with margin (`confident`) — are kept as samples, and a voice
  is also scored against the `top_k` samples closest to it (not their
  mean: a few takes from across the room would drown among the many from
  close by). A short turn that matched nobody counts too if the confident
  one comes later in the same conversation. The enrolled voiceprint is the
  anchor: samples whose mean drifts away from it (`min_anchor_similarity`)
  are thrown away. Only vectors are kept, never audio.

Distance changes a voice more than anything else here (2026-10-11: the same
speaker, 0.37 from across the room and 0.81 close by half a minute later),
so the wizard also takes `far_sentences` from where the user usually talks
and keeps them as a second enrolled voiceprint, never averaged with the
close one. `far_phrases` run that part alone for a voice already known.

Profiles are voiceprints (biometric data): they live under the gitignored
`data/`. This is recognition for convenience, not authentication — a
recording of the voice passes.
"""

from __future__ import annotations

import array
import datetime
import difflib
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
        # The adapted score is the mean over this many closest samples.
        "top_k": 3,
        # Mean of the samples vs the enrolled voiceprint: below this they
        # are dropped.
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
    # Said after the wake word by a known voice (or followed by its name),
    # these run only the takes from a distance. Checked before the above.
    "far_phrases": ["impara la mia voce da lontano", "riconoscimi da lontano",
                    "facciamo conoscenza da lontano", "impara a riconoscermi da lontano"],
    # One take per sentence.
    "sentences": [
        "Oggi è una bella giornata e ho voglia di ascoltare un po' di musica.",
        "Per favore, accendi la luce del salotto e abbassa il volume.",
        "Che tempo farà domani mattina, quando esco di casa?",
        "Ricordami di chiamare il dottore giovedì alle quattro e mezza.",
        "Mi piace cucinare la pasta con il pomodoro fresco e il basilico.",
    ],
    # After those, one take per sentence from where the user usually talks
    # ([] = no far takes).
    "far_sentences": [
        "Metti un po' di musica tranquilla mentre preparo la cena.",
        "Quanto manca alla fine del timer che ho messo prima?",
        "Spegni tutto, per favore, che stiamo uscendo di casa.",
    ],
    # A take shorter than this is asked again.
    "min_take_s": 2.0,
    # A take less similar than this to the ones before it is asked again
    # (somebody else talking, or mostly noise).
    "take_consistency": 0.35,
    # A take must also be the sentence that was asked: the STT text has to
    # be this close to it (difflib ratio on the letters, 0 = not checked).
    # A voice the STT can't make out is too faint to learn from — and from
    # that far the agent wouldn't understand a request either.
    "take_text_match": 0.6,
    # The same for a far take, against the close voiceprint or the far takes
    # before it, whichever is closer: distance alone costs a lot.
    "far_take_consistency": 0.25,
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
FAR_INTRO = ("[warm] Adesso allontanati e mettiti dove stai di solito quando mi parli. "
             "Parla da lì, con la voce di sempre.")
FAR_ONLY_INTRO = ("[warm] Va bene, imparo a riconoscerti anche da lontano. Allontanati e "
                  "mettiti dove stai di solito quando mi parli, poi parla da lì con la "
                  "voce di sempre.")
FAR_DONE = "[cheerful] Fatto! Ora ti riconosco anche da lontano."
FAR_CLOSER = "[warm] Da lì non ti sento bene. Avvicinati un po' e riprova."
FAR_SKIPPED = "[calm] Va bene, ci fermiamo qui. Da vicino ti riconosco già."
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
    for key in ("enroll_phrases", "far_phrases"):
        out[key] = [wake_word.normalize(p) for p in out[key] if wake_word.normalize(p)]
    return out


def take_prompt(sentence: str) -> str:
    return f"[calm] Ripeti dopo di me: {sentence}"


def prompts(scfg: dict) -> list[str]:
    """Every fixed line the wizard can say — what the bridge caches."""
    lines = ([INTRO, ASK_NAME, NAME_AGAIN, GREET, *GOOD, TOO_SHORT, UNCLEAR, DONE,
              CANCELLED, TIMEOUT, GIVE_UP, FAILED]
             + [take_prompt(s) for s in scfg["sentences"]])
    if scfg["far_sentences"]:
        lines += [FAR_INTRO, FAR_ONLY_INTRO, FAR_DONE, FAR_CLOSER, FAR_SKIPPED]
        lines += [take_prompt(s) for s in scfg["far_sentences"]]
    return list(dict.fromkeys(lines))


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
    (the best of `enrolled` — the close voiceprint — `far` and `adapted`,
    None for the ones that speaker doesn't have), the embedding and the
    utterance's length."""
    name: str
    score: float
    enrolled: float
    far: float | None
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
        self._far: dict[str, list[float]] = {}  # the same, from a distance
        self._adapt = scfg["adapt"]
        self._samples: dict[str, list[list[float]]] = {}
        self._adapted: set[str] = set()  # speakers whose samples are in use
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
        self._far = {name: _unit(p["far"]["embedding"])
                     for name, p in speakers.items() if p.get("far")}
        if not self.profiles:
            return
        log.info("Speaker id: %d known voice(s): %s", len(self.profiles),
                 ", ".join(n + (" (+ far)" if n in self._far else "") for n in self.profiles))
        if self._adapt["enabled"]:
            for name, p in speakers.items():
                self._samples[name] = (p.get("adapted") or {}).get("samples") or []
                self._refresh_adapted(name)

    def save_profile(self, name: str, embeddings) -> None:
        """Store `name` as the mean of its takes. It replaces an older
        profile, far takes included (the wizard asks them again); adapted
        samples stay if they are still like the new voiceprint."""
        data = self._read_file()
        mean = _mean(embeddings)
        now = datetime.datetime.now().isoformat(timespec="seconds")
        old = ((data["speakers"].get(name) or {}).get("adapted") or {}).get("samples") or []
        kept = [x for x in old if _cosine(x, mean) >= float(self._adapt["min_score"])]
        data["speakers"][name] = {"embedding": mean, "takes": len(embeddings), "updated": now}
        if kept:
            data["speakers"][name]["adapted"] = {"samples": kept, "updated": now}
        self._write_file(data)
        self.profiles[name] = mean
        self._far.pop(name, None)
        self._samples[name] = kept
        if self._adapt["enabled"]:
            self._refresh_adapted(name)
        log.info("Speaker id: saved the voice of %s (%d takes, %d of %d samples kept)",
                 name, len(embeddings), len(kept), len(old))

    def save_far(self, name: str, embeddings) -> None:
        """Add to `name`'s profile the mean of takes made from a distance: a
        second enrolled voiceprint, never averaged with the close one."""
        data = self._read_file()
        if name not in data["speakers"]:
            raise OSError(f"no profile for {name}")
        mean = _mean(embeddings)
        data["speakers"][name]["far"] = {
            "embedding": mean,
            "takes": len(embeddings),
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        self._write_file(data)
        self._far[name] = mean
        log.info("Speaker id: saved the voice of %s from a distance (%d takes, "
                 "%.2f to the close one)", name, len(embeddings),
                 _cosine(mean, self.profiles[name]))

    # -- adapted voiceprint ----------------------------------------------
    def _enrolled(self, embedding, name: str) -> tuple[float, float | None]:
        """`embedding` against `name`'s close and far voiceprints."""
        far = _cosine(embedding, self._far[name]) if name in self._far else None
        return _cosine(embedding, self.profiles[name]), far

    def _refresh_adapted(self, name: str) -> None:
        """Decide whether `name`'s samples are used; drop them if together
        they have drifted from the enrolled voiceprint."""
        samples = self._samples.get(name) or []
        self._adapted.discard(name)
        if len(samples) < int(self._adapt["min_samples"]):
            if samples:
                log.info("Speaker id: adapted voice of %s: %d of %d samples, not used yet",
                         name, len(samples), int(self._adapt["min_samples"]))
            return
        near = max(x for x in self._enrolled(_mean(samples), name) if x is not None)
        if near < float(self._adapt["min_anchor_similarity"]):
            log.warning("Speaker id: adapted voice of %s is %.2f from the enrolled one "
                        "(min %.2f), dropping its %d samples", name, near,
                        float(self._adapt["min_anchor_similarity"]), len(samples))
            self._samples[name] = []
            return
        self._adapted.add(name)
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

    def _collect(self, m: Match, mine: bool) -> None:
        """Keep `m` as a sample of its speaker if this conversation vouches
        for it: a turn in it matched an enrolled voiceprint with margin.
        Turns heard before that one wait in `_conv_pending`. `mine`: the
        turn may be `m.name`'s (attributed to them, or to nobody yet)."""
        a = self._adapt
        if not a["enabled"]:
            return
        enrolled = max(m.enrolled, m.far if m.far is not None else m.enrolled)
        if mine and m.seconds >= float(a["min_s"]) and enrolled >= float(a["min_score"]):
            self._conv_pending.append((m.name, m.embedding))
        if enrolled >= float(a["confident"]):
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

    def _score(self, embedding, name: str) -> tuple[float, float, float | None, float | None]:
        """(`score`, enrolled, far, adapted) of `embedding` against `name`."""
        enrolled, far = self._enrolled(embedding, name)
        adapted = None
        if name in self._adapted:
            top = sorted((_cosine(embedding, x) for x in self._samples[name]), reverse=True)
            top = top[:max(1, int(self._adapt["top_k"]))]
            adapted = sum(top) / len(top)
        return (max(x for x in (enrolled, far, adapted) if x is not None),
                enrolled, far, adapted)

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
        short = m.seconds <= float(self.cfg["carry_max_s"])
        note = ""
        if m.score >= self.threshold:
            if self._conv_name and m.name != self._conv_name:
                self.end_conversation()  # somebody else took over
            name = self._conv_name = m.name
        elif self._conv_name and short:
            name = self._conv_name
            note = f"; {m.seconds:.1f}s, kept from this conversation"
        else:
            name = None
            if not short:
                self.end_conversation()  # long enough to tell, and it is nobody
        if m.far is not None or m.adapted is not None:
            parts = [("close", m.enrolled), ("far", m.far), ("adapted", m.adapted)]
            note = "; " + ", ".join(f"{k} {v:.2f}" for k, v in parts if v is not None) + note
        log.info("Speaker: %s (closest %s, %.2f, threshold %.2f%s)",
                 name or self.unknown, m.name, m.score, self.threshold, note)
        # A short turn that matched nobody may still be theirs: it waits for
        # a confident turn later in this conversation.
        self._collect(m, mine=name == m.name or (name is None and short))
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
    sentence, save the profile, then the same from a distance
    (`far_sentences`). One session at a time."""

    def __init__(self, sid: SpeakerId, scfg: dict) -> None:
        self.sid = sid
        self.cfg = scfg
        self._lock = threading.Lock()
        self.active = False
        self._stage = ""
        self._name = ""
        self._takes: list[list[float]] = []
        self._far_takes: list[list[float]] = []
        self._far_only = False
        self._name_tries = self._bad = 0

    def is_far_request(self, text: str) -> bool:
        """`text` asks for the takes from a distance only."""
        return (bool(self.cfg["far_sentences"])
                and len(wake_word.normalize(text).split()) <= 8
                and wake_word.matches(text, self.cfg["far_phrases"]))

    def is_request(self, text: str) -> bool:
        """`text` (a transcript) asks to start the wizard."""
        return (len(wake_word.normalize(text).split()) <= 8
                and wake_word.matches(text, self.cfg["enroll_phrases"]))

    @property
    def wants_voice(self) -> bool:
        """The next utterance is a take: it needs its embedding."""
        return self.active and self._stage in ("takes", "far")

    def start(self, name: str = "", far_only: bool = False) -> Step:
        """`far_only`: just the takes from a distance, for a voice that is
        already known (anybody else gets the whole wizard)."""
        with self._lock:
            self.active = True
            self._name = _clean_name(name)
            self._takes, self._far_takes = [], []
            self._far_only = far_only and bool(self.cfg["far_sentences"])
            self._name_tries = self._bad = 0
            self.sid.warm()  # loads while the intro plays
            if self._name:
                log.info("Enrollment: started for %s%s", self._name,
                         " (far takes)" if self._far_only else "")
                return self._after_name([])
            log.info("Enrollment: started, asking the name")
            self._stage = "name"
            return Step([INTRO, ASK_NAME], True)

    def cancel(self, reason: str) -> Step:
        """Stop the session; `reason` "timeout" picks the line that says so."""
        with self._lock:
            if not self.active:
                return Step([], False)
            if self._stage == "far":
                return self._end(FAR_SKIPPED, f"{reason}, far takes skipped")
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
                if self._stage == "far":
                    return self._end(FAR_SKIPPED, "cancelled by voice, far takes skipped")
                return self._end(CANCELLED, "cancelled by voice")
            if self._stage == "name":
                return self._on_name(text)
            return self._on_take(text, embedding, seconds)

    # -- stages (called with the lock held) --------------------------------
    def _end(self, line: str, why: str) -> Step:
        log.info("Enrollment: ended (%s)", why)
        self.active = False
        self._stage = ""
        return Step([line], False)

    def _first_take(self, before: list) -> Step:
        self._stage = "takes"
        return Step(before + [take_prompt(self.cfg["sentences"][0])], True)

    def _first_far(self, before: list) -> Step:
        self._stage = "far"
        self._bad = 0
        return Step(before + [take_prompt(self.cfg["far_sentences"][0])], True)

    def _after_name(self, greeting: list) -> Step:
        """The name is known: far takes alone if that was asked and the
        close voiceprint is there, the whole wizard otherwise."""
        if self._far_only and self._name in self.sid.profiles:
            return self._first_far(greeting + [FAR_ONLY_INTRO])
        self._far_only = False
        return self._first_take(greeting or [INTRO])

    def _on_name(self, text: str) -> Step:
        name = _parse_name(text)
        if not name:
            self._name_tries += 1
            if self._name_tries >= _MAX_NAME_TRIES:
                return self._end(GIVE_UP, "no name understood")
            return Step([NAME_AGAIN], True)
        self._name = name
        log.info("Enrollment: name %r", name)
        return self._after_name([("tts", f"[cheerful] Piacere, {name}!", GREET)])

    def _on_take(self, text: str, embedding, seconds: float) -> Step:
        far = self._stage == "far"
        sentences = self.cfg["far_sentences" if far else "sentences"]
        takes = self._far_takes if far else self._takes
        again = take_prompt(sentences[len(takes)])
        if embedding is None and seconds >= float(self.cfg["min_take_s"]):
            return self._end(FAILED, "no voice model")
        heard = difflib.SequenceMatcher(
            None, wake_word.normalize(text).replace(" ", ""),
            wake_word.normalize(sentences[len(takes)]).replace(" ", "")).ratio()
        problem = None
        if embedding is None or seconds < float(self.cfg["min_take_s"]):
            problem = TOO_SHORT
        elif heard < float(self.cfg["take_text_match"]):
            log.info("Enrollment: %stake %d, heard %r, %.2f like the sentence asked",
                     "far " if far else "", len(takes) + 1, text, heard)
            problem = FAR_CLOSER if far else UNCLEAR
        elif far:
            close = _cosine(embedding, self.sid.profiles[self._name])
            others = _cosine(embedding, _mean(takes)) if takes else None
            log.info("Enrollment: far take %d, %.1fs, %.2f like the close voice%s",
                     len(takes) + 1, seconds, close,
                     f", {others:.2f} like the far takes before" if takes else "")
            if max(close, others if takes else close) < float(self.cfg["far_take_consistency"]):
                problem = UNCLEAR
        elif takes:
            score = _cosine(embedding, _mean(takes))
            log.info("Enrollment: take %d, %.1fs, %.2f like the previous ones",
                     len(takes) + 1, seconds, score)
            if score < float(self.cfg["take_consistency"]):
                problem = UNCLEAR
        if problem:
            self._bad += 1
            if self._bad >= _MAX_BAD_TAKES:
                return self._end(GIVE_UP, "too many bad takes")
            return Step([problem, again], True)
        takes.append(embedding)
        if len(takes) < len(sentences):
            return Step([random.choice(GOOD), take_prompt(sentences[len(takes)])], True)
        try:
            if far:
                self.sid.save_far(self._name, takes)
                return self._end(FAR_DONE, f"learned {self._name} from a distance")
            other, score = self.sid.best(_mean(takes))
            if other and other != self._name and score >= self.sid.threshold:
                log.warning("Enrollment: %s sounds like %s (%.2f) — they may be confused",
                            self._name, other, score)
            self.sid.save_profile(self._name, takes)
        except OSError as exc:
            log.error("Enrollment: cannot save the profile: %s", exc)
            return self._end(FAILED, "profile not saved")
        if self.cfg["far_sentences"]:
            # The close voiceprint is saved: whatever happens to the far
            # takes, this voice is known from here on.
            return self._first_far([random.choice(GOOD), FAR_INTRO])
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
