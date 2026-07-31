#!/usr/bin/env python3
"""Tests for the non-speech transcription guard — the predicate and its
wiring into the worker loop.

ElevenLabs STT doesn't return an empty string when the captured audio holds
no speech: it returns bracketed audio-event tags ("[click]",
"[rumore di fogli]", "[rumore di sottofondo]"). Those are non-empty, so they
sail past the `if not text` guard and reach the gateway as a real user turn.
On 2026-07-31 (16:11:27–16:12:10) that fed a three-round loop: the agent
answered "Sono qui.", the speaker replayed it into the still-open mic
(`play_pcm` unmutes for external speech), the mic transcribed more room noise,
and round two began. `_is_non_speech` drops those turns one stage before the
gateway — the same "nothing to say" decision as a NO_REPLY reply, without the
round-trip.

Two groups, both hardware-free and network-free:

  1. The predicate `_is_non_speech`  — a string is non-speech when it strips
                                        to only audio-event tags and
                                        punctuation. The three real incident
                                        tags, parenthesised variants, several
                                        tags in one string, and punctuation-
                                        only input are non-speech; real
                                        Italian speech (accents included), a
                                        tag mixed with real words, and a
                                        sentence with a parenthetical are NOT
                                        and must pass through untouched.

  2. The wiring                       — `_worker_loop` skips a non-speech
                                        transcription without ever calling the
                                        gateway or putting anything on
                                        `playback_q`; a real transcript DOES
                                        reach the gateway (positive control).

Run: .venv/bin/python tests/test_non_speech_filter.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_voice_bridge():
    """`voice-bridge.py`'s hyphen blocks normal `import`; load via
    importlib so the tests can reach `_is_non_speech` and construct the
    real `VoiceBridge` class."""
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_PROJECT_ROOT, "voice-bridge.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SAMPLE_RATE = 16000


def _cfg(**overrides) -> dict:
    """Minimal cfg for the worker loop — the openclaw gateway leg keys it
    reads plus the merge/rate knobs, nothing hardware-related."""
    base = {
        "sample_rate": _SAMPLE_RATE,
        "tts_sample_rate": 24000,
        "hid_mute_enabled": True,
        "gateway_backend": "openclaw",
        "gateway_base_url": "http://gateway.invalid",
        "gateway_token": "test-token",
        "voice_model": "openclaw",
        "session_key": "agent:main:voice-bridge",
    }
    base.update(overrides)
    return base


class _StubHid:
    """In-memory stand-in for `HidMuteMonitor` — the worker loop never
    touches it, but `VoiceBridge.__init__` wants an object."""

    def set_led(self, muted: bool) -> None:
        pass


def _make_bridge(cfg: dict) -> "tuple":
    vb = _load_voice_bridge()
    bridge = vb.VoiceBridge(cfg, stt=mock.Mock(), tts=mock.Mock(), hid=_StubHid())
    return bridge, vb


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# 1. The predicate
# ---------------------------------------------------------------------------
class IsNonSpeechTest(unittest.TestCase):
    """`_is_non_speech(text)` is True when `text` carries no words — only
    bracketed audio-event tags and punctuation survive. Anything with an
    actual word (letters or digits, accents included) is speech."""

    def setUp(self):
        self.vb = _load_voice_bridge()

    def test_real_incident_tags_are_non_speech(self):
        """The three tags ElevenLabs emitted in the 2026-07-31 loop — each
        on its own must be recognised as non-speech and dropped."""
        for tag in ("[click]", "[rumore di fogli]", "[rumore di sottofondo]"):
            self.assertTrue(self.vb._is_non_speech(tag),
                            f"{tag!r} must be treated as non-speech")

    def test_parenthesised_variants_are_non_speech(self):
        """Other engines wrap the same events in parentheses instead of
        square brackets — the predicate covers both bracket styles."""
        for tag in ("(click)", "(rumore di fogli)", "(background noise)"):
            self.assertTrue(self.vb._is_non_speech(tag),
                            f"{tag!r} must be treated as non-speech")

    def test_several_tags_in_one_string_are_non_speech(self):
        """A capture can transcribe as a run of tags with nothing else —
        still no words, still non-speech."""
        self.assertTrue(
            self.vb._is_non_speech("[click] [rumore di fogli] (background noise)"))

    def test_punctuation_only_is_non_speech(self):
        """Bare punctuation carries no words either."""
        for text in ("...", " . ", "?!", "—"):
            self.assertTrue(self.vb._is_non_speech(text),
                            f"{text!r} must be treated as non-speech")

    def test_plain_italian_speech_passes_through(self):
        """A normal request is speech — must NOT be dropped."""
        self.assertFalse(self.vb._is_non_speech("accendi la luce"))

    def test_accented_speech_passes_through(self):
        """Accented letters are alphanumeric — a word with an accent is
        still a word, so the turn reaches the gateway."""
        self.assertFalse(self.vb._is_non_speech("è già acceso, perché?"))

    def test_tag_mixed_with_real_speech_passes_through(self):
        """A tag glued onto a real request ("[click] accendi la luce") is
        speech — stripping the tag still leaves words behind."""
        self.assertFalse(self.vb._is_non_speech("[click] accendi la luce"))

    def test_sentence_with_a_parenthetical_passes_through(self):
        """A parenthetical inside a real sentence must not swallow the
        whole utterance — the words outside the brackets remain."""
        self.assertFalse(
            self.vb._is_non_speech("accendi la luce (per favore)"))


# ---------------------------------------------------------------------------
# 2. The wiring into the worker loop
# ---------------------------------------------------------------------------
class WorkerLoopGuardTest(unittest.TestCase):
    """`_worker_loop` runs STT, then — before touching the gateway —
    drops the turn if `_is_non_speech(text)`. This is what breaks the
    self-feeding loop: a "[click]" transcription never becomes a gateway
    round-trip, so the agent never answers a noise into the open mic."""

    def _run_worker_once(self, bridge, vb, gateway_mock):
        """Start the worker in a daemon thread, wait for it to consume the
        utterance queue, then shut it down. The gateway function on the
        module is patched with `gateway_mock` so a call is observable and
        never actually hits the network."""
        with mock.patch.object(vb, "gateway_chat_stream", gateway_mock):
            t = threading.Thread(target=bridge._worker_loop, daemon=True)
            t.start()
            try:
                # The utterance is already queued; wait for the worker to
                # pull it off before we assert on what it did.
                self.assertTrue(
                    _wait_until(lambda: bridge.utterance_q.empty(), timeout=2.0),
                    "worker never consumed the utterance")
                # Give the loop a beat to finish the turn (skip or gateway).
                time.sleep(0.3)
            finally:
                bridge.shutdown_event.set()
                t.join(timeout=1.5)

    def test_non_speech_transcription_never_reaches_the_gateway(self):
        """STT returns "[click]" (the incident's first tag). The worker
        must skip the turn: gateway untouched, nothing on `playback_q`."""
        bridge, vb = _make_bridge(_cfg())
        bridge.recording.set()
        bridge.stt.transcribe = mock.Mock(return_value="[click]")
        gateway_mock = mock.Mock(name="gateway_chat_stream")

        gen = bridge._current_gen()
        bridge.utterance_q.put((gen, b"noise-pcm", _SAMPLE_RATE))

        self._run_worker_once(bridge, vb, gateway_mock)

        gateway_mock.assert_not_called()
        self.assertTrue(bridge.playback_q.empty(),
                        "a non-speech turn must not enqueue any playback")

    def test_real_transcription_reaches_the_gateway(self):
        """Positive control: a genuine transcript flows through to the
        gateway (and its reply reaches the player), proving the guard
        drops only non-speech and doesn't wedge every turn."""
        bridge, vb = _make_bridge(_cfg())
        bridge.recording.set()
        bridge.stt.transcribe = mock.Mock(return_value="accendi la luce")
        gateway_mock = mock.Mock(name="gateway_chat_stream",
                                 return_value=iter(["accendi", " la luce"]))

        # A TTS that actually consumes the delta stream and emits PCM, so
        # the reply lands on playback_q like a real turn.
        def _fake_tts_stream(text_iter):
            list(text_iter)  # drain the gateway deltas
            yield b"reply-pcm"

        bridge.tts.synthesize_stream = _fake_tts_stream

        gen = bridge._current_gen()
        bridge.utterance_q.put((gen, b"speech-pcm", _SAMPLE_RATE))

        self._run_worker_once(bridge, vb, gateway_mock)

        gateway_mock.assert_called_once()
        # The user's transcript is the third positional arg to the openclaw
        # gateway leg (base_url, token, text, model, session_key).
        self.assertEqual(gateway_mock.call_args.args[2], "accendi la luce")
        self.assertFalse(bridge.playback_q.empty(),
                         "a real turn must enqueue the reply for the player")


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
