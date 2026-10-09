#!/usr/bin/env python3
"""Synchronous tests for `endpointer.Endpointer` — no threads, no sleeps.

The threaded behaviour around it (generations, ducking, auto-idle) is
covered by test_voice_bridge_endpointer.py; these pin the per-chunk state
machine itself.

Run: .venv/bin/python tests/test_endpointer.py
"""

from __future__ import annotations

import array
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import endpointer  # noqa: E402
from endpointer import Endpointer  # noqa: E402

_SR, _CHUNK = 16000, 1024  # 64 ms chunks


def _chunk(amp: int) -> bytes:
    return array.array("h", [amp, -amp] * (_CHUNK // 2)).tobytes()


LOUD, QUIET = _chunk(8000), _chunk(10)


def _ep(vad=None, **overrides) -> Endpointer:
    cfg = {
        "sample_rate": _SR, "chunk_size": _CHUNK, "vad_rms_threshold": 1e6,
        "silence_timeout_ms": 192,  # 3 chunks
        "silence_keep_ms": 0, "pre_speech_keep_ms": 0,
        **overrides,
    }
    return Endpointer(cfg, vad)


def _feed(ep, chunks):
    return [r for r in (ep.feed(c) for c in chunks) if r is not None]


class _Vad:
    def __init__(self, voiced):
        self.voiced = voiced

    def is_speech(self, frame, rate):
        return self.voiced


class EndpointerTest(unittest.TestCase):
    def test_silence_never_commits_and_counts(self):
        ep = _ep()
        self.assertEqual(_feed(ep, [QUIET] * 10), [])
        self.assertEqual(ep.silence, 10)
        self.assertFalse(ep.in_speech)

    def test_speech_then_pause_commits_trimmed_pcm(self):
        ep = _ep()
        results = _feed(ep, [LOUD] * 4 + [QUIET] * 3)
        self.assertEqual(results, [(endpointer.COMMITTED, LOUD * 4)])
        self.assertFalse(ep.in_speech)
        self.assertEqual(ep.silence, 0, "a commit restarts the idle window")

    def test_keep_and_pre_roll(self):
        ep = _ep(silence_keep_ms=64, pre_speech_keep_ms=128)
        (kind, pcm), = _feed(ep, [QUIET] * 5 + [LOUD] * 2 + [QUIET] * 3)
        self.assertEqual(kind, endpointer.COMMITTED)
        self.assertEqual(pcm, QUIET * 2 + LOUD * 2 + QUIET)

    def test_short_burst_dropped_and_silence_keeps_running(self):
        ep = _ep(min_speech_ms=192)
        self.assertEqual(_feed(ep, [LOUD] * 2 + [QUIET] * 3), [(endpointer.BURST, None)])
        self.assertEqual(ep.silence, 3, "noise must not postpone auto-idle")

    def test_no_voice_dropped(self):
        ep = _ep(_Vad(False), speech_filter={"min_voiced_ms": 200})
        self.assertEqual(_feed(ep, [LOUD] * 6 + [QUIET] * 3), [(endpointer.NO_VOICE, None)])

    def test_voice_committed(self):
        ep = _ep(_Vad(True), speech_filter={"min_voiced_ms": 200})
        self.assertEqual(_feed(ep, [LOUD] * 6 + [QUIET] * 3)[0][0], endpointer.COMMITTED)

    def test_too_long_utterance(self):
        ep = _ep(max_utterance_ms=640)  # 10 chunks
        self.assertEqual(_feed(ep, [LOUD] * 10), [(endpointer.TOO_LONG, None)])
        self.assertFalse(ep.in_speech, "the utterance is dropped, not kept")

    def test_force_commit(self):
        ep = _ep()
        self.assertIsNone(ep.force_commit(), "nothing to commit")
        _feed(ep, [LOUD] * 2)
        self.assertEqual(ep.force_commit(), (endpointer.COMMITTED, LOUD * 2))
        self.assertFalse(ep.in_speech)

    def test_reset_discards(self):
        ep = _ep()
        _feed(ep, [LOUD] * 2)
        ep.reset()
        self.assertFalse(ep.in_speech)
        self.assertEqual(_feed(ep, [QUIET] * 3), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
