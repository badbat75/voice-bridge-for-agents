#!/usr/bin/env python3
"""Tests for the shadow STT comparison (`stt_compare.SttComparer`).

Hardware-free and network-free: `wake_word.whistle_transcribe` is patched, so
Whistle is never loaded. Checks that `submit()` never blocks the worker (it
returns while Whistle is still busy, and drops rather than waits when the
queue is full), that each comparison becomes exactly one TSV row under a
header, and that tabs/newlines in either transcript are flattened.

Run: .venv/bin/python tests/test_stt_compare.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import stt_compare  # noqa: E402

PCM = b"\0\0" * 16000  # 1 s @ 16 kHz


def _wait_rows(path: str, n: int, timeout: float = 2.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            rows = open(path, encoding="utf-8").read().splitlines()
            if len(rows) >= n + 1:
                return rows
        time.sleep(0.01)
    return open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []


class SttComparerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "sub", "cmp.tsv")
        self.comparer = stt_compare.SttComparer({"log_path": self.path}, self.dir, "elevenlabs")

    def test_one_row_per_utterance_with_header(self):
        with mock.patch.object(stt_compare.wake_word, "whistle_transcribe", return_value="ciao\tcome va\n"):
            self.comparer.start()
            self.comparer.submit(PCM, 16000, "Ciao, come va?", 0.8)
            rows = _wait_rows(self.path, 1)
        self.assertEqual(rows[0], stt_compare.HEADER.rstrip("\n"))
        cols = rows[1].split("\t")
        self.assertEqual(cols[1:5], ["1.00", "elevenlabs", "0.80", "Ciao, come va?"])
        self.assertEqual(cols[6], "ciao come va")

    def test_submit_does_not_wait_for_whistle(self):
        release = threading.Event()
        with mock.patch.object(stt_compare.wake_word, "whistle_transcribe",
                               side_effect=lambda *a, **k: release.wait(5) and "x"):
            self.comparer.start()
            t0 = time.monotonic()
            for _ in range(stt_compare.MAX_PENDING + 5):  # overflow drops, never blocks
                self.comparer.submit(PCM, 16000, "x", 0.1)
            self.assertLess(time.monotonic() - t0, 0.5)
            release.set()

    def test_wrong_rate_skipped(self):
        self.comparer.submit(PCM, 48000, "x", 0.1)
        self.assertTrue(self.comparer._q.empty())


if __name__ == "__main__":
    unittest.main(verbosity=2)
