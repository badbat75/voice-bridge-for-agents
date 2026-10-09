#!/usr/bin/env python3
"""The recorder's mic-stream lifecycle, with a fake PyAudio.

The stream stays open through the pause for processing (the reply reopens
the mic seconds later) and for `_MIC_HOLD_S` after it stops being needed;
only then is it closed. Chunks read while held open are discarded.

Run: .venv/bin/python tests/test_recorder.py
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
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_PROJECT_ROOT, "voice-bridge.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VB = _load_voice_bridge()
S = VB.State


class _Stream:
    def __init__(self, log):
        self.log = log
        log.append("open")

    def read(self, n, exception_on_overflow=False):
        time.sleep(0.005)
        return b"\x00\x00" * n

    def close(self):
        self.log.append("close")


class _PA:
    log: list[str] = []

    def open(self, **_kw):
        return _Stream(_PA.log)

    def terminate(self):
        pass


class _Hid:
    def set_led(self, muted: bool) -> None:
        pass


def _wait_until(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


class RecorderHoldTest(unittest.TestCase):
    def setUp(self):
        _PA.log = []
        cfg = {"sample_rate": 16000, "chunk_size": 64, "tts_sample_rate": 24000,
               "output_device": "null", "hid_mute_enabled": False, "idle_timeout_ms": 0}
        self.bridge = VB.VoiceBridge(cfg, stt=mock.Mock(), tts=mock.Mock(), hid=_Hid(),
                                     deezer=mock.Mock())
        for p in (mock.patch.object(VB.pyaudio, "PyAudio", _PA),
                  mock.patch.object(VB, "find_input_device", lambda _pa: 0),
                  mock.patch.object(VB, "_MIC_HOLD_S", 0.3)):
            p.start()
            self.addCleanup(p.stop)
        self.t = threading.Thread(target=self.bridge._recorder_loop, daemon=True)
        self.t.start()
        self.addCleanup(self._stop)
        self.assertTrue(_wait_until(lambda: _PA.log == ["open"]))

    def _stop(self):
        self.bridge.shutdown_event.set()
        self.bridge.recording.set()
        self.t.join(2)

    def test_processing_keeps_the_stream_open_and_discards(self):
        self.bridge._set_state(S.PROCESSING, auto_idled=True)
        VB.VoiceBridge._drain_queue(self.bridge.audio_q)
        time.sleep(0.6)  # well past the hold
        self.assertEqual(_PA.log, ["open"], "no close/reopen during processing")
        self.assertTrue(self.bridge.audio_q.empty(), "held chunks are discarded")
        self.bridge._set_state(S.RECORDING)
        self.assertTrue(_wait_until(lambda: not self.bridge.audio_q.empty()))
        self.assertEqual(_PA.log, ["open"])

    def test_muted_closes_after_the_hold(self):
        self.bridge._set_state(S.MUTED)
        time.sleep(0.1)
        self.assertEqual(_PA.log, ["open"], "still held right after the mute")
        self.assertTrue(_wait_until(lambda: _PA.log == ["open", "close"]))

    def test_quick_resume_inside_the_hold_does_not_reopen(self):
        self.bridge._set_state(S.MUTED)
        time.sleep(0.1)
        self.bridge._set_state(S.RECORDING)
        time.sleep(0.5)
        self.assertEqual(_PA.log, ["open"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
