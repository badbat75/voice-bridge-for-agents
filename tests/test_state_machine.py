#!/usr/bin/env python3
"""The bridge's state machine, pinned as one table (state × event → state).

Each row puts a fresh `VoiceBridge` in a start state (plus the
`auto_idled` bit), fires one event through the real transition method and
checks the end state, the bit, the derived `recording` / `_wake_armed`
Events and the LED writes (one write iff the firmware mute changed). The
same table is in AGENTS.md — keep the two in step.

Run: .venv/bin/python tests/test_state_machine.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
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
REC, PROC, IDLE, MUTED = S.RECORDING, S.PROCESSING, S.IDLE_LISTENING, S.MUTED


class _Hid:
    def __init__(self):
        self.leds: list[bool] = []

    def set_led(self, muted: bool) -> None:
        self.leds.append(muted)


def _bridge(activation: str):
    cfg = {
        "sample_rate": 16000, "chunk_size": 1024, "tts_sample_rate": 24000,
        "output_device": "null", "hid_mute_enabled": True, "idle_timeout_ms": 320,
        "activation": activation,
        "wake_word": {"acks": [], "sleep_acks": []},
    }
    bridge = VB.VoiceBridge(cfg, stt=mock.Mock(), tts=mock.Mock(), hid=_Hid(),
                            deezer=mock.Mock())
    # No audio: acks, goodbyes and aplay are out of scope here.
    bridge._play_wake_ack = bridge._say_goodbye = lambda _clip: None
    bridge._kill_player = lambda: None
    return bridge


def _press_while_playing(bridge):
    with mock.patch.object(bridge, "_reply_playing", return_value=True):
        bridge._on_hid_press()


EVENTS = {
    "press": lambda b: b._on_hid_press(),
    "press_playing": _press_while_playing,
    "silence": lambda b: b._enter_idle("test"),
    "wake": lambda b: b._on_wake("hey binary"),
    "commit": lambda b: b._pause_for_processing(),
    "reply_starts": lambda b: b._unidle_for_reply(),
    "turn_ends_silent": lambda b: b._resume_after_processing(),
    "say_to_speaker": lambda b: b._unmute_for_external(),
    "whistle_fails": lambda b: b._disable_wake(),
}

# (activation, start, start auto_idled, event) → (end, end auto_idled)
TABLE = [
    ("button", REC, False, "press", MUTED, False),          # commit-and-mute
    ("button", REC, False, "press_playing", REC, False),    # stop reply, listen
    ("button", REC, False, "silence", MUTED, True),
    ("button", REC, False, "commit", PROC, True),
    ("button", REC, False, "reply_starts", REC, False),
    ("button", REC, False, "say_to_speaker", REC, False),
    ("button", PROC, True, "press", MUTED, False),          # privacy mute, reply still plays
    ("button", PROC, True, "press_playing", MUTED, False),
    ("button", PROC, True, "silence", PROC, True),
    ("button", PROC, True, "reply_starts", REC, False),
    ("button", PROC, True, "turn_ends_silent", REC, False),
    ("button", PROC, True, "say_to_speaker", REC, False),
    ("button", MUTED, False, "press", REC, False),
    ("button", MUTED, False, "press_playing", REC, False),
    ("button", MUTED, False, "reply_starts", MUTED, False),  # explicit mute wins
    ("button", MUTED, True, "reply_starts", REC, False),     # auto-idle resumes
    ("button", MUTED, True, "press", REC, False),
    ("button", MUTED, False, "silence", MUTED, False),
    ("button", MUTED, False, "commit", MUTED, False),
    ("button", MUTED, False, "turn_ends_silent", MUTED, False),
    ("button", MUTED, False, "say_to_speaker", REC, False),
    ("wake_word", REC, False, "silence", IDLE, True),
    ("wake_word", REC, False, "press", MUTED, False),
    ("wake_word", REC, False, "wake", REC, False),
    ("wake_word", REC, False, "whistle_fails", REC, False),
    ("wake_word", IDLE, False, "wake", REC, False),
    ("wake_word", IDLE, True, "wake", REC, False),
    ("wake_word", IDLE, False, "press", REC, False),
    ("wake_word", IDLE, True, "reply_starts", REC, False),
    ("wake_word", IDLE, False, "reply_starts", IDLE, False),
    ("wake_word", IDLE, False, "say_to_speaker", REC, False),
    ("wake_word", IDLE, True, "whistle_fails", MUTED, True),
    ("wake_word", MUTED, False, "wake", MUTED, False),       # privacy mute: button only
    ("wake_word", MUTED, False, "press", REC, False),
    ("wake_word", PROC, True, "press", MUTED, False),
    ("wake_word", PROC, True, "wake", PROC, True),
]


class StateTableTest(unittest.TestCase):
    def test_table(self):
        for activation, start, auto, event, end, end_auto in TABLE:
            with self.subTest(activation=activation, start=start.name, auto=auto, event=event):
                bridge = _bridge(activation)
                bridge._set_state(start, auto_idled=auto)
                bridge.hid.leds.clear()
                EVENTS[event](bridge)
                self.assertEqual(bridge._state, end)
                self.assertEqual(bridge._auto_idled, end_auto)
                records, armed, muted = VB._STATE_EFFECTS[end]
                self.assertEqual(bridge.recording.is_set(), records)
                self.assertEqual(bridge._wake_armed.is_set(), armed)
                changed = muted != VB._STATE_EFFECTS[start][2]
                self.assertEqual(bridge.hid.leds, [muted] if changed else [])

    def test_boot_states(self):
        self.assertEqual(_bridge("button")._state, MUTED)
        self.assertEqual(_bridge("wake_word")._state, IDLE)

    def test_idle_listening_keeps_firmware_mic_open(self):
        """Red means privacy mute only."""
        self.assertEqual([s for s in S if VB._STATE_EFFECTS[s][2]], [MUTED])


if __name__ == "__main__":
    unittest.main(verbosity=1)
