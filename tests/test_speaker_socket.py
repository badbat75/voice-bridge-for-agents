#!/usr/bin/env python3
"""The bridge's local speaker socket (speaker_socket.py): the MCP server's
`say_to_speaker` reaches `VoiceBridge.play_pcm` through it.

Run: .venv/bin/python tests/test_speaker_socket.py
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import speaker_socket  # noqa: E402


class SpeakerSocketTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "vb", "speaker.sock")
        self.calls = []

        def play_pcm(pcm, *, text=None):
            self.calls.append((pcm, text))
            if text == "boom":
                raise RuntimeError("player exploded")
            return len(pcm) / 48000

        self.reports = []
        self.server = speaker_socket.serve(
            self.path, play_pcm, poll_interval=0.1,
            report=lambda text, source="agent": self.reports.append((source, text)))
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = speaker_socket.SpeakerClient(self.path, timeout=5)

    def test_round_trip_returns_after_play(self):
        pcm = b"\x01\x02" * 24000
        self.assertAlmostEqual(self.client.play_pcm(pcm, text="Ciao?"), 1.0)
        self.assertEqual(self.calls, [(pcm, "Ciao?")])

    def test_report_is_handed_over_without_pcm(self):
        self.client.report("Aorus è acceso", source="worker")
        self.assertEqual(self.reports, [("worker", "Aorus è acceso")])
        self.assertEqual(self.calls, [])

    def test_report_refused_when_bridge_takes_none(self):
        self.server.report = None
        with self.assertRaisesRegex(RuntimeError, "no agent reports"):
            self.client.report("x")

    def test_socket_is_private(self):
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode) & 0o077, 0)

    def test_player_error_reaches_the_caller(self):
        with self.assertRaisesRegex(RuntimeError, "player exploded"):
            self.client.play_pcm(b"\x00\x00", text="boom")

    def test_bridge_not_running(self):
        client = speaker_socket.SpeakerClient(os.path.join(self.dir.name, "nope.sock"))
        with self.assertRaisesRegex(RuntimeError, "not running"):
            client.play_pcm(b"\x00\x00")

    def test_stale_socket_file_is_replaced(self):
        self.server.shutdown()
        self.server.server_close()
        server = speaker_socket.serve(self.path, lambda pcm, text=None: 0.5, poll_interval=0.1)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.assertEqual(self.client.play_pcm(b"\x00\x00"), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
