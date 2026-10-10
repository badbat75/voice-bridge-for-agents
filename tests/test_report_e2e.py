#!/usr/bin/env python3
"""End-to-end, offline: an async delegation leaves and its outcome returns.

Everything real except the edges: a real `VoiceBridge` (worker + player
threads), the real WebSocket gateway leg talking to a fake zeroclaw on
localhost, the real speaker socket and `SpeakerClient`, the real MCP tool
function `report_to_user`. Faked: STT, TTS, aplay, the HID device.

The scenario is the one this exists for:

    user: "accendi il computer"
    voice agent: "Sto accendendo, ti avviso."  + delegate(background=true)
                                         ... the turn ENDS here ...
    worker (later): report_to_user("Aorus è acceso")
    voice agent, same session: "Aorus è acceso."

Run: .venv/bin/python tests/test_report_e2e.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from websockets.sync.server import serve as ws_serve  # noqa: E402

import mcp_voice_server  # noqa: E402
import speaker_socket  # noqa: E402


def _load_voice_bridge():
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_PROJECT_ROOT, "voice-bridge.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VB = _load_voice_bridge()
State = VB.State
_TTS_RATE = 24000


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


class _Hid:
    def __init__(self):
        self.leds = []

    def set_led(self, muted):
        self.leds.append(muted)


class _FakeAplay:
    """Collects what would have been played."""
    played: list[bytes] = []

    def __init__(self, *_a, **_kw):
        class _Stdin:
            def write(self, data):
                _FakeAplay.played.append(bytes(data))

            def close(self):
                pass

            def fileno(self):
                raise OSError("fake")
        self.stdin = _Stdin()

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0

    def kill(self):
        pass


class _FakeZeroclaw:
    """A `/ws/chat` endpoint: one message per connection, scripted frames.
    `script(content) -> list of frames`; a float in the list is a sleep."""

    def __init__(self, script):
        self.script = script
        self.turns: list[dict] = []   # {"session", "agent", "content", "t"}
        self.server = ws_serve(self._handle, "127.0.0.1", 0)
        self.port = self.server.socket.getsockname()[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handle(self, ws):
        q = parse_qs(urlsplit(ws.request.path).query)
        msg = json.loads(ws.recv())
        self.turns.append({"session": q["session_id"][0], "agent": q["agent"][0],
                           "content": msg["content"], "t": time.monotonic()})
        for frame in self.script(msg["content"]):
            if isinstance(frame, float):
                time.sleep(frame)
            else:
                ws.send(json.dumps(frame))

    def close(self):
        self.server.shutdown()


def _script(content: str):
    if "non dall'utente" in content:           # the worker's report
        return [{"type": "chunk", "content": "... Fatto, Aorus è acceso. Serve altro?"},
                {"type": "done"}]
    return [{"type": "chunk", "content": "... Sto accendendo, ti avviso."},
            {"type": "tool_call", "name": "delegate",
             "args": {"agent": "worker", "background": True, "prompt": "accendi Aorus"}},
            {"type": "tool_result", "name": "delegate", "output": "task_id=42 running"},
            {"type": "done"}]


class AsyncDelegationE2E(unittest.TestCase):
    def setUp(self):
        _FakeAplay.played = []
        self.gw = _FakeZeroclaw(_script)
        self.addCleanup(self.gw.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        sock = os.path.join(self.tmp.name, "vb", "speaker.sock")

        cfg = {
            "sample_rate": 16000, "chunk_size": 1024, "tts_sample_rate": _TTS_RATE,
            "output_device": "null", "hid_mute_enabled": True,
            "vad_rms_threshold": 1e6, "silence_timeout_ms": 192, "idle_timeout_ms": 0,
            "gateway_backend": "zeroclaw_ws", "gateway_agent": "voice",
            "session_key": "agent:main:voice-bridge",
            "gateway_base_url": f"http://127.0.0.1:{self.gw.port}", "gateway_token": "t",
            "thinking_cue": {"enabled": False},
        }
        stt = mock.Mock()
        stt.transcribe.return_value = "accendi il computer"
        tts = mock.Mock()
        # One PCM blob per TTS segment, tagged with its text, so the test can
        # read what was spoken off the fake aplay.
        self.segments: list[str] = []

        def synth(text_iter):
            buf = ""
            for d in text_iter:
                if d == VB.TOOL_BOUNDARY:
                    if buf.strip():
                        self.segments.append(buf.strip())
                        yield buf.strip().encode()
                    buf = ""
                else:
                    buf += d
            if buf.strip():
                self.segments.append(buf.strip())
                yield buf.strip().encode()

        tts.synthesize_stream.side_effect = synth
        self.bridge = VB.VoiceBridge(cfg, stt=stt, tts=tts, hid=_Hid(), deezer=mock.Mock())

        p = mock.patch.object(VB, "_aplay_popen", side_effect=_FakeAplay)
        p.start()
        self.addCleanup(p.stop)
        for target in (self.bridge._worker_loop, self.bridge.player.run):
            threading.Thread(target=target, daemon=True).start()
        self.addCleanup(self.bridge.shutdown_event.set)

        server = speaker_socket.serve(sock, self.bridge.play_pcm, poll_interval=0.1,
                                      report=self.bridge.report)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        # The MCP tool, wired the way `mcp_voice_server.main()` wires it.
        old = mcp_voice_server._bridge
        mcp_voice_server._bridge = speaker_socket.SpeakerClient(sock, timeout=5)
        self.addCleanup(setattr, mcp_voice_server, "_bridge", old)

    def _spoken(self) -> str:
        return b" | ".join(_FakeAplay.played).decode(errors="replace")

    def _user_says(self):
        """A committed utterance, as the endpointer hands it over."""
        b = self.bridge
        b._set_state(State.RECORDING)
        gen = b._current_gen()
        self.assertTrue(b._enqueue_utterance(gen, b"\x00\x00" * 16000, 16000))
        b._pause_for_processing()

    def _quiet(self):
        b = self.bridge
        return _wait_until(lambda: not b._worker_busy.is_set() and not b._reply_playing()
                           and b.utterance_q.empty())

    def test_departure_then_return_in_the_same_session(self):
        b = self.bridge
        t0 = time.monotonic()
        self._user_says()

        # --- departure: the announcement is spoken and the turn is over
        self.assertTrue(_wait_until(lambda: "Sto accendendo" in self._spoken()))
        t_announce = time.monotonic()
        self.assertTrue(self._quiet())
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertEqual(len(self.gw.turns), 1)
        self.assertEqual(self.gw.turns[0]["content"], "accendi il computer")
        self.assertNotIn("Aorus è acceso", self._spoken())
        self.assertIs(b._state, State.RECORDING, "the user can keep talking meanwhile")

        # --- return: the worker reports through the MCP tool
        t_report = time.monotonic()
        res = mcp_voice_server.report_to_user("Aorus è acceso ed è in rete.", source="worker")
        self.assertEqual(res, {"ok": True, "queued": True})
        self.assertLess(time.monotonic() - t_report, 0.5, "the tool must not wait for the voice")
        self.assertTrue(_wait_until(lambda: "Aorus è acceso" in self._spoken()),
                        f"spoken so far: {self._spoken()!r}")
        t_spoken = time.monotonic()
        self.assertEqual(len(self.gw.turns), 2)
        first, second = self.gw.turns
        self.assertEqual(second["session"], first["session"], "same conversation")
        self.assertEqual(second["agent"], "voice")
        self.assertIn("«worker»", second["content"])
        self.assertTrue(second["content"].endswith("Aorus è acceso ed è in rete."))
        self.assertEqual(b.stt.transcribe.call_count, 1, "a report skips STT")
        self.assertEqual(self.segments, ["... Sto accendendo, ti avviso.",
                                         "... Fatto, Aorus è acceso. Serve altro?"])
        # Bridge overhead with instant STT / gateway / TTS: what the bridge
        # itself adds to each leg. Budgets are loose (Pi under load).
        print("\n  bridge overhead (fakes are instant):")
        print(f"    user turn: commit → spoken   {t_announce - t0:6.3f} s")
        print(f"    report: tool call → agent    {second['t'] - t_report:6.3f} s")
        print(f"    report: tool call → spoken   {t_spoken - t_report:6.3f} s")
        self.assertLess(t_announce - t0, 1.0)
        self.assertLess(second["t"] - t_report, 1.5, "delivery poll is 0.2 s / 1 s idle")
        self.assertLess(t_spoken - t_report, 2.0)

    def test_announcement_is_spoken_before_a_blocking_delegation_ends(self):
        # The old failure: "Chiamo il worker" came out after the worker was
        # done. Even if the agent delegates synchronously, what it said
        # before the tool call must be audible while the tool runs.
        self.gw.script = lambda _c: [
            {"type": "chunk", "content": "... Chiamo subito il worker"},
            {"type": "tool_call", "name": "delegate", "args": {"agent": "worker"}},
            1.0,
            {"type": "chunk", "content": " Fatto, è acceso."},
            {"type": "done"}]
        self._user_says()
        self.assertTrue(_wait_until(lambda: "Chiamo subito il worker" in self._spoken(), 0.8),
                        "the announcement must not wait for the tool result")
        self.assertNotIn("Fatto", self._spoken())
        self.assertTrue(_wait_until(lambda: "Fatto, è acceso." in self._spoken(), 3.0))

    def test_report_waits_for_the_users_turn_and_is_not_lost(self):
        b = self.bridge
        self.gw.script = lambda c: ([0.5] if "non dall'utente" not in c else []) + _script(c)
        self._user_says()
        self.assertTrue(_wait_until(lambda: len(self.gw.turns) == 1))
        mcp_voice_server.report_to_user("Aorus è acceso.", source="worker")  # mid-turn
        self.assertTrue(_wait_until(lambda: len(self.gw.turns) == 2))
        self.assertIn("Sto accendendo", self._spoken(),
                      "the user's reply is spoken before the report is delivered")
        self.assertGreaterEqual(self.gw.turns[1]["t"] - self.gw.turns[0]["t"], 0.5)
        self.assertTrue(_wait_until(lambda: "Aorus è acceso" in self._spoken()))

    def test_report_survives_a_resume_and_wakes_an_idle_bridge(self):
        b = self.bridge
        b._set_state(State.IDLE_LISTENING)
        b._user_speaking.set()                   # someone is mid-sentence
        mcp_voice_server.report_to_user("La lavatrice ha finito.", source="worker")
        time.sleep(0.4)
        self.assertEqual(self.gw.turns, [], "never talk over the user")
        b._gen += 1                              # what a wake/HID resume does
        b._user_speaking.clear()
        self.assertTrue(_wait_until(lambda: len(self.gw.turns) == 1))
        self.assertTrue(_wait_until(lambda: "Aorus è acceso" in self._spoken()))
        self.assertTrue(_wait_until(lambda: b._state is State.RECORDING),
                        "the reply opens the mic so the user can answer")

    def test_bridge_down_is_an_error_the_worker_can_see(self):
        mcp_voice_server._bridge = speaker_socket.SpeakerClient(
            os.path.join(self.tmp.name, "gone.sock"))
        with self.assertRaisesRegex(RuntimeError, "not running"):
            mcp_voice_server.report_to_user("x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
