#!/usr/bin/env python3
"""Tests for the user-experience fixes from the 2026-10-09 session review.

  1. Thinking cue          — tick after `delay_ms` of dead air, a spoken
                             "un attimo" on the first `tool_call`, never a
                             cue behind the reply, cues don't duck music.
     Hybrid TTS            — `tts_first_sentence_early`: first sentence
                             synthesized before the gateway finishes, the
                             rest in one call.
  2. Playback pre-buffer   — the first aplay write carries
                             `playback_prebuffer_ms` of PCM.
  3. Barge-in              — HID press while a reply plays stops it and
                             listens; external callers are released.
  4. Speech filter         — webrtcvad drops loud non-voice before STT.
  5. Fuzzy wake word       — "hey binally" wakes, aliases match, unrelated
                             speech doesn't.
  6. Idle windows          — wider idle window after a reply, wider still
                             after a question.
  7. WS gateway failures   — retry only when the message never left,
                             honest message otherwise, quiet after partial.
  8. Turn timing log       — one line per turn with the stage timings.

No PyAudio, no Jabra, no network: aplay, the gateway and the providers
are faked.

Run: .venv/bin/python tests/test_ux_improvements.py
"""

from __future__ import annotations

import array
import importlib.util
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import bridge_config  # noqa: E402
import wake_word  # noqa: E402
from elevenlabs_voice import ElevenLabsVoice  # noqa: E402


def _load_voice_bridge():
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_PROJECT_ROOT, "voice-bridge.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VB = _load_voice_bridge()
_SR, _CHUNK, _TTS_RATE = 16000, 1024, 24000


def _wait_until(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _silence(n=_CHUNK) -> bytes:
    return b"\x00\x00" * n


def _tone(n=_CHUNK, amp=8000) -> bytes:
    return array.array("h", (int(amp * math.sin(2 * math.pi * 440 * i / _SR))
                             for i in range(n))).tobytes()


class _Hid:
    def __init__(self):
        self.leds: list[bool] = []

    def set_led(self, muted: bool) -> None:
        self.leds.append(muted)


def _cfg(**over) -> dict:
    base = {
        "sample_rate": _SR, "chunk_size": _CHUNK, "tts_sample_rate": _TTS_RATE,
        "output_device": "null", "hid_mute_enabled": True,
        "vad_rms_threshold": 1e6, "silence_timeout_ms": 192,
        "silence_keep_ms": 0, "pre_speech_keep_ms": 0, "idle_timeout_ms": 320,
        "gateway_backend": "zeroclaw_ws", "gateway_base_url": "http://gw",
        "gateway_token": "t",
    }
    base.update(over)
    return base


def _bridge(**over):
    hid = _Hid()
    bridge = VB.VoiceBridge(_cfg(**over), stt=mock.Mock(), tts=mock.Mock(),
                            hid=hid, deezer=mock.Mock())
    return bridge, hid


def _recording(bridge) -> None:
    """Put the bridge in RECORDING (as a resume would) and forget that LED
    write, so a test asserts only the transitions it drives."""
    bridge._set_state(type(bridge._state).RECORDING)
    for attr in ("set_led_calls", "leds"):
        getattr(bridge.hid, attr, []).clear()


class _FakeAplay:
    """Records writes; `wait` returns at once unless `hold` is set (then it
    blocks until kill(), like a real aplay still draining audio)."""

    instances: list["_FakeAplay"] = []

    def __init__(self, *_a, **_kw):
        self.writes: list[bytes] = []
        self.killed = threading.Event()
        self.hold = False
        outer = self

        class _Stdin:
            def write(self, data):
                outer.writes.append(data)

            def close(self):
                pass

            def fileno(self):
                raise OSError("fake")

        self.stdin = _Stdin()
        _FakeAplay.instances.append(self)

    def wait(self, timeout=None):
        if self.hold and not self.killed.wait(timeout):
            import subprocess
            raise subprocess.TimeoutExpired("aplay", timeout)
        return 0

    def poll(self):
        return 0

    def kill(self):
        self.killed.set()


def _drain_q(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


# ---------------------------------------------------------------------------
# 1. Thinking cue + hybrid TTS
# ---------------------------------------------------------------------------
class ThinkingCueTest(unittest.TestCase):
    def _cue(self, **tc):
        bridge, _ = _bridge(thinking_cue={"enabled": True, "delay_ms": 50,
                                          "repeat_ms": 100, **tc})
        return bridge, VB._ThinkingCue(bridge, bridge._current_gen())

    def test_disabled_cue_never_plays(self):
        bridge, _ = _bridge(thinking_cue={"enabled": False, "delay_ms": 0})
        cue = VB._ThinkingCue(bridge, 0).start()
        time.sleep(0.15)
        cue.stop()
        self.assertTrue(bridge.playback_q.empty())

    def test_tick_after_delay_then_repeats(self):
        bridge, cue = self._cue()
        cue.start()
        self.assertTrue(_wait_until(lambda: cue.cues_played >= 2, 1.0))
        cue.stop()
        items = [it for _g, it in _drain_q(bridge.playback_q)]
        self.assertTrue(all(isinstance(it, VB._ExternalUtterance) for it in items))
        self.assertTrue(all(it.cue and not it.duck for it in items),
                        "cues must not duck the music or count as replies")
        self.assertEqual(items[0].pcm, VB._make_tick_pcm(_TTS_RATE))

    def test_tool_call_plays_spoken_cue_once(self):
        bridge, cue = self._cue(delay_ms=10_000, repeat_ms=10_000)
        bridge._thinking_acks = mock.Mock()
        bridge._thinking_acks.pick.return_value = ("elevenlabs", "Un attimo.", b"\x07\x07")
        cue.start()
        cue.on_event("thinking", {})
        time.sleep(0.1)
        self.assertEqual(cue.cues_played, 0, "only tool_call triggers the spoken cue")
        cue.on_event("tool_call", {"text_before": True})
        time.sleep(0.1)
        self.assertEqual(cue.cues_played, 0, "no spoken cue when the model already spoke")
        cue.on_event("tool_call", {})
        cue.on_event("tool_call", {})
        self.assertTrue(_wait_until(lambda: cue.cues_played == 1))
        time.sleep(0.15)
        cue.stop()
        items = [it for _g, it in _drain_q(bridge.playback_q)]
        self.assertEqual([it.pcm for it in items], [b"\x07\x07"])
        self.assertEqual(items[0].label, "thinking-ack")

    def test_spoken_cue_after_a_wait_even_without_tool_call_then_still_phrases(self):
        bridge, cue = self._cue(delay_ms=10_000, repeat_ms=10_000,
                                speak_after_ms=60, still_after_ms=80)
        bridge._thinking_acks = mock.Mock()
        bridge._thinking_acks.pick.return_value = ("elevenlabs", "Un attimo.", b"\x07\x07")
        bridge._still_acks = mock.Mock()
        bridge._still_acks.pick.return_value = ("elevenlabs", "Ci sto ancora lavorando.", b"\x08\x08")
        cue.start()
        self.assertTrue(_wait_until(lambda: cue.cues_played >= 3, 1.0))
        cue.on_event("tool_call", {})      # already spoke: no second "Un attimo"
        time.sleep(0.03)
        cue.stop()
        items = [it for _g, it in _drain_q(bridge.playback_q)]
        self.assertEqual([it.pcm for it in items[:3]], [b"\x07\x07", b"\x08\x08", b"\x08\x08"])
        self.assertEqual([it.label for it in items[:2]], ["thinking-ack", "thinking-still"])
        self.assertEqual(bridge._thinking_acks.pick.call_count, 1)

    def test_tool_call_speaks_before_the_wait_and_resets_it(self):
        bridge, cue = self._cue(delay_ms=10_000, repeat_ms=10_000,
                                speak_after_ms=150, still_after_ms=0)
        bridge._thinking_acks = mock.Mock()
        bridge._thinking_acks.pick.return_value = ("elevenlabs", "Un attimo.", b"\x07\x07")
        cue.start()
        cue.on_event("tool_call", {})
        self.assertTrue(_wait_until(lambda: cue.cues_played == 1, 0.1))
        time.sleep(0.3)
        cue.stop()
        self.assertEqual(cue.cues_played, 1, "still_after_ms 0: nothing spoken after the first")

    def test_no_cue_after_stop(self):
        bridge, cue = self._cue(delay_ms=0)
        cue.stop()
        self.assertFalse(cue._enqueue(b"\x01\x01", "x"))
        cue.start()
        time.sleep(0.15)
        self.assertTrue(bridge.playback_q.empty())

    def test_worker_cues_during_slow_gateway_but_never_after_reply(self):
        bridge, _ = _bridge(thinking_cue={"enabled": True, "delay_ms": 50, "repeat_ms": 60})
        bridge.stt.transcribe.return_value = "metti musica"

        def slow_gateway(*_a, on_event=None, **_kw):
            on_event("tool_call", {"name": "player_play"})
            time.sleep(0.3)
            yield "Fatto."

        bridge.tts.synthesize_stream.side_effect = lambda it: (b"R" * 10 for _ in it)
        gen = bridge._current_gen()
        with mock.patch.object(VB, "gateway_chat_stream_zeroclaw_ws", slow_gateway):
            cue = VB._ThinkingCue(bridge, gen).start()
            bridge._run_turn(gen, b"\x00" * 3200, _SR, time.monotonic(), {}, cue)
            cue.stop()
        time.sleep(0.15)  # a late cue thread would enqueue here
        items = [it for _g, it in _drain_q(bridge.playback_q)]
        first_reply = next(i for i, it in enumerate(items) if it == b"R" * 10)
        self.assertGreater(first_reply, 0, "a tick should have played during the wait")
        self.assertFalse(any(isinstance(it, VB._ExternalUtterance)
                             for it in items[first_reply:]),
                         "no cue may land behind the reply")

    def test_cue_plays_through_player_without_ducking(self):
        bridge, _ = _bridge()
        bridge.playback_q.put((bridge._current_gen(), VB._ExternalUtterance(
            b"\x01\x01", threading.Event(), label="thinking-tick", duck=False, cue=True)))
        with mock.patch.object(VB, "_aplay_popen", side_effect=_FakeAplay):
            t = threading.Thread(target=bridge.player.run, daemon=True)
            t.start()
            self.assertTrue(_wait_until(lambda: bridge.playback_q.empty() and not bridge._is_playing()))
            time.sleep(0.05)
            bridge.shutdown_event.set()
            t.join(1)
        bridge.deezer.duck.assert_not_called()
        self.assertEqual(bridge._replies_since_resume, 0, "a cue is not a reply")


class HybridTtsTest(unittest.TestCase):
    def _voice(self, early=True):
        return ElevenLabsVoice(api_key="k", voice_id="v", tts_model="eleven_v3",
                               tts_whole_reply=True, tts_first_sentence_early=early)

    def _run(self, voice, deltas, log):
        def text_iter():
            for d in deltas:
                log.append(("delta", d))
                yield d

        class _TTS:
            def stream(self, **kw):
                log.append(("tts", kw["text"]))
                yield b"pcm:" + kw["text"].encode()

        client = mock.Mock(text_to_speech=_TTS())
        with mock.patch("elevenlabs_voice.elevenlabs.ElevenLabs", return_value=client):
            return list(voice.synthesize_stream(text_iter()))

    def test_first_sentence_before_rest_of_stream(self):
        log = []
        out = self._run(self._voice(), ["Ci penso io. ", "Ho trovato ", "la canzone. ", "Parte ora."], log)
        self.assertEqual([e for e in log if e[0] == "tts"],
                         [("tts", "Ci penso io."), ("tts", "Ho trovato la canzone. Parte ora.")])
        self.assertLess(log.index(("tts", "Ci penso io.")), log.index(("delta", "Ho trovato ")),
                        "first sentence must be synthesized before the rest arrives")
        self.assertEqual(len(out), 2)

    def test_breath_prefix_is_not_a_first_sentence(self):
        # The agent opens every reply with "... " — that must not be taken as
        # the first sentence (one TTS call for three dots, the real first
        # sentence then waits for the whole reply).
        log = []
        self._run(self._voice(), ["... ", "[calm] Ci penso ", "io. ", "Ho trovato la canzone."], log)
        self.assertEqual([e for e in log if e[0] == "tts"],
                         [("tts", "... [calm] Ci penso io."), ("tts", "Ho trovato la canzone.")])
        self.assertLess(log.index(("tts", "... [calm] Ci penso io.")),
                        log.index(("delta", "Ho trovato la canzone.")))

    def test_sentence_mode_keeps_prefix_with_first_sentence(self):
        voice = ElevenLabsVoice(api_key="k", voice_id="v", tts_model="eleven_v3")
        log = []
        self._run(voice, ["... [warm] ", "Ciao. ", "Dimmi."], log)
        self.assertEqual([e for e in log if e[0] == "tts"],
                         [("tts", "... [warm] Ciao."), ("tts", "Dimmi.")])

    def test_reply_without_boundary_is_one_call(self):
        log = []
        self._run(self._voice(), ["Fatto", " subito"], log)
        self.assertEqual([e for e in log if e[0] == "tts"], [("tts", "Fatto subito")])

    def test_tool_boundary_speaks_each_segment_as_it_ends(self):
        # "Chiamo il worker." then a 70 s tool call: the sentence must play
        # while the tool runs, not after it, in every HTTP mode.
        TB = VB.TOOL_BOUNDARY
        for voice in (self._voice(), self._voice(early=False),
                      ElevenLabsVoice(api_key="k", voice_id="v", tts_model="eleven_v3")):
            log = []
            self._run(voice, ["... Chiamo il ", "worker", TB, " Fatto, è acceso. ", "Serve altro?"], log)
            tts = [e for e in log if e[0] == "tts"]
            self.assertEqual(tts[0], ("tts", "... Chiamo il worker"))
            self.assertLess(log.index(tts[0]), log.index(("delta", " Fatto, è acceso. ")),
                            "the segment must be synthesized before the tool result arrives")
            self.assertEqual("".join(t for _k, t in tts[1:]).replace(" ", ""),
                             "Fatto,èacceso.Servealtro?")

    def test_flag_off_keeps_whole_reply(self):
        log = []
        self._run(self._voice(early=False), ["Uno. ", "Due."], log)
        self.assertEqual([e for e in log if e[0] == "tts"], [("tts", "Uno. Due.")])


# ---------------------------------------------------------------------------
# 2. Pre-buffer
# ---------------------------------------------------------------------------
class PrebufferTest(unittest.TestCase):
    def setUp(self):
        _FakeAplay.instances.clear()
        p = mock.patch.object(VB, "_aplay_popen", side_effect=_FakeAplay)
        p.start()
        self.addCleanup(p.stop)

    def test_first_write_holds_prebuffer_worth_of_pcm(self):
        bridge, _ = _bridge(playback_prebuffer_ms=100)  # 4800 bytes @ 24 kHz
        gen = bridge._current_gen()
        for _ in range(4):
            bridge.playback_q.put((gen, b"\x01" * 2000))
        bridge.playback_q.put((gen, VB._END_OF_UTTERANCE))
        self.assertTrue(bridge.player.play_streamed(b"\x01" * 1000, "null", _TTS_RATE, gen))
        writes = _FakeAplay.instances[0].writes
        self.assertGreaterEqual(len(writes[0]), 4800)
        self.assertEqual(sum(map(len, writes)), 9000, "no PCM lost")

    def test_short_reply_ends_inside_prebuffer(self):
        bridge, _ = _bridge(playback_prebuffer_ms=500)
        gen = bridge._current_gen()
        bridge.playback_q.put((gen, b"\x02" * 100))
        bridge.playback_q.put((gen, VB._END_OF_UTTERANCE))
        self.assertTrue(bridge.player.play_streamed(b"\x02" * 100, "null", _TTS_RATE, gen))
        self.assertEqual(_FakeAplay.instances[0].writes, [b"\x02" * 200])

    def test_slow_stream_starts_after_cap(self):
        bridge, _ = _bridge(playback_prebuffer_ms=50)
        gen = bridge._current_gen()
        t0 = time.monotonic()
        threading.Timer(1.6, lambda: bridge.playback_q.put((gen, VB._END_OF_UTTERANCE))).start()
        bridge.player.play_streamed(b"\x03" * 10, "null", _TTS_RATE, gen)
        self.assertEqual(_FakeAplay.instances[0].writes[0], b"\x03" * 10)
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_external_utterance_mid_reply_is_deferred_not_written(self):
        bridge, _ = _bridge()
        gen = bridge._current_gen()
        ext = VB._ExternalUtterance(b"\x09" * 4, threading.Event())
        bridge.playback_q.put((gen, ext))
        bridge.playback_q.put((gen, b"\x01" * 4))
        bridge.playback_q.put((gen, VB._END_OF_UTTERANCE))
        bridge.player.play_streamed(b"\x01" * 4, "null", _TTS_RATE, gen)
        self.assertEqual(_FakeAplay.instances[0].writes, [b"\x01" * 4, b"\x01" * 4])
        self.assertEqual(list(bridge.player.backlog), [(gen, ext)])


# ---------------------------------------------------------------------------
# 3. Barge-in
# ---------------------------------------------------------------------------
class BargeInTest(unittest.TestCase):
    def test_press_while_reply_plays_stops_it_and_listens(self):
        bridge, hid = _bridge()
        _recording(bridge)  # player un-idled for the reply
        proc = _FakeAplay()
        bridge.player.proc = proc
        gen0 = bridge._current_gen()
        ext = VB._ExternalUtterance(b"\x01", threading.Event())
        bridge.playback_q.put((gen0, b"rest of reply"))
        bridge.playback_q.put((gen0, ext))
        bridge.utterance_q.put((gen0, b"echo", _SR))

        bridge._on_hid_press()

        self.assertGreater(bridge._current_gen(), gen0)
        self.assertTrue(proc.killed.is_set(), "aplay must be killed")
        self.assertTrue(bridge.playback_q.empty())
        self.assertTrue(bridge.utterance_q.empty())
        self.assertTrue(ext.done.is_set(), "a blocked say_to_speaker must be released")
        self.assertTrue(bridge.recording.is_set(), "mic open right away")
        self.assertFalse(bridge._force_commit.is_set())
        self.assertEqual(bridge._state, VB.State.RECORDING)
        self.assertEqual(hid.leds, [], "LED was already off")

    def test_press_while_muted_and_reply_plays_also_stops(self):
        bridge, hid = _bridge()
        bridge.player.proc = _FakeAplay()
        bridge._on_hid_press()
        self.assertTrue(bridge.recording.is_set())
        self.assertEqual(hid.leds[-1], False)

    def test_press_while_processing_still_mutes(self):
        bridge, hid = _bridge()
        bridge._set_state(VB.State.PROCESSING, auto_idled=True)
        bridge.player.proc = _FakeAplay()  # a thinking tick is audible
        gen0 = bridge._current_gen()
        bridge._on_hid_press()
        self.assertEqual(bridge._current_gen(), gen0)
        self.assertEqual(hid.leds[-1], True)

    def test_player_thread_survives_barge_in(self):
        bridge, _ = _bridge()
        _FakeAplay.instances.clear()

        def popen(*a, **kw):
            p = _FakeAplay()
            p.hold = True
            return p

        with mock.patch.object(VB, "_aplay_popen", side_effect=popen):
            t = threading.Thread(target=bridge.player.run, daemon=True)
            t.start()
            gen = bridge._current_gen()
            bridge.playback_q.put((gen, b"\x01" * 100))
            bridge.playback_q.put((gen, VB._END_OF_UTTERANCE))
            self.assertTrue(_wait_until(bridge._is_playing))
            bridge._on_hid_press()
            self.assertTrue(_wait_until(lambda: not bridge._is_playing()))
            self.assertEqual(bridge._replies_since_resume, 0, "a cut reply is not a turn")
            # next reply still plays
            gen = bridge._current_gen()
            bridge.playback_q.put((gen, b"\x02" * 100))
            bridge.playback_q.put((gen, VB._END_OF_UTTERANCE))
            self.assertTrue(_wait_until(lambda: len(_FakeAplay.instances) == 2))
            _FakeAplay.instances[1].kill()
            bridge.shutdown_event.set()
            t.join(2)
        self.assertFalse(t.is_alive())


# ---------------------------------------------------------------------------
# 4. Speech filter
# ---------------------------------------------------------------------------
class _FakeVad:
    """Voiced iff the frame's first sample is non-zero."""

    def is_speech(self, frame, rate):
        return frame[:2] != b"\x00\x00"


class SpeechFilterTest(unittest.TestCase):
    def test_voiced_ms_counts_30ms_frames(self):
        frame = 480 * 2  # 30 ms at 16 kHz
        pcm = b"\x01\x00" * 480 * 3 + b"\x00\x00" * 480 * 2
        self.assertEqual(VB._voiced_ms(pcm, _SR, _FakeVad()), 90)
        self.assertEqual(len(pcm) // frame, 5)

    def test_unsupported_rate_skips_check(self):
        self.assertIsNone(VB._voiced_ms(b"\x01\x00" * 1000, 22050, _FakeVad()))

    def test_real_webrtcvad_finds_no_voice_in_silence(self):
        vad = VB._make_speech_vad({"speech_filter": {"enabled": True, "aggressiveness": 2}})
        self.assertIsNotNone(vad, "webrtcvad must be installed in the venv")
        self.assertEqual(VB._voiced_ms(_silence(16000), _SR, vad), 0)

    def test_disabled_filter_builds_no_vad(self):
        self.assertIsNone(VB._make_speech_vad({"speech_filter": {"enabled": False}}))

    def _run_endpointer(self, voiced: bool):
        bridge, _ = _bridge(idle_timeout_ms=5000,
                            speech_filter={"enabled": False, "min_voiced_ms": 200})
        bridge._speech_vad = mock.Mock()
        bridge._speech_vad.is_speech.return_value = voiced
        bridge.cfg["speech_filter"]["min_voiced_ms"] = 200
        _recording(bridge)
        t = threading.Thread(target=bridge._endpointer_loop, daemon=True)
        t.start()
        gen = bridge._current_gen()
        for c in [_tone()] * 6 + [_silence()] * 5:
            bridge.audio_q.put((gen, c))
        time.sleep(0.4)
        bridge.shutdown_event.set()
        t.join(1)
        return bridge

    def test_loud_non_voice_is_dropped_before_stt(self):
        bridge = self._run_endpointer(voiced=False)
        self.assertTrue(bridge.utterance_q.empty())
        self.assertTrue(bridge.recording.is_set(), "no processing pause for noise")
        bridge.deezer.unduck.assert_called()

    def test_voice_is_committed(self):
        bridge = self._run_endpointer(voiced=True)
        self.assertFalse(bridge.utterance_q.empty())


class NoReplySentinelTest(unittest.TestCase):
    def test_sentinel_with_breath_prefix_is_swallowed(self):
        for deltas in (["... NO_REPLY"], ["...", " NO_", "REPLY"], ["… NOREPLY\n"]):
            self.assertEqual(list(VB._filter_no_reply(iter(deltas))), [], deltas)

    def test_real_reply_with_prefix_passes(self):
        out = "".join(VB._filter_no_reply(iter(["... ", "[warm] Ciao ", "Gabriele!"])))
        self.assertEqual(out, "... [warm] Ciao Gabriele!")

    def test_is_no_reply(self):
        self.assertTrue(VB._is_no_reply("... NO_REPLY"))
        self.assertFalse(VB._is_no_reply("... No, replico dopo."))

    def test_worker_beeps_low_on_prefixed_sentinel(self):
        bridge, _ = _bridge()
        bridge.stt.transcribe.return_value = "parlano tra loro"
        bridge.tts.synthesize_stream.side_effect = lambda it: (b"X" for _ in it)

        def gw(*_a, **_kw):
            yield "... NO_REPLY"

        gen = bridge._current_gen()
        with mock.patch.object(VB, "gateway_chat_stream_zeroclaw_ws", gw):
            bridge._run_turn(gen, b"\x00" * 3200, _SR, time.monotonic(), {},
                             VB._ThinkingCue(bridge, gen))
        items = [it for _g, it in _drain_q(bridge.playback_q)]
        self.assertNotIn(b"X", items, "the sentinel must never reach TTS")
        self.assertEqual(items[0], VB._make_beep_pcm(_TTS_RATE, freq=220, duration=0.18))


# ---------------------------------------------------------------------------
# 5. Fuzzy wake word
# ---------------------------------------------------------------------------
class FuzzyWakeTest(unittest.TestCase):
    P = ["hey binary"]

    def test_close_mishearings_wake(self):
        for heard in ["Hey, binally.", "Okay Binary!", "hey bin ary"]:
            self.assertTrue(wake_word.matches(heard, self.P, fuzzy=0.8), heard)

    def test_unrelated_speech_does_not_wake(self):
        for heard in ["Hey, bud.", "Great!", "Good luck.", "Hey baby", "Hey Siri",
                      "I'm gonna stop her landing telephone.", "Yeah."]:
            self.assertFalse(wake_word.matches(heard, self.P, fuzzy=0.8), heard)

    def test_fuzzy_zero_is_exact_only(self):
        self.assertFalse(wake_word.matches("hey binally", self.P, fuzzy=0.0))

    def test_aliases_match_exactly(self):
        self.assertTrue(wake_word.matches("Hey, bye, Harry.", self.P, aliases=["hey bye harry"]))
        self.assertFalse(wake_word.matches("bye harry", self.P, aliases=["hey bye harry"]))

    def test_detector_uses_config_and_keeps_aliases_out_of_bias(self):
        wcfg = wake_word.wake_config({"wake_word": {"aliases": ["Hey, bye Ali"]}})
        det = wake_word.WakeDetector(wcfg)
        self.assertTrue(det.heard("hey bye ali"))
        self.assertTrue(det.heard("hey binally"))
        with mock.patch.object(wake_word, "whistle_transcribe", return_value="") as wt:
            det.transcribe(b"")
        self.assertEqual(wt.call_args.args[2], ["hey binary"])


# ---------------------------------------------------------------------------
# 6. Idle windows
# ---------------------------------------------------------------------------
class IdleWindowTest(unittest.TestCase):
    def test_question_detection(self):
        ea = VB._expects_answer
        self.assertTrue(ea("Vuoi che la metta?"))
        self.assertTrue(ea("Che ne dici? [curious]"))
        self.assertTrue(ea('Intendi "No Doubt?"'))
        self.assertFalse(ea("Fatto, parte ora."))
        self.assertFalse(ea(""))
        self.assertFalse(ea("... [warm] "))

    def test_question_in_the_last_two_sentences(self):
        ea = VB._expects_answer
        self.assertTrue(ea("Ho trovato la remix. Vuoi quella? Te la metto se dici sì."))
        self.assertFalse(ea("Era questa? No, era l'altra. L'ho messa. Buon ascolto."))

    def test_invitation_without_question_mark(self):
        ea = VB._expects_answer
        self.assertTrue(ea("... [cheerful] Ma se ti va una canzone, quella la so fare eccome. "
                           "Dimmi tu il titolo e la metto!"))
        self.assertTrue(ea("La frase si è persa a metà. Rifammi il pensiero, che ci sono."))
        self.assertFalse(ea("Dimmi pure cosa vuoi. Intanto metto la musica."))

    def test_resume_uses_its_own_window(self):
        bridge, _ = _bridge(idle_after_resume_ms=6000)
        bridge._resume()
        self.assertEqual(bridge._idle_window_ms, 6000)

    def test_after_reply_windows(self):
        bridge, _ = _bridge(idle_after_reply_ms=8000, idle_after_question_ms=12000)
        bridge._after_reply()
        self.assertEqual(bridge._idle_window_ms, 8000)
        bridge._reply_is_question = True
        bridge._after_reply()
        self.assertEqual(bridge._idle_window_ms, 12000)
        self.assertEqual(bridge._replies_since_resume, 2)
        bridge._resume()
        self.assertEqual(bridge._idle_window_ms, 320)

    def test_config_defaults_preserve_old_behaviour(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "vb.json")
            with open(path, "w") as f:
                json.dump({"idle_timeout_ms": 3000}, f)
            with mock.patch.object(bridge_config, "CONFIG_PATH", path), \
                    mock.patch.object(bridge_config, "SECRETS_PATH", os.path.join(d, "none")):
                cfg = VB.load_config()
        self.assertEqual(cfg["idle_after_reply_ms"], 3000)
        self.assertEqual(cfg["idle_after_question_ms"], 3000)
        self.assertEqual(cfg["playback_prebuffer_ms"], 0)
        self.assertEqual(cfg["idle_after_resume_ms"], 3000)
        self.assertEqual(cfg["max_utterance_ms"], 0)
        self.assertFalse(cfg["speech_filter"]["enabled"])
        self.assertFalse(cfg["thinking_cue"]["enabled"])
        self.assertFalse(cfg["tts_first_sentence_early"])

    def test_endpointer_waits_for_the_widened_window(self):
        bridge, _ = _bridge(idle_timeout_ms=320, idle_after_reply_ms=960)
        _recording(bridge)
        bridge._idle_window_ms = 960  # a reply just finished
        t = threading.Thread(target=bridge._endpointer_loop, daemon=True)
        t.start()
        gen = bridge._current_gen()
        for _ in range(8):  # ~512 ms: past 320, short of 960
            bridge.audio_q.put((gen, _silence()))
        time.sleep(0.3)
        self.assertTrue(bridge.recording.is_set(), "must not idle at the base window")
        for _ in range(10):
            bridge.audio_q.put((gen, _silence()))
        self.assertTrue(_wait_until(lambda: not bridge.recording.is_set()))
        bridge.shutdown_event.set()
        t.join(1)


class MaxUtteranceTest(unittest.TestCase):
    def test_side_conversation_is_dropped_and_goes_idle(self):
        bridge, hid = _bridge(max_utterance_ms=640, idle_timeout_ms=0)  # 10 chunks
        _recording(bridge)
        t = threading.Thread(target=bridge._endpointer_loop, daemon=True)
        t.start()
        gen = bridge._current_gen()
        for _ in range(12):
            bridge.audio_q.put((gen, _tone()))
        self.assertTrue(_wait_until(lambda: not bridge.recording.is_set()))
        self.assertTrue(bridge.utterance_q.empty(), "the chatter must never reach STT")
        self.assertEqual(hid.leds[-1], True)  # button mode: back to mute
        bridge.shutdown_event.set()
        t.join(1)

    def test_short_request_is_committed(self):
        bridge, _ = _bridge(max_utterance_ms=640, idle_timeout_ms=0)
        _recording(bridge)
        t = threading.Thread(target=bridge._endpointer_loop, daemon=True)
        t.start()
        gen = bridge._current_gen()
        for c in [_tone()] * 5 + [_silence()] * 4:
            bridge.audio_q.put((gen, c))
        self.assertTrue(_wait_until(lambda: not bridge.utterance_q.empty()))
        bridge.shutdown_event.set()
        t.join(1)


# ---------------------------------------------------------------------------
# Agent reports (report_to_user)
# ---------------------------------------------------------------------------
class AgentReportTest(unittest.TestCase):
    def _bridge(self):
        bridge, hid = _bridge(thinking_cue={"enabled": True, "delay_ms": 0, "repeat_ms": 100})
        bridge.tts.synthesize_stream.side_effect = lambda it: (b"R" * 10 for _ in it)
        self.sent = []

        def gateway(_url, _tok, text, *_a, **_kw):
            self.sent.append(text)
            yield "Aorus è acceso."

        p = mock.patch.object(VB, "gateway_chat_stream_zeroclaw_ws", gateway)
        p.start()
        self.addCleanup(p.stop)
        return bridge

    def _run_worker(self, bridge, until):
        t = threading.Thread(target=bridge._worker_loop, daemon=True)
        t.start()
        ok = _wait_until(until, 3.0)
        bridge.shutdown_event.set()
        t.join(2)
        return ok

    def test_report_becomes_a_turn_in_the_agent_session(self):
        bridge = self._bridge()
        State = type(bridge._state)
        bridge._set_state(State.MUTED)
        bridge.report("Aorus è acceso ed è in rete.", source="worker")
        self.assertTrue(self._run_worker(bridge, lambda: not bridge.playback_q.empty()))
        self.assertEqual(len(self.sent), 1)
        self.assertIn("worker", self.sent[0])
        self.assertIn("non dall'utente", self.sent[0])
        self.assertTrue(self.sent[0].endswith("Aorus è acceso ed è in rete."))
        bridge.stt.transcribe.assert_not_called()
        items = [it for _g, it in _drain_q(bridge.playback_q)]
        self.assertEqual(items[0], b"R" * 10, "no thinking cue before a report's reply")
        self.assertTrue(bridge._auto_idled, "the reply must be able to open the mic")
        bridge._unidle_for_reply()
        self.assertIs(bridge._state, State.RECORDING)

    def test_report_waits_while_the_user_is_talking(self):
        bridge = self._bridge()
        _recording(bridge)
        bridge._user_speaking.set()
        bridge.report("Fatto.", source="worker")
        self.assertIsNone(bridge._next_report())
        bridge._user_speaking.clear()
        bridge._set_state(type(bridge._state).PROCESSING, auto_idled=True)
        self.assertIsNone(bridge._next_report(), "a user turn in flight goes first")
        _recording(bridge)
        self.assertEqual(bridge._next_report(), ("worker", "Fatto."))

    def test_silent_agent_does_not_beep_on_a_report(self):
        bridge = self._bridge()
        with mock.patch.object(VB, "gateway_chat_stream_zeroclaw_ws",
                               lambda *a, **k: iter(["... NO_REPLY"])):
            bridge._run_report("worker", "niente da dire")
        items = [it for _g, it in _drain_q(bridge.playback_q)]
        self.assertEqual(items, [VB._END_OF_UTTERANCE])

    def test_empty_report_is_rejected(self):
        bridge = self._bridge()
        with self.assertRaises(ValueError):
            bridge.report("   ")


# ---------------------------------------------------------------------------
# 7. WS gateway failures
# ---------------------------------------------------------------------------
class _FakeWs:
    def __init__(self, frames, fail_at=None):
        self.frames = list(frames)
        self.fail_at = fail_at
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def send(self, data):
        self.sent.append(json.loads(data))

    def recv(self, timeout=None):
        if self.fail_at is not None and len(self.frames) <= self.fail_at:
            raise ConnectionError("no close frame received or sent")
        return json.dumps(self.frames.pop(0))


class WsGatewayTest(unittest.TestCase):
    def _run(self, connect, on_event=None):
        return list(VB.gateway_chat_stream_zeroclaw_ws(
            "http://gw", "tok", "ciao", "voice", "s", on_event=on_event, connect_fn=connect))

    def test_chunks_yielded_and_events_reported(self):
        ws = _FakeWs([{"type": "thinking", "content": "x"},
                      {"type": "tool_call", "name": "play"},
                      {"type": "chunk", "content": "Fatto."},
                      {"type": "done"}])
        events = []
        out = self._run(lambda *a, **k: ws, on_event=lambda t, f: events.append(t))
        self.assertEqual(out, ["Fatto."])
        self.assertEqual(events, ["thinking", "tool_call", "done"])
        self.assertEqual(ws.sent, [{"type": "message", "content": "ciao"}])

    def test_tool_call_after_text_flushes_a_sentence_boundary(self):
        # "Verifico subito." then a tool call: the whitespace yielded on the
        # tool_call lets the TTS sentence splitter commit the sentence now,
        # and the cue is told text already preceded the tool.
        ws = _FakeWs([{"type": "chunk", "content": "... Verifico subito."},
                      {"type": "tool_call", "name": "shell"},
                      {"type": "chunk", "content": " Fatto."},
                      {"type": "done"}])
        events = []
        out = self._run(lambda *a, **k: ws, on_event=lambda t, f: events.append((t, f.get("text_before"))))
        self.assertEqual(out, ["... Verifico subito.", VB.TOOL_BOUNDARY, " Fatto."])
        self.assertEqual(events, [("tool_call", True), ("done", None)])
        self.assertTrue(VB._is_no_reply("... NO_REPLY" + VB.TOOL_BOUNDARY))

    def test_connect_failure_is_retried_once(self):
        calls = []
        ws = _FakeWs([{"type": "chunk", "content": "Eccomi."}, {"type": "done"}])

        def connect(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("connection refused")
            return ws

        self.assertEqual(self._run(connect), ["Eccomi."])
        self.assertEqual(len(calls), 2)

    def test_connect_failing_twice_says_unreachable(self):
        calls = []

        def connect(*a, **k):
            calls.append(1)
            raise OSError("down")

        self.assertEqual(self._run(connect), [VB.GATEWAY_UNREACHABLE_REPLY])
        self.assertEqual(len(calls), 2)

    def test_drop_after_send_is_not_retried_and_says_so(self):
        calls = []

        def connect(*a, **k):
            calls.append(1)
            return _FakeWs([{"type": "tool_call"}], fail_at=0)

        self.assertEqual(self._run(connect), [VB.GATEWAY_LOST_REPLY])
        self.assertEqual(len(calls), 1, "the agent may already have acted: no retry")

    def test_drop_after_partial_answer_stays_quiet(self):
        ws = _FakeWs([{"type": "chunk", "content": "Allora, "}], fail_at=0)
        self.assertEqual(self._run(lambda *a, **k: ws), ["Allora, "])

    def test_error_frame_without_answer_is_spoken(self):
        ws = _FakeWs([{"type": "error", "message": "boom"}])
        self.assertEqual(self._run(lambda *a, **k: ws), [VB.GATEWAY_LOST_REPLY])


# ---------------------------------------------------------------------------
# 8. Turn timing log
# ---------------------------------------------------------------------------
class TurnTimingTest(unittest.TestCase):
    def test_worker_logs_stage_timings(self):
        bridge, _ = _bridge()
        bridge.stt.transcribe.return_value = "ciao"
        bridge.tts.synthesize_stream.side_effect = lambda it: (b"P" for _ in it)

        def gw(*_a, **_kw):
            yield "Ciao!"

        gen = bridge._current_gen()
        with mock.patch.object(VB, "gateway_chat_stream_zeroclaw_ws", gw), \
                self.assertLogs("voice-bridge", "INFO") as logs:
            cue = VB._ThinkingCue(bridge, gen)
            bridge._run_turn(gen, b"\x00" * 3200, _SR, time.monotonic(), {}, cue)
        line = next(m for m in logs.output if "Turn timing" in m)
        for key in ("stt=", "first_token=", "gateway_done=", "first_pcm="):
            self.assertIn(key, line)
        self.assertFalse(bridge._reply_is_question)


if __name__ == "__main__":
    unittest.main(verbosity=2)
