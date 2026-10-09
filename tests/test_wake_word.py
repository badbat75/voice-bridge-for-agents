#!/usr/bin/env python3
"""Tests for wake-word activation — the energy gate, the phrase matcher, and
the VoiceBridge state transitions that `"activation": "wake_word"` adds.

Three groups, all hardware-free and network-free (Whistle is never loaded):

  1. `SpeechGate`   — silence never opens a segment; a loud burst followed by
                      `hang_ms` of quiet yields one segment that includes the
                      pre-roll; a click shorter than `min_loud_ms` is dropped;
                      a long burst is cut at `max_segment_ms`.

  2. `matches`      — the wake phrase is found through punctuation and case
                      ("Hey, Binary!") but only on word boundaries.

  3. Transitions    — wake mode boots idle-but-listening (armed, firmware
                      unmuted); auto-idle re-arms WITHOUT firmware-muting; an
                      HID mute disarms and firmware-mutes; a wake hit plays an
                      ack and resumes (gen bump, LED off); a hit while disarmed
                      is ignored; auto-idle says a goodbye clip and only then
                      unducks, but stays silent while a reply is in flight.
                      Button mode is unchanged: never armed, and auto-idle
                      firmware-mutes.

Run: .venv/bin/python tests/test_wake_word.py
"""

from __future__ import annotations

import array
import importlib.util
import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import wake_word  # noqa: E402


def _load_voice_bridge():
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_PROJECT_ROOT, "voice-bridge.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SR, _CHUNK = 16000, 1600  # 100 ms chunks keep the ms arithmetic readable


def _chunk(amplitude: int) -> bytes:
    return array.array("h", [amplitude, -amplitude] * (_CHUNK // 2)).tobytes()


QUIET, LOUD = _chunk(100), _chunk(10000)


def _gate(**overrides) -> wake_word.SpeechGate:
    wcfg = {**wake_word.DEFAULTS, **overrides}
    return wake_word.SpeechGate(_SR, _CHUNK, wcfg)


def _feed(gate, chunks) -> list[bytes]:
    return [seg for seg in (gate.feed(c) for c in chunks) if seg is not None]


# ---------------------------------------------------------------------------
# 1. SpeechGate
# ---------------------------------------------------------------------------
class SpeechGateTest(unittest.TestCase):
    def test_silence_never_opens(self):
        self.assertEqual(_feed(_gate(), [QUIET] * 50), [])

    def test_burst_yields_one_segment_with_preroll(self):
        # 3 quiet (pre-roll) + 5 loud + 3 quiet (hang closes it).
        segs = _feed(_gate(), [QUIET] * 10 + [LOUD] * 5 + [QUIET] * 3)
        self.assertEqual(len(segs), 1)
        self.assertEqual(len(segs[0]), (3 + 5 + 3) * _CHUNK * 2)

    def test_click_shorter_than_min_loud_is_dropped(self):
        self.assertEqual(_feed(_gate(), [QUIET] * 5 + [LOUD] * 2 + [QUIET] * 5), [])

    def test_long_burst_cut_at_max_segment(self):
        segs = _feed(_gate(), [LOUD] * 30)
        self.assertEqual(len(segs), 1)
        self.assertEqual(len(segs[0]), 30 * _CHUNK * 2)

    def test_short_dip_does_not_close(self):
        # A 200 ms dip between words (< 300 ms hang) keeps one segment.
        segs = _feed(_gate(), [LOUD] * 4 + [QUIET] * 2 + [LOUD] * 4 + [QUIET] * 3)
        self.assertEqual(len(segs), 1)


# ---------------------------------------------------------------------------
# 2. Phrase matcher
# ---------------------------------------------------------------------------
class MatchesTest(unittest.TestCase):
    def test_punctuation_and_case(self):
        for text in ("Hey, binary.", "Hey, Binary!", "hey binary", "ok HEY BINARY dimmi"):
            self.assertTrue(wake_word.matches(text, ["hey binary"]), text)

    def test_word_boundaries(self):
        for text in ("Thank you.", "Vielen Dank.", "hey binaryx", "they binary", ""):
            self.assertFalse(wake_word.matches(text, ["hey binary"]), text)

    def test_config_normalizes_phrases(self):
        wcfg = wake_word.wake_config({"wake_word": {"phrases": ["Hey, Binary!", "  "]}})
        self.assertEqual(wcfg["phrases"], ["hey binary"])


class ValidateLanguageTest(unittest.TestCase):
    def test_whistle_languages_and_autodetect_accepted(self):
        for lang in (*wake_word.WHISTLE_LANGUAGES, None, ""):
            wake_word.validate_language("wake_word.language", lang)

    def test_unsupported_language_rejected(self):
        for lang in ("pt", "ita", "EN"):
            with self.assertRaises(ValueError):
                wake_word.validate_language("wake_word.language", lang)

    def test_matches_whistle_package(self):
        try:
            from needle.agent.whistle import LANGUAGES
        except ImportError:
            self.skipTest("cactus-needle not installed")
        self.assertEqual(tuple(LANGUAGES), wake_word.WHISTLE_LANGUAGES)


class AckBankTest(unittest.TestCase):
    """Cold cache → synthesized and cached; warm cache → no TTS calls;
    a failed clip is retried until it lands; a truncated file isn't trusted;
    the cache is keyed to the voice, so a voice change re-downloads and
    prunes the old voice's clips."""

    def setUp(self):
        import tempfile
        self.cache = tempfile.mkdtemp()
        self.cfg = {"tts_sample_rate": 24000, "deepgram_key": "k", "deepgram_tts_model": "aura-2-livia-it"}

    def _bank(self, tts, cfg=None, kind="wake", phrases=("Dimmi.", "Eccomi.")):
        return wake_word.AckBank(cfg or self.cfg, phrases, ["deepgram"], self.cache,
                                 lambda name: tts, kind=kind)

    @staticmethod
    def _echo_tts():
        tts = mock.Mock()
        tts.synthesize.side_effect = lambda text: text.encode()
        return tts

    def _files(self):
        return sorted(os.path.relpath(os.path.join(d, f), self.cache)
                      for d, _, fs in os.walk(self.cache) for f in fs)

    def test_cold_then_warm_cache(self):
        tts = self._echo_tts()
        self._bank(tts).prepare()
        self.assertEqual(tts.synthesize.call_count, 2)
        warm = mock.Mock()
        bank = self._bank(warm)
        bank.prepare()
        warm.synthesize.assert_not_called()
        self.assertEqual(sorted(c[2] for c in bank._clips), [b"Dimmi.", b"Eccomi."])

    def test_clips_live_in_a_per_voice_folder(self):
        self._bank(self._echo_tts()).prepare()
        self.assertTrue(all(f.startswith("wake/deepgram-aura-2-livia-it/") for f in self._files()))

    def test_voice_change_redownloads_and_prunes_old_voice(self):
        self._bank(self._echo_tts()).prepare()
        tts = self._echo_tts()
        self._bank(tts, cfg={**self.cfg, "deepgram_tts_model": "aura-2-giulia-it"}).prepare()
        self.assertEqual(tts.synthesize.call_count, 2)
        files = self._files()
        self.assertEqual(len(files), 2)
        self.assertTrue(all("aura-2-giulia-it" in f for f in files))

    def test_voice_settings_change_redownloads(self):
        cfg = {**self.cfg, "elevenlabs_key": "k", "elevenlabs_voice": "v", "elevenlabs_model": "m",
               "elevenlabs_voice_settings": {"stability": 0.7}}
        mk = lambda c, t: wake_word.AckBank(c, ["Dimmi."], ["elevenlabs"], self.cache, lambda n: t)
        mk(cfg, self._echo_tts()).prepare()
        tts = self._echo_tts()
        mk({**cfg, "elevenlabs_voice_settings": {"stability": 0.3}}, tts).prepare()
        self.assertEqual(tts.synthesize.call_count, 1)

    def test_wake_and_sleep_banks_do_not_prune_each_other(self):
        self._bank(self._echo_tts(), kind="wake").prepare()
        self._bank(self._echo_tts(), kind="sleep", phrases=("A dopo.",)).prepare()
        warm = mock.Mock()
        self._bank(warm, kind="wake").prepare()
        warm.synthesize.assert_not_called()
        self.assertEqual(len(self._files()), 3)

    def test_failed_clip_is_retried(self):
        calls = {"Dimmi.": 0}
        def synth(text):
            if text == "Dimmi.":
                calls["Dimmi."] += 1
                if calls["Dimmi."] == 1:
                    raise OSError("network down")
            return text.encode()
        tts = mock.Mock()
        tts.synthesize.side_effect = synth
        bank = self._bank(tts)
        with mock.patch.object(wake_word, "RETRY_MIN_S", 0.01):
            bank.prepare()
        self.assertEqual(calls["Dimmi."], 2)
        self.assertEqual(len(bank._clips), 2)

    def test_stop_ends_retries(self):
        import threading
        tts = mock.Mock()
        tts.synthesize.side_effect = OSError("down")
        stop = threading.Event()
        stop.set()
        self._bank(tts).prepare(stop)  # one round, then returns
        self.assertEqual(tts.synthesize.call_count, 2)

    def test_empty_cache_file_is_resynthesized(self):
        tts = self._echo_tts()
        bank = self._bank(tts)
        path = bank._path("deepgram", "Dimmi.")
        os.makedirs(os.path.dirname(path))
        open(path, "wb").close()
        bank.prepare()
        self.assertEqual(tts.synthesize.call_count, 2)


# ---------------------------------------------------------------------------
# 3. VoiceBridge transitions
# ---------------------------------------------------------------------------
class _RecordingHid:
    def __init__(self):
        self.leds: list[bool] = []

    def set_led(self, muted: bool) -> None:
        self.leds.append(muted)


def _bridge(activation: str):
    vb = _load_voice_bridge()
    cfg = {
        "sample_rate": _SR, "chunk_size": 1024, "tts_sample_rate": 24000,
        "output_device": "null", "hid_mute_enabled": True,
        "activation": activation,
        "wake_word": {"acks": ["Dimmi."], "sleep_acks": ["A dopo."], "ack_providers": []},
    }
    hid = _RecordingHid()
    bridge = vb.VoiceBridge(cfg, stt=mock.Mock(), tts=mock.Mock(), hid=hid, deezer=mock.Mock())
    return bridge, hid, vb


class WakeTransitionsTest(unittest.TestCase):
    def test_wake_mode_boots_armed_and_idle(self):
        bridge, _, _ = _bridge("wake_word")
        self.assertTrue(bridge._wake_armed.is_set())
        self.assertFalse(bridge.recording.is_set())

    def test_auto_idle_rearms_without_firmware_mute(self):
        bridge, hid, _ = _bridge("wake_word")
        bridge._resume()
        hid.leds.clear()
        bridge._enter_idle("test")
        self.assertTrue(bridge._wake_armed.is_set())
        self.assertNotIn(True, hid.leds)

    def test_hid_mute_disarms_and_mutes(self):
        bridge, hid, _ = _bridge("wake_word")
        bridge._resume()
        bridge._on_hid_press()
        self.assertFalse(bridge._wake_armed.is_set())
        self.assertFalse(bridge.recording.is_set())
        self.assertEqual(hid.leds[-1], True)

    def test_wake_hit_plays_ack_then_resumes(self):
        bridge, hid, vb = _bridge("wake_word")
        bridge._wake_acks._clips.append(("elevenlabs", "Dimmi.", b"\0\0" * 10))
        gen = bridge._current_gen()
        with mock.patch.object(vb, "play_audio") as play:
            bridge._on_wake("Hey, binary.")
        play.assert_called_once_with("null", b"\0\0" * 10, 24000)
        self.assertTrue(bridge.recording.is_set())
        self.assertFalse(bridge._wake_armed.is_set())
        self.assertEqual(bridge._current_gen(), gen + 1)
        self.assertEqual(hid.leds[-1], False)

    def test_wake_without_acks_beeps(self):
        bridge, _, vb = _bridge("wake_word")
        with mock.patch.object(vb, "play_beep") as beep, mock.patch.object(vb, "play_audio") as play:
            bridge._on_wake("hey binary")
        beep.assert_called_once()
        play.assert_not_called()

    def test_wake_ignored_when_disarmed(self):
        bridge, _, vb = _bridge("wake_word")
        bridge._wake_armed.clear()
        with mock.patch.object(vb, "play_beep") as beep:
            bridge._on_wake("hey binary")
        beep.assert_not_called()
        self.assertFalse(bridge.recording.is_set())

    def _with_goodbye(self):
        bridge, hid, vb = _bridge("wake_word")
        bridge._sleep_acks._clips.append(("deepgram", "A dopo.", b"\1\1"))
        bridge._resume()
        return bridge, hid, vb

    def test_auto_idle_says_goodbye_then_unducks(self):
        bridge, _, vb = self._with_goodbye()
        order = []
        bridge.deezer.unduck.side_effect = lambda: order.append("unduck")
        with mock.patch.object(vb, "play_audio", side_effect=lambda *a: order.append("goodbye")):
            bridge._enter_idle("test")
            for t in [t for t in __import__("threading").enumerate() if t.name == "vb-goodbye"]:
                t.join(2)
        self.assertEqual(order, ["goodbye", "unduck"])

    def test_no_goodbye_while_reply_in_flight(self):
        bridge, _, vb = self._with_goodbye()
        bridge._worker_busy.set()
        with mock.patch.object(vb, "play_audio") as play:
            bridge._enter_idle("test")
        play.assert_not_called()
        bridge.deezer.unduck.assert_called_once()

    def test_button_mode_unchanged(self):
        bridge, hid, _ = _bridge("button")
        self.assertFalse(bridge._wake_armed.is_set())
        bridge._resume()
        bridge._enter_idle("test")
        self.assertFalse(bridge._wake_armed.is_set())
        self.assertEqual(hid.leds[-1], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
