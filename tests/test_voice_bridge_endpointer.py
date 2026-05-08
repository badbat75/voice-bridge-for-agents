#!/usr/bin/env python3
"""Tests for VoiceBridge's endpointer (sound gate), commit/idle timings,
and the HID-press mute trigger.

Three groups, exercising the three pieces of state-machine logic that
matter most for end-user feel — all on synthetic audio + a stub HID,
no PyAudio, no Jabra, no network.

  1. Sound gate (RMS VAD)        — `_endpointer_loop` distinguishes
                                    silence from speech using the
                                    legacy `sum(s²)/sqrt(N)` metric;
                                    only the speech path emits PCM.

  2. Activation/deactivation     — commit fires after `silence_timeout_ms`
     timings                       of trailing silence; auto-idle fires
                                    after `idle_timeout_ms` of pure
                                    silence (and is blocked while a
                                    reply is still playing).

  3. Mute trigger (HID press)    — `_on_hid_press` flips the bridge
                                    between recording and idle. While
                                    recording: soft mute — endpointer
                                    force-commits any in-progress speech
                                    buffer (so the user's last utterance
                                    is still sent), recorder closes,
                                    LED on. The worker/TTS/player keep
                                    flowing; queues are NOT drained,
                                    gen is NOT bumped. While idle:
                                    resume — bumps gen (sheds
                                    stragglers), sets recording, LED
                                    off, clears `_auto_idled`.

Run: .venv/bin/python tests/test_voice_bridge_endpointer.py
"""

from __future__ import annotations

import array
import importlib.util
import math
import os
import queue
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
    importlib so the tests can construct the real `VoiceBridge` class."""
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_PROJECT_ROOT, "voice-bridge.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic PCM helpers — match the recorder's S16LE mono format.
# ---------------------------------------------------------------------------
_SAMPLE_RATE = 16000
_CHUNK_SIZE = 1024  # ~64 ms at 16 kHz, same as production


def _silence_chunk(n: int = _CHUNK_SIZE) -> bytes:
    """All-zeros PCM. RMS metric (sum(s²)/sqrt(N)) is 0 — well below
    any sensible threshold."""
    return b"\x00\x00" * n


def _speech_chunk(amp: int = 8000, freq: float = 440.0,
                  n: int = _CHUNK_SIZE, sr: int = _SAMPLE_RATE) -> bytes:
    """Sine wave loud enough to clear a 1e6 RMS-metric threshold by
    ~3 orders of magnitude (amp=8000 → metric ≈ 1e9 with N=1024)."""
    arr = array.array("h")
    for i in range(n):
        val = int(amp * math.sin(2 * math.pi * freq * i / sr))
        arr.append(max(-32768, min(32767, val)))
    return arr.tobytes()


def _quiet_chunk(amp: int = 100, n: int = _CHUNK_SIZE) -> bytes:
    """Below-threshold sine: present, but the RMS metric stays ~1e5,
    still under the 1e6 default threshold. Used to confirm the gate
    rejects low-energy noise rather than just rejecting digital zero."""
    return _speech_chunk(amp=amp, n=n)


# ---------------------------------------------------------------------------
# Test fixture: minimal cfg + stub HID + a way to drive `_endpointer_loop`
# in a daemon thread without bringing up PyAudio.
# ---------------------------------------------------------------------------
def _cfg(**overrides) -> dict:
    """Default cfg: 16 kHz / 1024-chunk recorder, threshold tuned to
    accept `_speech_chunk(amp=8000)` and reject everything quieter.
    Timings short enough to keep the suite under a few seconds."""
    base = {
        "sample_rate": _SAMPLE_RATE,
        "chunk_size": _CHUNK_SIZE,
        "vad_rms_threshold": 1e6,
        # 192 ms ≈ 3 chunks; commit fires after 3 contiguous silent chunks.
        "silence_timeout_ms": 192,
        # Trim trailing silence aggressively so commit math is predictable.
        "silence_keep_ms": 0,
        "pre_speech_keep_ms": 0,
        # 320 ms ≈ 5 chunks; idle fires after 5 silent chunks (no speech).
        "idle_timeout_ms": 320,
        "hid_mute_enabled": True,
    }
    base.update(overrides)
    return base


class _StubHid:
    """In-memory stand-in for `HidMuteMonitor`. Records every `set_led`
    call so tests can assert the LED/firmware-mute flips at the right
    transitions."""

    def __init__(self) -> None:
        self.set_led_calls: list[bool] = []

    def set_led(self, muted: bool) -> None:
        self.set_led_calls.append(muted)


def _make_bridge(cfg: dict) -> "tuple":
    vb = _load_voice_bridge()
    hid = _StubHid()
    bridge = vb.VoiceBridge(cfg, stt=mock.Mock(), tts=mock.Mock(), hid=hid)
    return bridge, hid, vb


def _start_endpointer(bridge) -> threading.Thread:
    t = threading.Thread(target=bridge._endpointer_loop, daemon=True)
    t.start()
    return t


def _stop_endpointer(bridge, t: threading.Thread) -> None:
    bridge.shutdown_event.set()
    t.join(timeout=1.5)


def _push(bridge, gen: int, chunks: list[bytes]) -> None:
    for c in chunks:
        bridge.audio_q.put((gen, c))


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# 1. Sound gate (RMS VAD)
# ---------------------------------------------------------------------------
class SoundGateTest(unittest.TestCase):
    """The endpointer's first job: distinguish silence from speech.
    These tests pin the VAD's contract — pure silence never commits,
    speech-then-silence does, and a quiet-but-nonzero chunk is still
    treated as silence (it's below the threshold)."""

    def test_silence_alone_never_commits_an_utterance(self):
        """No speech ever crossed the threshold → no utterance can be
        committed. Even a long silent burst should leave the utterance
        queue empty."""
        cfg = _cfg(idle_timeout_ms=0)  # disable idle so silence just sits
        bridge, _hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        t = _start_endpointer(bridge)
        try:
            _push(bridge, bridge._current_gen(), [_silence_chunk()] * 30)
            time.sleep(0.5)
            self.assertTrue(bridge.utterance_q.empty(),
                            "silence-only audio must never produce an utterance")
        finally:
            _stop_endpointer(bridge, t)

    def test_speech_followed_by_silence_commits_pcm(self):
        """Loud sine for a few chunks, then enough silence to satisfy
        `silence_timeout_ms` → exactly one utterance lands on the
        utterance queue, with non-empty PCM tagged with the right gen."""
        cfg = _cfg(idle_timeout_ms=0)
        bridge, _hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        t = _start_endpointer(bridge)
        try:
            gen = bridge._current_gen()
            # 4 speech chunks + 5 silence (commit_chunks=3 at 192ms/64ms).
            _push(bridge, gen, [_speech_chunk()] * 4 + [_silence_chunk()] * 5)

            try:
                gen_out, pcm, sr = bridge.utterance_q.get(timeout=2.0)
            except queue.Empty:
                self.fail("expected an utterance commit, got none")

            self.assertEqual(gen_out, gen)
            self.assertEqual(sr, _SAMPLE_RATE)
            self.assertGreater(len(pcm), 0,
                               "committed PCM should contain the speech buffer")
            # Only one utterance — the trailing silence after the cut
            # must NOT spawn a second commit.
            self.assertTrue(bridge.utterance_q.empty())
        finally:
            _stop_endpointer(bridge, t)

    def test_below_threshold_chunk_does_not_count_as_speech(self):
        """A small-amplitude sine (amp=100) sits well below the default
        1e6 threshold. The endpointer must treat it as silence — i.e.
        no in_speech transition, no commit."""
        cfg = _cfg(idle_timeout_ms=0)
        bridge, _hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        t = _start_endpointer(bridge)
        try:
            _push(bridge, bridge._current_gen(),
                  [_quiet_chunk()] * 20 + [_silence_chunk()] * 5)
            time.sleep(0.5)
            self.assertTrue(bridge.utterance_q.empty(),
                            "below-threshold audio must not commit")
        finally:
            _stop_endpointer(bridge, t)


# ---------------------------------------------------------------------------
# 2. Activation / deactivation timings
# ---------------------------------------------------------------------------
class ActivationTimingsTest(unittest.TestCase):
    """How long the endpointer waits before firing each transition.

    `silence_timeout_ms` controls the speech-then-silence pause that
    triggers a commit; `idle_timeout_ms` controls the cumulative-silence
    timer that closes the mic. The latter must be suppressed while a
    reply is still playing — otherwise the bridge would idle in the
    middle of its own audible response.
    """

    def test_commit_waits_for_full_silence_timeout(self):
        """With silence_timeout_ms=192 (≈3 chunks at 64 ms/chunk), 2
        silence chunks after speech must NOT commit; the 3rd should.
        This pins the threshold so a future tweak to chunk_ms doesn't
        accidentally halve the commit pause."""
        cfg = _cfg(silence_timeout_ms=192, idle_timeout_ms=0)
        bridge, _hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        t = _start_endpointer(bridge)
        try:
            gen = bridge._current_gen()
            # Speech + 2 silence: not enough to commit.
            _push(bridge, gen, [_speech_chunk()] * 3 + [_silence_chunk()] * 2)
            try:
                bridge.utterance_q.get(timeout=0.4)
                self.fail("commit fired before silence_timeout_ms reached")
            except queue.Empty:
                pass
            # One more silence chunk pushes us over commit_chunks.
            _push(bridge, gen, [_silence_chunk()] * 3)
            try:
                bridge.utterance_q.get(timeout=2.0)
            except queue.Empty:
                self.fail("commit did not fire after silence_timeout_ms reached")
        finally:
            _stop_endpointer(bridge, t)

    def test_idle_fires_after_idle_timeout_of_pure_silence(self):
        """Silence with no speech ever — after `idle_timeout_ms` worth of
        chunks the endpointer calls `_enter_idle`, which clears
        `recording` and writes the LED to muted=True via the stub."""
        cfg = _cfg(idle_timeout_ms=192)  # 3 chunks
        bridge, hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        t = _start_endpointer(bridge)
        try:
            _push(bridge, bridge._current_gen(), [_silence_chunk()] * 8)
            self.assertTrue(_wait_until(lambda: not bridge.recording.is_set(),
                                        timeout=2.0),
                            "auto-idle never fired after idle_timeout_ms of silence")
            # set_led(muted=True) is part of `_enter_idle`'s contract:
            # the firmware capture mute must follow the bridge's intent.
            self.assertTrue(any(call is True for call in hid.set_led_calls),
                            f"expected set_led(muted=True), got {hid.set_led_calls}")
        finally:
            _stop_endpointer(bridge, t)

    def test_idle_blocked_while_player_is_active(self):
        """If the player still has aplay running, idle must NOT fire —
        the endpointer would otherwise close the mic mid-reply on a
        long TTS playback. We simulate an active player by parking a
        Mock in `_player_proc`."""
        cfg = _cfg(idle_timeout_ms=128)  # 2 chunks
        bridge, _hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        bridge._player_proc = mock.Mock()  # _is_playing() → True
        t = _start_endpointer(bridge)
        try:
            _push(bridge, bridge._current_gen(), [_silence_chunk()] * 12)
            time.sleep(0.6)  # several idle-windows worth
            self.assertTrue(bridge.recording.is_set(),
                            "idle must not fire while playback is in progress")
        finally:
            _stop_endpointer(bridge, t)

    def test_commit_resets_silence_count_so_idle_starts_after_commit(self):
        """The commit path zeros `silence_count` on its way out — the
        idle timer therefore measures silence *after* the last
        transaction, not just after the last speech onset. Without this
        the bridge would idle the moment a commit landed (since the
        trailing silence already exceeded `idle_timeout_ms` if
        silence_timeout_ms ≤ idle_timeout_ms)."""
        cfg = _cfg(silence_timeout_ms=128, idle_timeout_ms=320)
        bridge, _hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        t = _start_endpointer(bridge)
        try:
            gen = bridge._current_gen()
            _push(bridge, gen, [_speech_chunk()] * 3 + [_silence_chunk()] * 3)
            try:
                bridge.utterance_q.get(timeout=2.0)
            except queue.Empty:
                self.fail("expected commit before idle could fire")
            # Right after commit: still recording (idle starts NOW, not
            # at the speech onset 200 ms ago).
            self.assertTrue(bridge.recording.is_set(),
                            "must still be recording immediately after commit")
        finally:
            _stop_endpointer(bridge, t)


# ---------------------------------------------------------------------------
# 3. Mute trigger (HID press)
# ---------------------------------------------------------------------------
class MuteTriggerTest(unittest.TestCase):
    """`_on_hid_press` is the only path that toggles the mic. While
    idle: bump gen + set recording + write LED off + clear
    `_auto_idled`. While recording: soft mute — set `_force_commit`
    (the endpointer flushes any in-progress speech), clear `recording`,
    write LED on. The press-to-mute is intentionally non-destructive:
    no gen bump, no queue drain, no aplay kill — the worker/TTS/player
    keep going so the user's last utterance is still sent and its
    reply played back."""

    def test_press_while_idle_resumes_and_bumps_gen(self):
        """Boot is muted (`hid_mute_enabled=True`). First press sets
        `recording`, bumps the generation counter (so any straggler
        chunks from before are dropped), clears `_auto_idled`, and
        writes set_led(False)."""
        bridge, hid, _vb = _make_bridge(_cfg())
        # Boot state: muted, gen=0.
        self.assertFalse(bridge.recording.is_set())
        gen0 = bridge._current_gen()
        # Pretend the bridge had auto-idled — the resume must clear it.
        bridge._auto_idled.set()

        bridge._on_hid_press()

        self.assertTrue(bridge.recording.is_set())
        self.assertGreater(bridge._current_gen(), gen0,
                           "press-to-resume must bump the generation counter")
        self.assertFalse(bridge._auto_idled.is_set(),
                         "press-to-resume must clear the auto-idle flag")
        self.assertEqual(hid.set_led_calls[-1], False,
                         "LED must turn off when bridge resumes recording")

    def test_press_while_recording_soft_mutes_without_draining(self):
        """Press during a live session = soft commit-and-mute. Every
        queue must be left intact (worker/TTS/player keep flowing), the
        gen must NOT bump (in-flight items stay valid), `_force_commit`
        must be set (endpointer will flush its in-progress buffer), and
        the LED must be back on."""
        bridge, hid, _vb = _make_bridge(_cfg())
        bridge.recording.set()
        gen0 = bridge._current_gen()

        # Pre-load each queue so we can prove they were NOT drained.
        bridge.audio_q.put((gen0, b"audio"))
        bridge.utterance_q.put((gen0, b"utt", _SAMPLE_RATE))
        bridge.playback_q.put((gen0, b"pcm"))

        bridge._on_hid_press()

        self.assertFalse(bridge.recording.is_set())
        self.assertEqual(bridge._current_gen(), gen0,
                         "press-to-mute must NOT bump the generation counter")
        self.assertTrue(bridge._force_commit.is_set(),
                        "press-to-mute must signal the endpointer to commit")
        self.assertFalse(bridge.audio_q.empty(),
                         "audio_q must NOT be drained on press-to-mute")
        self.assertFalse(bridge.utterance_q.empty(),
                         "utterance_q must NOT be drained on press-to-mute")
        self.assertFalse(bridge.playback_q.empty(),
                         "playback_q must NOT be drained on press-to-mute")
        self.assertEqual(hid.set_led_calls[-1], True,
                         "LED must turn on (firmware mute) on press-to-mute")

    def test_press_force_commits_in_progress_speech(self):
        """User pressed mute mid-speech (in_speech=True, partial buf in
        the endpointer, no silence reached yet). The press must commit
        that partial buf to utterance_q with the gen it was captured
        under — same shape as the silence_timeout commit. Without this,
        whatever the user just finished saying gets dropped on press."""
        cfg = _cfg(idle_timeout_ms=0)
        bridge, _hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        gen0 = bridge._current_gen()
        t = _start_endpointer(bridge)
        try:
            # Speech with no trailing silence — the endpointer enters
            # in_speech but cannot auto-commit yet (silence_timeout_ms
            # not reached).
            _push(bridge, gen0, [_speech_chunk()] * 5)
            self.assertTrue(_wait_until(lambda: bridge.audio_q.empty(),
                                        timeout=2.0))
            self.assertTrue(bridge.utterance_q.empty(),
                            "endpointer must not auto-commit while in_speech")

            bridge._on_hid_press()

            try:
                gen_out, pcm, sr = bridge.utterance_q.get(timeout=2.0)
            except queue.Empty:
                self.fail("HID press did not force-commit the in-progress "
                          "speech buffer")
            self.assertEqual(gen_out, gen0,
                             "force-committed utterance must keep its gen")
            self.assertGreater(len(pcm), 0)
            self.assertEqual(sr, _SAMPLE_RATE)
            # Press did not bump gen → worker can still trust gen0.
            self.assertEqual(bridge._current_gen(), gen0)
        finally:
            _stop_endpointer(bridge, t)

    def test_press_with_no_speech_buffer_just_mutes(self):
        """If the user presses mute while quiet (no in-progress speech),
        the press is a no-op for the endpointer — `_force_commit` fires
        but there's nothing to commit. Just the soft mute side-effects
        remain (recording cleared, LED on)."""
        cfg = _cfg(idle_timeout_ms=0)
        bridge, hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        t = _start_endpointer(bridge)
        try:
            # Push only silence so the endpointer never enters in_speech.
            _push(bridge, bridge._current_gen(), [_silence_chunk()] * 3)
            self.assertTrue(_wait_until(lambda: bridge.audio_q.empty(),
                                        timeout=2.0))

            bridge._on_hid_press()

            # No utterance lands — there was nothing to commit.
            time.sleep(0.4)
            self.assertTrue(bridge.utterance_q.empty(),
                            "press with no in-progress speech must not commit")
            self.assertFalse(bridge.recording.is_set())
            self.assertEqual(hid.set_led_calls[-1], True)
        finally:
            _stop_endpointer(bridge, t)

    def test_auto_idle_sets_auto_idled_flag(self):
        """`_enter_idle` is the only path that should set `_auto_idled`
        — that's how the player tells "auto-idle while in-flight" apart
        from "user explicitly muted" when deciding whether to un-idle on
        first PCM."""
        cfg = _cfg(idle_timeout_ms=192)
        bridge, _hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        self.assertFalse(bridge._auto_idled.is_set())
        t = _start_endpointer(bridge)
        try:
            _push(bridge, bridge._current_gen(), [_silence_chunk()] * 8)
            self.assertTrue(_wait_until(lambda: not bridge.recording.is_set(),
                                        timeout=2.0),
                            "auto-idle never fired")
            self.assertTrue(bridge._auto_idled.is_set(),
                            "auto-idle must set the _auto_idled flag")
        finally:
            _stop_endpointer(bridge, t)

    def test_press_to_mute_does_not_set_auto_idled(self):
        """The user's explicit press is NOT auto-idle — the player must
        not later re-open the mic mid-playback. Only the timeout-driven
        path sets the flag."""
        bridge, _hid, _vb = _make_bridge(_cfg())
        bridge.recording.set()
        self.assertFalse(bridge._auto_idled.is_set())

        bridge._on_hid_press()

        self.assertFalse(bridge._auto_idled.is_set(),
                         "press-to-mute must not set the auto-idle flag")

    def test_press_toggles_back_and_forth(self):
        """Sanity: press → record, press → idle, press → record again.
        Only the resume direction bumps gen — the soft mute keeps gen
        intact so the in-flight pipeline stays valid. LED writes
        alternate off/on/off."""
        bridge, hid, _vb = _make_bridge(_cfg())
        g0 = bridge._current_gen()

        bridge._on_hid_press()  # idle → record (resume bumps gen)
        g1 = bridge._current_gen()
        self.assertTrue(bridge.recording.is_set())
        self.assertGreater(g1, g0)

        bridge._on_hid_press()  # record → idle (soft mute, no bump)
        g2 = bridge._current_gen()
        self.assertFalse(bridge.recording.is_set())
        self.assertEqual(g2, g1, "press-to-mute must NOT bump the gen")

        bridge._on_hid_press()  # idle → record (resume bumps gen)
        g3 = bridge._current_gen()
        self.assertTrue(bridge.recording.is_set())
        self.assertGreater(g3, g2)

        # LED writes follow the recording state: off, on, off.
        self.assertEqual(hid.set_led_calls, [False, True, False])

    def test_endpointer_drops_audio_with_stale_gen(self):
        """The gen bump is only useful if downstream stages actually
        check it. A press-cancel mid-utterance bumps gen; any straggler
        chunks pushed under the old gen must be dropped, so resuming
        doesn't accidentally re-commit a half-finished utterance from
        before the cancel."""
        cfg = _cfg(idle_timeout_ms=0)
        bridge, _hid, _vb = _make_bridge(cfg)
        bridge.recording.set()
        t = _start_endpointer(bridge)
        try:
            stale = bridge._current_gen()
            bridge._bump_gen()
            fresh = bridge._current_gen()

            # Stale-gen speech + silence: should be dropped, no commit.
            _push(bridge, stale,
                  [_speech_chunk()] * 4 + [_silence_chunk()] * 5)
            try:
                bridge.utterance_q.get(timeout=0.5)
                self.fail("endpointer committed audio from a stale generation")
            except queue.Empty:
                pass

            # Fresh-gen audio commits as normal.
            _push(bridge, fresh,
                  [_speech_chunk()] * 4 + [_silence_chunk()] * 5)
            try:
                gen_out, _pcm, _sr = bridge.utterance_q.get(timeout=2.0)
            except queue.Empty:
                self.fail("endpointer dropped fresh-gen audio too")
            self.assertEqual(gen_out, fresh)
        finally:
            _stop_endpointer(bridge, t)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
