#!/usr/bin/env python3
"""
Binary Voice Bridge v3 — always-on mic, async pipeline.

Trigger: `activation` = "button" (Jabra HID press unmutes) or "wake_word"
         (RMS-gated Whistle keyword spotting while idle; see wake_word.py).
         Either way the HID button mutes, and voice activity drives turns.
STT:     Deepgram or ElevenLabs Scribe (configurable).
TTS:     Deepgram Aura or ElevenLabs (configurable, streaming).
Output:  ALSA aplay.

Four worker threads connected by queues:

    Recorder ──audio_q──▶ Endpointer ──utt_q──▶ Worker ──playback_q──▶ Player

Recorder keeps PyAudio open while `recording` is set; endpointer runs
RMS VAD per chunk and commits utterances on a configurable pause; worker
drives STT → gateway SSE → TTS streaming; player drives one aplay
subprocess per utterance. The bridge auto-idles (closes the mic stream)
after `idle_timeout_ms` of pure silence; an HID press resumes it. CPU is
low by design — recorder blocks in `stream.read`, endpointer does a
single RMS per ~64 ms chunk, worker/player are idle off-turn.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import logging
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Iterable, Iterator

import pyaudio

import endpointer
import wake_word
from deezer_connect_plugin import DeezerConnectPlugin
from endpointer import Endpointer
from jabra_hid import HidMuteMonitor
from stt_compare import SttComparer

# The helpers live in their own modules; they are imported by name here so
# `VoiceBridge` resolves them through this module (tests patch them on it)
# and so `voice-bridge.py` keeps exposing the same surface for ad-hoc use.
from audio_io import (  # noqa: F401  (re-exported)
    _aplay_popen,
    _apply_output_volume,
    _drain_aplay,
    _make_beep_pcm,
    _make_sleep_tone_pcm,
    _make_speech_vad,
    _make_tick_pcm,
    _voiced_ms,
    find_input_device,
    play_audio,
    play_audio_stream,
    play_beep,
)
from bridge_config import (  # noqa: F401  (re-exported)
    CONFIG_PATH,
    SECRETS_PATH,
    _HERE,
    _build_voice_provider,
    load_config,
)
from gateway import (  # noqa: F401  (re-exported)
    GATEWAY_FALLBACK_REPLY,
    GATEWAY_LOST_REPLY,
    GATEWAY_UNREACHABLE_REPLY,
    _expects_answer,
    _filter_no_reply,
    _is_no_reply,
    _is_non_speech,
    gateway_chat,
    gateway_chat_stream,
    gateway_chat_stream_zeroclaw,
    gateway_chat_stream_zeroclaw_ws,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("voice-bridge")
logging.basicConfig(
    level=logging.INFO,
    format="[voice-bridge] %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Async pipeline orchestrator
# ---------------------------------------------------------------------------
class State(enum.Enum):
    """The bridge's mode. `VoiceBridge._set_state` derives everything else
    from it: the `recording` / `_wake_armed` Events the recorder routes on,
    and the LED / firmware capture mute. Transition table: AGENTS.md."""
    RECORDING = "recording"            # mic → endpointer, LED off
    PROCESSING = "processing"          # mic paused for the agent, LED off
    IDLE_LISTENING = "idle_listening"  # wake word armed, LED off (mic open)
    MUTED = "muted"                    # firmware-muted, LED red


# Derived from the state: (records, wake armed, firmware-muted).
_STATE_EFFECTS = {
    State.RECORDING: (True, False, False),
    State.PROCESSING: (False, False, False),
    State.IDLE_LISTENING: (False, True, False),
    State.MUTED: (False, False, True),
}


def _transition(method):
    """Run a `VoiceBridge` state transition under `_state_lock`.

    The bridge state is `_state` + `_auto_idled` plus plain fields
    (`_idle_window_ms`, `_replies_since_resume`, `_quiet_idle`,
    `_reply_is_question`) written from the HID, endpointer, worker, player
    and MCP threads. Each transition reads several of them and then writes
    several; the lock (re-entrant: transitions call each other) makes every
    such check-then-act atomic. Transitions must not block while holding it."""
    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)
    return locked


# Sentinel pushed into `playback_q` after each utterance's audio chunks
# so the player thread knows to close the current aplay process and
# wait for the next utterance. Plain None would conflict with empty-
# chunk filtering elsewhere; an explicit object is unambiguous.
class _EndOfUtterance:
    pass


_END_OF_UTTERANCE = _EndOfUtterance()
# Returned by `_next_reply_item` when the reply was cut by a gen change.
_STALE = object()

# Most idle audio `wake_q` holds; Whistle normally drains it in well under 1 s.
_WAKE_Q_SECONDS = 10.0


# A whole externally-supplied utterance (the MCP `say_to_speaker` tool),
# enqueued as ONE atomic item rather than chunk+marker so it can never
# interleave with the worker's streamed reply chunks on `playback_q`. The
# player plays `pcm` end-to-end as its own utterance and fires `done` when
# the playback (and no-clip drain) finishes, releasing a blocked caller.
class _ExternalUtterance:
    """`duck`: lower the music while it plays. `cue`: a thinking cue (tick
    or "un attimo") — it doesn't count as a reply for the idle window.
    `question`: the speech expects an answer (see `_after_external`)."""

    def __init__(self, pcm: bytes, done: "threading.Event", *,
                 label: str = "say_to_speaker", duck: bool = True,
                 cue: bool = False, question: bool = False) -> None:
        self.pcm = pcm
        self.done = done
        self.label = label
        self.duck = duck
        self.cue = cue
        self.question = question


class _ThinkingCue:
    """Per-turn "still working" feedback between pick-up and the first
    reply audio, so the user never sits through 10–25 s of dead air.

    - the first `tool_call` frame plays a spoken cue ("Un attimo.") once;
    - after `delay_ms` with no reply audio, a soft tick, repeated every
      `repeat_ms`.

    Cues go through the player queue (serialized, never mixed with the
    reply) and don't duck the music. `stop()` is called before the first
    reply chunk is queued; the lock makes "check stopped + enqueue"
    atomic, so no cue can land behind the reply."""

    def __init__(self, bridge: "VoiceBridge", gen: int) -> None:
        self._bridge = bridge
        self._gen = gen
        tc = bridge.cfg.get("thinking_cue") or {}
        self._enabled = bool(tc.get("enabled"))
        self._delay = max(0, int(tc.get("delay_ms", 3000))) / 1000.0
        self._repeat = max(100, int(tc.get("repeat_ms", 5000))) / 1000.0
        self._stop = threading.Event()
        self._tool = threading.Event()
        # Wakes `_run` early (tool_call or stop) instead of polling.
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self.cues_played = 0

    def start(self) -> "_ThinkingCue":
        if self._enabled:
            threading.Thread(target=self._run, name="vb-thinking", daemon=True).start()
        return self

    def on_event(self, ftype: str, _frame: dict) -> None:
        if ftype == "tool_call":
            self._tool.set()
            self._wake.set()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
        self._wake.set()

    def _enqueue(self, pcm: bytes | None, label: str) -> bool:
        if not pcm:
            return False
        with self._lock:
            if self._stop.is_set():
                return False
            self._bridge.playback_q.put((self._gen, _ExternalUtterance(
                pcm, threading.Event(), label=label, duck=False, cue=True)))
            self.cues_played += 1
            return True

    def _run(self) -> None:
        rate = int(self._bridge.cfg["tts_sample_rate"])
        next_tick = time.monotonic() + self._delay
        spoke = False
        while True:
            self._wake.wait(max(0.0, next_tick - time.monotonic()))
            self._wake.clear()
            if self._stop.is_set():
                return
            now = time.monotonic()
            if self._tool.is_set() and not spoke:
                spoke = True
                bank = self._bridge._thinking_acks
                clip = bank.pick() if bank else None
                if self._enqueue(clip[2] if clip else None, "thinking-ack"):
                    log.info("Thinking cue: %r", clip[1])
                    next_tick = now + self._repeat
                    continue
            if now >= next_tick:
                self._enqueue(_make_tick_pcm(rate), "thinking-tick")
                next_tick = now + self._repeat


class VoiceBridge:
    """Always-on mic + 4-stage async pipeline + HID mute/resume button.

    Threads (all daemons):

      - `_hid_loop`        polls HidMuteMonitor; triggers state toggles
      - `_recorder_loop`   PyAudio open/read while `recording` is set
      - `_endpointer_loop` RMS VAD; emits utterances on `silence_timeout_ms`
                           pauses, triggers auto-idle on `idle_timeout_ms`
      - `_worker_loop`     STT → gateway SSE → TTS streaming
      - `_player_loop`     one aplay subprocess per utterance

    Pipeline items carry the generation (`_gen`) they were produced
    under; downstream stages drop anything whose generation has been
    superseded. Only a resume (`_resume`: HID press, wake word) or a
    barge-in on a playing reply (`_stop_reply`, which also drains the
    player queues and kills aplay) bumps it. Every other stop is soft:
    an HID press while recording commits what was said and mutes, and
    auto-idle closes the mic — neither bumps `_gen`, so the reply the
    user was waiting for still plays.
    """

    def __init__(self, cfg: dict, stt, tts, hid: HidMuteMonitor,
                 deezer: DeezerConnectPlugin | None = None) -> None:
        self.cfg = cfg
        self.stt = stt
        self.tts = tts
        self.hid = hid
        # Optional deezer-connect ducking plugin. No-op unless enabled
        # in voice-bridge.json under `deezer_connect`. Constructed by
        # main() so tests can inject a fake.
        self.deezer = deezer or DeezerConnectPlugin(cfg.get("deezer_connect"))
        # Ducking holds — see `_duck_acquire`.
        self._duck_lock = threading.Lock()
        self._duck_holds = 0
        # Set by `start()`: volume calls go to the `vb-duck` thread.
        self._duck_q: "queue.Queue | None" = None
        self._duck_thread: threading.Thread | None = None

        # `audio_q` is unbounded: the endpointer is O(N) over a 1024-
        # sample chunk per ~64 ms — easily faster than the recorder, so
        # the queue should stay near-empty in practice. The other queues
        # are also unbounded; backpressure is naturally bounded by an
        # utterance's duration (~10s of audio = ~250KB at 24kHz).
        self.audio_q: "queue.Queue[tuple[int, bytes]]" = queue.Queue()
        self.utterance_q: "queue.Queue[tuple[int, bytes, int]]" = queue.Queue()
        self.playback_q: "queue.Queue[tuple[int, bytes | _EndOfUtterance]]" = queue.Queue()

        self.shutdown_event = threading.Event()
        # The bridge's mode; only `_set_state` writes it. With HID enabled
        # (the canonical deployment) the bridge boots MUTED — the
        # HidMuteMonitor's engage write puts the device into firmware-mute
        # (LED red, USB capture silenced) until the user presses the button.
        # If HID is disabled, fall back to "always-on" boot so there's
        # still a way to use the bridge — without HID there's nothing to
        # un-mute it from a muted boot. Wake mode boots IDLE_LISTENING
        # (set below). `recording` is derived: it gates the recorder thread.
        self._state = State.RECORDING if not cfg.get("hid_mute_enabled") else State.MUTED
        self.recording = threading.Event()
        if self._state is State.RECORDING:
            self.recording.set()

        self._gen = 0
        self._gen_lock = threading.Lock()
        # Serializes the state transitions — see `_transition`.
        self._state_lock = threading.RLock()

        # Set by the player after a playback completes so the endpointer
        # resets its silence counter — otherwise the 10s playback eats
        # into the idle window and the bridge auto-idles right after the
        # reply finishes. Effect: idle timer measures silence *after* the
        # last interaction (user speech OR our reply), not just user speech.
        self._idle_reset_pending = threading.Event()

        # Set by an HID-press while recording to tell the endpointer to
        # commit any in-progress speech buffer right now, instead of
        # waiting for `silence_timeout_ms`. The companion to "soft mute":
        # the user pressed mute, so we still send what they were saying
        # but stop listening for new input.
        self._force_commit = threading.Event()

        # Rides along with `_state` (set by `_set_state`): the bridge left
        # RECORDING on its own — auto-idle on silence, or PROCESSING. The
        # player checks it when the first PCM chunk of a reply arrives and
        # un-idles so the user can talk back the moment the reply ends. If
        # the user explicitly muted via HID it is clear and the player
        # respects the press — playback happens, the mic stays muted.
        self._auto_idled = False

        # The active aplay process, if any. Held under `_player_lock`
        # so a hard-cancel from the HID thread can `kill()` it without
        # racing the player thread's setup/teardown.
        self._player_proc: subprocess.Popen | None = None
        self._player_lock = threading.Lock()

        # Wake-word activation. `_wake_armed` (derived: set exactly in
        # IDLE_LISTENING) = idle but listening for the wake phrase: the
        # firmware mic stays unmuted and the recorder routes chunks to
        # `wake_q` instead of `audio_q`. Always clear in "button" mode.
        self.wake_mode = cfg.get("activation") == "wake_word"
        # Bounded: if the wake loop stalls or dies, the recorder drops the
        # oldest audio instead of growing the queue until the OOM killer.
        chunk_s = int(cfg.get("chunk_size", 1024)) / int(cfg.get("sample_rate", 16000))
        self.wake_q: "queue.Queue[bytes]" = queue.Queue(
            maxsize=max(1, int(_WAKE_Q_SECONDS / chunk_s)))
        self._wake_armed = threading.Event()
        # Set while a wake/sleep clip plays so the wake loop ignores our own
        # voice coming back through the mic.
        self._speaking_ack = threading.Event()
        # Set by the worker from picking an utterance up until it loops back
        # for the next one — i.e. a turn is in flight. A goodbye on auto-idle
        # is skipped while a reply is still on its way.
        self._worker_busy = threading.Event()
        self._wake_cfg = self._wake_detector = self._wake_acks = self._sleep_acks = None
        # Acks speak in the reply voice: they reuse the reply's TTS instance.
        voice = cfg.get("tts_provider", "elevenlabs")
        ack_dir = wake_word.wake_config(cfg)["ack_cache_dir"]
        if self.wake_mode:
            self._wake_cfg = wake_word.wake_config(cfg)
            self._wake_detector = wake_word.WakeDetector(self._wake_cfg)
            w = self._wake_cfg
            self._wake_acks = wake_word.AckBank(cfg, w["acks"], voice, ack_dir, tts, kind="wake")
            self._sleep_acks = wake_word.AckBank(
                cfg, w["sleep_acks"], voice, ack_dir, tts, kind="sleep")
            if self._state is State.MUTED:
                self._state = State.IDLE_LISTENING
                self._wake_armed.set()

        # Spoken "un attimo" clips for the thinking cue (any activation mode).
        self._thinking_acks = None
        tphrases = (cfg.get("thinking_cue") or {}).get("phrases") or []
        if (cfg.get("thinking_cue") or {}).get("enabled") and tphrases:
            self._thinking_acks = wake_word.AckBank(
                cfg, tphrases, voice, ack_dir, tts, kind="thinking")

        # Frame-level voice check at commit (None = off).
        self._speech_vad = _make_speech_vad(cfg)

        # Idle window currently in force (ms). Reset to `idle_timeout_ms` on
        # resume/commit; widened to `idle_after_reply_ms` /
        # `idle_after_question_ms` when a reply finishes playing.
        self._idle_window_ms = int(cfg.get("idle_timeout_ms", 0))
        # Set by the worker when the reply text is complete: does it end
        # with a question? Read by the player when the reply finishes.
        self._reply_is_question = False
        # Replies played since the last wake/resume — a goodbye is only
        # spoken after an actual conversation (wake_word mode).
        self._replies_since_resume = 0
        # Set after a say_to_speaker announcement that asked nothing: the
        # next auto-idle goes back to sleep without a goodbye or tone.
        self._quiet_idle = False
        # Pick-up time of the turn in flight, for the timing log.
        self._turn_t0: float | None = None
        # External utterances that arrived while a streamed reply was
        # playing; the player plays them right after it.
        self._player_backlog: "deque[tuple[int, object]]" = deque()

        # Shadow STT comparison (`stt_compare.enabled`): the remote STT stays
        # authoritative; Whistle's take on the same PCM is only logged.
        ccfg = cfg.get("stt_compare") or {}
        self._stt_compare = (
            SttComparer(ccfg, _HERE, cfg.get("stt_provider", "?")) if ccfg.get("enabled") else None
        )

        self._threads: list[threading.Thread] = []

    # -- generation helpers --------------------------------------------
    def _current_gen(self) -> int:
        with self._gen_lock:
            return self._gen

    def _bump_gen(self) -> int:
        with self._gen_lock:
            self._gen += 1
            return self._gen

    @staticmethod
    def _drain_queue(q: "queue.Queue") -> int:
        n = 0
        try:
            while True:
                q.get_nowait()
                n += 1
        except queue.Empty:
            return n

    def _kill_player(self) -> None:
        with self._player_lock:
            proc = self._player_proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception as exc:
            log.warning("kill aplay failed: %s", exc)

    def _is_playing(self) -> bool:
        with self._player_lock:
            return self._player_proc is not None

    # -- state transitions ---------------------------------------------
    # -- deezer ducking --------------------------------------------------
    # Music is ducked while someone is talking: the user (the endpointer
    # holds from the first above-threshold chunk until the utterance is
    # committed to STT) or the bridge (one hold per aplay — reply, ack,
    # goodbye, beep, say_to_speaker). Holds overlap (talking over a reply,
    # an ack racing the player), so they're counted: duck on 0→1, unduck
    # on 1→0. Mute/idle state no longer affects ducking.
    #
    # The volume I/O (HTTP to the deezer-connect BFF, up to 2 × timeout)
    # runs on the `vb-duck` thread once `start()` has launched it, so the
    # endpointer and the player never wait on the network. Commands keep
    # their order through one FIFO; before `start()` (tests) they run inline.
    def _duck_acquire(self, reason: str) -> None:
        with self._duck_lock:
            self._duck_holds += 1
            if self._duck_holds == 1:
                log.info("Ducking on (%s)", reason)
                self._duck_send(self.deezer.duck)

    def _duck_release(self, reason: str) -> None:
        with self._duck_lock:
            self._duck_holds = max(0, self._duck_holds - 1)
            if self._duck_holds == 0:
                log.info("Ducking off (%s)", reason)
                self._duck_send(self.deezer.unduck)

    def _duck_send(self, fn) -> None:
        """Run a volume call on the duck thread (inline when not started).
        Called under `_duck_lock`, so commands are queued in hold order."""
        if self._duck_q is None:
            fn()
        else:
            self._duck_q.put(fn)

    def _duck_loop(self) -> None:
        while True:
            fn = self._duck_q.get()
            if fn is None:
                return
            try:
                fn()
            except Exception:
                log.exception("deezer-connect volume call failed")

    @contextlib.contextmanager
    def _ducked(self, reason: str):
        self._duck_acquire(reason)
        try:
            yield
        finally:
            self._duck_release(reason)

    @_transition
    def _set_state(self, new: State, *, auto_idled: bool = False) -> None:
        """Enter `new` and derive the rest: `recording`, `_wake_armed` and
        the LED / firmware mute (written only when it changes).

        Event order keeps every mic chunk on one route: entering RECORDING
        sets `recording` before clearing `_wake_armed` (the recorder checks
        `recording` first), leaving it arms the wake word before clearing
        `recording`."""
        old_muted = _STATE_EFFECTS[self._state][2]
        self._state = new
        self._auto_idled = auto_idled
        records, armed, muted = _STATE_EFFECTS[new]
        if records:
            self.recording.set()
        (self._wake_armed.set if armed else self._wake_armed.clear)()
        if not records:
            self.recording.clear()
        if muted != old_muted:
            self.hid.set_led(muted=muted)

    @_transition
    def _enter_idle(self, source: str) -> None:
        """Hard idle: close mic, firmware-mute, LED red.

        Fires after `idle_timeout_ms` of silence following the last
        "transaction" — either a user utterance commit or the end of
        a TTS playback. The 10 s window is owned by the endpointer's
        `silence_count`, which is reset on both commit and playback
        end so the timer always measures silence *after* the last
        interaction, not just after the last user speech.

        We deliberately don't bump `_gen` here, so any utterance the
        worker is processing (and any audio the player is still
        flushing) finishes naturally. The audio_q is drained because
        anything captured after the silence threshold won't change
        the outcome; sparing the endpointer the work of filtering it
        out chunk-by-chunk on resume.
        """
        if self._state is not State.RECORDING:
            return
        log.info("Idle (%s): closing mic, in-flight pipeline continues", source)
        # auto_idled: the player un-idles when the in-flight reply starts
        # playing — see `_player_loop`. In wake mode idle-but-listening:
        # the firmware mic must stay open or the wake loop would hear only
        # zeros, so the LED stays off too.
        self._set_state(State.IDLE_LISTENING if self.wake_mode else State.MUTED,
                        auto_idled=True)
        self._drain_queue(self.audio_q)
        if self.wake_mode:
            # Say goodbye — unless a reply is still on its way, in which
            # case the player un-idles for it and "a dopo" would be a lie.
            # Only after a conversation, though: a wake that got no reply
            # (nobody spoke, or only noise) dozes off with a soft tone.
            if self._quiet_idle:
                # Only an announcement was spoken: no conversation to close.
                self._quiet_idle = False
            elif not self._turn_in_flight():
                clip = None
                if (self._replies_since_resume > 0
                        or not self._wake_cfg.get("goodbye_after_turn_only", True)):
                    clip = self._sleep_acks.pick() if self._sleep_acks else None
                if clip is None:
                    rate = int(self.cfg["tts_sample_rate"])
                    clip = ("tone", "(sleep tone)", _make_sleep_tone_pcm(rate))
                threading.Thread(target=self._say_goodbye, args=(clip,),
                                 name="vb-goodbye", daemon=True).start()

    def _turn_in_flight(self) -> bool:
        """A turn is being processed, waiting, or its reply is playing."""
        return (self._worker_busy.is_set() or not self.utterance_q.empty()
                or self._reply_playing())

    def _play_clip(self, clip) -> None:
        provider, text, pcm = clip
        log.info("Ack (%s): %r", provider, text)
        self._speaking_ack.set()
        try:
            with self._ducked("ack"):
                play_audio(self.cfg["output_device"], pcm, int(self.cfg["tts_sample_rate"]))
        finally:
            self._speaking_ack.clear()

    def _say_goodbye(self, clip) -> None:
        try:
            self._play_clip(clip)
        except Exception:
            log.exception("Goodbye playback failed")

    def _reply_playing(self) -> bool:
        """A reply (or other player output) is audible or queued to be."""
        return self._is_playing() or not self.playback_q.empty() or bool(self._player_backlog)

    def _drain_playback(self) -> None:
        """Empty the player queues, releasing any blocked `play_pcm` caller."""
        items = list(self._player_backlog)
        self._player_backlog.clear()
        while True:
            try:
                items.append(self.playback_q.get_nowait())
            except queue.Empty:
                break
        for _gen, item in items:
            if isinstance(item, _ExternalUtterance):
                item.done.set()

    def _stop_reply(self) -> None:
        """HID press during playback: cut the reply and listen right away.

        Bumping the gen first stops the worker from queueing more of the
        reply (its tee checks the gen) and makes the player drop whatever
        is in flight; then the queues are drained and aplay is killed."""
        log.info("HID press during playback: stop reply, listening")
        self._resume()
        self._drain_playback()
        self._drain_queue(self.utterance_q)
        self._kill_player()

    @_transition
    def _on_hid_press(self) -> None:
        if self._state is State.PROCESSING:
            # Mic is only paused for the agent; a press here means "mute".
            # No gen bump, so the reply still plays — the mic stays muted.
            log.info("HID press while processing: mute (reply still plays)")
            self._set_state(State.MUTED)
            return
        if self._state is State.RECORDING:
            if self._reply_playing():
                self._stop_reply()
                return
            log.info("HID press: commit-and-mute (in-flight pipeline continues)")
            # Soft mute: tell the endpointer to commit any in-progress
            # speech buffer right now (don't wait for silence_timeout_ms),
            # stop the recorder, write LED on. The worker still picks the
            # committed utterance off `utterance_q` and runs STT → gateway
            # → TTS as usual; the player still plays the reply. We just
            # stop listening for new input until the next press resumes.
            # No queue drain, no gen bump, no aplay kill — those would
            # discard the very thing the user pressed mute to send.
            self._force_commit.set()
            # An explicit mute is a privacy mute in wake mode too: firmware
            # silenced, wake word off, only the button resumes.
            self._set_state(State.MUTED)
        elif self._reply_playing():
            # Muted earlier (press while processing) and the reply is now
            # playing: a second press means "stop talking, I'm listening".
            self._stop_reply()
        else:
            log.info("HID press: resume recording")
            self._resume()

    @_transition
    def _resume(self, prefill: "Iterable[bytes]" = ()) -> None:
        """Idle/muted → recording. Shared by HID press and wake word.

        `prefill`: mic chunks captured before the resume (the wake word's
        trailing command) that the endpointer must see first."""
        # Bump gen so any stragglers from before (e.g. an old
        # in-progress speech buffer the endpointer might have under
        # the previous gen) are shed by downstream stages.
        self._replies_since_resume = 0
        self._quiet_idle = False
        self._set_idle_window("idle_after_resume_ms")
        gen = self._bump_gen()
        # Queued before `recording` is set, so they land ahead of the
        # recorder's first live chunk.
        for chunk in prefill:
            self.audio_q.put((gen, chunk))
        self._set_state(State.RECORDING)

    @_transition
    def _on_wake(self, text: str, segment: bytes = b"", backlog: "Iterable[bytes]" = (),
                 command: str = "") -> None:
        """Wake phrase heard while idle: listen at once, ack in parallel.

        The mic opens before the ack plays, so "Hey Binary… metti la
        musica" said in one breath isn't cut. `backlog` (audio queued while
        Whistle transcribed) always goes to the endpointer; `segment` (the
        wake segment itself) too when Whistle heard words after the phrase
        (`command`) — STT then gets the whole sentence. With a command under
        way a soft tick replaces the spoken ack, so it doesn't talk over it."""
        if self._state is not State.IDLE_LISTENING:
            return
        log.info("Wake word: %r%s", text, f" + command {command!r}" if command else "")
        prefill = list(self._split_chunks(segment)) if command else []
        prefill += list(backlog)
        speaking = self._is_playing()
        self._resume(prefill=prefill)
        if speaking:
            return  # a reply is already speaking; the resume is the ack
        rate = int(self.cfg["tts_sample_rate"])
        if command:
            clip = ("tone", "(wake tick)", _make_tick_pcm(rate))
        else:
            clip = self._wake_acks.pick() if self._wake_acks else None
            if clip is None:
                clip = ("tone", "(beep)", _make_beep_pcm(rate, freq=880, duration=0.08))
        threading.Thread(target=self._play_wake_ack, args=(clip,),
                         name="vb-wake-ack", daemon=True).start()

    def _play_wake_ack(self, clip) -> None:
        try:
            self._play_clip(clip)
        except Exception:
            # The mic is already open: a failed ack costs nothing else.
            log.exception("Wake ack playback failed")

    def _split_chunks(self, pcm: bytes) -> "Iterator[bytes]":
        """`pcm` in recorder-sized chunks (the VAD metric depends on N)."""
        step = int(self.cfg["chunk_size"]) * 2
        for i in range(0, len(pcm), step):
            yield pcm[i:i + step]

    def play_pcm(self, pcm: bytes, *, block: bool = True, text: str | None = None) -> float:
        """Play externally-supplied PCM through the bridge's player so it
        behaves like a normal spoken reply.

        Used by the MCP `say_to_speaker` tool. The PCM (S16LE mono at
        `tts_sample_rate`) is enqueued as one utterance under the current
        generation, so it serializes behind any reply already playing and
        reuses the player's no-clip drain and the `_is_playing()` guard (the
        endpointer won't auto-idle mid-speech).

        Like a real reply the device unmutes (mic open, LED off) and the
        player ducks deezer-connect for exactly the playback window.
        Afterwards the player's end-of-playback reset starts the idle window,
        so the usual `idle_timeout_ms` silence re-mutes — exactly the tail of
        a normal speech. Ducking is a no-op unless `deezer_connect` is enabled.

        `text` (what the PCM says) decides the tail: a question gets the
        answer window, a plain announcement goes back to sleep without a
        goodbye — see `_after_external`.

        Returns the audio duration in seconds; when `block` (the default),
        waits until the player has finished this utterance (bounded so a
        stuck player can't pin the caller).
        """
        sample_rate = int(self.cfg["tts_sample_rate"])
        seconds = len(pcm) / (sample_rate * 2) if pcm else 0.0
        if not pcm:
            return 0.0

        self._unmute_for_external()

        # One atomic item under the current gen → serializes behind any reply
        # already playing, never interleaves with the worker's chunks. The
        # player fires `done` when playback finishes (or immediately if a HID
        # press bumps the gen and the item goes stale).
        done = threading.Event()
        self.playback_q.put((self._current_gen(), _ExternalUtterance(
            pcm, done, question=_expects_answer(text or ""))))

        if block:
            # Generous bound: playback is realtime, plus drain + margin.
            done.wait(timeout=seconds + 15.0)
        return seconds

    # -- thread loops --------------------------------------------------
    def _hid_loop(self) -> None:
        # Blocks on the monitor's press event; the timeout only bounds how
        # long a shutdown can go unnoticed.
        while not self.shutdown_event.is_set():
            if self.hid.wait_press(0.5):
                self._on_hid_press()

    def _recorder_loop(self) -> None:
        pa = pyaudio.PyAudio()
        stream: pyaudio.Stream | None = None
        sr = self.cfg["sample_rate"]
        chunk = self.cfg["chunk_size"]

        def _safe_terminate(inst) -> None:
            if inst is None:
                return
            try:
                inst.terminate()
            except Exception:
                pass

        def _rebuild_pa(old):
            # Tear the old PyAudio instance down and build a fresh one so a
            # hot-plugged Jabra becomes visible (PyAudio snapshots its device
            # list at construction). Both the teardown and the rebuild are
            # guarded and the rebuild retries: a transient ALSA error during an
            # unplug must not kill the recorder thread — that would leave the
            # mic dead until a service restart. Returns the new instance, or
            # None when shutdown is requested while retrying.
            _safe_terminate(old)
            while not self.shutdown_event.wait(1.0):
                try:
                    return pyaudio.PyAudio()
                except Exception as exc:
                    log.warning("Recorder: PyAudio rebuild failed: %s — retry in 1s", exc)
            return None

        try:
            while not self.shutdown_event.is_set():
                listening = self.recording.is_set() or self._wake_armed.is_set()
                if not listening:
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
                        stream = None
                        log.info("Recorder: stream closed")
                    # Park on `recording`. A timeout lets us notice
                    # shutdown even if no toggle ever arrives.
                    self.recording.wait(0.2)
                    continue

                if stream is None:
                    idx = find_input_device(pa)
                    if idx is None:
                        # A Jabra plugged in after we started is invisible until
                        # we rebuild the instance — same reconnect philosophy the
                        # HID monitor uses, so a hot-plug needs no restart.
                        log.warning("Recorder: Jabra input not found — retry in 1s")
                        pa = _rebuild_pa(pa)
                        if pa is None:
                            return
                        continue
                    try:
                        stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=sr,
                            input=True,
                            input_device_index=idx,
                            frames_per_buffer=chunk,
                        )
                        log.info("Recorder: opened (idx=%s rate=%dHz chunk=%d)",
                                 idx, sr, chunk)
                    except Exception as exc:
                        log.warning("Recorder: cannot open: %s — retry in 1s", exc)
                        pa = _rebuild_pa(pa)
                        if pa is None:
                            return
                        continue

                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                except Exception as exc:
                    log.warning("Recorder: read failed: %s — reopening", exc)
                    try:
                        stream.close()
                    except Exception:
                        pass
                    stream = None
                    continue
                if self.recording.is_set():
                    self.audio_q.put((self._current_gen(), data))
                elif self._wake_armed.is_set():
                    self._put_wake_chunk(data)
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            _safe_terminate(pa)

    def _put_wake_chunk(self, data: bytes) -> None:
        """Queue an idle chunk for the wake loop, dropping the oldest when full."""
        while True:
            try:
                self.wake_q.put_nowait(data)
                return
            except queue.Full:
                try:
                    self.wake_q.get_nowait()
                except queue.Empty:
                    pass

    @_transition
    def _disable_wake(self) -> None:
        """Whistle is unusable: fall back to button activation for good.

        Idle-but-listening becomes a firmware mute, so the mic is not kept
        open (and recorded into `wake_q`) for a wake word nobody can hear."""
        self.wake_mode = False
        self._drain_queue(self.wake_q)
        if self._state is State.IDLE_LISTENING:
            self._set_state(State.MUTED, auto_idled=self._auto_idled)

    def _wake_loop(self) -> None:
        """Idle wake-word spotting: gate chunks on energy, Whistle the rest."""
        try:
            self._wake_detector.load()
        except Exception:
            log.exception("Wake word: Whistle failed to load — falling back to the HID button")
            self._disable_wake()
            return
        log.info("Wake word: listening for %s (lang=%s, rms>%g)",
                 self._wake_cfg["phrases"], self._wake_cfg["language"],
                 self._wake_cfg["rms_threshold"])
        gate = wake_word.SpeechGate(self.cfg["sample_rate"], self.cfg["chunk_size"], self._wake_cfg)
        while not self.shutdown_event.is_set():
            try:
                chunk = self.wake_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self._wake_armed.is_set() or self._speaking_ack.is_set():
                gate.reset()
                self._drain_queue(self.wake_q)
                continue
            segment = gate.feed(chunk)
            if segment is None:
                continue
            t0 = time.monotonic()
            try:
                text = self._wake_detector.transcribe(segment)
            except Exception as exc:
                log.warning("Wake word: transcribe failed: %s", exc)
                continue
            command = self._wake_detector.detect(text)
            hit = command is not None
            log.info("Wake segment %.1fs → %r in %.2fs%s",
                     len(segment) / (2 * self.cfg["sample_rate"]), text,
                     time.monotonic() - t0, " (wake)" if hit else "")
            if hit:
                # Audio queued while Whistle transcribed is the start of
                # what the user says next: hand it to the endpointer.
                backlog = []
                while True:
                    try:
                        backlog.append(self.wake_q.get_nowait())
                    except queue.Empty:
                        break
                self._on_wake(text, segment, backlog, command)
                gate.reset()

    def _endpointer_loop(self) -> None:
        """Drive the `Endpointer` (see endpointer.py) from `audio_q`.

        The endpointer commits utterances on `silence_timeout_ms` pauses;
        this loop adds what needs bridge state: generation changes, the
        HID force-commit, ducking while the user talks, and auto-idle after
        `self._idle_window_ms` of silence (`idle_timeout_ms` = 0 disables it).
        """
        ep = Endpointer(self.cfg, self._speech_vad)
        sr = ep.sample_rate
        idle_enabled = int(self.cfg.get("idle_timeout_ms", 0)) > 0
        seen_gen = self._current_gen()
        # Music is ducked while the endpointer is inside an utterance: from
        # the first above-threshold chunk until it is committed or dropped.
        ducked = False

        def follow_ducking(reason: str) -> None:
            nonlocal ducked
            if ep.in_speech != ducked:
                ducked = ep.in_speech
                (self._duck_acquire if ducked else self._duck_release)(reason)

        def committed(pcm: bytes, gen: int) -> None:
            if self._enqueue_utterance(gen, pcm, sr):
                self._pause_for_processing()

        while not self.shutdown_event.is_set():
            # A gen bump (resume) sheds any half-built utterance.
            cur_gen = self._current_gen()
            if cur_gen != seen_gen:
                ep.reset()
                follow_ducking(endpointer.DISCARDED)
                seen_gen = cur_gen

            # Playback just ended: the idle window starts now. Drop the
            # chunks queued while aplay ran too — the user was listening, and
            # processing them back-to-back would burn the fresh window down
            # before the first wall-clock-fresh chunk arrives.
            if self._idle_reset_pending.is_set():
                self._idle_reset_pending.clear()
                ep.silence = 0
                self._drain_queue(self.audio_q)

            # HID press while recording = "send what I said and stop
            # listening": commit now instead of waiting for the pause.
            if self._force_commit.is_set():
                self._force_commit.clear()
                result = ep.force_commit()
                follow_ducking(endpointer.COMMITTED)
                if result:
                    self._enqueue_utterance(cur_gen, result[1], sr)

            try:
                gen, data = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != cur_gen:
                continue

            result = ep.feed(data)
            follow_ducking(result[0] if result else "speech")
            if result:
                kind, pcm = result
                if kind == endpointer.COMMITTED:
                    self._set_idle_window("idle_timeout_ms")
                    committed(pcm, cur_gen)
                elif kind == endpointer.TOO_LONG:
                    # In wake mode only the wake word brings it back.
                    self._enter_idle(source=f"utterance>{self.cfg.get('max_utterance_ms')}ms")

            idle_ms = self._idle_window_ms
            idle_chunks = int(idle_ms / ep.chunk_ms) if idle_enabled and idle_ms > 0 else 0
            if (not ep.in_speech
                    and idle_chunks > 0
                    and ep.silence >= idle_chunks
                    and not self._is_playing()
                    and not self._idle_reset_pending.is_set()):
                # The `_idle_reset_pending` guard closes a race at the tail
                # of a reply: the player sets it just before clearing
                # `_player_proc`, so "not playing" with a pending reset means
                # the next tick zeroes the stale playback-long silence first.
                self._enter_idle(source=f"silence>{idle_ms}ms")
                ep.silence = 0

    def _enqueue_utterance(self, gen: int, pcm: bytes, sr: int) -> bool:
        """Hand a committed utterance to the worker, holding at most ONE
        message in line behind the turn in flight. While the worker is busy
        (gateway or playback), the first utterance waits; anything said after
        it is dropped — otherwise interjections, side talk and speaker echo
        pile into a muddled turn."""
        if self._worker_busy.is_set() and not self.utterance_q.empty():
            log.info("Endpointer: one message already queued, dropping %.2fs utterance",
                     len(pcm) / (sr * 2))
            return False
        self.utterance_q.put((gen, pcm, sr))
        return True

    @_transition
    def _unidle_for_reply(self) -> None:
        """A reply's first audio is about to play: if the bridge went idle
        on its own meanwhile (auto-idle, or the pause for processing), open
        the mic again so the user can talk back the moment it ends."""
        if not self._auto_idled:
            return  # an explicit HID mute wins: play, but stay muted
        log.info("Player: un-idling for playback (auto-idle, "
                 "resume mic + LED off)")
        self._set_state(State.RECORDING)
        self._idle_reset_pending.set()

    @_transition
    def _unmute_for_external(self) -> None:
        """say_to_speaker: unmute like a normal speech (mirrors the HID
        resume and the player's auto-resume); idempotent if unmuted."""
        if self._state is not State.RECORDING:
            log.info("play_pcm: unmuting for external speech (mic open, LED off)")
        self._set_state(State.RECORDING)

    @_transition
    def _pause_for_processing(self) -> None:
        """Stop recording while the committed utterance is processed.

        The player resumes the mic when the reply starts (via `_auto_idled`,
        same as an auto-idle), and the worker resumes it if the turn ends
        without a reply (noise, empty STT, gateway error). The LED and
        firmware mute stay off: this is not a mute, just not listening.
        """
        if self._state is not State.RECORDING:
            return
        log.info("Processing: mic paused until the reply starts")
        self._set_state(State.PROCESSING, auto_idled=True)
        self._drain_queue(self.audio_q)

    @_transition
    def _resume_after_processing(self) -> None:
        """Turn ended with nothing to play: listen again."""
        if self._state is not State.PROCESSING:
            return
        log.info("Processing: no reply, resuming mic")
        self._idle_reset_pending.set()
        self._set_state(State.RECORDING)

    def _discard_utterances(self, gen: int) -> int:
        """Drop every queued utterance; returns how many were under `gen`
        (stale-gen ones, left by a resume, are dropped without counting)."""
        dropped = 0
        while True:
            try:
                g, _pcm, _sr = self.utterance_q.get_nowait()
            except queue.Empty:
                return dropped
            dropped += g == gen

    def _worker_loop(self) -> None:
        while not self.shutdown_event.is_set():
            # Every `continue` below lands back here, so busy spans exactly
            # pick-up → reply handed to the player (or turn dropped).
            self._worker_busy.clear()
            if (self._state is State.PROCESSING and self.utterance_q.empty()
                    and not self._reply_playing()):
                self._resume_after_processing()
            try:
                gen, pcm, sr = self.utterance_q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._worker_busy.set()
            if gen != self._current_gen():
                continue

            # Only ONE message waits behind the reply being delivered: hold
            # this utterance until the previous reply has finished playing,
            # and drop anything else said meanwhile (the endpointer already
            # refuses a second one while we're busy; this catches the rest).
            # Well-behaved turn-taking pays no latency: when no reply is in
            # flight the wait loop doesn't run and the utterance is sent
            # immediately.
            extra = self._discard_utterances(gen)
            while self._reply_playing():
                if self.shutdown_event.is_set() or gen != self._current_gen():
                    break
                self.shutdown_event.wait(0.1)
                extra += self._discard_utterances(gen)
            if gen != self._current_gen():
                continue
            if extra:
                log.info("Worker: dropped %d extra utterance(s) queued behind the reply",
                         extra)

            t0 = time.monotonic()
            self._turn_t0 = t0
            timing: dict[str, float] = {}
            # Dead-air feedback from pick-up until the first reply audio.
            cue = _ThinkingCue(self, gen).start()
            try:
                self._run_turn(gen, pcm, sr, t0, timing, cue)
            finally:
                cue.stop()

    def _run_turn(self, gen: int, pcm: bytes, sr: int, t0: float,
                  timing: dict, cue: "_ThinkingCue") -> None:
        """One turn: STT → gateway → TTS → playback_q. Returns early (turn
        dropped) on empty/non-speech STT or a stale generation."""
        log.info("Worker: STT (%d bytes ≈ %.2fs)", len(pcm), len(pcm) / (sr * 2))
        t_stt = time.monotonic()
        text = self.stt.transcribe(pcm, sr)
        timing["stt"] = time.monotonic() - t0
        if self._stt_compare:
            self._stt_compare.submit(pcm, sr, text, time.monotonic() - t_stt)
        if not text:
            log.info("Worker: empty transcription, skipping")
            return
        if _is_non_speech(text):
            log.info("Worker: non-speech transcription %s, skipping", text)
            return
        if gen != self._current_gen():
            return
        log.info("User: %s", text)

        backend = self.cfg.get("gateway_backend", "openclaw")
        if backend == "zeroclaw_ws":
            log.info("Worker: → gateway %s (backend=zeroclaw_ws agent=%s session=%s)",
                     self.cfg["gateway_base_url"],
                     self.cfg.get("gateway_agent", "default"),
                     self.cfg.get("session_key", "voice-bridge"))
            text_stream = gateway_chat_stream_zeroclaw_ws(
                self.cfg["gateway_base_url"],
                self.cfg["gateway_token"],
                text,
                self.cfg.get("gateway_agent", "default"),
                self.cfg.get("session_key", "voice-bridge"),
                on_event=cue.on_event,
            )
        elif backend == "zeroclaw":
            log.info("Worker: → gateway %s (backend=zeroclaw)",
                     self.cfg["gateway_base_url"])
            text_stream = gateway_chat_stream_zeroclaw(
                self.cfg["gateway_base_url"],
                self.cfg["gateway_token"],
                text,
            )
        else:
            log.info("Worker: → gateway %s (backend=openclaw model=%s session=%s)",
                     self.cfg["gateway_base_url"],
                     self.cfg["voice_model"],
                     self.cfg.get("session_key", "voice-bridge"))
            text_stream = gateway_chat_stream(
                self.cfg["gateway_base_url"],
                self.cfg["gateway_token"],
                text,
                self.cfg["voice_model"],
                self.cfg.get("session_key", "voice-bridge"),
            )
        collected: list[str] = []

        def _tee(s: Iterable[str]) -> Iterator[str]:
            for delta in s:
                if gen != self._current_gen():
                    return
                timing.setdefault("first_token", time.monotonic() - t0)
                collected.append(delta)
                yield delta
            timing["gateway_done"] = time.monotonic() - t0

        try:
            for chunk in self.tts.synthesize_stream(_filter_no_reply(_tee(text_stream))):
                if gen != self._current_gen():
                    break
                if not chunk:
                    continue
                if "first_pcm" not in timing:
                    timing["first_pcm"] = time.monotonic() - t0
                    cue.stop()  # before the first chunk: no cue lands behind it
                self.playback_q.put((gen, chunk))
        except Exception as exc:
            log.error("Worker: TTS pipeline error: %s", exc)
        finally:
            cue.stop()
            full_reply = "".join(collected).strip()
            self._reply_is_question = _expects_answer(full_reply)
            if _is_no_reply(full_reply) and gen == self._current_gen():
                # Agent said "stay silent" — play a short low beep so
                # the user gets feedback that the turn was processed
                # but nothing needed saying.
                log.info("Binary: %s (sentinel — low beep)", full_reply)
                beep = _make_beep_pcm(
                    self.cfg["tts_sample_rate"],
                    freq=220,
                    duration=0.18,
                )
                self.playback_q.put((gen, beep))
            # Always emit the end-of-utterance marker (even after
            # cancel) so the player can release the current aplay
            # cleanly. Stale gen → player drops it harmlessly.
            self.playback_q.put((gen, _END_OF_UTTERANCE))

        if collected and not _is_no_reply(full_reply):
            log.info("Binary: %s", "".join(collected)[:200])
        log.info("Turn timing (s from pick-up): %s, cues=%d",
                 " ".join(f"{k}={v:.2f}" for k, v in timing.items()) or "n/a",
                 cue.cues_played)

    def _next_playback(self, timeout: float):
        """Next player item: deferred external utterances first, then the queue."""
        if self._player_backlog:
            return self._player_backlog.popleft()
        return self.playback_q.get(timeout=timeout)

    def _set_idle_window(self, key: str) -> None:
        """Put the idle window named by config `key` in force (any of the
        `idle_*_ms` keys); a missing key falls back to `idle_timeout_ms`."""
        self._idle_window_ms = int(self.cfg.get(key, self.cfg.get("idle_timeout_ms", 0)))

    @_transition
    def _after_reply(self, *, cue: bool = False) -> None:
        """A reply (or say_to_speaker speech) finished playing: widen the
        idle window so the user has time to answer — more if it asked."""
        if cue:
            return
        self._replies_since_resume += 1
        self._quiet_idle = False
        self._set_idle_window("idle_after_question_ms" if self._reply_is_question
                              else "idle_after_reply_ms")
        self._reply_is_question = False

    @_transition
    def _after_external(self, question: bool) -> None:
        """say_to_speaker finished. A question opens a conversation: the
        answer window, and a goodbye later if nobody answers. A plain
        announcement keeps the normal window and, unless a conversation was
        already going, dozes off silently — no "Ciao ciao!" after "La lavatrice
        ha finito"."""
        if question:
            self._replies_since_resume += 1
            self._quiet_idle = False
            self._set_idle_window("idle_after_question_ms")
            return
        self._set_idle_window("idle_timeout_ms")
        if not self._replies_since_resume:
            self._quiet_idle = True

    def _player_loop(self) -> None:
        device = self.cfg["output_device"]
        sample_rate = self.cfg["tts_sample_rate"]
        while not self.shutdown_event.is_set():
            try:
                gen, item = self._next_playback(timeout=0.2)
            except queue.Empty:
                continue
            # Externally-supplied speech (MCP say_to_speaker) and thinking
            # cues arrive as one atomic item; play it as its own utterance
            # via the same path, then release the blocked caller.
            # play_pcm() already did the unmute transition before enqueuing.
            if isinstance(item, _ExternalUtterance):
                if gen == self._current_gen():
                    ctx = self._ducked(item.label) if item.duck else contextlib.nullcontext()
                    with ctx:
                        self._play_blob(item.pcm, device, sample_rate)
                    if gen == self._current_gen() and not item.cue:  # not cut by a barge-in
                        self._after_external(item.question)
                item.done.set()
                continue

            # Skip end-of-utterance markers that arrive with no
            # preceding audio (e.g. worker bailed before producing any
            # PCM), and stale-gen anything.
            if isinstance(item, _EndOfUtterance) or gen != self._current_gen():
                continue

            # If the bridge auto-idled while this turn was still being
            # processed (worker → TTS), the device is currently muted
            # and the mic is closed. Resume recording before playing the
            # reply so the user can talk back the moment it ends; reset
            # the idle silence counter so the next idle window measures
            # from end-of-playback. Only fires for *auto*-idle — if the
            # user explicitly pressed HID to mute, `_auto_idled` is
            # clear and we leave the mic muted (the press wins).
            self._unidle_for_reply()

            with self._ducked("reply"):
                if self._play_streamed(item, device, sample_rate, gen):
                    self._after_reply()

    def _next_reply_item(self, gen: int, timeout: float):
        """Next item of the reply streaming under `gen`: PCM bytes,
        `_END_OF_UTTERANCE`, `_STALE` (cut by a gen change) or None (nothing
        yet). External utterances met meanwhile are deferred to the backlog
        (played right after the reply); stale ones release their caller."""
        try:
            gen2, item = self.playback_q.get(timeout=timeout)
        except queue.Empty:
            return None
        if gen2 != gen or gen2 != self._current_gen():
            if isinstance(item, _ExternalUtterance):
                item.done.set()
            return _STALE
        if isinstance(item, _ExternalUtterance):
            self._player_backlog.append((gen2, item))
            return None
        return item

    def _prebuffer(self, first: bytes, gen: int, sample_rate: int) -> "tuple[list[bytes], bool, bool]":
        """Collect `playback_prebuffer_ms` of PCM before the first aplay
        write. ElevenLabs v3 often sends one chunk and then stalls ~0.5 s;
        writing that chunk alone makes aplay start and underrun (click).
        Waits at most prebuffer + 1 s, so a slow stream still starts.

        Returns (chunks, ended, stale): `ended` when the end-of-utterance
        marker was reached, `stale` when the gen changed meanwhile."""
        ms = int(self.cfg.get("playback_prebuffer_ms", 0))
        chunks = [first]
        if ms <= 0:
            return chunks, False, False
        want = sample_rate * 2 * ms // 1000
        have = len(first)
        deadline = time.monotonic() + ms / 1000.0 + 1.0
        while have < want and not self.shutdown_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            item = self._next_reply_item(gen, min(0.05, remaining))
            if item is None:
                continue
            if item is _STALE:
                return chunks, False, True
            if item is _END_OF_UTTERANCE:
                return chunks, True, False
            chunks.append(item)
            have += len(item)
        return chunks, False, False

    def _play_streamed(self, item: bytes, device: str, sample_rate: int,
                       gen: int | None = None) -> bool:
        """Play one reply: `item` is its first PCM chunk, the rest is pulled
        off `playback_q` until the end-of-utterance marker. Returns True when
        the reply played to its end (not cut by a stale gen / barge-in)."""
        if gen is None:
            gen = self._current_gen()
        chunks, ended, stale = self._prebuffer(item, gen, sample_rate)
        if stale:
            return False
        completed = False
        with self._aplay_session(device, sample_rate) as proc:
            if self._turn_t0 is not None:
                log.info("Turn: first audio %.2fs after pick-up", time.monotonic() - self._turn_t0)
                self._turn_t0 = None
            if not self._write_chunk(proc, b"".join(chunks)):
                return False
            if ended:
                return True
            while not self.shutdown_event.is_set():
                item = self._next_reply_item(gen, 0.2)
                if item is None:
                    continue
                if item is _STALE:
                    # Barge-in mid-utterance: aplay has likely been killed
                    # already; break so it is closed cleanly.
                    break
                if item is _END_OF_UTTERANCE:
                    completed = True
                    break
                if not self._write_chunk(proc, item):
                    break
        return completed and gen == self._current_gen()

    def _play_blob(self, pcm: bytes, device: str, sample_rate: int) -> None:
        """Play one complete PCM blob as a single utterance (external
        speech, thinking cues) — same aplay session as a streamed reply."""
        with self._aplay_session(device, sample_rate) as proc:
            self._write_chunk(proc, pcm)

    @contextlib.contextmanager
    def _aplay_session(self, device: str, sample_rate: int):
        """One aplay for one utterance, registered as `_player_proc`.

        While registered, `_is_playing()` is true, so the endpointer can't
        auto-idle and a barge-in can `_kill_player()` it. On exit the tail is
        played out without chopping it (bounded so a hung aplay can't pin
        the thread — see `_drain_aplay`: a fixed timeout + kill would clip
        the reply and, via `sw_dmix`'s mixing, overlap the next utterance).
        `_player_proc` stays set throughout the drain, so the audible tail
        still counts as playing.

        Order matters at the end: the idle reset is signalled BEFORE the
        handle is cleared. The endpointer's idle check also gates on
        `_idle_reset_pending`, so whenever `_player_proc` becomes None the
        flag is already set and the endpointer can never see "not playing +
        stale silence_count" and auto-idle the instant the reply ends; its
        next tick zeroes `silence_count`, restarting the idle window here."""
        proc = _aplay_popen(device, sample_rate, bufsize=0)
        with self._player_lock:
            self._player_proc = proc
        try:
            yield proc
        finally:
            _drain_aplay(proc, sample_rate, abort=self.shutdown_event)
            self._idle_reset_pending.set()
            with self._player_lock:
                self._player_proc = None

    @staticmethod
    def _write_chunk(proc: subprocess.Popen, chunk: bytes) -> bool:
        if not chunk:
            return True
        try:
            proc.stdin.write(chunk)
            return True
        except BrokenPipeError:
            return False

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        # Pin deezer-connect to its configured baseline volume on boot
        # (no-op unless the plugin is enabled).
        self.deezer.apply_default_volume()
        self._duck_q = queue.Queue()
        self._duck_thread = threading.Thread(target=self._duck_loop, name="vb-duck", daemon=True)
        self._duck_thread.start()
        loops = [
            ("hid", self._hid_loop),
            ("recorder", self._recorder_loop),
            ("endpointer", self._endpointer_loop),
            ("worker", self._worker_loop),
            ("player", self._player_loop),
        ]
        if self.wake_mode:
            # Boot idle-but-listening: firmware mic open (the HID monitor
            # boots muted), acks synthesized/loaded off the hot path.
            if self._state is State.IDLE_LISTENING:
                self.hid.set_led(muted=False)
            loops += [("wake", self._wake_loop),
                      ("wake-acks", lambda: self._wake_acks.prepare(self.shutdown_event)),
                      ("sleep-acks", lambda: self._sleep_acks.prepare(self.shutdown_event))]
        if self._thinking_acks:
            loops.append(("thinking-acks", lambda: self._thinking_acks.prepare(self.shutdown_event)))
        if self._stt_compare:
            self._stt_compare.start()
        for name, fn in loops:
            t = threading.Thread(target=fn, name=f"vb-{name}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self.shutdown_event.set()
        # Wake any thread parked on `recording.wait()` so it notices
        # shutdown immediately instead of waiting out its timeout.
        self.recording.set()
        # Kill the active aplay so the player loop's wait returns
        # without hitting the 2s timeout.
        self._kill_player()
        for t in self._threads:
            t.join(timeout=3.0)
        # Let queued volume calls finish, then go back to inline calls.
        with self._duck_lock:
            q, self._duck_q = self._duck_q, None
            self._duck_holds = 0
        if q is not None:
            q.put(None)
            self._duck_thread.join(timeout=3.0)
        # Restore deezer-connect's volume if we were ducked when stop
        # arrived (SIGTERM mid-playback or mid-speech). No-op when the
        # plugin is disabled or wasn't currently ducking.
        self.deezer.unduck()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()

    # Validate only the API keys actually needed for the chosen
    # providers (voice-bridge.json decides which).
    needed_providers = {cfg["stt_provider"], cfg["tts_provider"]}
    if "elevenlabs" in needed_providers and not cfg.get("elevenlabs_key"):
        log.error("ElevenLabs selected but no ElevenLabs API key in gateway config")
        sys.exit(1)
    if cfg["tts_provider"] == "elevenlabs":
        if not cfg.get("elevenlabs_voice"):
            log.error("messages.tts.providers.elevenlabs.voiceId missing in openclaw config")
            sys.exit(1)
        if not cfg.get("elevenlabs_model"):
            log.error("messages.tts.providers.elevenlabs.modelId missing in openclaw config")
            sys.exit(1)
    if "deepgram" in needed_providers and not cfg.get("deepgram_key"):
        log.error("Deepgram selected but no Deepgram API key (env DEEPGRAM_API_KEY or gateway config)")
        sys.exit(1)
    if not cfg.get("gateway_token"):
        log.error("No gateway token in config")
        sys.exit(1)

    log.info("Config loaded")
    log.info("STT provider: %s", cfg["stt_provider"])
    log.info("TTS provider: %s (voice=%s model=%s rate=%dHz stream=%s whole_reply=%s)",
             cfg["tts_provider"], cfg["elevenlabs_voice"], cfg["elevenlabs_model"],
             cfg["tts_sample_rate"], cfg["tts_streaming_mode"], cfg["tts_whole_reply"])
    log.info("Output: %s @ %d Hz", cfg["output_device"], cfg["tts_sample_rate"])
    _apply_output_volume(cfg)
    log.info("VAD: rms_threshold=%g pause_commit=%dms idle=%dms",
             cfg["vad_rms_threshold"], cfg["silence_timeout_ms"], cfg["idle_timeout_ms"])
    if not cfg.get("hid_mute_enabled") and cfg["idle_timeout_ms"] > 0:
        log.warning("HID disabled but idle_timeout_ms>0 — auto-idle will be unrecoverable; "
                    "set idle_timeout_ms=0 or hid_mute_enabled=true")

    stt = _build_voice_provider("stt", cfg)
    tts = _build_voice_provider("tts", cfg)

    hid = HidMuteMonitor()
    if cfg.get("hid_mute_enabled"):
        hid.start()

    bridge = VoiceBridge(cfg, stt, tts, hid)

    def _sigterm(_signum, _frame):
        log.info("Shutdown requested")
        bridge.shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    bridge.start()
    if cfg["activation"] == "wake_word":
        log.info("Ready — say the wake word to begin (Jabra button mutes/resumes)")
    elif cfg.get("hid_mute_enabled"):
        log.info("Ready — device starts muted, press the Jabra button to begin")
    else:
        log.info("Ready — listening (always-on mic, HID button disabled)")

    # Serve the MCP voice tools from inside this process when
    # `mcp_server.enabled` is true, sharing this process's config and the
    # same provider instances. Best-effort: a failure to start the MCP server
    # is caught and logged so the always-on voice client keeps running.
    if (cfg.get("mcp_server") or {}).get("enabled"):
        try:
            import mcp_voice_server
            mcp_voice_server.configure(cfg, stt=stt, tts=tts, bridge=bridge)
            mcp_voice_server.serve_background()
            log.info("MCP server: http://%s:%d",
                     mcp_voice_server._HOST, mcp_voice_server._PORT)
        except Exception:
            log.exception(
                "MCP server failed to start; voice client continues"
            )

    # Block here until SIGTERM/SIGINT. Worker threads do all the work;
    # main is only around to own the signal handlers and the cleanup.
    try:
        while not bridge.shutdown_event.wait(1.0):
            pass
    finally:
        bridge.stop()
        hid.stop()
        log.info("Stopped")


if __name__ == "__main__":
    main()
