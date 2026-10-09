"""Reply playback: one aplay per utterance, fed from a queue.

`Player` owns everything that leaves through the speaker on the bridge's
behalf: the `q` the worker (reply PCM), the thinking cue and `play_pcm`
(`ExternalUtterance`) write to, the backlog of external utterances that
arrived mid-reply, and the active aplay process. It knows nothing of the
bridge's state machine: `VoiceBridge` hands it the callbacks it needs
(current generation, ducking, "reply starts", "reply done", ...).

Items are `(gen, payload)`; anything whose gen is no longer the current
one is dropped (a barge-in bumped it). A streamed reply is its PCM chunks
followed by `END_OF_UTTERANCE`; an `ExternalUtterance` is one atomic item.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from typing import Callable

from audio_io import _aplay_popen, _drain_aplay

log = logging.getLogger("voice-bridge")


class EndOfUtterance:
    """Marks the end of a streamed reply on the queue. Plain None would
    conflict with empty-chunk filtering; an explicit object is unambiguous."""


END_OF_UTTERANCE = EndOfUtterance()
# Returned by `_next_reply_item` when the reply was cut by a gen change.
_STALE = object()


class ExternalUtterance:
    """A whole externally-supplied utterance (MCP `say_to_speaker`, thinking
    cues), enqueued as ONE item so it can never interleave with a streamed
    reply's chunks. The player plays `pcm` as its own utterance and sets
    `done` when it finishes (or goes stale), releasing a blocked caller.

    `duck`: lower the music while it plays. `cue`: a thinking cue — it
    doesn't count as a reply. `question`: the speech expects an answer."""

    def __init__(self, pcm: bytes, done: "threading.Event", *,
                 label: str = "say_to_speaker", duck: bool = True,
                 cue: bool = False, question: bool = False) -> None:
        self.pcm = pcm
        self.done = done
        self.label = label
        self.duck = duck
        self.cue = cue
        self.question = question


class Player:
    """Plays the queue, one aplay per utterance (see module doc).

    Callbacks, all supplied by `VoiceBridge`:
      current_gen()             the generation in force
      ducked(reason)            context manager holding the music down
      on_reply_start()          a streamed reply's first PCM is about to play
      on_reply_done()           a streamed reply played to its end
      on_external_done(q)       an ExternalUtterance (not a cue) played; q =
                                it expects an answer
      on_audio_end()            an aplay session ended — set BEFORE `proc` is
                                cleared, so "not playing" is never observed
                                without it
      popen(device, rate, bufsize=)  spawns aplay (tests fake it)
    """

    def __init__(self, cfg: dict, shutdown: "threading.Event", *,
                 current_gen: Callable[[], int],
                 ducked: Callable[[str], "contextlib.AbstractContextManager"],
                 on_reply_start: Callable[[], None],
                 on_reply_done: Callable[[], None],
                 on_external_done: Callable[[bool], None],
                 on_audio_end: Callable[[], None],
                 popen: Callable[..., subprocess.Popen] = _aplay_popen) -> None:
        self._cfg = cfg
        self._shutdown = shutdown
        self._gen = current_gen
        self._ducked = ducked
        self._on_reply_start = on_reply_start
        self._on_reply_done = on_reply_done
        self._on_external_done = on_external_done
        self._on_audio_end = on_audio_end
        self._popen = popen
        # Unbounded: an utterance is ~10 s of audio (~500 KB at 24 kHz).
        self.q: "queue.Queue[tuple[int, object]]" = queue.Queue()
        # External utterances met while a streamed reply was playing; they
        # play right after it.
        self.backlog: "deque[tuple[int, object]]" = deque()
        # The active aplay, under `_lock` so a barge-in from another thread
        # can kill it without racing setup/teardown.
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        # Pick-up time of the turn in flight (set by the worker), for the
        # "first audio" log line.
        self.turn_t0: float | None = None

    # -- queries / control from other threads ----------------------------
    def is_playing(self) -> bool:
        with self._lock:
            return self.proc is not None

    def busy(self) -> bool:
        """Something is audible or queued to be."""
        return self.is_playing() or not self.q.empty() or bool(self.backlog)

    def kill(self) -> None:
        with self._lock:
            proc = self.proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception as exc:
            log.warning("kill aplay failed: %s", exc)

    def wake(self) -> None:
        """Wake an idle `run()` (shutdown): a stale marker it ignores."""
        self.q.put((-1, END_OF_UTTERANCE))

    def drain(self) -> None:
        """Empty the queues, releasing any blocked `ExternalUtterance` caller."""
        items = list(self.backlog)
        self.backlog.clear()
        while True:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                break
        for _gen, item in items:
            if isinstance(item, ExternalUtterance):
                item.done.set()

    # -- the thread --------------------------------------------------------
    def run(self) -> None:
        device = self._cfg["output_device"]
        sample_rate = self._cfg["tts_sample_rate"]
        while not self._shutdown.is_set():
            try:
                # Blocks; `wake()` (bridge shutdown) or the 1 s safety timeout
                # gets it back to the shutdown check.
                gen, item = self._next(timeout=1.0)
            except queue.Empty:
                continue
            if isinstance(item, ExternalUtterance):
                if gen == self._gen():
                    ctx = self._ducked(item.label) if item.duck else contextlib.nullcontext()
                    with ctx:
                        self.play_blob(item.pcm, device, sample_rate)
                    if gen == self._gen() and not item.cue:  # not cut by a barge-in
                        self._on_external_done(item.question)
                item.done.set()
                continue
            # A marker with no audio before it (the worker bailed before any
            # PCM), or anything stale.
            if isinstance(item, EndOfUtterance) or gen != self._gen():
                continue
            self._on_reply_start()
            with self._ducked("reply"):
                if self.play_streamed(item, device, sample_rate, gen):
                    self._on_reply_done()

    def _next(self, timeout: float):
        """Deferred external utterances first, then the queue."""
        if self.backlog:
            return self.backlog.popleft()
        return self.q.get(timeout=timeout)

    def _next_reply_item(self, gen: int, timeout: float):
        """Next item of the reply streaming under `gen`: PCM bytes,
        `END_OF_UTTERANCE`, `_STALE` (cut by a gen change) or None (nothing
        yet). External utterances met meanwhile go to the backlog; stale
        ones release their caller."""
        try:
            gen2, item = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        if gen2 != gen or gen2 != self._gen():
            if isinstance(item, ExternalUtterance):
                item.done.set()
            return _STALE
        if isinstance(item, ExternalUtterance):
            self.backlog.append((gen2, item))
            return None
        return item

    def _prebuffer(self, first: bytes, gen: int, sample_rate: int) -> "tuple[list[bytes], bool, bool]":
        """Collect `playback_prebuffer_ms` of PCM before the first aplay
        write. ElevenLabs v3 often sends one chunk and then stalls ~0.5 s;
        writing that chunk alone makes aplay start and underrun (click).
        Waits at most prebuffer + 1 s, so a slow stream still starts.
        Returns (chunks, ended, stale): `ended` when the end-of-utterance
        marker was reached, `stale` when the gen changed meanwhile."""
        ms = int(self._cfg.get("playback_prebuffer_ms", 0))
        chunks = [first]
        if ms <= 0:
            return chunks, False, False
        want = sample_rate * 2 * ms // 1000
        have = len(first)
        deadline = time.monotonic() + ms / 1000.0 + 1.0
        while have < want and not self._shutdown.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            item = self._next_reply_item(gen, min(0.05, remaining))
            if item is None:
                continue
            if item is _STALE:
                return chunks, False, True
            if item is END_OF_UTTERANCE:
                return chunks, True, False
            chunks.append(item)
            have += len(item)
        return chunks, False, False

    def play_streamed(self, item: bytes, device: str, sample_rate: int,
                      gen: int | None = None) -> bool:
        """Play one reply: `item` is its first PCM chunk, the rest is pulled
        off the queue until the end-of-utterance marker. True when the reply
        played to its end (not cut by a stale gen / barge-in)."""
        if gen is None:
            gen = self._gen()
        chunks, ended, stale = self._prebuffer(item, gen, sample_rate)
        if stale:
            return False
        completed = False
        with self._session(device, sample_rate) as proc:
            if self.turn_t0 is not None:
                log.info("Turn: first audio %.2fs after pick-up", time.monotonic() - self.turn_t0)
                self.turn_t0 = None
            if not self._write(proc, b"".join(chunks)):
                return False
            if ended:
                return True
            while not self._shutdown.is_set():
                item = self._next_reply_item(gen, 0.2)
                if item is None:
                    continue
                if item is _STALE:
                    # Barge-in mid-utterance: aplay has likely been killed
                    # already; break so it is closed cleanly.
                    break
                if item is END_OF_UTTERANCE:
                    completed = True
                    break
                if not self._write(proc, item):
                    break
        return completed and gen == self._gen()

    def play_blob(self, pcm: bytes, device: str, sample_rate: int) -> None:
        """One complete PCM blob as a single utterance — same aplay session
        as a streamed reply."""
        with self._session(device, sample_rate) as proc:
            self._write(proc, pcm)

    @contextlib.contextmanager
    def _session(self, device: str, sample_rate: int):
        """One aplay for one utterance, registered as `proc`.

        While registered, `is_playing()` is true (the bridge can't auto-idle,
        a barge-in can `kill()` it). On exit the tail is played out without
        chopping it, bounded so a hung aplay can't pin the thread (see
        `_drain_aplay`: a fixed timeout + kill would clip the reply and, via
        `sw_dmix`, overlap the next one). `proc` stays set through the drain,
        so the audible tail still counts as playing.

        `on_audio_end` fires BEFORE `proc` is cleared: the bridge's idle check
        gates on it, so it never sees "not playing + stale silence count"."""
        proc = self._popen(device, sample_rate, bufsize=0)
        with self._lock:
            self.proc = proc
        try:
            yield proc
        finally:
            _drain_aplay(proc, sample_rate, abort=self._shutdown)
            self._on_audio_end()
            with self._lock:
                self.proc = None

    @staticmethod
    def _write(proc: subprocess.Popen, chunk: bytes) -> bool:
        if not chunk:
            return True
        try:
            proc.stdin.write(chunk)
            return True
        except BrokenPipeError:
            return False
