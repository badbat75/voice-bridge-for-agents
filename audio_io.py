"""Audio plumbing: aplay playback, generated tones, mic lookup, voice check.

The bridge's audio leaves through `aplay` subprocesses (raw S16LE mono at
`tts_sample_rate`, see `_aplay_popen` / `_drain_aplay`) and enters through
PyAudio (`find_input_device`). `_voiced_ms` / `_make_speech_vad` are the
webrtcvad check the endpointer runs at commit.
"""

from __future__ import annotations

import array
import contextlib
import fcntl
import functools
import logging
import math
import os
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pyaudio

log = logging.getLogger("voice-bridge")

def find_input_device(pa: pyaudio.PyAudio) -> int | None:
    """Find Jabra SPEAK 510 input device index."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "jabra" in info["name"].lower() and info["maxInputChannels"] > 0:
            return i
    return None


_AUDIO_DEBUG = os.environ.get("VOICE_BRIDGE_DEBUG_AUDIO") == "1"


def _drain_aplay_stderr(stream) -> None:
    """Forward aplay's stderr line-by-line to the Python logger.

    Runs as a daemon thread; exits when aplay closes stderr.
    """
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                log.warning("aplay: %s", line)
    except ValueError:
        pass  # the stream was closed under us when aplay was reaped
    finally:
        with contextlib.suppress(Exception):
            stream.close()


def _aplay_popen(device: str, sample_rate: int, *, bufsize: int = -1) -> subprocess.Popen:
    """Spawn aplay for raw S16LE mono PCM at `sample_rate`.

    With `VOICE_BRIDGE_DEBUG_AUDIO=1`, drops `-q` and forwards aplay's
    stderr to the logger — that's where ALSA prints `underrun!!!` lines.
    """
    cmd = ["aplay", "-D", device, "-f", "S16_LE", "-r", str(sample_rate), "-c", "1"]
    if not _AUDIO_DEBUG:
        cmd.insert(1, "-q")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE if _AUDIO_DEBUG else subprocess.DEVNULL,
        bufsize=bufsize,
    )
    if _AUDIO_DEBUG and proc.stderr is not None:
        threading.Thread(
            target=_drain_aplay_stderr,
            args=(proc.stderr,),
            daemon=True,
        ).start()
    return proc


def _drain_aplay(
    proc: subprocess.Popen,
    sample_rate: int,
    abort: "threading.Event | None" = None,
) -> None:
    """Close aplay's stdin and wait for it to play out its buffered PCM,
    then make sure it has exited.

    Network TTS feeds PCM faster than realtime, so at stdin-close the
    unplayed tail is the kernel pipe buffer (queried exactly via
    `F_GETPIPE_SZ`) plus aplay's ALSA ring buffer. We derive a
    generous-but-bounded drain budget from that buffer size and the stream
    rate, then poll until aplay exits on its own — we never kill a
    still-draining process, which would chop the reply's tail and, because
    `voice_out` runs through `sw_dmix` (which *mixes* streams), briefly
    overlap the next utterance's aplay. The budget scales with
    `sample_rate`, so it stays correct if `tts_sample_rate` changes.

    `abort`, if given, cuts the wait short (bridge shutdown). A genuinely
    hung aplay is killed once the budget is spent, so the caller can never
    block forever.
    """
    bytes_per_sec = sample_rate * 2  # S16LE mono
    try:
        pipe_bytes = fcntl.fcntl(proc.stdin.fileno(), fcntl.F_GETPIPE_SZ)
    except Exception:
        pipe_bytes = 65536  # Linux default pipe capacity
    # pipe drain time + ALSA buffer headroom & safety margin.
    drain_budget = pipe_bytes / bytes_per_sec + 2.0
    with contextlib.suppress(Exception):
        proc.stdin.close()
    deadline = time.monotonic() + drain_budget
    while abort is None or not abort.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            proc.wait(timeout=min(0.2, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    if proc.poll() is None:
        # Budget exhausted (hung aplay) or aborted — drop it so the
        # caller can move on.
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=1.0)


def _apply_output_volume(cfg: dict) -> None:
    """Re-assert the configured softvol level on the output control.

    The `voice_out` softvol control gives the bridge a volume independent
    from deezer-connect's player gain. ALSA creates that control lazily on
    first PCM open and resets it to 100% after a reboot, so we (1) open the
    device with a brief silent buffer to instantiate the control, then
    (2) set it with `amixer`. Best-effort: any failure is logged and
    swallowed — a missing mixer must never take the bridge down (it just
    means the level stays at ALSA's 100% default)."""
    control = cfg.get("output_volume_control")
    card = cfg.get("output_volume_card")
    percent = int(cfg.get("output_volume_percent", 100))
    if not control or card is None:
        return
    device = cfg["output_device"]
    rate = int(cfg["tts_sample_rate"])
    # ~50 ms of silence: enough to make ALSA open the PCM and create the
    # softvol control element, inaudible since every sample is zero.
    silence = b"\x00\x00" * (rate // 20)
    try:
        proc = _aplay_popen(device, rate)
        proc.communicate(input=silence, timeout=5)
    except Exception as exc:
        log.warning("output volume: could not prime %s (%s); "
                    "amixer set may fail", device, exc)
    try:
        subprocess.run(
            ["amixer", "-c", str(card), "sset", control, f"{percent}%"],
            check=True, capture_output=True, timeout=5,
        )
        log.info("Output volume: %s on card %s → %d%%", control, card, percent)
    except Exception as exc:
        log.warning("output volume: amixer set %s on card %s failed: %s",
                    control, card, exc)


def play_audio(device: str, audio_data: bytes, sample_rate: int) -> None:
    """Play raw S16LE mono PCM bytes through aplay at `sample_rate`.

    `sample_rate` MUST match what the TTS provider produced (we ask it
    for `pcm_<rate>` / `linear16` at the same number) — otherwise aplay
    plays back at the wrong speed/pitch."""
    proc = _aplay_popen(device, sample_rate)
    # Not `communicate()`: with VOICE_BRIDGE_DEBUG_AUDIO=1 a drain thread
    # already owns aplay's stderr, and two readers race to EBADF.
    with contextlib.suppress(BrokenPipeError):
        proc.stdin.write(audio_data)
    _drain_aplay(proc, sample_rate)


def play_audio_stream(device: str, audio_iter: Iterable[bytes], sample_rate: int) -> bool:
    """Pipe audio chunks through aplay as they arrive.

    aplay is started immediately (so ALSA acquires the device early) and
    each chunk is written to its stdin the moment it's pulled from
    `audio_iter`. ALSA's own period buffer absorbs short pauses in the
    producer (e.g. waiting on the next sentence from TTS). With
    `bufsize=0`, every write goes straight to the pipe — no Python-level
    buffering between TTS chunks and ALSA.

    Returns True if at least one chunk was written (i.e. something was
    actually played), so callers can distinguish "speech happened" from
    "stream produced nothing".
    """
    proc = _aplay_popen(device, sample_rate, bufsize=0)
    wrote_any = False
    try:
        for chunk in audio_iter:
            if not chunk:
                continue
            try:
                proc.stdin.write(chunk)
            except BrokenPipeError:
                # aplay died (device gone, ALSA error). Stop pulling
                # from the upstream iterators — still need to drain the
                # process below.
                break
            wrote_any = True
    finally:
        # Bounded drain (see `_drain_aplay`): play out the buffered tail
        # without chopping it, but don't hang on a dead aplay.
        _drain_aplay(proc, sample_rate)
    return wrote_any


@functools.lru_cache(maxsize=16)
def _make_beep_pcm(sample_rate: int, freq: float, duration: float, amplitude: int = 16000) -> bytes:
    """Generate a fade-out sine beep as S16LE mono PCM bytes.

    Cached: the same few tones (tick, beep, sleep tone) are replayed many
    times, and the sine loop is pure Python."""
    n_samples = int(sample_rate * duration)
    buf = array.array("h")
    for i in range(n_samples):
        env = 1.0 - (i / n_samples)
        val = int(amplitude * env * (0.5 + 0.5 * math.sin(2 * math.pi * freq * i / sample_rate)))
        buf.append(max(-32768, min(32767, val)))
    return buf.tobytes()


def _make_tick_pcm(sample_rate: int) -> bytes:
    """Soft, short "still working" tick — quiet enough to sit under music."""
    return _make_beep_pcm(sample_rate, freq=520, duration=0.06, amplitude=4000)


def _make_sleep_tone_pcm(sample_rate: int) -> bytes:
    """Two soft falling notes: the bridge dozed off without a conversation."""
    return (_make_beep_pcm(sample_rate, freq=660, duration=0.09, amplitude=6000)
            + _make_beep_pcm(sample_rate, freq=440, duration=0.12, amplitude=6000))


def play_beep(device: str, sample_rate: int) -> None:
    """Play a short ~80 ms 880 Hz beep at the given output sample rate."""
    play_audio(device, _make_beep_pcm(sample_rate, freq=880, duration=0.08), sample_rate)


_VAD_RATES = (8000, 16000, 32000, 48000)
_VAD_FRAME_MS = 30


def _voiced_ms(pcm: bytes, sample_rate: int, vad) -> int | None:
    """Milliseconds of voiced 30 ms frames in S16LE mono `pcm`, per a
    `webrtcvad.Vad`. None when the rate is one webrtcvad can't take, so the
    caller can skip the check instead of dropping speech."""
    if sample_rate not in _VAD_RATES:
        return None
    frame_bytes = sample_rate * _VAD_FRAME_MS // 1000 * 2
    voiced = 0
    for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        if vad.is_speech(pcm[i:i + frame_bytes], sample_rate):
            voiced += _VAD_FRAME_MS
    return voiced


def _make_speech_vad(cfg: dict):
    """A `webrtcvad.Vad` when `speech_filter.enabled`, else None. A missing
    module only logs: the filter is an optimization, never a hard dep."""
    sf = cfg.get("speech_filter") or {}
    if not sf.get("enabled"):
        return None
    try:
        import webrtcvad
    except ImportError:
        log.warning("speech_filter enabled but webrtcvad is not installed — filter off")
        return None
    return webrtcvad.Vad(int(sf.get("aggressiveness", 2)))
