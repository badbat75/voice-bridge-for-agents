"""Jabra SPEAK 510 HID button monitor + mute control.

Watches /dev/hidraw* for the device's mute button and surfaces presses as
edge-triggered wake events on a background daemon thread. Also drives the
device's mute LED via output report 0x03 — and that write controls more
than just the LED: bit 2 of byte 1 in the same report is the firmware's
USB-Audio capture mute switch, verified by PCM probe (0/14400 non-zero
samples in 1 s of capture with the bit set, vs ~14400/14400 with it
clear). So `set_led()` is mis-named-but-keep-it: turning the LED on also
silences the mic at the device level, not just visually.

The monitor owns the full lifecycle of the hidraw fd: on USB disconnect
or read failure it closes the fd and loops on `_open_device()` with a 2 s
backoff until the Jabra reappears or `stop()` is called. The bridge sees
one continuous monitor across plug cycles — no exit, no external
supervisor needed. The mute preference (`_muted_pref`) is remembered
across reconnects so the device snaps back to the bridge's intended
state on replug.

Idle CPU is ~0 by design. The reconnect backoff blocks in
`threading.Event.wait()`, not `time.sleep()`, so the thread is parked
on the kernel wait-queue and `stop()` wakes it immediately. While the
device is attached, the read loop's BlockingIOError sleep is also a
short event-wait, for the same reason.

The mute button is a *momentary* HID Telephony button (bit 4 of byte 1
of report 0x03, usage 0x2F "Mic Mute") — high while held, low when
released. A single press emits a 1→0 sequence on the wire, so the
monitor triggers on the rising edge: one wake event per press,
regardless of the release.

The Jabra SPEAK 510 firmware does NOT emit telephony input reports until
the host writes an output report putting the device into off-hook state.
`_engage()` performs that write (combined with the current mute
preference) each time the fd is (re)opened. Without it no mute press
ever reaches userspace. See docs/JABRA.md for the full protocol
reference.

Device discovery matches USB vendor 0B0E / product 0422 via
/sys/class/hidraw/*/device/uevent.

Set JABRA_HID_DEBUG=1 in the environment to log the first 16 bytes of
every incoming HID report — useful when porting to a different Jabra
variant or diagnosing an unexpected layout.
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger(__name__)

# Backoff between reconnect attempts when no Jabra is present. The poll
# thread parks in `_shutdown.wait()` for this interval, so the cost
# while the device is absent is one cheap `_find_device()` (a single
# listdir) every 2 s — effectively zero CPU. `stop()` sets the event
# and the thread returns immediately.
_RECONNECT_BACKOFF_S = 2.0
# Idle wait between non-blocking read attempts when the fd is open but
# has no data. Same Event-based wait (kernel-blocked, not spin) so
# stop() wakes it instantly.
_READ_IDLE_S = 0.05


class HidMuteMonitor:
    """Detect Jabra SPEAK 510 mute-button presses via /dev/hidraw, and
    drive the device's mute LED + firmware capture mute via the same
    output report channel.
    """

    _FEATURE_RPT = 0x03
    _BUTTON_BIT = 0x10  # bit 4 of byte 1 = HID Telephony usage 0x2F (Mic Mute)
    # Output report 0x03 byte 1 bits we drive:
    #   0x01 = LED Off-Hook  → tells firmware "in a call", starts emitting
    #          telephony input reports for the buttons.
    #   0x04 = LED Mute      → both lights the red ring AND mutes the
    #          USB-Audio capture inside the firmware. Setting/clearing
    #          is the bridge's authoritative mute switch.
    _OFFHOOK_BIT = 0x01
    _MUTE_BIT = 0x04

    def __init__(self) -> None:
        self._fd: int | None = None
        self._button_down: bool = False
        self._unmute_event = threading.Event()
        # Set by stop() to wake the poll thread out of any blocking wait
        # (read-idle sleep or reconnect backoff) and tear it down.
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._device: str | None = None
        # Authoritative mute preference. The bridge boots muted (LED red,
        # firmware capture silenced) so the user has to press the button
        # before any audio is recorded — and the device's mute state is
        # deterministic at startup, not whatever the previous session left
        # it in. Re-applied on every reconnect via _engage().
        self._muted_pref: bool = True

    def consume_unmute_event(self) -> bool:
        """Non-blocking check: was the mute button pressed since last call?"""
        if self._unmute_event.is_set():
            self._unmute_event.clear()
            return True
        return False

    @staticmethod
    def _find_device() -> str | None:
        for name in os.listdir("/dev"):
            if not name.startswith("hidraw"):
                continue
            path = f"/dev/{name}"
            try:
                with open(f"/sys/class/hidraw/{name}/device/uevent") as f:
                    uevent = f.read()
                if "0B0E" in uevent.upper() and "0422" in uevent:
                    return path
            except (FileNotFoundError, PermissionError):
                continue
        return None

    def _build_payload(self, muted: bool) -> bytes:
        """Output report 0x03 byte 1 = Off-Hook | (Mute if muted)."""
        b1 = self._OFFHOOK_BIT | (self._MUTE_BIT if muted else 0)
        return bytes([0x03, b1, 0x00])

    def _engage(self, fd: int) -> bool:
        """Put the Jabra into off-hook state and apply the current mute pref.

        Off-hook is required for the device to emit telephony input
        reports at all; the mute bit is bundled in so reconnects don't
        snap the device back to unmuted on top of whatever the bridge
        wanted. Returns True if the SET_REPORT write succeeded.
        """
        with self._lock:
            payload = self._build_payload(self._muted_pref)
        try:
            os.write(fd, payload)
            return True
        except OSError as exc:
            log.warning("Off-hook engage write failed: %s — buttons may stay silent", exc)
            return False

    def set_led(self, muted: bool) -> None:
        """Set device mute state (LED + firmware capture mute, both at once).

        Writing this report immediately silences/unsilences the USB-Audio
        capture endpoint at the firmware level — empirically verified
        with PCM probes. The preference is also remembered, so a USB
        unplug/replug or any future _engage() restores the bridge's
        intent rather than reverting to the device's last physical state.

        Thread-safe. No-op (but the preference is still recorded) if the
        device is not currently attached; the next successful
        _open_device() will pick up the value via _engage().
        """
        with self._lock:
            self._muted_pref = muted
            fd = self._fd
            payload = self._build_payload(muted)
        if fd is None:
            return
        try:
            os.write(fd, payload)
        except OSError as exc:
            log.warning("LED/mute write failed: %s", exc)

    def _open_device(self) -> bool:
        """Find, open O_RDWR, and engage the Jabra. Sets self._fd / self._device.

        Returns True on success. On failure (no device, open denied) leaves
        self._fd as None and returns False — the poll loop retries after
        the reconnect backoff.
        """
        path = self._find_device()
        if not path:
            return False
        try:
            fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        except OSError as exc:
            log.warning("Cannot open %s O_RDWR: %s", path, exc)
            return False
        self._engage(fd)
        self._fd = fd
        self._device = path
        return True

    def _close_fd(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def start(self) -> None:
        """Start the poll thread. Returns immediately.

        The thread acquires the device on its own — if the Jabra isn't
        plugged in yet, it parks in the reconnect backoff (zero CPU)
        until the device appears or `stop()` is called. Plug state is
        treated as runtime state, not init state, so there is no
        startup-failure path."""
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread:
            # Generous timeout: a backoff sleep can be in-flight; the
            # event wakes it immediately so the join almost always
            # returns well under a second.
            self._thread.join(timeout=2.0 + _RECONNECT_BACKOFF_S)
        self._close_fd()

    def _poll_loop(self) -> None:
        debug = os.environ.get("JABRA_HID_DEBUG", "").lower() not in ("", "0", "false")
        while not self._shutdown.is_set():
            if self._fd is None:
                if not self._open_device():
                    # Device not present (or unopenable). Park on the
                    # shutdown event for the backoff interval — kernel
                    # wait, no CPU. Returns True if stop() fired.
                    if self._shutdown.wait(_RECONNECT_BACKOFF_S):
                        return
                    continue
                log.info("HID button monitor on %s", self._device)
                # Reset held-state on every fresh attach: the device
                # always comes up with the button released, so the next
                # press is a clean rising edge.
                with self._lock:
                    self._button_down = False

            try:
                n = os.read(self._fd, 64)
            except BlockingIOError:
                # No data available — non-blocking fd, normal idle.
                if self._shutdown.wait(_READ_IDLE_S):
                    return
                continue
            except OSError as exc:
                if not self._shutdown.is_set():
                    log.warning("HID read failed: %s — reconnecting", exc)
                self._close_fd()
                continue
            if not n:
                # EOF on a hidraw fd means the device was unplugged.
                if not self._shutdown.is_set():
                    log.warning("HID read returned EOF — reconnecting")
                self._close_fd()
                continue
            if len(n) < 2:
                continue
            if debug:
                log.info("HID raw[len=%d]: %s", len(n), n[:16].hex(" "))
            if n[0] != self._FEATURE_RPT:
                continue
            is_pressed = bool(n[1] & self._BUTTON_BIT)
            with self._lock:
                was_pressed = self._button_down
                self._button_down = is_pressed
            if is_pressed and not was_pressed:
                log.info("HID: button press → wake")
                self._unmute_event.set()
