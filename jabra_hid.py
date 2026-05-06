"""Jabra SPEAK 510 HID button monitor.

Watches /dev/hidraw* for the device's mute button and surfaces presses as
edge-triggered wake events on a background daemon thread. On USB
disconnect or read failure it sets `device_lost` and exits — the bridge
is expected to notice this, finish any in-flight turn, and exit cleanly
(systemd then waits for the udev `add` rule to bring it back up when
the device is reconnected).

The mute button is a *momentary* HID Telephony button (bit 4 of byte 1
of report 0x03, usage 0x2F "Mic Mute") — high while held, low when
released. A single press emits a 1→0 sequence on the wire, so the
monitor triggers on the rising edge: one wake event per press,
regardless of the release.

The Jabra SPEAK 510 firmware does NOT emit telephony input reports until
the host writes an output report putting the device into off-hook state.
`_engage()` performs that write each time the fd is (re)opened. Without
it no mute press ever reaches userspace. See docs/JABRA.md for the full
protocol reference.

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
import time

log = logging.getLogger(__name__)


class HidMuteMonitor:
    """Detect Jabra SPEAK 510 mute-button presses via /dev/hidraw."""

    _FEATURE_RPT = 0x03
    _BUTTON_BIT = 0x10  # bit 4 of byte 1 = HID Telephony usage 0x2F (Mic Mute)
    # Output report 0x03 with byte 1 bit 0 (LED Off-Hook) set. Tells the
    # device firmware "we're in an active call" so it starts sending input
    # reports for telephony buttons. See docs/JABRA.md.
    _ENGAGE_PAYLOAD = bytes([0x03, 0x01, 0x00])

    def __init__(self) -> None:
        self._fd: int | None = None
        self._button_down: bool = False
        self._unmute_event = threading.Event()
        # Set when the read loop detects the device is gone (USB unplug,
        # power-off, ENODEV/EOF). Bridge polls this and exits cleanly.
        self._device_lost = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._device: str | None = None

    def consume_unmute_event(self) -> bool:
        """Non-blocking check: was the mute button pressed since last call?"""
        if self._unmute_event.is_set():
            self._unmute_event.clear()
            return True
        return False

    def device_lost(self) -> bool:
        """Set when the read loop has observed the Jabra disappear (USB
        unplug, power-off, ENODEV/EOF). Bridge polls this and shuts down."""
        return self._device_lost.is_set()

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

    @classmethod
    def _engage(cls, fd: int) -> bool:
        """Put the Jabra into off-hook state so telephony buttons emit reports.

        Returns True if the SET_REPORT write succeeded. A failure here is
        recoverable in principle (the read loop will still run) but in
        practice it means no mute presses will be seen.
        """
        try:
            os.write(fd, cls._ENGAGE_PAYLOAD)
            return True
        except OSError as exc:
            log.warning("Off-hook engage write failed: %s — buttons may stay silent", exc)
            return False

    def _open_device(self) -> bool:
        """Find, open O_RDWR, and engage the Jabra. Sets self._fd / self._device.

        Returns True on success. On failure (no device, open denied) leaves
        self._fd as None and returns False — the caller decides whether to
        retry or give up.
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

    def start(self) -> bool:
        """Find, open, engage the Jabra and start the read thread.

        Returns False (without starting the thread) if the device isn't
        present at startup — the caller is expected to treat that as a
        clean shutdown signal."""
        if not self._open_device():
            log.warning("No Jabra SPEAK 510 found at startup")
            return False
        self._running = True
        self._device_lost.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("HID button monitor on %s", self._device)
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _poll_loop(self) -> None:
        debug = os.environ.get("JABRA_HID_DEBUG", "").lower() not in ("", "0", "false")
        while self._running:
            try:
                n = os.read(self._fd, 64)
            except BlockingIOError:
                # No data available right now — non-blocking fd, normal idle.
                time.sleep(0.05)
                continue
            except OSError as exc:
                if self._running:
                    log.warning("HID read failed: %s — device lost", exc)
                    self._device_lost.set()
                return
            if not n:
                # EOF on a hidraw fd means the device was unplugged.
                if self._running:
                    log.warning("HID read returned EOF — device lost")
                    self._device_lost.set()
                return
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
