#!/usr/bin/env python3
"""Tests for jabra_hid.HidMuteMonitor.

The Jabra SPEAK 510 mute control is a *momentary* HID button: bit 4 of
byte 1 of report 0x03 (HID Telephony usage 0x2F "Mic Mute") is high
while held, low when released. A single press looks like a 1→0 sequence
on the wire, NOT a mute-state flip. The monitor must trigger on the
rising edge (press) and produce exactly one wake event per press,
regardless of releases.

Real reports also have bit 0 set permanently after engage (the device
echoes back its Hook Switch state); we mirror that in the synthetic
fixtures so the bytes match what real hardware sends.

We feed synthetic HID reports through an os.pipe() acting as
/dev/hidraw and bypass start()'s device discovery by injecting the
read end directly.

Run: .venv/bin/python test_hid_mute.py
"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jabra_hid import HidMuteMonitor  # noqa: E402


def _btn(pressed: bool) -> bytes:
    """Build a 64-byte report 0x03 with bit 4 of byte 1 = mute button state.

    Real device reports have bit 0 (Hook Switch) and bit 1 (Line Busy Tone)
    permanently set after engage; we mirror that as `0x03` baseline so the
    parser is exercised against realistic bytes (mute pressed = 0x13,
    released = 0x03).
    """
    base = 0x03
    return bytes([0x03, base | (0x10 if pressed else 0x00)]) + b"\x00" * 62


def _wait(predicate, timeout: float = 1.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class HidMuteMonitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.r, self.w = os.pipe()
        self.mon = HidMuteMonitor()
        # Inject the pipe as the already-open device fd. The poll loop
        # checks `if self._fd is None` and skips _open_device() when it's
        # already set, so the read logic runs against our pipe directly.
        self.mon._fd = self.r
        self.mon._device = "<test-pipe>"
        self.mon._thread = threading.Thread(target=self.mon._poll_loop, daemon=True)
        self.mon._thread.start()

    def tearDown(self) -> None:
        self.mon._shutdown.set()
        # Wake the blocked os.read so the loop re-checks _shutdown.
        try:
            os.write(self.w, bytes([0xFF]) + b"\x00" * 63)
        except OSError:
            pass
        self.mon._thread.join(timeout=1.0)
        for fd in (self.w, self.r):
            try:
                os.close(fd)
            except OSError:
                pass

    def _send_button(self, pressed: bool) -> None:
        os.write(self.w, _btn(pressed))

    def _press_cycle(self) -> None:
        """Simulate a real press: button down then released."""
        self._send_button(pressed=True)
        self._send_button(pressed=False)

    # -----------------------------------------------------------------------

    def test_initial_state_has_no_pending_wake_event(self) -> None:
        time.sleep(0.05)  # let the poll loop spin up
        self.assertFalse(self.mon.consume_unmute_event())

    def test_full_press_cycle_fires_exactly_one_wake_event(self) -> None:
        """The bug: old code fired UNMUTE on the release edge AND a phantom
        MUTE on the next press's down edge. Now: one event per press, period."""
        self._press_cycle()

        self.assertTrue(
            _wait(self.mon.consume_unmute_event),
            "wake event never fired on press",
        )
        time.sleep(0.05)  # give the release report time to be processed
        self.assertFalse(
            self.mon.consume_unmute_event(),
            "release after press incorrectly fired a second wake event",
        )

    def test_button_down_alone_fires_wake_event(self) -> None:
        """Wake fires on the rising edge — no need to wait for the release."""
        self._send_button(pressed=True)

        self.assertTrue(_wait(self.mon.consume_unmute_event))

    def test_release_without_prior_press_fires_nothing(self) -> None:
        self._send_button(pressed=False)
        time.sleep(0.05)

        self.assertFalse(self.mon.consume_unmute_event())

    def test_three_press_cycles_fire_three_wake_events(self) -> None:
        for cycle in range(3):
            self._press_cycle()
            self.assertTrue(
                _wait(self.mon.consume_unmute_event),
                f"cycle {cycle}: wake event missing",
            )
            time.sleep(0.02)
            self.assertFalse(
                self.mon.consume_unmute_event(),
                f"cycle {cycle}: spurious second event",
            )

    def test_repeated_button_down_reports_do_not_re_fire(self) -> None:
        """If the device sends the same 'pressed' report twice without a
        release in between, only the first 0→1 edge counts."""
        self._send_button(pressed=True)
        self.assertTrue(_wait(self.mon.consume_unmute_event))

        self._send_button(pressed=True)  # still held — no new edge
        time.sleep(0.05)
        self.assertFalse(self.mon.consume_unmute_event())

    def test_reports_with_other_ids_are_ignored(self) -> None:
        # Report ID 0x07 with bit 4 set on byte 1 — must not move state.
        os.write(self.w, bytes([0x07, 0x10]) + b"\x00" * 62)
        time.sleep(0.05)

        self.assertFalse(self.mon.consume_unmute_event())
        # And a real press still works after a noise report.
        self._press_cycle()
        self.assertTrue(_wait(self.mon.consume_unmute_event))


class HidMuteMonitorReconnectTest(unittest.TestCase):
    """When the Jabra disappears mid-poll, the monitor closes the fd and
    keeps trying to reopen — the thread does NOT exit. The bridge sees
    one continuous monitor across plug cycles. `stop()` must wake the
    backoff sleep immediately (Event-based wait, not time.sleep)."""

    def _patch_open_to_fail(self, mon: HidMuteMonitor) -> list[int]:
        """Replace _open_device on the instance so reconnect attempts
        fail without touching real /dev/hidraw* (which on the Pi might
        belong to a Jabra owned by the running production service).
        Returns a list whose length grows with every attempt."""
        attempts: list[int] = []

        def fake_open() -> bool:
            attempts.append(1)
            return False

        mon._open_device = fake_open  # type: ignore[method-assign]
        return attempts

    def test_eof_on_read_does_not_kill_thread(self) -> None:
        """Closing the writer side makes os.read return b'' — the
        disconnect signal. The loop closes the fd and re-enters the
        reconnect backoff; the thread stays alive."""
        r, w = os.pipe()
        mon = HidMuteMonitor()
        mon._fd = r
        attempts = self._patch_open_to_fail(mon)

        thread = threading.Thread(target=mon._poll_loop, daemon=True)
        thread.start()

        os.close(w)  # signals EOF to the reader
        # Give the loop time to notice EOF and call _open_device once.
        self.assertTrue(
            _wait(lambda: len(attempts) >= 1, timeout=2.0),
            "loop didn't attempt reconnect after EOF",
        )
        self.assertTrue(thread.is_alive(), "thread exited on EOF — should reconnect")

        # Stop() must wake the backoff Event-wait immediately.
        t0 = time.monotonic()
        mon._shutdown.set()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive(), "stop() did not wake the backoff")
        self.assertLess(
            time.monotonic() - t0, 0.5,
            "stop() took >0.5s — backoff is sleeping instead of waiting on Event",
        )

    def test_oserror_on_read_does_not_kill_thread(self) -> None:
        """A real I/O error (closed fd → EBADF) is treated the same as
        a USB unplug — close the fd and try to reconnect, do not exit."""
        r, w = os.pipe()
        mon = HidMuteMonitor()
        mon._fd = r
        attempts = self._patch_open_to_fail(mon)

        thread = threading.Thread(target=mon._poll_loop, daemon=True)
        thread.start()
        os.close(r)
        os.close(w)

        self.assertTrue(
            _wait(lambda: len(attempts) >= 1, timeout=2.0),
            "loop didn't attempt reconnect after OSError",
        )
        self.assertTrue(thread.is_alive(), "thread exited on OSError — should reconnect")

        mon._shutdown.set()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())

    def test_blocking_io_error_does_not_kill_thread(self) -> None:
        """EAGAIN / BlockingIOError is normal idle on a non-blocking fd
        and must NOT trigger reconnect (otherwise an idle bridge would
        churn fds in a tight loop)."""
        r, w = os.pipe()
        os.set_blocking(r, False)
        mon = HidMuteMonitor()
        mon._fd = r
        attempts = self._patch_open_to_fail(mon)

        thread = threading.Thread(target=mon._poll_loop, daemon=True)
        thread.start()
        time.sleep(0.2)  # plenty of EAGAIN spins
        self.assertEqual(attempts, [], "BlockingIOError must not trigger reconnect")
        self.assertTrue(thread.is_alive())

        mon._shutdown.set()
        os.close(w)
        thread.join(timeout=1.0)
        try:
            os.close(r)
        except OSError:
            pass

    def test_reconnect_succeeds_when_open_eventually_works(self) -> None:
        """After a disconnect, once _open_device() succeeds the read
        loop resumes processing button presses normally."""
        r1, w1 = os.pipe()
        r2, w2 = os.pipe()
        mon = HidMuteMonitor()
        mon._fd = r1

        # _open_device fails N times, then attaches r2 as the new fd —
        # simulates the device coming back after a couple of polls.
        attempts = []

        def fake_open() -> bool:
            attempts.append(1)
            if len(attempts) >= 2:
                mon._fd = r2
                mon._device = "<test-pipe-2>"
                return True
            return False

        mon._open_device = fake_open  # type: ignore[method-assign]

        thread = threading.Thread(target=mon._poll_loop, daemon=True)
        thread.start()

        os.close(w1)  # disconnect on fd1

        self.assertTrue(
            _wait(lambda: len(attempts) >= 2, timeout=6.0),
            "loop didn't reach a successful reconnect within two backoffs",
        )
        # Reconnect is in. A press on the new fd should produce a wake.
        os.write(w2, _btn(True))
        self.assertTrue(
            _wait(mon.consume_unmute_event, timeout=1.0),
            "wake event missing on reconnected fd",
        )

        mon._shutdown.set()
        os.close(w2)
        thread.join(timeout=2.0)
        try:
            os.close(r1)
        except OSError:
            pass
        try:
            os.close(r2)
        except OSError:
            pass


class HidMuteMonitorEngageTest(unittest.TestCase):
    """The off-hook engage must run before reads, with the exact bytes
    the device firmware expects. Regression-protect that wire format."""

    def test_engage_writes_off_hook_payload(self) -> None:
        r, w = os.pipe()
        try:
            ok = HidMuteMonitor._engage(w)
            self.assertTrue(ok)
            data = os.read(r, 8)
            # Output report 0x03 with byte 1 bit 0 (LED Off-Hook) set.
            self.assertEqual(data, bytes([0x03, 0x01, 0x00]))
        finally:
            os.close(r)
            os.close(w)

    def test_engage_returns_false_on_write_failure(self) -> None:
        # A closed fd makes os.write raise — _engage must catch it and
        # report False rather than crashing the start() path.
        r, w = os.pipe()
        os.close(w)
        os.close(r)
        self.assertFalse(HidMuteMonitor._engage(w))


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
