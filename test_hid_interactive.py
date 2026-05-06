#!/usr/bin/env python3
"""Interactive smoke test for HidMuteMonitor against a real Jabra SPEAK 510.

Unlike test_hid_mute.py (synthetic reports via os.pipe), this exercises the
actual device. Run on the target machine with the Jabra connected and follow
the prompts. Verifies:

  - The device is discoverable and openable (udev / permissions OK).
  - Each button press produces exactly one wake event.
  - No two wakes fire close enough in time to suggest a phantom duplicate
    (the symptom of a wire-protocol regression).
  - No wakes fire while the user does nothing.

Run: .venv/bin/python test_hid_interactive.py [N]   (default N=3 presses)
"""

import logging
import os
import sys
import time

# Always log raw HID bytes during the interactive test so the operator can
# correlate button presses with the actual report stream from the device.
# Set JABRA_HID_DEBUG=0 before invocation to suppress.
os.environ.setdefault("JABRA_HID_DEBUG", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jabra_hid import HidMuteMonitor  # noqa: E402

N_PRESSES_DEFAULT = 3
PRESS_TIMEOUT_S = 60.0
SPURIOUS_GUARD_S = 2.0
MIN_GAP_MS = 100  # human presses can't realistically be closer than this


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="[%(name)s] %(levelname)s %(message)s"
    )
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PRESSES_DEFAULT

    mon = HidMuteMonitor()
    if not mon.start():
        print("FAIL: no Jabra SPEAK 510 detected.", file=sys.stderr)
        print("  - `ls /dev/hidraw*` should list the device.", file=sys.stderr)
        print(
            "  - The current user needs read access (udev rule, plugdev group, ...).",
            file=sys.stderr,
        )
        return 1

    try:
        print(f"\nPress the Jabra mute button {n} times. (Ctrl+C to abort.)\n", flush=True)

        timestamps: list[float] = []
        deadline = time.monotonic() + PRESS_TIMEOUT_S
        while len(timestamps) < n and time.monotonic() < deadline:
            if mon.consume_unmute_event():
                timestamps.append(time.monotonic())
                print(f"  [{len(timestamps)}/{n}] wake event observed", flush=True)
            time.sleep(0.02)

        if len(timestamps) < n:
            print(
                f"\nFAIL: saw {len(timestamps)}/{n} presses within {PRESS_TIMEOUT_S:.0f}s.",
                file=sys.stderr,
            )
            return 1

        # Phantom-duplicate guard: human presses can't land within ~100ms.
        bad: list[tuple[int, float]] = []
        for i in range(1, n):
            gap_ms = (timestamps[i] - timestamps[i - 1]) * 1000
            if gap_ms < MIN_GAP_MS:
                bad.append((i, gap_ms))
        if bad:
            print("\nFAIL: phantom-duplicate wake events:", file=sys.stderr)
            for i, gap_ms in bad:
                print(
                    f"  - event {i + 1} fired only {gap_ms:.1f}ms after event {i}",
                    file=sys.stderr,
                )
            return 1

        print(
            f"\nHold still — watching for spurious events for {SPURIOUS_GUARD_S:.0f}s...\n",
            flush=True,
        )
        spurious = 0
        end = time.monotonic() + SPURIOUS_GUARD_S
        while time.monotonic() < end:
            if mon.consume_unmute_event():
                spurious += 1
            time.sleep(0.02)

        if spurious:
            print(
                f"FAIL: {spurious} spurious event(s) with no button activity.",
                file=sys.stderr,
            )
            return 1

        print(f"\nPASS — {n} presses, clean timing, no spurious events.")
        return 0

    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130
    finally:
        mon.stop()


if __name__ == "__main__":
    sys.exit(main())
