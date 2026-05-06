# Jabra SPEAK 510 — HID button capture reference

Practical reference for whoever picks up `jabra_hid.py` next. Captures
everything we learned diagnosing why the mute button on this specific
device went silent on Linux despite hidraw being present and other
buttons working.

## TL;DR — the operational reality

1. The Jabra SPEAK 510 (USB `0B0E:0422`) is a **HID Telephony** device.
   By default the device firmware **does NOT emit input reports for the
   mute / hook / hangup buttons**. It only emits volume / play-pause via
   the Consumer page (Report `0x01`).
2. The host must put the device into "active call" / off-hook state by
   writing an **output report** to engage the telephony state machine.
   Until that happens, pressing the mute button does literally nothing
   on the wire — no hidraw report, no input event, no ALSA control.
3. After engage, telephony input reports (Report `0x03`) start flowing
   on `/dev/hidraw0`. The mute button maps to **byte 1 bit 4** of that
   report (HID Telephony usage `0x2F` "Mic Mute", mask `0x10`). Verified
   on real hardware — see "Engage confirmed" below for raw byte capture.
4. **The mute button is momentary, not a toggle.** The device does not
   maintain or report a persistent mute state. Each press emits a 1→0
   pulse on bit 4 (button held → button released). Mute *state* must be
   tracked by the host. See "Tracking mute state".
5. The kernel `hid-input` driver doesn't surface the mute press as a
   key on `/dev/input/event*` because the descriptor flags those bits
   as `Constant Variable Relative` (item type `0x07`), and Constant
   fields are skipped by the input layer. **hidraw is the right path.**

## Device IDs / paths

| Resource | Path | Notes |
|---|---|---|
| USB IDs | vendor `0x0B0E`, product `0x0422` | match in `/sys/class/hidraw/*/device/uevent` |
| HID raw | `/dev/hidraw0` | mode `666`, opens RW without root |
| HID legacy | `/dev/usb/hiddev0` | mode `600` root-only, **don't bother** |
| Input devices | `/dev/input/event{4,5,6}` | `event4`=Consumer Control (volume+media), `event5`=ABS axis, `event6`=programmable + dial pad + LED_MUTE — none surface the mute key |
| ALSA card | `/proc/asound/card<N>/usbid` containing `0b0e:0422` | typically card 3 on the Pi |
| ALSA controls | `/dev/snd/controlC<N>` | mic mute is `numid=5 'Headset Capture Switch'` — but it does NOT change when the user presses mute (kernel doesn't bridge HID telephony → ALSA on this device) |

## Engage sequence

Open hidraw `O_RDWR`, then write the output report (Report ID `0x03`)
with the LED-Off-Hook bit set:

```python
fd = os.open("/dev/hidraw0", os.O_RDWR | os.O_NONBLOCK)
os.write(fd, bytes([0x03, 0x01, 0x00]))   # report 3 payload [bit0=Off-Hook, 0]
```

After this single write the device starts emitting input reports for
all telephony buttons. The current `start()` in `jabra_hid.py` does NOT
do this (its only existing engage attempt is a `HIDIOCSFEATURE` call
with a malformed magic number `0x401C4800` that silently fails).

Whether you also need to write again periodically to keep the device
"engaged" is unverified — the Jabra mailing list mentions that without
a softphone *acknowledging* a call the device may "time out and emit a
hangup report". For our use case we don't care about hangup; we just
want the mute button responsive. Empirically test whether writing once
at startup is enough, or whether you need to re-engage after some
timeout.

### Engage confirmed

`test_hid_engage.py` was run on real hardware: after the engage write,
each mute-button press produced exactly two reports in rapid succession
(microseconds apart) on `/dev/hidraw0`:

```
+t.0   len=3   03 13 00     ← press (byte 1 bit 4 set: Mic Mute held)
+t.0   len=3   03 03 00     ← release (bit 4 clear)
```

`byte 1 = 0x03` baseline = bits 0 and 1 set (`0x20` Hook Switch
+ `0x97` Line Busy Tone). Hook Switch high confirms the off-hook state
took effect. `0x13 = 0x03 | 0x10` = baseline + Mic Mute.

So:
- Edge-trigger on byte 1 bit 4 going `0→1` for "user pressed the button".
- Discard the `1→0` release transition (no useful information for our
  purposes; it just means the user let go).

## Input report 0x03 layout (what to parse)

Report ID byte = `0x03`. Then byte 1 contains the telephony button
state, bit-packed as Const+Var so the kernel input layer skips it but
hidraw delivers it raw:

| Bit | Mask | HID usage | Meaning |
|---|---|---|---|
| 0 | `0x01` | `0x20` Hook Switch | active call |
| 1 | `0x02` | `0x97` Line Busy Tone | |
| 2 | `0x04` | `0x2B` Speaker Phone | |
| 3 | `0x08` | `0x2A` Line | |
| 4 | `0x10` | `0x2F` Mic Mute | **← the mute button** |
| 5 | `0x20` | `0x21` Flash | |
| 6 | `0x40` | `0x24` Redial | |
| 7 | `0x80` | `0x50` Speed Dial | |

Byte 2 carries: a 4-bit programmable-button array (low nibble), the
Programmable Button 1 bit (bit 4), and 3 padding bits.

`jabra_hid.py` had `_BUTTON_BIT = 0x04` (bit 2 = Speaker Phone). That
was wrong even when it *seemed* to work — change to `0x10`.

The mute button is **momentary**: bit 4 goes 1 while the user holds the
button, 0 when released. A press emits a `1→0` sequence on the wire,
so edge-trigger on the rising 0→1 to fire one wake event per press.
Don't model the bit as a persistent mute state.

## Output report 0x03 layout (what we just wrote)

Byte 1 (LED page 0x08), 7 bits low nibble first:

| Bit | Mask | LED usage |
|---|---|---|
| 0 | `0x01` | `0x17` Off Hook |
| 1 | `0x02` | `0x1E` (vendor / unspecified) |
| 2 | `0x04` | `0x09` Mute |
| 3 | `0x08` | `0x18` Ring |
| 4 | `0x10` | `0x20` Hold |
| 5 | `0x20` | `0x21` Microphone |
| 6 | `0x40` | `0x2A` (vendor) |
| 7 | `0x80` | `0x9E` Ringer (Telephony page) |

Bits 8-15 are padding. So the engage write `[0x03, 0x01, 0x00]` sets
*only* Off-Hook LED on. To also light the Mute LED when we have an
internal mute state to mirror, write `[0x03, 0x05, 0x00]` (Off-Hook +
Mute LEDs). This is optional — useful UX, not required for button
detection.

## Tracking mute state

The Jabra mute button is a **momentary** physical switch (push-to-make,
release-to-break). Pressing it does not change a persistent flag inside
the device firmware. The device reports only "button is held now"
(bit 4 = 1) and "button is released now" (bit 4 = 0). Whatever "the mic
is currently muted" means, the host has to track and decide it.

Three patterns, pick based on what `voice-bridge.py` actually needs:

### A. Host-side toggle + LED feedback (softphone-like)

Tracks `_muted: bool` on the host, toggles on each rising edge of bit 4,
and writes the LED state back so the device's red ring mirrors it.

```python
# rising edge handler in _poll_loop, after edge detection:
self._muted = not self._muted
self._unmute_event.set()  # wake event for the bridge

# mirror to device LED
led_byte = 0x01 | (0x04 if self._muted else 0x00)  # Off-Hook | Mute LED
try:
    os.write(self._fd, bytes([0x03, led_byte, 0x00]))
except OSError:
    pass
```

This matches user expectation (LED reflects the logical state) and is
what every softphone does. Caveat: if some other process writes the
output report (we don't observed any doing it on this system, but it's
possible) the LED can drift from `_muted`. Cheap mitigation: rewrite
the LED on every received input report.

### B. Bind to ALSA mixer state

Drive the `Headset Capture Switch` (numid=5 on the Jabra card) on/off
as part of the toggle. Pros: the rest of the audio stack sees the mic
as muted (PipeWire reflects it, recording apps respect it). Cons: the
mixer is a system-wide audio concern; the bridge becomes a side-effect
producer over global audio state. Probably not worth it unless you
specifically want the mic actually muted at the OS level when the user
toggles.

### C. Don't track state at all (current `voice-bridge.py`)

For the bridge's actual flow — wake → record → STT → LLM → TTS → idle
— the bridge doesn't need to know "is the mic muted right now". Each
press just means "user wants to start a turn now". The mainloop after
the previous edits already operates this way (`consume_unmute_event()`
drives the wake; nothing reads a `muted` property). If we don't add UX
that requires the user to "be in mute mode", option C is the simplest
working baseline. Option A only matters once we want to give visual
feedback or interpret presses asymmetrically (press-during-record =
abort, press-during-idle = wake, etc.).

## Why the "obvious" paths don't work

Documented for next person who wonders "why not just use X":

- **`/dev/input/event*`** — the kernel `hid-input` driver maps
  Telephony usages 0x20/0x2F/etc. to `KEY_HOOK_SWITCH` / `KEY_MICMUTE`
  starting from kernel commit `2275ce8`(2016). However, the SPEAK 510
  descriptor flags the relevant bits as `Constant Variable Relative`
  (item type `0x07`). The input layer treats Constant items as
  reserved/padding and never registers a KEY for them. Verified by
  checking `/sys/class/input/event*/device/capabilities/key` on a
  working setup — `KEY_MUTE` (113) and `KEY_MICMUTE` (248) are absent.
- **ALSA control events on `/dev/snd/controlC<N>`** — the kernel
  doesn't bridge Telephony Mic Mute → mixer `Headset Capture Switch`
  on this kernel. We confirmed by subscribing to control events
  (`SNDRV_CTL_IOCTL_SUBSCRIBE_EVENTS` ioctl `0xC0045516`) and pressing
  mute: zero events fired. The control exists and pipewire updates it,
  but only when *software* changes mute, not when the physical button
  is pressed.
- **PulseAudio / PipeWire `pactl subscribe`** — same reason: the
  daemons don't see the press either.
- **`/dev/usb/hiddev0`** — root-only on this system, and even with
  access the engage problem still applies.
- **HIDIOCGFEATURE / HIDIOCGINPUT polling** — probably works (the
  state should be queryable on demand) but burns CPU for nothing if
  hidraw push works after engage. Not worth implementing.

## Tests in the tree

| Script | What it does |
|---|---|
| `test_hid_mute.py` | Automated unit tests for `HidMuteMonitor` over `os.pipe()`. No hardware. Covers edge detection on bit 4 and the engage write payload. |
| `test_hid_interactive.py` | Real hardware: count wake events on N presses, check timing/spurious. The end-to-end smoke test. |

The earlier diagnostic scripts (`test_hid_diagnose.py`,
`test_alsa_diagnose.py`, `test_hid_engage.py`, `test_hid_poll.py`)
were removed once their hypotheses were resolved. If something breaks
in a way that doesn't show up in the unit tests, this doc has enough
detail to rebuild a focused probe in 30 lines.

## Current implementation in `jabra_hid.py`

Reflecting what's now in code:

- `HidMuteMonitor._ENGAGE_PAYLOAD = bytes([0x03, 0x01, 0x00])` — the
  off-hook output report. Written by `_engage(fd)` once during
  `start()`, after the fd is opened in `O_RDWR`.
- `HidMuteMonitor._BUTTON_BIT = 0x10` — bit 4 of byte 1 of report
  `0x03`, the HID Telephony Mic Mute usage.
- `_engage()` swallows `OSError` and logs a warning. The read loop
  still runs even on failure, so the monitor stays alive; the operator
  sees the warning in logs and knows why presses aren't observed.
- The previous broken `HIDIOCSFEATURE` ioctl loop (`0x401C4800` magic
  number, malformed) has been removed.

What is **not** done yet, deliberately:

- **No periodic re-engage.** A single write at startup (and after each
  reconnect) is the validated minimum. If we ever observe presses
  going silent after some idle period, add a keep-alive on a timer or
  piggy-back on each received input report.
- **No LED feedback.** `voice-bridge.py` doesn't track persistent mute
  state; option C from "Tracking mute state" applies. To switch to
  option A, write `bytes([0x03, 0x05, 0x00])` to flip the Mute LED on
  alongside Off-Hook, mirroring whatever `_muted` boolean we maintain.

### Reconnect on USB unplug/replug

Verified failure mode: pulling the Jabra USB cable while the bridge runs
makes the open hidraw fd unusable; subsequent `os.read` returns EOF or
raises `OSError(ENODEV)`. The monitor handles this:

- `_poll_loop` distinguishes `BlockingIOError` (no data, normal idle)
  from `OSError` / empty read (device gone) and routes the latter to
  `_reopen()`.
- `_reopen()` closes the dead fd, resets `_button_down` to False (so a
  new fd doesn't fire a phantom press from stale state), then loops
  on `_find_device()` with `_BACKOFF_S = 2.0` seconds between attempts,
  using `_shutdown.wait()` so `stop()` can short-circuit the wait.
- On success it re-engages and the read loop resumes. Logged as
  `HID button monitor reconnected on /dev/hidraw0`.

Same path is taken when `start()` runs while the device is absent —
`_open_device()` returns False, the monitor thread starts in reconnect
mode and waits for the device to appear instead of erroring out.

## Open questions / things still unverified

Capture answers here as they're learned.

- **Engage persistence**: does the device stay "off-hook" after the
  single write at startup, or does it time out? Symptom of timeout:
  presses work for a few minutes after start then go silent.
- **Conflict with PipeWire / WirePlumber**: if a softphone or wireplumber
  module decides to also engage the device, do the engage states
  conflict? Worth testing with pipewire stopped vs running.
- **Reboot resilience**: after a cold start of the Pi, does the engage
  still work, or is there a USB enumeration race? If yes, retry with
  backoff in `start()`.
- **LED feedback in option A**: does the device respect the Mute LED
  bit (write `03 05 00`), or only the Off-Hook bit? (Some Jabra firmware
  drives the LED ring color from internal state regardless of host
  writes.) Easy to test once option A is wired up.
- **Other buttons**: this doc focuses on mute. If we ever want to use
  the call/hangup/redial buttons too, the same engage applies and they
  appear in the same report at the bit positions documented above.

## Sources

- [Re: [PATCH] HID: Remove Jabra speakerphone devices from ignore list — Linux Input mailing list](https://www.spinics.net/lists/linux-input/msg53112.html) — explains why telephony buttons are silent by default and that the host must SET_REPORT to wake them.
- [HID: Support telephony devices — Linux kernel patchwork](https://patchwork.kernel.org/project/linux-input/patch/1470214307-29441-1-git-send-email-nolsen@jabra.com/) — kernel commit (2016) that added Telephony usage → KEY_* mappings, including `0x2F → KEY_MICMUTE`, `0x17 → LED_OFF_HOOK`. Useful for understanding what the kernel *would* do if the descriptor weren't marking the bits Constant.
- [pehandersen-jabra/telephony-webhid-demo (index.html)](https://github.com/pehandersen-jabra/telephony-webhid-demo/blob/master/index.html) — Jabra's own WebHID reference. Confirms the engage pattern: write LED Off-Hook = 1, then read input reports.
- [HIDRAW kernel docs](https://docs.kernel.org/hid/hidraw.html) — for `os.write` semantics on hidraw fds and the `HIDIOC*` ioctl numbering used by `test_hid_poll.py` (`HIDIOCGFEATURE(64) = 0xC0404807`, `HIDIOCGINPUT(64) = 0x8040480A`, `HIDIOCSFEATURE(64) = 0xC0404806`).
