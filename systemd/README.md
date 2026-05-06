# Deploying the voice-bridge as a (user) systemd service

Two files in this directory:

- `openclaw-voicebridge.service` — the unit. Runs `voice-bridge.py`
  from the in-tree `.venv`. **This is a user unit**: it lives under
  `~/.config/systemd/user` and is managed with `systemctl --user`,
  no `User=` directive. Edit the absolute paths in `WorkingDirectory=`
  and `ExecStart=` if your deployment isn't at
  `/home/openclaw/.openclaw/workspace/voice-bridge`.
- `99-openclaw-voicebridge.rules` — udev rule that grants the
  `plugdev` group read/write on the Jabra SPEAK 510's `/dev/hidraw*`
  node. The rule does NOT trigger the service — the bridge handles
  device attach/detach via its internal reconnect loop.

The `openclaw-` prefix matches the user's sudo policy for this Pi
(passwordless `systemctl <verb> openclaw-*` for the `openclaw` user),
though with a user unit you generally won't need sudo at all.

## Prerequisites

- The user must be in groups `plugdev` (for hidraw access via the
  udev rule) and `audio` (for ALSA / `aplay`).
- Linger must be enabled (`sudo loginctl enable-linger $USER`) so
  the user-mode systemd starts at boot without a login session.

`install.sh` checks both and prompts you with the exact `usermod`
command if a group is missing; it enables linger automatically.

## Install / uninstall

```bash
./install.sh              # install (link unit + rule, reload, enable+start)
./install.sh --status     # show current install state + last 20 log lines
./install.sh --uninstall  # remove links and reload
```

The script asks `sudo` only for the privileged steps (udev rule, linger,
and a one-time cleanup if the older system-mode install is still
present). The unit itself is linked with `systemctl --user link`, no
sudo. Both links point back to this directory, so editing the source
files is enough — no re-copy needed, just a reload:

```bash
systemctl --user daemon-reload                         # after editing the .service
sudo udevadm control --reload-rules                    # after editing the .rules
systemctl --user restart openclaw-voicebridge          # if it was running
```

The install also runs `udevadm trigger --subsystem-match=hidraw
--action=add`, which re-emits `add` events for already-plugged
hidraw devices — so the new MODE/GROUP take effect without unplug
/replug.

## How it behaves

- **Boot, no Jabra plugged in**: the unit starts (linger pulls up the
  user systemd at boot). The bridge's HID monitor parks in its
  reconnect-backoff `Event.wait()` — kernel-blocked, ~0 CPU. It tries
  `_find_device()` once every 2 s.
- **Jabra plugged**: udev sets `plugdev:0660` on the new
  `/dev/hidraw*`. Within 2 s the bridge's reconnect loop opens it,
  engages off-hook, and starts watching for button presses.
- **Jabra unplugged while bridge runs**: `os.read` returns EOF (or
  raises ENODEV); `_poll_loop` logs `HID read returned EOF —
  reconnecting`, closes the fd, and re-enters the 2 s backoff. The
  unit stays active.
- **Jabra re-plugged**: udev re-applies permissions; the reconnect
  loop opens it on the next poll (≤ 2 s).
- **Bridge crash**: `Restart=on-failure` plus `StartLimitBurst=5`
  over 60 s — restarts up to 5 times, then gives up to avoid a tight
  crash loop. Check `journalctl --user -u openclaw-voicebridge` to
  see why.

## Verify

```bash
systemctl --user status openclaw-voicebridge
journalctl --user -u openclaw-voicebridge -f      # watch live logs
```

You should see `[jabra_hid] INFO HID button monitor on /dev/hidraw0`
once the device is detected, then `HID: button press → wake` on each
press.

## Manual control

```bash
systemctl --user start openclaw-voicebridge       # start
systemctl --user stop openclaw-voicebridge        # graceful shutdown (SIGTERM)
systemctl --user enable openclaw-voicebridge      # start at boot (needs linger)
systemctl --user disable openclaw-voicebridge     # remove from boot
```

## Why a single long-running unit instead of udev-triggered

Earlier versions started the unit via `SYSTEMD_WANTS=` from the udev
rule and exited the bridge on disconnect. That mechanism only works
with system-mode systemd (udev runs as root and can't address user
units). To go user-mode without losing the "auto-start when device
is plugged" property, the bridge now does that work itself with an
in-process reconnect loop. The cost while the device is absent is
~0 CPU (Event-based wait), so a long-running unit isn't wasteful.
