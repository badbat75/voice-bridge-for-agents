# Deploying the voice-bridge as a systemd service

Two files in this directory:

- `openclaw-voicebridge.service` — the unit. Runs `voice-bridge.py`
  from the in-tree `.venv`. Edit `User=` and the absolute paths before
  installing if your deployment isn't at
  `/home/openclaw/.openclaw/workspace/voice-bridge`.
- `99-openclaw-voicebridge.rules` — udev rule that triggers the
  service on `/dev/hidraw*` add for the Jabra SPEAK 510 (USB `0b0e:0422`).

The `openclaw-` prefix matches the user's sudo policy for this Pi
(passwordless `systemctl <verb> openclaw-*` for the `openclaw` user).

## Install / uninstall

```bash
./install.sh              # install (link unit + rule, reload, trigger)
./install.sh --status     # show current install state + last 20 log lines
./install.sh --uninstall  # remove links and reload
```

The script asks `sudo` for the privileged steps. It uses
`systemctl link` for the unit (creates the right symlink in
`/etc/systemd/system/` and registers it with systemd in one step) and
`ln -sf` for the udev rule (udev has no equivalent verb). Both point
back to this directory, so editing the source files is enough — no
re-copy needed, just a reload:

```bash
sudo systemctl daemon-reload                       # after editing the .service
sudo udevadm control --reload-rules                # after editing the .rules
sudo systemctl restart openclaw-voicebridge        # if it was running
```

The install also runs `udevadm trigger --subsystem-match=hidraw`,
which re-emits `add` events for already-plugged hidraw devices — so
if the Jabra is connected when you install, the unit starts
immediately without unplug/replug.

## How it behaves

- **Boot, no Jabra plugged in**: nothing runs.
- **Jabra plugged**: udev fires `SYSTEMD_WANTS=openclaw-voicebridge.service`,
  systemd starts the unit. The bridge engages off-hook and begins
  watching for button presses.
- **Jabra unplugged while bridge runs**: read on the hidraw fd raises
  EOF/`ENODEV`. The bridge logs `HID read returned EOF — reconnecting`
  and enters a 2-second backoff loop calling `_find_device()` until it
  reappears. The unit stays active.
- **Jabra re-plugged**: either the bridge's reconnect loop sees the new
  hidraw node first, or udev re-fires (the unit is already active so
  systemd no-ops). Either way the bridge re-engages.
- **Bridge crash**: `Restart=on-failure` plus `StartLimitBurst=5` over
  60s — restarts up to 5 times, then gives up to avoid a tight crash
  loop. Check `journalctl -u openclaw-voicebridge` to see why.

## Verify

```bash
systemctl status openclaw-voicebridge
journalctl -u openclaw-voicebridge -f       # watch live logs
```

You should see `[jabra_hid] INFO HID button monitor on /dev/hidraw0`
on a successful start, then `HID: button press → wake` on each press.

## Manual control

```bash
sudo systemctl start openclaw-voicebridge      # start without unplug/replug
sudo systemctl stop openclaw-voicebridge       # graceful shutdown (handles SIGTERM)
sudo systemctl enable openclaw-voicebridge     # also start at boot if Jabra is there
sudo systemctl disable openclaw-voicebridge    # remove from boot (udev still triggers)
```

## Why the bridge runs even without the device

Because USB devices come and go and re-emitting boot-time service
dependencies is fragile. The bridge's internal reconnect loop is the
authoritative state machine; udev is just a convenience starter. If you
want the unit to truly stop when the device is unplugged you'd add a
`BindsTo=` to a `.device` unit, but that adds complexity for limited
benefit — the running bridge with no device costs effectively zero CPU
(it's idle in `_shutdown.wait(2.0)` between `_find_device()` calls).
